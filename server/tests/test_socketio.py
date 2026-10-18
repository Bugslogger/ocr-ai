"""Exercise Engine.IO/Socket.IO wire protocol through the real ASGI mount."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import Mock, patch
import json
import time

from fastapi.testclient import TestClient

from server.app import create_app
from server.services.invoice_jobs import InvoiceJobService
from server.tests.test_invoices import pdf_bytes, extracted_result


class SocketIOTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.jobs = InvoiceJobService(root=self.directory.name)
        self.app = create_app()
        self.transport = self.app.state.socketio
        self.transport.jobs = self.jobs
        self.transport.gpu = Mock()
        self.transport.gpu.snapshot.return_value.model_dump.return_value = {"available": True}
        self.client = self.enterContext(TestClient(self.app))

    @contextmanager
    def connection(self):
        with self.client.websocket_connect("/socket.io/?EIO=4&transport=websocket", headers={"upgrade": "websocket", "connection": "Upgrade"}) as ws:
            opening = ws.receive_text()
            self.assertTrue(opening.startswith("0"), opening)
            ws.send_text("40")
            while True:
                packet = ws.receive_text()
                if packet.startswith("40"):
                    break
            try:
                yield ws
            finally:
                ws.send_text("41")

    def event(self, ws, name):
        for _ in range(30):
            packet = ws.receive_text()
            if packet == "2":
                ws.send_text("3")
            elif packet.startswith("42"):
                event, payload = json.loads(packet[2:])
                if event == name:
                    return payload
        self.fail("Expected event " + name)

    def send(self, ws, name, data):
        ws.send_text("42" + json.dumps([name, data]))

    def terminal(self, job_id):
        for _ in range(250):
            if self.jobs.get(job_id).status in {"succeeded", "failed"}:
                return
            time.sleep(.02)
        self.fail("Job failed to finish")

    def test_polling_handshake_and_existing_http_routes(self):
        response = self.client.get("/socket.io/?EIO=4&transport=polling")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.text.startswith("0"))
        self.assertEqual(self.client.get("/api/v1/health").json(), {"status": "ok"})

    def test_binary_upload_progress_gpu_and_reconnect_on_one_connection(self):
        started, release = Event(), Event()

        def pipeline(*args, **options):
            options["progress"]("running_ocr")
            started.set()
            if not release.wait(5):
                raise RuntimeError("Inference not released")
            return extracted_result()

        job_id = None
        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            try:
                with self.connection() as ws:
                    ws.send_text('451-["invoice:submit",{"filename":"invoice.pdf","pdf":{"_placeholder":true,"num":0}}]')
                    ws.send_bytes(pdf_bytes())
                    accepted = self.event(ws, "invoice:accepted")
                    job_id = accepted["job"]["job_id"]
                    self.assertTrue(started.wait(2))
                    self.assertEqual(self.event(ws, "invoice:progress")["job"]["job_id"], job_id)
                    self.send(ws, "gpu:subscribe", {"interval": 1})
                    self.assertTrue(self.event(ws, "gpu:status")["data"]["available"])
                    self.send(ws, "invoice:get", {"job_id": job_id})
                    details = self.event(ws, "invoice:details")["job"]
                    self.assertEqual(details["stage"], "running_ocr")
                    self.assertEqual(details["queue_position"], 0)
                    self.assertEqual(details["progress_percent"], 20)
                self.assertEqual(self.jobs.get(job_id).status, "processing")
                release.set()
                self.terminal(job_id)
                with self.connection() as ws:
                    self.send(ws, "invoice:subscribe", {"job_id": job_id})
                    final = self.event(ws, "invoice:completed")["job"]
                    self.assertEqual(final["result"], extracted_result())
                    self.assertEqual(final["progress_percent"], 100)
                    # Terminal invoice events leave the connection available for GPU events.
                    self.send(ws, "gpu:get", {})
                    self.assertTrue(self.event(ws, "gpu:status")["data"]["available"])
            finally:
                release.set()
                if job_id:
                    self.terminal(job_id)

    def test_validation_errors_and_missing_job_do_not_close_connection(self):
        with self.connection() as ws:
            for event, data, status in [
                ("invoice:submit", {"filename": "invoice.pdf", "pdf": "base64-is-not-binary"}, 422),
                ("invoice:get", {"job_id": "a" * 32}, 404),
                ("gpu:subscribe", {"interval": 0}, 422),
            ]:
                self.send(ws, event, data)
                error = self.event(ws, "request:error")
                self.assertEqual(error["status"], status)
                self.assertEqual(error["event"], event)
            self.send(ws, "gpu:get", {})
            self.assertTrue(self.event(ws, "gpu:status")["ok"])

    def test_acknowledgement_and_unsubscribe(self):
        with self.connection() as ws:
            ws.send_text('421["gpu:subscribe",{"interval":1}]')
            while True:
                packet = ws.receive_text()
                if packet.startswith("431"):
                    self.assertTrue(json.loads(packet[3:])[0]["ok"])
                    break
            ws.send_text('422["gpu:unsubscribe",{}]')
            while True:
                packet = ws.receive_text()
                if packet.startswith("432"):
                    self.assertTrue(json.loads(packet[3:])[0]["ok"])
                    break
            self.assertTrue(all(not session["tasks"] for session in self.transport.sessions.values()))
