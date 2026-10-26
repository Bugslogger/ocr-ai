"""Exercise Engine.IO/Socket.IO wire protocol through the real ASGI mount."""
from contextlib import contextmanager
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch
import json
import time
import asyncio

from fastapi.testclient import TestClient

from server.app import create_app
from server.api.socketio import InvoiceSocketIO
from server.schemas import InvoiceJobResponse
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
                self.assertFalse((Path(self.directory.name) / job_id).exists())
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

    def test_final_lookup_cleans_files_and_disconnected_lookup_keeps_them(self):
        with patch("AI.main.doc_pipeline", return_value=extracted_result()):
            from io import BytesIO
            job = self.jobs.create(BytesIO(pdf_bytes()), "invoice.pdf")
            self.terminal(job.job_id)
        folder = Path(self.directory.name) / job.job_id
        asyncio.run(self.transport.get("disconnected-client", {"job_id": job.job_id}))
        self.assertTrue((folder / "job.json").exists())
        with self.connection() as ws:
            self.send(ws, "invoice:get", {"job_id": job.job_id})
            self.assertEqual(self.event(ws, "invoice:details")["job"]["result"], extracted_result())
            self.send(ws, "gpu:get", {})
            self.event(ws, "gpu:status")
        self.assertFalse(folder.exists())

    def test_extra_arguments_return_error_and_ack_without_closing_connection(self):
        with self.connection() as ws:
            ws.send_text('421["invoice:subscribe",{"job_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},{}]')
            error = self.event(ws, "request:error")
            self.assertEqual(error["status"], 422)
            self.assertEqual(error["event"], "invoice:subscribe")
            self.assertIn("one JSON object", error["detail"])
            packet = ws.receive_text()
            self.assertTrue(packet.startswith("431"), packet)
            self.assertEqual(json.loads(packet[3:]), [error])
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

    def test_subscribe_accepts_objects_and_json_text(self):
        for as_text in (False, True):
            with self.subTest(as_text=as_text), patch("AI.main.doc_pipeline", return_value=extracted_result()):
                job = self.jobs.create(BytesIO(pdf_bytes()), "invoice.pdf")
                self.terminal(job.job_id)
                with self.connection() as ws:
                    payload = {"job_id": job.job_id, "interval": 2}
                    self.send(ws, "invoice:subscribe", json.dumps(payload) if as_text else payload)
                    subscribed = self.event(ws, "invoice:subscribed")
                    self.assertTrue(subscribed["ok"])
                    self.assertEqual(subscribed["interval"], 2)
                    self.assertEqual(subscribed["job"]["job_id"], job.job_id)
                    self.assertEqual(self.event(ws, "invoice:completed")["job"]["result"], extracted_result())

    def test_subscription_validation_identifies_invalid_fields(self):
        with self.connection() as ws:
            for payload, field in [({}, "job_id"), ({"job_id": "short"}, "job_id"),
                                   ({"job_id": "a" * 32, "interval": 0}, "interval"),
                                   ("{invalid json", "payload")]:
                self.send(ws, "invoice:subscribe", payload)
                error = self.event(ws, "request:error")
                self.assertEqual(error["status"], 422)
                self.assertEqual(error["errors"][0]["field"], field)
                self.assertNotIn("PDF", error["detail"])
            self.send(ws, "gpu:get", {})
            self.assertTrue(self.event(ws, "gpu:status")["ok"])

    def test_invalid_stored_job_is_server_error_not_payload_error(self):
        try:
            InvoiceJobResponse.model_validate({})
        except Exception as exc:
            invalid_job = exc
        with patch.object(self.jobs, "get", side_effect=invalid_job), self.assertLogs(
            "server.api.socketio", level="ERROR"
        ), self.connection() as ws:
            self.send(ws, "invoice:subscribe", {"job_id": "a" * 32, "interval": 2})
            self.assertEqual(self.event(ws, "request:error")["status"], 502)

    def test_stage_updates_push_without_waiting_for_poll_interval(self):
        started, advance, finish = Event(), Event(), Event()

        def pipeline(*args, **options):
            options["progress"]("running_ocr")
            started.set()
            if not advance.wait(5):
                raise RuntimeError("Stage change not released")
            options["progress"]("cleaning_text")
            options["progress"]("extracting_json")
            if not finish.wait(5):
                raise RuntimeError("Completion not released")
            return extracted_result()

        async def watch(job_id):
            transport = InvoiceSocketIO(jobs=self.jobs)
            stages = []
            initial, updated, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def emitted(sid, event, payload):
                if event == "invoice:progress":
                    stages.append(payload["job"]["stage"])
                    initial.set()
                    if len(stages) == 3:
                        updated.set()
                elif event == "invoice:completed":
                    completed.set()
                return True

            transport.emit = AsyncMock(side_effect=emitted)
            await transport.connect("test", {})
            try:
                await transport.subscribe("test", {"job_id": job_id, "interval": 60})
                await asyncio.wait_for(initial.wait(), timeout=2)
                advance.set()
                await asyncio.wait_for(updated.wait(), timeout=2)
                self.assertEqual(stages, ["running_ocr", "cleaning_text", "extracting_json"])
                finish.set()
                await asyncio.wait_for(completed.wait(), timeout=2)
            finally:
                await transport.close()
            self.assertEqual(self.jobs._subscribers, {})

        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            job_id = self.jobs.create(BytesIO(pdf_bytes()), "invoice.pdf").job_id
            try:
                self.assertTrue(started.wait(2))
                asyncio.run(watch(job_id))
                self.assertFalse((Path(self.directory.name) / job_id).exists())
            finally:
                advance.set()
                finish.set()
                self.terminal(job_id)

    def test_backend_polling_detects_changes_and_suppresses_duplicate_snapshots(self):
        now = datetime.now(timezone.utc)
        initial = InvoiceJobResponse(job_id="a" * 32, status="queued", stage="queued",
                                     filename="invoice.pdf", created_at=now, updated_at=now)
        final = initial.model_copy(update={"status": "succeeded", "stage": "completed",
                                           "progress_percent": 100})
        jobs = Mock(spec=InvoiceJobService)
        jobs.subscribe.return_value = initial
        # No local notification: polling first sees unchanged data, then completion.
        jobs.get.side_effect = [initial, final]
        transport = InvoiceSocketIO(jobs=jobs)
        transport.emit = AsyncMock(return_value=True)

        async def watch():
            try:
                await asyncio.wait_for(transport.job_stream("test", initial.job_id, .01), timeout=2)
            finally:
                await transport.close()

        asyncio.run(watch())
        self.assertEqual([call.args[1] for call in transport.emit.await_args_list],
                         ["invoice:progress", "invoice:completed"])
        self.assertEqual(jobs.get.call_count, 2)
        jobs.unsubscribe.assert_called_once()
        jobs.cleanup_delivered.assert_called_once_with(initial.job_id)
