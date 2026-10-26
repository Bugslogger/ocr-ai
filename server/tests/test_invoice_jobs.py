"""Background uploads finish after the HTTP response and persist final JSON."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4
import json
import os
import time
import asyncio

from fastapi.testclient import TestClient

from server.app import create_app
from server.schemas import InvoiceJobResponse, JobStatus
from server.services.invoice_jobs import InvoiceJobService
from server.tests.test_invoices import pdf_bytes, extracted_result


class InvoiceJobTests(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = InvoiceJobService(root=self.directory.name)
        self.route_patch = patch("server.routes.invoices.invoice_jobs", self.service)
        self.route_patch.start()
        self.addCleanup(self.route_patch.stop)
        socket_patch = patch("server.api.invoice_job_websocket.invoice_jobs", self.service)
        socket_patch.start()
        self.addCleanup(socket_patch.stop)
        self.client = TestClient(create_app())
        self.addCleanup(self.client.close)

    def terminal(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.service.get(job_id)
            if job.status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
                return job
            time.sleep(0.02)
        self.fail("Invoice job did not finish")

    def submit(self, content=None, filename="invoice.pdf"):
        return self.client.post("/api/v1/invoices/jobs", files={"file": (filename, content if content is not None else pdf_bytes())})

    def test_202_and_polling_do_not_wait_for_inference(self):
        started = Event()
        release = Event()
        expected = extracted_result()

        def pipeline(document, output, **options):
            options["progress"]("running_ocr")
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release inference")
            return expected

        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            try:
                response = self.submit()
                self.assertEqual(response.status_code, 202, response.text)
                job_id = response.json()["job_id"]
                self.assertTrue(started.wait(2))
                self.assertFalse(release.is_set())
                self.assertEqual(response.headers["location"], "/api/v1/invoices/jobs/"+job_id)
                progress = self.client.get(response.headers["location"])
                self.assertEqual(progress.status_code, 200)
                self.assertEqual(progress.json()["status"], "processing")
                self.assertEqual(progress.json()["stage"], "running_ocr")
                self.assertEqual(progress.json()["progress_percent"], 20)
                self.assertEqual(progress.json()["queue_position"], 0)
                self.assertIsNone(progress.json()["result"])
                self.assertEqual(progress.headers["cache-control"], "no-store")
            finally:
                release.set()
            final = self.terminal(job_id)
        self.assertEqual(final.status, JobStatus.SUCCEEDED)
        self.assertEqual(final.progress_percent, 100)
        self.assertIsNone(final.queue_position)
        self.assertEqual(final.result.model_dump(), expected)
        self.assertFalse((Path(self.directory.name)/job_id/"invoice.pdf").exists())
        # Completed results remain available after recreating the registry.
        restored = InvoiceJobService(root=self.directory.name).get(job_id)
        self.assertEqual(restored.result.model_dump(), expected)

    def test_extract_returns_job_without_waiting_for_inference(self):
        started, release = Event(), Event()

        def pipeline(*args, **options):
            options["progress"]("running_ocr")
            started.set()
            if not release.wait(5):
                raise RuntimeError("Inference not released")
            return extracted_result()

        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            job_id = None
            try:
                response = self.client.post("/api/v1/invoices/extract", files={"file": ("invoice.pdf", pdf_bytes())})
                self.assertEqual(response.status_code, 202, response.text)
                job_id = response.json()["job_id"]
                self.assertTrue(started.wait(2))
                self.assertFalse(release.is_set())
                self.assertEqual(response.headers["location"], f"/api/v1/invoices/jobs/{job_id}")
                self.assertEqual(response.headers["retry-after"], "2")
                with self.client.websocket_connect(f"/ws/invoices/jobs/{job_id}") as socket:
                    self.assertEqual(socket.receive_json()["job"]["stage"], "running_ocr")
            finally:
                release.set()
                if job_id:
                    self.terminal(job_id)
        self.assertEqual(self.client.get(response.headers["location"]).json()["result"], extracted_result())

    def test_final_http_response_removes_only_delivered_job_files(self):
        with patch("AI.main.doc_pipeline", return_value=extracted_result()):
            first = self.submit().json()["job_id"]
            second = self.submit().json()["job_id"]
            self.terminal(first)
            self.terminal(second)
        first_folder = Path(self.directory.name) / first
        second_folder = Path(self.directory.name) / second
        self.assertTrue((first_folder / "job.json").exists())
        (first_folder / "extra-artifact.txt").write_text("private invoice data")
        response = self.client.get(f"/api/v1/invoices/jobs/{first}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"], extracted_result())
        self.assertFalse(first_folder.exists())
        self.assertTrue((second_folder / "job.json").exists())
        # Current-process reconnects work without recreating files on disk.
        self.assertEqual(self.client.get(f"/api/v1/invoices/jobs/{first}").json()["result"], extracted_result())
        self.assertFalse(first_folder.exists())
        with self.assertRaises(FileNotFoundError):
            InvoiceJobService(root=self.directory.name).get(first)

    def test_cleanup_keeps_processing_and_failed_jobs(self):
        started, release = Event(), Event()

        def pipeline(*args, **options):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Inference not released")
            raise RuntimeError("Extraction failed")

        with patch("AI.main.doc_pipeline", side_effect=pipeline), self.assertLogs(
            "server.services.invoice_jobs", level="ERROR"
        ):
            job_id = self.submit().json()["job_id"]
            folder = Path(self.directory.name) / job_id
            try:
                self.assertTrue(started.wait(2))
                self.service.cleanup_delivered(job_id)
                self.assertTrue((folder / "invoice.pdf").exists())
            finally:
                release.set()
                self.terminal(job_id)
            self.service.cleanup_delivered(job_id)
            self.assertTrue((folder / "job.json").exists())
            self.assertEqual(self.client.get(f"/api/v1/invoices/jobs/{job_id}").json()["status"], "failed")
            self.assertTrue((folder / "job.json").exists())

    def test_failed_websocket_send_preserves_result_for_retry(self):
        from server.api.invoice_job_websocket import invoice_job_events

        with patch("AI.main.doc_pipeline", return_value=extracted_result()):
            job_id = self.submit().json()["job_id"]
            self.terminal(job_id)
        socket = AsyncMock()
        socket.send_json.side_effect = OSError("Disconnected before final result")
        asyncio.run(invoice_job_events(socket, job_id))
        self.assertTrue((Path(self.directory.name) / job_id / "job.json").exists())
        self.assertEqual(self.service.get(job_id).result.model_dump(), extracted_result())

    def test_cleanup_failure_keeps_result_readable_and_can_retry(self):
        with patch("AI.main.doc_pipeline", return_value=extracted_result()):
            job_id = self.submit().json()["job_id"]
            self.terminal(job_id)
        with patch("server.services.invoice_jobs.shutil.rmtree", side_effect=OSError("Busy")), self.assertLogs(
            "server.services.invoice_jobs", level="ERROR"
        ):
            self.service.cleanup_delivered(job_id)
        self.assertEqual(self.service.get(job_id).result.model_dump(), extracted_result())
        self.service.cleanup_delivered(job_id)
        self.service.cleanup_delivered(job_id)
        self.assertFalse((Path(self.directory.name) / job_id).exists())

    def test_extract_default_validates_before_acceptance(self):
        for filename, content, status in [("invoice.txt", b"text", 415), ("invoice.pdf", b"broken", 422)]:
            with self.subTest(filename=filename), patch("AI.main.doc_pipeline") as run:
                response = self.client.post("/api/v1/invoices/extract", files={"file": (filename, content)})
                self.assertEqual(response.status_code, status)
                run.assert_not_called()

    def test_pipeline_failure_is_pollable_and_upload_is_deleted(self):
        with patch("AI.main.doc_pipeline", side_effect=RuntimeError("private credentials")), self.assertLogs(
            "server.services.invoice_jobs", level="ERROR"
        ):
            response = self.submit()
            self.assertEqual(response.status_code, 202)
            final = self.terminal(response.json()["job_id"])
        self.assertEqual(final.status, JobStatus.FAILED)
        self.assertIsNone(final.result)
        self.assertNotIn("private credentials", final.error)
        self.assertFalse((Path(self.directory.name)/final.job_id/"invoice.pdf").exists())

    def test_invalid_ai_response_is_failed_job(self):
        with patch("AI.main.doc_pipeline", return_value={"invoices": []}), self.assertLogs(
            "server.services.invoice_jobs", level="ERROR"
        ):
            response = self.submit()
            self.assertEqual(self.terminal(response.json()["job_id"]).status, JobStatus.FAILED)

    def test_upload_validation_occurs_before_job_is_accepted(self):
        for name, content, status in [("invoice.txt", b"text", 415), ("invoice.pdf", b"", 422),
                                      ("invoice.pdf", b"not PDF", 422), ("invoice.pdf", pdf_bytes(True), 422)]:
            with self.subTest(filename=name, status=status), patch("AI.main.doc_pipeline") as run:
                response = self.submit(content, name)
                self.assertEqual(response.status_code, status, response.text)
                run.assert_not_called()
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_oversize_upload_is_not_queued(self):
        with patch.dict(os.environ, {"INVOICE_MAX_UPLOAD_BYTES": "10"}):
            self.assertEqual(self.submit().status_code, 413)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_missing_file_and_unknown_or_invalid_job_id(self):
        self.assertEqual(self.client.post("/api/v1/invoices/jobs").status_code, 422)
        for job_id in [uuid4().hex, "not-an-id"]:
            self.assertEqual(self.client.get("/api/v1/invoices/jobs/"+job_id).status_code, 404)

    def test_restarted_worker_is_marked_interrupted(self):
        job_id = uuid4().hex
        folder = Path(self.directory.name)/job_id
        folder.mkdir()
        (folder/"invoice.pdf").write_bytes(pdf_bytes())
        now = datetime.now(timezone.utc)
        state = InvoiceJobResponse(job_id=job_id, status=JobStatus.PROCESSING, stage="running_ocr",
                                   filename="invoice.pdf", created_at=now, updated_at=now).model_dump(mode="json")
        state.update(_owner_pid=os.getpid(), _owner_instance="previous-instance")
        self.service._save(job_id, state)
        response = self.client.get("/api/v1/invoices/jobs/"+job_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "failed")
        self.assertEqual(response.json()["stage"], "interrupted")
        self.assertFalse((folder/"invoice.pdf").exists())

    def test_queue_is_serial_and_another_request_can_be_accepted(self):
        first_started = Event()
        release = Event()
        calls = []

        def pipeline(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                first_started.set()
                if not release.wait(5):
                    raise RuntimeError("Test did not release first job")
            return extracted_result()

        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            try:
                first = self.submit().json()["job_id"]
                self.assertTrue(first_started.wait(2))
                response = self.submit()
                self.assertEqual(response.status_code, 202)
                second = response.json()["job_id"]
                self.assertEqual(self.service.get(second).status, JobStatus.QUEUED)
                self.assertEqual(self.service.get(second).queue_position, 1)
                third = self.submit().json()["job_id"]
                self.assertEqual(self.service.get(third).queue_position, 2)
                self.assertEqual(len(calls), 1)
            finally:
                release.set()
            self.assertEqual(self.terminal(first).status, JobStatus.SUCCEEDED)
            self.assertEqual(self.terminal(second).status, JobStatus.SUCCEEDED)
            self.assertEqual(self.terminal(third).status, JobStatus.SUCCEEDED)

    def test_failed_ocr_does_not_poison_next_queued_invoice(self):
        from AI.utils import extract_doc

        started, release = Event(), Event()
        model = Mock()
        model.config = SimpleNamespace(sliding_window=128)

        def infer(*args, **kwargs):
            self.assertEqual(model.config.sliding_window, 128)
            self.assertFalse(hasattr(model.config, "_ring_window"))
            model.config.sliding_window = None
            model.config._ring_window = 128
            if model.infer_multi.call_count == 1:
                started.set()
                if not release.wait(5):
                    raise RuntimeError("Test did not release first invoice")
                raise RuntimeError("Injected OCR generation failure")
            return "Second invoice text", 3

        model.infer_multi.side_effect = infer

        def pipeline(pdf, output, **options):
            options["progress"]("running_ocr")
            extract_doc(pdf, output, {"ocr_model": model, "ocr_tokenizer": None})
            return extracted_result()

        with patch("AI.main.doc_pipeline", side_effect=pipeline), self.assertLogs(
            "server.services.invoice_jobs", level="ERROR"
        ):
            try:
                first = self.submit().json()["job_id"]
                self.assertTrue(started.wait(2))
                second = self.submit().json()["job_id"]
                self.assertEqual(self.service.get(second).status, JobStatus.QUEUED)
            finally:
                release.set()
            self.assertEqual(self.terminal(first).status, JobStatus.FAILED)
            final = self.terminal(second)
            self.assertEqual(final.status, JobStatus.SUCCEEDED)
            self.assertEqual(final.result.model_dump(), extracted_result())
        self.assertEqual(model.infer_multi.call_count, 2)
        self.assertEqual(model.config.sliding_window, 128)
        self.assertFalse(hasattr(model.config, "_ring_window"))

    def test_job_socket_streams_progress_and_persisted_result_on_reconnect(self):
        started, release = Event(), Event()

        def pipeline(*args, **options):
            options["progress"]("extracting_json")
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release inference")
            return extracted_result()

        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            job_id = self.submit().json()["job_id"]
            try:
                self.assertTrue(started.wait(2))
                with self.client.websocket_connect(f"/ws/invoices/jobs/{job_id}") as socket:
                    message = socket.receive_json()
                    self.assertEqual(message["event"], "progress")
                    self.assertEqual(message["job"]["job_id"], job_id)
                    self.assertEqual(message["job"]["progress_percent"], 55)
                    self.assertTrue(message["job"]["progress_is_estimate"])
                    self.assertIn("structured", message["job"]["stage_description"])
                    release.set()
                    self.terminal(job_id)
                    message = socket.receive_json()
                    self.assertEqual(message["event"], "completed")
                    self.assertEqual(message["job"]["result"], extracted_result())
            finally:
                release.set()
                self.terminal(job_id)
        with self.client.websocket_connect(f"/ws/invoices/jobs/{job_id}") as socket:
            message = socket.receive_json()
            self.assertEqual(message["event"], "completed")
            self.assertEqual(message["job"]["progress_percent"], 100)
        self.assertFalse((Path(self.directory.name) / job_id).exists())

    def test_socket_disconnect_leaves_job_running_and_can_reconnect(self):
        started, release = Event(), Event()

        def pipeline(*args, **options):
            options["progress"]("running_ocr")
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release inference")
            return extracted_result()

        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            job_id = self.submit().json()["job_id"]
            try:
                self.assertTrue(started.wait(2))
                with self.client.websocket_connect(f"/ws/invoices/jobs/{job_id}") as socket:
                    self.assertEqual(socket.receive_json()["event"], "progress")
                self.assertEqual(self.service.get(job_id).status, JobStatus.PROCESSING)
                with self.client.websocket_connect(f"/ws/invoices/jobs/{job_id}") as socket:
                    socket.receive_json()
                    socket.send_text("unexpected")
                    self.assertEqual(socket.receive_json()["event"], "error")
                self.assertEqual(self.service.get(job_id).status, JobStatus.PROCESSING)
            finally:
                release.set()
                self.terminal(job_id)

    def test_socket_unknown_job_and_failed_job(self):
        with self.client.websocket_connect("/ws/invoices/jobs/not-an-id") as socket:
            message = socket.receive_json()
            self.assertEqual(message["status"], 404)
        with patch("AI.main.doc_pipeline", side_effect=RuntimeError("private details")), self.assertLogs(
            "server.services.invoice_jobs", level="ERROR"
        ):
            job_id = self.submit().json()["job_id"]
            self.terminal(job_id)
        with self.client.websocket_connect(f"/ws/invoices/jobs/{job_id}") as socket:
            message = socket.receive_json()
            self.assertEqual(message["event"], "failed")
            self.assertNotIn("private details", message["job"]["error"])
            self.assertLess(message["job"]["progress_percent"], 100)

    def test_socket_queue_position_advances_when_job_ahead_starts(self):
        first_started, second_started = Event(), Event()
        release_first, release_second = Event(), Event()
        calls = []

        def pipeline(*args, **options):
            calls.append(True)
            if len(calls) == 1:
                first_started.set()
                if not release_first.wait(5):
                    raise RuntimeError("First job not released")
            elif len(calls) == 2:
                second_started.set()
                if not release_second.wait(5):
                    raise RuntimeError("Second job not released")
            return extracted_result()

        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            first = self.submit().json()["job_id"]
            try:
                self.assertTrue(first_started.wait(2))
                second = self.submit().json()["job_id"]
                third = self.submit().json()["job_id"]
                with self.client.websocket_connect(f"/ws/invoices/jobs/{third}") as socket:
                    self.assertEqual(socket.receive_json()["job"]["queue_position"], 2)
                    release_first.set()
                    self.assertTrue(second_started.wait(2))
                    self.assertEqual(socket.receive_json()["job"]["queue_position"], 1)
                    detail = self.client.get(f"/api/v1/invoices/jobs/{third}").json()
                    self.assertEqual(detail["queue_position"], 1)
            finally:
                release_first.set()
                release_second.set()
                self.terminal(first)
                if "second" in locals():
                    self.terminal(second)
                if "third" in locals():
                    self.terminal(third)


if __name__ == "__main__":
    main()
