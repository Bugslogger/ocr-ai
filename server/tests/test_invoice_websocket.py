"""WebSocket PDF protocol and shared pipeline integration tests."""
from copy import deepcopy
from pathlib import Path
from tempfile import SpooledTemporaryFile
from unittest import TestCase, main
from unittest.mock import Mock, patch
import json
import os

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from AI.tariff import CatalogRetriever
from server.app import create_app
from server.tests.test_invoices import pdf_bytes, extracted_result


class InvoiceWebSocketTests(TestCase):
    def setUp(self):
        self.client = TestClient(create_app())
        self.addCleanup(self.client.close)
        env = patch.dict(os.environ, {"INVOICE_TARIFF_BACKEND": "weaviate",
                                      "INVOICE_MAX_UPLOAD_BYTES": str(25 * 1024 * 1024)})
        env.start()
        self.addCleanup(env.stop)

    def receive_terminal(self, socket):
        events = []
        while True:
            message = socket.receive_json()
            events.append(message)
            if message["event"] in {"completed", "error"}:
                return events

    def assert_close(self, socket, code):
        with self.assertRaises(WebSocketDisconnect) as closed:
            socket.receive_json()
        self.assertEqual(closed.exception.code, code)

    def test_chunked_upload_returns_same_json_as_http_and_reports_progress(self):
        expected = extracted_result()
        captured = []
        data = pdf_bytes()

        def pipeline(document_path, output_path, **options):
            captured.append(document_path.parent)
            self.assertEqual(document_path.read_bytes(), data)
            self.assertEqual(options["backend"], "weaviate")
            if "progress" in options:
                options["progress"]("running_ocr")
                options["progress"]("extracting_json")
            return deepcopy(expected)

        with patch("AI.main.doc_pipeline", side_effect=pipeline):
            http_result = self.client.post("/api/v1/invoices/extract", files={"file": ("invoice.pdf", data)})
            with self.client.websocket_connect("/ws/invoices/extract") as socket:
                socket.send_json({"filename": "../../invoice.PDF", "size": len(data)})
                self.assertEqual(socket.receive_json()["event"], "ready")
                for offset in range(0, len(data), 128):
                    socket.send_bytes(data[offset:offset + 128])
                events = self.receive_terminal(socket)
                self.assertEqual(events[0], {"event": "processing"})
                self.assertEqual([e["stage"] for e in events if e["event"] == "progress"], ["running_ocr", "extracting_json"])
                self.assertEqual(events[-1], {"event": "completed", "result": http_result.json()})
                self.assert_close(socket, 1000)
        self.assertEqual(len(captured), 2)
        self.assertNotEqual(captured[0], captured[1])
        self.assertTrue(all(not path.exists() for path in captured))

    def test_runs_real_pipeline_with_fake_models(self):
        payload = extracted_result()
        generate = Mock(side_effect=[json.dumps(payload), json.dumps({
            "hs6": "854442", "needs_review": True, "confidence": "MEDIUM",
            "classification_reason": "Cable candidate", "origin_national_code": None,
            "destination_national_code": None,
        })])
        ocr = Mock()
        ocr.infer_multi.return_value = ("<PAGE>Invoice cable", 3)
        retriever = CatalogRetriever([{"hscode": "854442", "description": "Insulated electric cable"}])
        data = pdf_bytes()
        with patch.dict(os.environ, {"INVOICE_TARIFF_BACKEND": "catalog"}), patch(
            "AI.main.download_models", return_value={"ocr": {"ocr_model": ocr, "ocr_tokenizer": None}, "qwen": {}}
        ), patch("AI.main.make_generator", return_value=generate), patch(
            "AI.main.CatalogRetriever.load", return_value=retriever
        ):
            with self.client.websocket_connect("/ws/invoices/extract") as socket:
                socket.send_json({"filename": "invoice.pdf", "size": len(data)})
                self.assertEqual(socket.receive_json()["event"], "ready")
                socket.send_bytes(data)
                events = self.receive_terminal(socket)
                self.assertEqual(events[-1]["event"], "completed")
                self.assertEqual(events[-1]["result"]["invoices"][0]["item_level"][0]["hs6_code"], "854442")
                stages = [e["stage"] for e in events if e["event"] == "progress"]
                self.assertIn("running_ocr", stages)
                self.assertIn("classifying_items", stages)
        self.assertFalse(Path(ocr.infer_multi.call_args.kwargs["output_path"]).exists())

    def test_invalid_metadata(self):
        for value in ["not json", "[]", '{}', '{"filename":"invoice.pdf","size":0}',
                      '{"filename":"invoice.pdf","size":true}', '{"filename":"invoice.pdf","size":"123"}']:
            with self.subTest(value=value), patch("AI.main.doc_pipeline") as pipeline:
                with self.client.websocket_connect("/ws/invoices/extract") as socket:
                    socket.send_text(value)
                    self.assertEqual(socket.receive_json()["status"], 422)
                    self.assert_close(socket, 1008)
                pipeline.assert_not_called()

    def test_binary_before_metadata_rejected(self):
        with self.client.websocket_connect("/ws/invoices/extract") as socket:
            socket.send_bytes(pdf_bytes())
            self.assertEqual(socket.receive_json()["status"], 422)
            self.assert_close(socket, 1008)

    def test_non_pdf_filename_rejected(self):
        with self.client.websocket_connect("/ws/invoices/extract") as socket:
            socket.send_json({"filename": "invoice.txt", "size": 100})
            self.assertEqual(socket.receive_json()["status"], 415)
            self.assert_close(socket, 1008)

    def test_declared_size_over_limit_rejected_before_ready(self):
        with patch.dict(os.environ, {"INVOICE_MAX_UPLOAD_BYTES": "10"}):
            with self.client.websocket_connect("/ws/invoices/extract") as socket:
                socket.send_json({"filename": "invoice.pdf", "size": 11})
                self.assertEqual(socket.receive_json()["status"], 413)
                self.assert_close(socket, 1009)

    def test_actual_size_over_limit_rejected(self):
        with patch.dict(os.environ, {"INVOICE_MAX_UPLOAD_BYTES": "10"}):
            with self.client.websocket_connect("/ws/invoices/extract") as socket:
                socket.send_json({"filename": "invoice.pdf", "size": 10})
                socket.receive_json()
                socket.send_bytes(b"X" * 11)
                self.assertEqual(socket.receive_json()["status"], 413)
                self.assert_close(socket, 1009)

    def test_binary_length_over_declared_size_rejected(self):
        with self.client.websocket_connect("/ws/invoices/extract") as socket:
            socket.send_json({"filename": "invoice.pdf", "size": 5})
            socket.receive_json()
            socket.send_bytes(b"X" * 6)
            self.assertEqual(socket.receive_json()["status"], 422)

    def test_text_or_empty_messages_during_upload_rejected(self):
        for text in (True, False):
            with self.subTest(text=text):
                with self.client.websocket_connect("/ws/invoices/extract") as socket:
                    socket.send_json({"filename": "invoice.pdf", "size": 5})
                    socket.receive_json()
                    if text:
                        socket.send_text("invoice data")
                    else:
                        socket.send_bytes(b"")
                    self.assertEqual(socket.receive_json()["status"], 422)

    def test_invalid_or_encrypted_pdf_rejected_before_pipeline(self):
        for data in [b"not PDF", pdf_bytes(encrypted=True)]:
            with self.subTest(encrypted=data.startswith(b"%PDF")), patch("AI.main.doc_pipeline") as pipeline:
                with self.client.websocket_connect("/ws/invoices/extract") as socket:
                    socket.send_json({"filename": "invoice.pdf", "size": len(data)})
                    socket.receive_json()
                    socket.send_bytes(data)
                    self.assertEqual(self.receive_terminal(socket)[-1]["status"], 422)
                pipeline.assert_not_called()

    def test_pipeline_error_closes_and_cleans_upload(self):
        captured = {}
        data = pdf_bytes()

        def fail(document_path, output_path, **options):
            captured["folder"] = document_path.parent
            raise RuntimeError("private error detail")

        with patch("AI.main.doc_pipeline", side_effect=fail), self.assertLogs("server.api.invoice_websocket", level="ERROR"):
            with self.client.websocket_connect("/ws/invoices/extract") as socket:
                socket.send_json({"filename": "invoice.pdf", "size": len(data)})
                socket.receive_json()
                socket.send_bytes(data)
                failure = self.receive_terminal(socket)[-1]
                self.assertEqual(failure["status"], 502)
                self.assertNotIn("private error detail", failure["detail"])
                self.assert_close(socket, 1011)
        self.assertFalse(captured["folder"].exists())

    def test_disconnect_mid_upload_closes_spooled_file(self):
        streams = []

        def spool(*args, **kwargs):
            stream = SpooledTemporaryFile(*args, **kwargs)
            streams.append(stream)
            return stream

        with patch("server.api.invoice_websocket.SpooledTemporaryFile", side_effect=spool), patch("AI.main.doc_pipeline") as pipeline:
            with self.client.websocket_connect("/ws/invoices/extract") as socket:
                socket.send_json({"filename": "invoice.pdf", "size": 100})
                socket.receive_json()
                socket.send_bytes(b"%PDF")
                socket.close()
            pipeline.assert_not_called()
        self.assertEqual(len(streams), 1)
        self.assertTrue(streams[0].closed)


if __name__ == "__main__":
    main()
