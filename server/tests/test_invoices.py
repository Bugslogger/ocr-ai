"""API integration checks with real PDFs and fake inference/database boundaries."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock, patch
import json
import os

import pymupdf
from fastapi.testclient import TestClient

from AI.schema import ORDER_FIELDS, ITEM_FIELDS, NUMBER_FIELDS
from AI.tariff import CatalogRetriever
from server.app import create_app


def pdf_bytes(encrypted=False):
    with pymupdf.open() as document:
        page = document.new_page(width=72, height=72)
        page.insert_text((5, 20), "Invoice", fontsize=8)
        if encrypted:
            return document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                    owner_pw="owner", user_pw="password")
        return document.tobytes()


def extracted_result():
    order = {key: None for key in ORDER_FIELDS}
    order.update(invoice_number="INV-001", receiver_country="Finland", total_amount=20)
    item = {key: None for key in ITEM_FIELDS + NUMBER_FIELDS}
    item.update(sku_code="CABLE-1", product_description="insulated electric cable",
                number_of_units_for_this_sku=2, unit_value_for_this_sku=10,
                extended_value_for_this_sku=20)
    return {"invoices": [{"order_level": order, "item_level": [item],
                          "shipment": {"invoice_number": "INV-001", "invoice_date": None, "currency": None}}]}


class InvoiceUploadTests(TestCase):
    def setUp(self):
        self.client = TestClient(create_app())
        self.environment = patch.dict(os.environ, {"INVOICE_TARIFF_BACKEND": "weaviate",
                                                    "INVOICE_MAX_UPLOAD_BYTES": str(25 * 1024 * 1024)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(self.client.close)

    def test_upload_returns_pipeline_json_and_cleans_files(self):
        captured = {}
        expected = extracted_result()
        expected["invoices"][0]["item_level"][0].update(hs6_code="854442", needs_review=True)

        def pipeline(document_path, output_path, **options):
            captured.update(pdf=document_path, output=output_path, options=options)
            self.assertTrue(document_path.is_file())
            self.assertTrue(document_path.read_bytes().startswith(b"%PDF"))
            output_path.mkdir()
            (output_path / "partial.json").write_text("{}")
            return deepcopy(expected)

        with patch("AI.main.doc_pipeline", side_effect=pipeline) as run:
            response = self.client.post("/api/v1/invoices/extract?wait=true", files={
                "file": ("../../invoice.PDF", pdf_bytes(), "application/pdf")
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), expected)
        run.assert_called_once()
        self.assertEqual(captured["options"], {"backend": "weaviate"})
        self.assertEqual(captured["pdf"].name, "invoice.pdf")
        self.assertFalse(captured["pdf"].parent.exists())

    def test_api_runs_real_pipeline_with_fake_models(self):
        payload = extracted_result()
        payload["invoices"].append(deepcopy(payload["invoices"][0]))
        generate = Mock(side_effect=[json.dumps(payload), *[
            json.dumps({"hs6": "854442", "origin_national_code": None,
                        "destination_national_code": None, "needs_review": True,
                        "classification_reason": "Cable candidate matches", "confidence": "MEDIUM"})
            for _ in range(2)
        ]])
        ocr = Mock()
        ocr.infer_multi.return_value = ("<PAGE>Invoice item cable", 5)
        retriever = CatalogRetriever([{"hscode": "854442", "description": "Insulated electric cable"}])
        with patch.dict(os.environ, {"INVOICE_TARIFF_BACKEND": "catalog"}), patch(
            "AI.main.download_models", return_value={"ocr": {"ocr_model": ocr, "ocr_tokenizer": None}, "qwen": {}}
        ), patch("AI.main.make_generator", return_value=generate), patch(
            "AI.main.CatalogRetriever.load", return_value=retriever
        ):
            response = self.client.post("/api/v1/invoices/extract?wait=true", files={"file": ("invoice.pdf", pdf_bytes())})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["invoices"]), 2)
        for invoice in response.json()["invoices"]:
            self.assertEqual(invoice["item_level"][0]["hs6_code"], "854442")
            self.assertTrue(invoice["item_level"][0]["needs_review"])
        self.assertEqual(generate.call_count, 3)
        self.assertFalse(Path(ocr.infer_multi.call_args.kwargs["output_path"]).exists())

    def test_missing_file_is_rejected(self):
        response = self.client.post("/api/v1/invoices/extract")
        self.assertEqual(response.status_code, 422)

    def test_non_pdf_filename_rejected_before_pipeline(self):
        with patch("AI.main.doc_pipeline") as run:
            response = self.client.post("/api/v1/invoices/extract?wait=true", files={"file": ("invoice.txt", b"text")})
        self.assertEqual(response.status_code, 415)
        run.assert_not_called()

    def test_empty_and_malformed_files_rejected_before_pipeline(self):
        for content in (b"", b"not a PDF", b"%PDF-1.7\nbroken body"):
            with self.subTest(content=content), patch("AI.main.doc_pipeline") as run:
                response = self.client.post("/api/v1/invoices/extract?wait=true", files={"file": ("invoice.pdf", content)})
                self.assertEqual(response.status_code, 422, response.text)
                run.assert_not_called()

    def test_encrypted_pdf_is_rejected(self):
        with patch("AI.main.doc_pipeline") as run:
            response = self.client.post("/api/v1/invoices/extract?wait=true", files={"file": ("invoice.pdf", pdf_bytes(True))})
        self.assertEqual(response.status_code, 422)
        self.assertIn("Password", response.json()["detail"])
        run.assert_not_called()

    def test_oversized_file_is_rejected(self):
        with patch.dict(os.environ, {"INVOICE_MAX_UPLOAD_BYTES": "10"}), patch("AI.main.doc_pipeline") as run:
            response = self.client.post("/api/v1/invoices/extract?wait=true", files={"file": ("invoice.pdf", pdf_bytes())})
        self.assertEqual(response.status_code, 413)
        run.assert_not_called()

    def test_pipeline_failure_returns_502_and_cleans_files(self):
        captured = {}

        def fail(document_path, output_path, **options):
            captured["directory"] = document_path.parent
            output_path.mkdir()
            (output_path / "ocr.txt").write_text("partial invoice")
            raise RuntimeError("private failure details")

        with patch("AI.main.doc_pipeline", side_effect=fail), self.assertLogs("server.routes.invoices", level="ERROR"):
            response = self.client.post("/api/v1/invoices/extract?wait=true", files={"file": ("invoice.pdf", pdf_bytes())})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("private failure details", response.text)
        self.assertFalse(captured["directory"].exists())

    def test_invalid_ai_response_returns_502(self):
        with patch("AI.main.doc_pipeline", return_value={"invoices": []}), self.assertLogs("server.routes.invoices", level="ERROR"):
            response = self.client.post("/api/v1/invoices/extract?wait=true", files={"file": ("invoice.pdf", pdf_bytes())})
        self.assertEqual(response.status_code, 502)

    def test_openapi_and_existing_routes(self):
        self.assertEqual(self.client.get("/api/v1/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/api/v1/models/status").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/api/v1/jobs", schema["paths"])
        operation = schema["paths"]["/api/v1/invoices/extract"]["post"]
        self.assertIn("multipart/form-data", operation["requestBody"]["content"])
        self.assertIn("200", operation["responses"])


if __name__ == "__main__":
    main()
