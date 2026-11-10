"""Hybrid native-text/OCR routing with real PDF pages and fake inference."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import pymupdf

from AI.config import Oconfig
from AI.utils import extract_doc, clean_extracted_text


NATIVE_TEXT = "Invoice INV-001 Electric cable quantity 2 total USD 20.00 receiver Finland"


def add_native_page(document):
    page = document.new_page()
    page.insert_text((40, 60), NATIVE_TEXT, fontsize=10)
    return page


def scan_image():
    from PIL import Image
    stream = BytesIO()
    Image.new("RGB", (600, 800), "white").save(stream, format="PNG")
    return stream.getvalue()


class PDFTextTests(TestCase):
    def setUp(self):
        config = patch.dict(Oconfig, {"pdf_text_first": True, "pdf_text_min_chars": 40})
        config.start()
        self.addCleanup(config.stop)

    def test_native_pdf_skips_rendering_and_model_loading(self):
        with TemporaryDirectory() as directory:
            pdf = Path(directory) / "native.pdf"
            with pymupdf.open() as document:
                add_native_page(document)
                document.save(pdf)
            with patch("AI.model.download_ocr") as load, patch("AI.utils.pdf_to_images") as render:
                extract_doc(pdf, directory)
            load.assert_not_called()
            render.assert_not_called()
            text = clean_extracted_text(directory)
            self.assertIn("INV-001", text)
            self.assertIn("20.00", text)
            self.assertIn("\t", text)

    def test_mixed_document_only_renders_scanned_page_and_preserves_order(self):
        with TemporaryDirectory() as directory:
            pdf = Path(directory) / "mixed.pdf"
            with pymupdf.open() as document:
                add_native_page(document)
                page = document.new_page()
                page.insert_image(page.rect, stream=scan_image())
                add_native_page(document)
                document.save(pdf)
            model = Mock()
            model.predict.return_value = [{"rec_texts": ["Scanned middle page"], "rec_boxes": [[0, 0, 100, 20]]}]
            stages = []
            with patch("AI.model.download_ocr", return_value={"ocr_model": model}) as load:
                raw = extract_doc(pdf, directory, progress=stages.append)
            load.assert_called_once()
            model.predict.assert_called_once()
            self.assertTrue(model.predict.call_args.kwargs["input"].endswith("page_0002.png"))
            self.assertFalse(Path(model.predict.call_args.kwargs["input"]).exists())
            pages = raw.split("\n<PAGE>\n")
            self.assertEqual(len(pages), 3)
            self.assertIn("INV-001", pages[0])
            self.assertEqual(pages[1], "Scanned middle page")
            self.assertIn("INV-001", pages[2])
            self.assertEqual(stages, ["extracting_pdf_text", "loading_ocr_model", "running_ocr"])

    def test_native_header_does_not_hide_a_large_scanned_body(self):
        with TemporaryDirectory() as directory:
            pdf = Path(directory) / "header.pdf"
            with pymupdf.open() as document:
                page = add_native_page(document)
                page.insert_image(pymupdf.Rect(20, 150, 550, 750), stream=scan_image())
                document.save(pdf)
            model = Mock()
            model.predict.return_value = [{"rec_texts": ["Header plus scanned body"], "rec_boxes": [[0, 0, 100, 20]]}]
            raw = extract_doc(pdf, directory, {"ocr_model": model})
            model.predict.assert_called_once()
            self.assertEqual(raw, "Header plus scanned body")

    def test_native_text_can_be_disabled(self):
        with TemporaryDirectory() as directory:
            pdf = Path(directory) / "native.pdf"
            with pymupdf.open() as document:
                add_native_page(document)
                document.save(pdf)
            model = Mock()
            model.predict.return_value = [{"rec_texts": ["Forced OCR"], "rec_boxes": [[0, 0, 100, 20]]}]
            raw = extract_doc(pdf, directory, {"ocr_model": model}, force_ocr=True)
            self.assertEqual(raw, "Forced OCR")
