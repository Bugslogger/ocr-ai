"""PaddleOCR geometry, malformed results, and literal invoice text."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

from AI.ocr import page_text
from AI.utils import clean_extracted_text, extract_doc


class OCRLayoutTests(TestCase):
    def test_shuffled_boxes_keep_rows_and_columns(self):
        result = {
            "rec_texts": ["20.00", "Qty", "Cable", "Amount", "2", "Description"],
            "rec_boxes": [[200, 50, 250, 70], [130, 10, 160, 30], [0, 50, 100, 70],
                          [200, 12, 260, 32], [130, 51, 150, 71], [0, 10, 100, 30]],
        }
        self.assertEqual(page_text(result), "Description\tQty\tAmount\nCable\t2\t20.00")

    def test_malformed_results_are_rejected(self):
        for result in (None, {}, {"rec_texts": ["x"], "rec_boxes": []},
                       {"rec_texts": ["x"], "rec_boxes": [[0, 0, 0, 20]]},
                       {"rec_texts": ["x"], "rec_boxes": [[0, float('nan'), 10, 20]]}):
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                page_text(result)

    def test_page_count_mismatch_and_failure_do_not_leave_partial_output(self):
        import pymupdf
        with TemporaryDirectory() as directory:
            pdf = Path(directory) / "invoice.pdf"
            with pymupdf.open() as doc:
                doc.new_page()
                doc.save(pdf)
            for result in ([], [{}, {}]):
                model = Mock()
                model.predict.return_value = result
                with self.assertRaisesRegex(RuntimeError, "Expected one OCR result"):
                    extract_doc(pdf, directory, {"ocr_model": model})
                self.assertFalse((Path(directory) / "detailed_invoice_output.txt").exists())
                self.assertFalse(list(Path(directory).glob("pages_*")))

    def test_literal_angle_brackets_and_entities_survive_cleanup(self):
        import pymupdf
        with TemporaryDirectory() as directory:
            pdf = Path(directory) / "invoice.pdf"
            with pymupdf.open() as doc:
                doc.new_page()
                doc.save(pdf)
            model = Mock()
            text = "Size <10mm> & A&amp;B"
            model.predict.return_value = [{"rec_texts": [text], "rec_boxes": [[0, 0, 100, 20]]}]
            extract_doc(pdf, directory, {"ocr_model": model})
            self.assertEqual(clean_extracted_text(directory), text)
