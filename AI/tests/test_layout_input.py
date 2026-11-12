"""Layout evidence reaches extraction unchanged; inference is mocked, not evaluated."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import pymupdf
from PIL import Image

from AI.main import doc_pipeline
from AI.model_config import EXTRACTION_CONFIG
from AI.ocr import text_blocks
from AI.utils import extract_doc, format_json


class LayoutInputTests(TestCase):
    def test_ocr_columns_order_confidence_unicode_and_page_boundaries(self):
        # Recognition order deliberately differs from geometric row order.
        results = [
            {"rec_texts": ["España", "Country of export", "日本", "Country of origin"],
             "rec_boxes": [[300, 60, 400, 80], [300, 20, 480, 40],
                           [20, 60, 100, 80], [20, 20, 200, 40]],
             "rec_scores": [.97, .98, .91, .99]},
            {"rec_texts": ["Description", "Qty", "Amount", "ケーブル", "2", "20.00"],
             "rec_boxes": [[10, 10, 100, 30], [150, 10, 180, 30], [250, 10, 310, 30],
                           [10, 50, 100, 70], [150, 50, 180, 70], [250, 50, 310, 70]]},
        ]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            pdf = root / "invoice.pdf"
            with pymupdf.open() as document:
                document.new_page(width=144, height=216)
                document.new_page(width=216, height=144)
                document.save(pdf)
            model = Mock()
            model.predict.side_effect = [[result] for result in results]
            extract_doc(pdf, root, {"ocr_model": model}, force_ocr=True)
            layout = json.loads((root / "layout_aware_text.txt").read_text())
            self.assertEqual(layout, json.loads((root / "document_layout.json").read_text()))
            self.assertEqual([page["page_number"] for page in layout["pages"]], [1, 2])
            for page, original, call in zip(layout["pages"], results, model.predict.call_args_list):
                self.assertEqual(page["source"], "ocr")
                self.assertEqual(page["coordinate_unit"], "pixels")
                self.assertEqual([block["text"] for block in page["blocks"]], original["rec_texts"])
                self.assertEqual([block["bbox"] for block in page["blocks"]], original["rec_boxes"])
                self.assertEqual([block["reading_order"] for block in page["blocks"]],
                                 list(range(len(original["rec_texts"]))))
                self.assertFalse(Path(call.kwargs["input"]).exists())
            self.assertEqual(layout["pages"][0]["blocks"][0]["confidence"], .97)
            self.assertIsNone(layout["pages"][1]["blocks"][0]["confidence"])
            # Use actual render dimensions, including rounding, rather than PDF dimensions.
            self.assertEqual((layout["pages"][0]["width"], layout["pages"][0]["height"]), (500, 750))
            self.assertEqual((layout["pages"][1]["width"], layout["pages"][1]["height"]), (750, 500))
            plain = (root / "extracted_text.txt").read_text()
            self.assertIn("Country of origin\tCountry of export\n日本\tEspaña", plain)
            self.assertIn("Description\tQty\tAmount\nケーブル\t2\t20.00", plain)
            self.assertEqual((root / "detailed_invoice_output.txt").read_text().count("<PAGE>"), 1)
            generate = Mock(return_value='{"invoices":[{"item_level":[{}]}]}')
            with patch.dict(EXTRACTION_CONFIG, {"layout_aware": True}):
                format_json(cleaned_text=plain, layout_text=(root / "layout_aware_text.txt").read_text(),
                            generate=generate)
            messages = generate.call_args.args[0]
            self.assertIn(plain, messages[1]["content"])
            self.assertEqual(json.loads(messages[1]["content"].split("Layout-aware OCR (JSON):\n")[1]), layout)

    def test_native_and_scanned_pages_keep_their_own_coordinate_units(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            pdf = root / "mixed.pdf"
            image = root / "scan.png"
            Image.new("RGB", (100, 100), "white").save(image)
            with pymupdf.open() as document:
                page = document.new_page(width=400, height=500)
                page.insert_text((20, 30), "Invoice item description quantity amount currency reference", fontsize=10)
                page = document.new_page(width=300, height=200)
                page.insert_image(page.rect, filename=str(image))
                document.save(pdf)
            model = Mock()
            model.predict.return_value = [{"rec_texts": ["Scanned item"], "rec_boxes": [[10, 20, 100, 40]]}]
            extract_doc(pdf, root, {"ocr_model": model})
            pages = json.loads((root / "document_layout.json").read_text())["pages"]
            self.assertEqual([page["source"] for page in pages], ["pdf_text", "ocr"])
            self.assertEqual([page["coordinate_unit"] for page in pages], ["pdf_points", "pixels"])
            self.assertEqual((pages[0]["width"], pages[0]["height"]), (400, 500))
            self.assertEqual(pages[0]["blocks"][0]["text"], "Invoice")
            self.assertEqual(pages[0]["blocks"][0]["bbox"][0], 20)
            self.assertIsNone(pages[0]["blocks"][0]["confidence"])
            self.assertEqual(pages[1]["page_number"], 2)
            model.predict.assert_called_once()

    def test_pipeline_passes_both_inputs_and_disabled_mode_keeps_plain_input(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            pdf = root / "native.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((20, 30), "Invoice description quantity total reference currency seller")
                document.save(pdf)
            for enabled in (True, False):
                with self.subTest(enabled=enabled), patch.dict(EXTRACTION_CONFIG, {"layout_aware": enabled}):
                    generate = Mock(return_value='{"invoices":[{"item_level":[{}]}]}')
                    doc_pipeline(pdf, root / str(enabled), models={}, generate=generate,
                                 retriever=Mock(retrieve=Mock(return_value={
                                     "hs_candidates": [], "origin_candidates": [], "destination_candidates": [],
                                     "origin_iso": None, "destination_iso": None, "source": "test"})))
                    messages = generate.call_args_list[0].args[0]
                    plain = (root / str(enabled) / "extracted_text.txt").read_text()
                    self.assertEqual("Layout-aware OCR (JSON):" in messages[1]["content"], enabled)
                    self.assertEqual("SPATIAL INPUT INTERPRETATION:" in messages[0]["content"], enabled)
                    self.assertTrue((root / str(enabled) / "document_layout.json").exists())
                    if not enabled:
                        self.assertEqual(messages[1]["content"], "OCR Text:\n\n" + plain)

    def test_retry_keeps_plain_and_layout_evidence(self):
        generate = Mock(side_effect=["invalid", '{"invoices":[{"item_level":[{}]}]}'])
        with patch.dict(EXTRACTION_CONFIG, {"layout_aware": True}):
            format_json(cleaned_text="Original plain text", layout_text='{"pages":[]}', generate=generate)
        first, retry = [call.args[0] for call in generate.call_args_list]
        self.assertEqual(first[0], retry[0])
        self.assertTrue(retry[1]["content"].startswith(first[1]["content"]))

    def test_malformed_confidence_is_not_silently_associated_with_wrong_block(self):
        for scores in ([], [float("nan")], [1.5]):
            with self.subTest(scores=scores), self.assertRaises(RuntimeError):
                text_blocks({"rec_texts": ["text"], "rec_boxes": [[0, 0, 10, 10]], "rec_scores": scores})
