"""Exact reconstruction of model layout input, without coordinate rounding."""
import json
from unittest import TestCase
from unittest.mock import Mock, patch

from AI.layout_input import compact_layout_text
from AI.model_config import EXTRACTION_CONFIG
from AI.utils import format_json


def expand_layout(text):
    layout = json.loads(text)
    if layout.get("encoding") != "indexed_axes_v1":
        return layout
    pages = []
    for page in layout["pages"]:
        blocks = []
        for order, text, x1, y1, x2, y2, confidence in page["blocks"]:
            blocks.append({"reading_order": order, "text": text,
                           "bbox": [page["x_coordinates"][x1], page["y_coordinates"][y1],
                                    page["x_coordinates"][x2], page["y_coordinates"][y2]],
                           "confidence": confidence})
        pages.append({**{k: v for k, v in page.items() if k not in
                        ("x_coordinates", "y_coordinates", "blocks")}, "blocks": blocks})
    return {"coordinate_system": layout["coordinate_system"], "pages": pages}


class LayoutCompactionTests(TestCase):
    def layout(self):
        return {"coordinate_system": "top_left_origin_x_right_y_down", "pages": [
            {"page_number": page, "width": 595.3200073242188, "height": 841.9199829101562,
             "coordinate_unit": "pdf_points" if page == 1 else "pixels", "source": "test",
             "blocks": [{"reading_order": 400 - i, "text": f"商品 {i} 日本 <&>\nsecond line",
                         "bbox": [10.123456789012345, 20.987654321098765 + (i // 8) * 12,
                                  100.12345678901235, 30.987654321098765 + (i // 8) * 12],
                         "confidence": None if page == 1 else .9876543210987654} for i in range(401)]}
            for page in (1, 2)]}

    def test_large_multi_page_document_roundtrips_every_field_exactly(self):
        layout = self.layout()
        original = json.dumps(layout, ensure_ascii=False, separators=(",", ":"))
        compact = compact_layout_text(original)
        self.assertLess(len(compact), len(original) * .6)
        self.assertEqual(expand_layout(compact), layout)
        self.assertEqual(len(expand_layout(compact)["pages"][1]["blocks"]), 401)
        self.assertEqual(expand_layout(compact)["pages"][1]["blocks"][-1]["text"],
                         "商品 400 日本 <&>\nsecond line")

    def test_unknown_metadata_or_shapes_are_preserved(self):
        for location in ("root", "block"):
            layout = self.layout()
            if location == "root":
                layout["new_field"] = "preserve"
            else:
                layout["pages"][0]["blocks"][0]["new_field"] = "preserve"
            original = json.dumps(layout)
            self.assertEqual(compact_layout_text(original), original)
        for text in ('{"pages":[]}', 'not JSON', '[]'):
            self.assertEqual(compact_layout_text(text), text)

    def test_extraction_keeps_plain_text_and_every_layout_block(self):
        layout = self.layout()
        original = json.dumps(layout, ensure_ascii=False)
        plain = "Full plain text\n\nLast page and last item"
        generate = Mock(return_value='{"invoices":[{"item_level":[{}]}]}')
        with patch.dict(EXTRACTION_CONFIG, {"layout_aware": True}):
            format_json(cleaned_text=plain, layout_text=original, generate=generate)
        messages = generate.call_args.args[0]
        user = messages[1]["content"]
        self.assertTrue(user.startswith("OCR Text:\n\n" + plain))
        encoded = user.split("Layout-aware OCR (JSON):\n", 1)[1]
        self.assertEqual(expand_layout(encoded), layout)
        self.assertIn("indexed_axes_v1", messages[0]["content"])

    def test_disabled_mode_does_not_send_layout(self):
        generate = Mock(return_value='{"invoices":[{"item_level":[{}]}]}')
        with patch.dict(EXTRACTION_CONFIG, {"layout_aware": False}):
            format_json(cleaned_text="All text", layout_text=json.dumps(self.layout()), generate=generate)
        self.assertEqual(generate.call_args.args[0][1]["content"], "OCR Text:\n\nAll text")
