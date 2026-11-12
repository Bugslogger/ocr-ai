"""Country fields survive extraction/API validation without cross-role fallbacks.

Model responses are fixtures: these tests exercise the contract and input wiring,
not the accuracy of live Qwen inference.
"""
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

import pymupdf

from AI.model_config import EXTRACTION_CONFIG
from AI.utils import extract_doc, format_json, update_invoice
from server.schemas import InvoiceExtractionResponse
from AI.tests.test_layout_compaction import expand_layout


class CountryRoleTests(TestCase):
    def extract(self, source, payload, layout=None, expected_shippers=None):
        original = deepcopy(payload)
        generate = Mock(return_value=json.dumps(payload, ensure_ascii=False))
        with patch.dict(EXTRACTION_CONFIG, {"layout_aware": True}):
            result = format_json(cleaned_text=source, layout_text=layout, generate=generate)
        # Tariff updates consume countries without overwriting them.
        matches = [dict(invoice_index=i, item_index=j, hs_candidates=[], origin_candidates=[],
                        destination_candidates=[], origin_iso=None, destination_iso=None, source="test")
                   for i, inv in enumerate(result["invoices"]) for j, _ in enumerate(inv["item_level"])]
        update_invoice(result, matches, generate)
        api = InvoiceExtractionResponse.model_validate(result).model_dump()
        for i, (before, after) in enumerate(zip(original["invoices"], api["invoices"])):
            for field in ("shipper_country", "receiver_country", "destination_country", "export_country", "origin_country"):
                expected = (expected_shippers[i] if field == "shipper_country" and expected_shippers is not None
                            else before.get("order_level", {}).get(field))
                self.assertEqual(after["order_level"][field], expected)
            self.assertEqual([item["country_of_manufacture"] for item in after["item_level"]],
                             [item.get("country_of_manufacture") for item in before["item_level"]])
        generate.assert_called_once()
        return api, generate.call_args.args[0]

    def test_addresses_only_leave_origin_export_and_manufacturing_null(self):
        api, _ = self.extract("From: Sender, Australia\nShip To: Buyer, New Zealand", {
            "invoices": [{"order_level": {"shipper_country": "AUSTRALIA", "receiver_country": "NEW ZEALAND",
                                          "destination_country": "NEW ZEALAND"}, "item_level": [{}]}]})
        order = api["invoices"][0]["order_level"]
        self.assertIsNone(order["origin_country"])
        self.assertIsNone(order["export_country"])
        self.assertEqual(order["shipper_country"], "AUSTRALIA")

    def test_item_only_origins_are_not_promoted_even_if_all_items_agree(self):
        for origins in (("JAPAN", "GERMANY"), ("JAPAN", "JAPAN")):
            with self.subTest(origins=origins):
                api, _ = self.extract("Item | Country of origin\nA | " + origins[0] + "\nB | " + origins[1], {
                    "invoices": [{"order_level": {"shipper_country": "AUSTRALIA"},
                                  "item_level": [{"sku_code": "A", "country_of_manufacture": origins[0]},
                                                 {"sku_code": "B", "country_of_manufacture": origins[1]}]}]})
                self.assertIsNone(api["invoices"][0]["order_level"]["origin_country"])

    def test_invoice_origin_default_and_item_exception_remain_independent(self):
        self.extract("Origin of all goods: Japan\nItem A\nItem B Made in Germany\nItem C", {
            "invoices": [{"order_level": {"origin_country": "JAPAN"},
                          "item_level": [{"sku_code": "A", "country_of_manufacture": "JAPAN"},
                                         {"sku_code": "B", "country_of_manufacture": "GERMANY"},
                                         {"sku_code": "C", "country_of_manufacture": "JAPAN"}]}]})

    def test_export_only_and_origin_only_do_not_fill_each_other_or_shipper(self):
        for field, source in (("export_country", "Country of Export: Australia"),
                              ("origin_country", "Country of Origin: Japan")):
            with self.subTest(field=field):
                self.extract(source, {"invoices": [{"order_level": {field: source.split(": ")[1]},
                                                    "item_level": [{}]}]})

    def test_conflicting_invoice_declarations_keep_item_evidence_and_null_global(self):
        self.extract("Goods origin declarations conflict: Japan; Germany. Item A Made in Japan.", {
            "invoices": [{"order_level": {"origin_country": None},
                          "item_level": [{"country_of_manufacture": "JAPAN"}, {}]}]})

    def test_multiple_invoices_keep_item_origin_and_shipping_roles_separate(self):
        self.extract("Invoice A: From Australia, to New Zealand, goods made in Japan.\n\n"
                     "Invoice B: From France, to Germany, item made in Spain.", {
            "invoices": [
                {"order_level": {"origin_country": "JAPAN", "shipper_country": "AUSTRALIA",
                                 "destination_country": "NEW ZEALAND"},
                 "item_level": [{"sku_code": "DUPLICATE", "country_of_manufacture": "JAPAN"}]},
                {"order_level": {"shipper_country": "FRANCE", "destination_country": "GERMANY"},
                 "item_level": [{"sku_code": "DUPLICATE", "country_of_manufacture": "SPAIN"}]}]}, expected_shippers=[None, "FRANCE"])

    def test_wrapped_country_columns_reach_model_with_geometry_and_correct_api_roles(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            pdf = root / "columns.pdf"
            with pymupdf.open() as document:
                page = document.new_page(width=700, height=900)
                page.insert_text((35, 35), "From: Sender, Australia")
                # Test fixture only: neither values nor coordinates are production rules.
                for x, label, value in ((40, "EXPORT", "AUSTRALIA"), (260, "ORIGIN", "JAPAN"),
                                        (470, "DESTINATION", "NEW ZEALAND")):
                    page.insert_text((x, 100), "COUNTRY OF")
                    page.insert_text((x, 115), label)
                    page.insert_text((x, 140), value)
                document.save(pdf)
            with patch("AI.model.download_ocr") as load:
                extract_doc(pdf, root)
            load.assert_not_called()
            plain = (root / "extracted_text.txt").read_text()
            layout = (root / "layout_aware_text.txt").read_text()
            _, messages = self.extract(plain, {"invoices": [{
                "order_level": {"origin_country": "JAPAN", "export_country": "AUSTRALIA",
                                "shipper_country": "AUSTRALIA", "destination_country": "NEW ZEALAND"},
                "item_level": [{"country_of_manufacture": "JAPAN"}]}]}, layout, expected_shippers=[None])
            self.assertIn(plain, messages[1]["content"])
            self.assertEqual(expand_layout(messages[1]["content"].split("Layout-aware OCR (JSON):\n")[1]),
                             json.loads(layout))
            blocks = json.loads(layout)["pages"][0]["blocks"]
            origin = next(block for block in blocks if block["text"] == "ORIGIN")
            value = next(block for block in blocks if block["text"] == "JAPAN")
            self.assertAlmostEqual(origin["bbox"][0], value["bbox"][0])
            self.assertLess(origin["bbox"][1], value["bbox"][1])

    def test_unicode_item_origins_and_absent_shipper_remain_unchanged(self):
        self.extract("商品 A 原産国 日本\n商品 B 原産国 ドイツ", {"invoices": [{
            "item_level": [{"sku_code": "A", "country_of_manufacture": "日本"},
                           {"sku_code": "B", "country_of_manufacture": "ドイツ"}]}]})

    def test_shipper_policy_is_applied_to_saved_extraction_and_api(self):
        payload = {"invoices": [{"order_level": {
            "origin_country": "JAPAN", "export_country": "AUSTRALIA",
            "shipper_country": "AUSTRALIA", "destination_country": "NEW ZEALAND"},
            "item_level": [{"country_of_manufacture": "JAPAN"}]}]}
        with TemporaryDirectory() as folder:
            parsed = format_json(cleaned_text="Invoice country declarations", output_path=folder,
                                 generate=Mock(return_value=json.dumps(payload)))
            saved = json.loads((Path(folder) / "extracted_invoice.json").read_text())
            api = InvoiceExtractionResponse.model_validate(parsed).model_dump()
            for result in (parsed, saved, api):
                order = result["invoices"][0]["order_level"]
                self.assertIsNone(order["shipper_country"])
                self.assertEqual(order["origin_country"], "JAPAN")
                self.assertEqual(order["export_country"], "AUSTRALIA")
                self.assertEqual(order["destination_country"], "NEW ZEALAND")
                self.assertEqual(result["invoices"][0]["item_level"][0]["country_of_manufacture"], "JAPAN")

    def test_agreeing_shipper_aliases_and_missing_values(self):
        for origin, shipper, expected in (
            ("Japan", "JP", "JP"), ("JAPAN", "japan", "japan"),
            ("日本", "日本", "日本"), ("JAPAN", None, None),
            (None, "AUSTRALIA", "AUSTRALIA"), ("JAPAN", "unknown", None),
        ):
            with self.subTest(origin=origin, shipper=shipper):
                api, _ = self.extract("Country evidence", {"invoices": [{
                    "order_level": {"origin_country": origin, "shipper_country": shipper},
                    "item_level": [{}]}]}, expected_shippers=[expected])
                self.assertEqual(api["invoices"][0]["order_level"]["shipper_country"], expected)
