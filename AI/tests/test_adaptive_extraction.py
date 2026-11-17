"""Contract tests for unfamiliar invoice fields; model responses are fixtures."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

from AI.schema import validate_invoice
from AI.tests.test_extraction_chunks import document, fragment
from AI.utils import ContextBudgetError, format_json, update_invoice
from server.schemas import InvoiceExtractionResponse


def field(label, value, page=1):
    return {"label": label, "value": value, "source_text": value, "page_number": page}


class AdaptiveExtractionTests(TestCase):
    def test_unfamiliar_fields_survive_extraction_classification_and_api(self):
        payload = {"invoices": [{
            "order_level": {"additional_fields": [field("VAT rate", "18%"),
                                                   field("银行账户", "0012345")]},
            "shipment": {"additional_fields": [field("Container", "ABCU1234567")]},
            "item_level": [{"sku_code": "A", "product_description": "Cable",
                            "additional_fields": [field("Net wt (kg)", "2.50"),
                                                  field("Lot", "001-23")]},
                           {"sku_code": "A", "product_description": "Cable",
                            "additional_fields": [field("Net wt (kg)", "4.00")]}],
        }]}
        original = deepcopy(payload)
        source = "VAT rate 18%\n银行账户 0012345\nContainer ABCU1234567\nA Cable 2.50 001-23\nA Cable 4.00"
        generate = Mock(return_value=json.dumps(payload))
        with TemporaryDirectory() as folder:
            parsed = format_json(cleaned_text=source, generate=generate, output_path=folder)
            self.assertEqual(json.loads((Path(folder) / "extracted_invoice.json").read_text()), parsed)
        matches = [dict(invoice_index=0, item_index=i, hs_candidates=[], origin_candidates=[],
                        destination_candidates=[], origin_iso=None, destination_iso=None, source="test")
                   for i in range(2)]
        result = InvoiceExtractionResponse.model_validate(update_invoice(parsed, matches, generate)).model_dump()
        inv = result["invoices"][0]
        for scope in ("order_level", "shipment"):
            self.assertEqual(inv[scope]["additional_fields"], original["invoices"][0][scope]["additional_fields"])
        for item, expected in zip(inv["item_level"], original["invoices"][0]["item_level"]):
            self.assertEqual(item["additional_fields"], expected["additional_fields"])
            self.assertIsNone(item["gross_value"])
        self.assertIsNone(inv["order_level"]["net_weight"])

    def test_continuations_preserve_repeated_extra_occurrences_at_each_scope(self):
        plain, layout = document(pages=2)
        first = fragment(items=[{"additional_fields": [field("Net wt", "2 kg")]}],
                         additional_fields=[field("Note", "Keep dry")])
        second = fragment(continuation=True,
                          items=[{"additional_fields": [field("Net wt", "3 kg", 2)]}],
                          additional_fields=[field("Note", "Keep dry", 2)])
        for page, value in enumerate((first, second), 1):
            value["invoices"][0]["shipment"]["additional_fields"] = [field("Seal", "001", page)]
        generate = Mock(side_effect=[ContextBudgetError("split"), json.dumps(first), json.dumps(second)])
        result = format_json(cleaned_text=plain, layout_text=layout, generate=generate)
        inv = result["invoices"][0]
        self.assertEqual([f["page_number"] for f in inv["order_level"]["additional_fields"]], [1, 2])
        self.assertEqual([f["page_number"] for f in inv["shipment"]["additional_fields"]], [1, 2])
        self.assertEqual([i["additional_fields"][0]["value"] for i in inv["item_level"]], ["2 kg", "3 kg"])

    def test_extension_evidence_and_types_are_validated(self):
        good = field("Lot", "001", None)
        for change in ({"label": ""}, {"source_text": ""}, {"value": 23},
                       {"page_number": 0}, {"invented": "unsupported"}):
            with self.subTest(change=change):
                payload = {"invoices": [{"item_level": [{"additional_fields": [{**good, **change}]}]}]}
                with self.assertRaisesRegex(ValueError, "additional_fields"):
                    validate_invoice(payload)
        validate_invoice({"invoices": [{"item_level": [{"additional_fields": [good]}]}]})

    def test_extra_arrays_remain_optional_and_do_not_hide_core_type_errors(self):
        parsed = validate_invoice({"invoices": [{"item_level": [{}]}]})
        self.assertNotIn("additional_fields", parsed["invoices"][0]["item_level"][0])
        with self.assertRaisesRegex(ValueError, "number_of_units_for_this_sku"):
            validate_invoice({"invoices": [{"item_level": [{"number_of_units_for_this_sku": "ten",
                "additional_fields": [field("Net wt", "2 kg")]}]}]})

    def test_failed_unknown_key_retry_requests_preservation_instead_of_discarding(self):
        bad = {"invoices": [{"item_level": [{"net_weight": 2}]}]}
        good = {"invoices": [{"item_level": [{"additional_fields": [field("Net wt", "2 kg")]}]}]}
        generate = Mock(side_effect=[json.dumps(bad), json.dumps(good)])
        with self.assertLogs("AI.utils", level="WARNING"):
            result = format_json(cleaned_text="Net wt: 2 kg", generate=generate)
        self.assertEqual(result["invoices"][0]["item_level"][0]["additional_fields"],
                         good["invoices"][0]["item_level"][0]["additional_fields"])
        messages = generate.call_args.args[0]
        self.assertIn("DOCUMENT ANALYSIS BEFORE FIELD MAPPING", messages[0]["content"])
        self.assertIn("Preserve unfamiliar source fields", messages[1]["content"])
