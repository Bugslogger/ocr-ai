"""Oversized-document regression tests with bounded fake inference."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from AI.extraction_chunks import _text, validate_chunk, chunk_schema
from AI.model_config import EXTRACTION_CONFIG, Qruntime
from AI.tests.test_layout_compaction import expand_layout
from AI.utils import (ContextBudgetError, OutputTokenLimitError, format_json, make_generator)


def document(pages=4, rows=3):
    layout = {"coordinate_system": "top_left_origin_x_right_y_down", "pages": [
        {"page_number": n, "width": 500, "height": 800, "source": "pdf_text",
         "coordinate_unit": "pdf_points", "blocks": [
             {"reading_order": i * 2 + col, "text": f"商品-{n}-{i}" if col == 0 else "2",
              "bbox": [col * 200, i * 20, col * 200 + 100, i * 20 + 10], "confidence": None}
             for i in range(rows) for col in range(2)]} for n in range(1, pages + 1)]}
    return "\n\n".join(_text(p) for p in layout["pages"]), json.dumps(layout, ensure_ascii=False)


def fragment(number="INV-1", continuation=False, items=None, **order):
    return {"invoices": [{"continues_previous": continuation,
                          "order_level": {"invoice_number": number, **order},
                          "shipment": {}, "item_level": [{}] if items is None else items}]}


def spatial(messages):
    encoded = messages[1]["content"].split("Layout-aware OCR (JSON):\n")[1].split(
        "\n\nPrevious invoice metadata", 1)[0]
    return expand_layout(encoded)


class ChunkExtractionTests(TestCase):
    def setUp(self):
        self.enterContext(patch.dict(EXTRACTION_CONFIG, {"layout_aware": True}))

    def test_reported_57644_token_input_is_rejected_by_cpu_preflight(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = range(57644)
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1048576))
        generate = make_generator(tokenizer, model)
        with patch.dict(Qruntime, {"max_context_tokens": 32768, "context_safety_margin": 512}):
            with self.assertRaisesRegex(ContextBudgetError, "57644 input tokens"):
                generate.check_budget([{"role": "user", "content": "source"}], 10000)
        tokenizer.assert_not_called()  # No tensor allocation or CUDA inference.

    def test_preflight_reserves_full_extraction_output_and_model_limit(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = range(25000)
        generate = make_generator(tokenizer, SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768)))
        with self.assertRaises(ContextBudgetError):
            generate.check_budget([], 10000)
        generate.check_budget([], 700)

    def test_recursive_page_chunks_cover_every_block_once_and_preserve_duplicate_skus(self):
        plain, layout = document()
        accepted = []

        def budget(messages, reserve):
            if len(spatial(messages)["pages"]) > 1:
                raise ContextBudgetError("split")

        def infer(messages, **kwargs):
            pages = spatial(messages)["pages"]
            accepted.extend(pages)
            return json.dumps(fragment(continuation=len(accepted) > 1,
                                      items=[{"sku_code": "DUP", "product_description": "same"}] * 2))

        generate = Mock(side_effect=infer)
        generate.check_budget = budget
        with TemporaryDirectory() as folder:
            result = format_json(cleaned_text=plain, layout_text=layout, generate=generate, output_path=folder)
            self.assertEqual(json.loads((Path(folder) / "extracted_invoice.json").read_text()), result)
            self.assertEqual(len(list((Path(folder) / "extraction_chunks").glob("*/extracted_chunk.json"))), 4)
        self.assertEqual(accepted, json.loads(layout)["pages"])
        self.assertEqual(len(result["invoices"]), 1)
        self.assertEqual(len(result["invoices"][0]["item_level"]), 8)
        self.assertNotIn("continues_previous", result["invoices"][0])
        self.assertEqual(generate.call_count, 4)
        context = generate.call_args.args[0][1]["content"].split("Previous invoice metadata (reference only):\n")[1]
        self.assertNotIn("item_level", json.loads(context))

    def test_oversized_page_splits_between_rows_retaining_columns_and_exact_geometry(self):
        plain, layout = document(pages=1, rows=8)
        accepted = []

        def budget(messages, reserve):
            if len(spatial(messages)["pages"][0]["blocks"]) > 4:
                raise ContextBudgetError("split")

        def infer(messages, **kwargs):
            blocks = spatial(messages)["pages"][0]["blocks"]
            accepted.extend(blocks)
            for i in range(0, len(blocks), 2):
                self.assertEqual(blocks[i]["bbox"][1], blocks[i + 1]["bbox"][1])
            return json.dumps(fragment(continuation=len(accepted) > len(blocks)))

        generate = Mock(side_effect=infer)
        generate.check_budget = budget
        format_json(cleaned_text=plain, layout_text=layout, generate=generate)
        self.assertEqual(accepted, json.loads(layout)["pages"][0]["blocks"])

    def test_output_exhaustion_also_splits_instead_of_returning_partial_json(self):
        plain, layout = document(pages=2)
        generate = Mock(side_effect=[OutputTokenLimitError("token limit"),
                                    json.dumps(fragment()), json.dumps(fragment(continuation=True))])
        result = format_json(cleaned_text=plain, layout_text=layout, generate=generate)
        self.assertEqual(len(result["invoices"][0]["item_level"]), 2)

    def test_header_only_fragment_can_join_items_and_fill_later_metadata(self):
        plain, layout = document(pages=2)
        generate = Mock(side_effect=[ContextBudgetError("split"),
                                    json.dumps(fragment(items=[], company_name="Seller")),
                                    json.dumps(fragment(continuation=True, currency="USD"))])
        inv = format_json(cleaned_text=plain, layout_text=layout, generate=generate)["invoices"][0]
        self.assertEqual(inv["order_level"]["company_name"], "Seller")
        self.assertEqual(inv["order_level"]["currency"], "USD")
        self.assertEqual(len(inv["item_level"]), 1)

    def test_unassociated_metadata_is_preserved_without_creating_empty_invoice(self):
        from server.schemas import InvoiceExtractionResponse
        plain, layout = document(pages=4)
        first = fragment(items=[{"sku_code": "DUP"}] * 2, total_amount=100)
        second = fragment(items=[{"sku_code": "DUP"}] * 2, total_amount=500)
        footer = fragment(items=[], total_amount=500)
        generate = Mock(side_effect=[ContextBudgetError("split"), json.dumps(first),
                                    ContextBudgetError("split again"), json.dumps(second), json.dumps(footer)])
        with TemporaryDirectory() as folder:
            result = format_json(cleaned_text=plain, layout_text=layout, generate=generate, output_path=folder)
            self.assertEqual(json.loads((Path(folder) / "extracted_invoice.json").read_text()), result)
        self.assertEqual(len(result["invoices"]), 2)
        self.assertEqual(sum(len(inv["item_level"]) for inv in result["invoices"]), 4)
        self.assertEqual([inv["order_level"]["total_amount"] for inv in result["invoices"]], [100, 500])
        metadata = result["metadata_only_fragments"][0]
        self.assertEqual(metadata["order_level"]["total_amount"], 500)
        self.assertEqual(metadata["source_chunks"], [4])
        self.assertEqual(metadata["source_pages"], [4])
        self.assertEqual(len(result["extraction_warnings"]), 2)
        # Both API serialization and persisted-job reload retain the diagnostics.
        api = InvoiceExtractionResponse.model_validate(result).model_dump(mode="json")
        self.assertEqual(api, result)
        self.assertEqual(InvoiceExtractionResponse.model_validate_json(json.dumps(api)).model_dump(), result)

    def test_header_for_a_different_invoice_is_not_attached_to_previous_items(self):
        plain, layout = document(pages=2)
        generate = Mock(side_effect=[ContextBudgetError("split"), json.dumps(fragment(number="A")),
                                    json.dumps(fragment(number="B", items=[], total_amount=900))])
        result = format_json(cleaned_text=plain, layout_text=layout, generate=generate)
        self.assertEqual(result["invoices"][0]["order_level"]["invoice_number"], "A")
        self.assertIsNone(result["invoices"][0]["order_level"]["total_amount"])
        self.assertEqual(result["metadata_only_fragments"][0]["order_level"]["invoice_number"], "B")

    def test_all_metadata_and_no_items_still_fails_without_saving_final(self):
        plain, layout = document(pages=2)
        generate = Mock(side_effect=[ContextBudgetError("split"), json.dumps(fragment(items=[])),
                                    json.dumps(fragment(number="OTHER", items=[]))])
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "No invoice line items.*manual review"):
                format_json(cleaned_text=plain, layout_text=layout, generate=generate, output_path=folder)
            self.assertFalse((Path(folder) / "extracted_invoice.json").exists())
            self.assertEqual(len(list(Path(folder).glob("extraction_chunks/*/extracted_chunk.json"))), 2)

    def test_new_invoices_and_unknown_identities_remain_separate(self):
        for second in ("INV-2", None):
            with self.subTest(second=second):
                plain, layout = document(pages=2)
                generate = Mock(side_effect=[ContextBudgetError("split"), json.dumps(fragment()),
                                            json.dumps(fragment(number=second))])
                result = format_json(cleaned_text=plain, layout_text=layout, generate=generate)
                self.assertEqual(len(result["invoices"]), 2)

    def test_conflicting_continuation_cannot_save_a_partial_final_invoice(self):
        for fields in ({"number": "INV-2"}, {"company_name": "Different Seller"}, {"total_amount": 99}):
            plain, layout = document(pages=2)
            generate = Mock(side_effect=[ContextBudgetError("split"),
                                        json.dumps(fragment(company_name="Seller", total_amount=100)),
                                        json.dumps(fragment(continuation=True, **fields))])
            with self.subTest(fields=fields), TemporaryDirectory() as folder:
                with self.assertRaisesRegex(ValueError, "Conflicting"):
                    format_json(cleaned_text=plain, layout_text=layout, generate=generate, output_path=folder)
                self.assertFalse((Path(folder) / "extracted_invoice.json").exists())

    def test_mismatched_source_refuses_to_discard_unrepresented_text(self):
        plain, layout = document()
        generate = Mock(side_effect=ContextBudgetError("split"))
        with self.assertRaisesRegex(ValueError, "matching original page layout"):
            format_json(cleaned_text=plain + " extra source", layout_text=layout, generate=generate)

    def test_indivisible_row_fails_explicitly_and_does_not_loop(self):
        plain, layout = document(pages=1, rows=1)
        generate = Mock(side_effect=ContextBudgetError("split"))
        with self.assertRaisesRegex(ValueError, "single invoice text row"):
            format_json(cleaned_text=plain, layout_text=layout, generate=generate)
        self.assertEqual(generate.call_count, 1)

    def test_disabled_layout_still_splits_using_page_boundaries(self):
        plain, layout = document(pages=2)
        generate = Mock(side_effect=[ContextBudgetError("split"), json.dumps(fragment()),
                                    json.dumps(fragment(continuation=True))])
        with patch.dict(EXTRACTION_CONFIG, {"layout_aware": False}):
            format_json(cleaned_text=plain, layout_text=layout, generate=generate)
        for call in generate.call_args_list:
            self.assertNotIn("Layout-aware OCR (JSON):", call.args[0][1]["content"])

    def test_unrelated_generation_errors_are_not_retried_as_size_failures(self):
        plain, layout = document()
        generate = Mock(side_effect=RuntimeError("CUDA failure"))
        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            format_json(cleaned_text=plain, layout_text=layout, generate=generate)
        self.assertEqual(generate.call_count, 1)


class ChunkSchemaRecoveryTests(TestCase):
    def test_reported_item_gross_weight_error_regenerates_chunk_without_changing_values(self):
        plain, layout = document(pages=2)
        item = {"sku_code": "DFS00031", "product_description": "DF Master Card 220V DF800",
                "simplified_description": "Master Card", "country_of_manufacture": "Israel",
                "gross_value": 866.04, "number_of_units_for_this_sku": 10,
                "unit_value_for_this_sku": 86.604, "extended_value_for_this_sku": 8660.4}
        invalid = fragment(items=[{**item, "gross_weight": 23}])
        valid = fragment(items=[item])
        generate = Mock(side_effect=[ContextBudgetError("split"), json.dumps(invalid),
                                    json.dumps(valid), json.dumps(fragment(continuation=True))])
        with TemporaryDirectory() as folder, self.assertLogs("AI.utils", level="WARNING") as logs:
            result = format_json(cleaned_text=plain, layout_text=layout, generate=generate, output_path=folder)
            diagnostic = next(Path(folder).glob("extraction_chunks/*/invalid_invoice_response_1.txt"))
            self.assertEqual(json.loads(diagnostic.read_text()), invalid)
            self.assertTrue((Path(folder) / "extracted_invoice.json").exists())
        self.assertEqual(generate.call_count, 4)
        extracted = result["invoices"][0]["item_level"][0]
        for field, value in item.items():
            self.assertEqual(extracted[field], value)
        self.assertNotIn("gross_weight", extracted)
        self.assertIsNone(result["invoices"][0]["order_level"]["gross_weight"])
        first, retry = [call.args[0] for call in generate.call_args_list[1:3]]
        self.assertEqual(first[0], retry[0])
        self.assertTrue(retry[1]["content"].startswith(first[1]["content"]))
        self.assertIn("invoices.0.item_level.0", retry[1]["content"])
        self.assertIn("gross_weight", retry[1]["content"])
        self.assertNotIn("DFS00031", " ".join(logs.output))

    def test_chunk_prompt_contains_exact_chunk_schema_once(self):
        from AI.utils import _format_json_single
        from AI.prompt import extraction_prompt, EXTRACT_SYSTEM_PROMPT
        self.assertEqual(extraction_prompt(), EXTRACT_SYSTEM_PROMPT)
        generate = Mock(return_value=json.dumps(fragment()))
        _format_json_single(cleaned_text="Source", layout_text=None, system_prompt=None,
                            generate=generate, chunk_context={})
        prompt = generate.call_args.args[0][0]["content"]
        encoded = prompt.split("JSON Schema:\n", 1)[1]
        schema, _ = json.JSONDecoder().raw_decode(encoded)
        self.assertEqual(schema, chunk_schema())
        self.assertEqual(prompt.count('"continues_previous": {"type": "boolean"}'), 1)
        self.assertEqual(schema["properties"]["invoices"]["items"]["properties"]["item_level"]["minItems"], 0)

    def test_repeated_invalid_chunk_schema_stops_after_two_attempts_without_saving_final(self):
        plain, layout = document(pages=2, rows=1)
        invalid = json.dumps(fragment(items=[{"gross_weight": 23}]))
        generate = Mock(side_effect=[ContextBudgetError("split"), invalid, invalid])
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "invalid invoice schema after 2 attempts"):
                format_json(cleaned_text=plain, layout_text=layout, generate=generate, output_path=folder)
            self.assertFalse((Path(folder) / "extracted_invoice.json").exists())
            self.assertEqual(len(list(Path(folder).glob("extraction_chunks/*/invalid_invoice_response_*.txt"))), 2)
        self.assertEqual(generate.call_count, 3)

    def test_reported_net_weight_failure_recovers_with_smaller_chunks(self):
        plain, layout = document(pages=4, rows=2)
        accepted = []
        invalid = fragment(items=[{"sku_code": "DUP", "net_weight": 23}])

        def budget(messages, reserve):
            # Reproduce both preflight failures in the report before generation.
            if len(spatial(messages)["pages"]) > 1:
                raise ContextBudgetError("input plus reserved output exceeds context")

        def infer(messages, **kwargs):
            page = spatial(messages)["pages"][0]
            if len(page["blocks"]) > 2:
                return json.dumps(invalid)
            accepted.extend(page["blocks"])
            return json.dumps(fragment(continuation=len(accepted) > 2,
                                       items=[{"sku_code": "DUP", "gross_value": 100}]))

        generate = Mock(side_effect=infer)
        generate.check_budget = budget
        with TemporaryDirectory() as folder, self.assertLogs("AI.utils", level="WARNING"):
            result = format_json(cleaned_text=plain, layout_text=layout,
                                 generate=generate, output_path=folder)
            self.assertEqual(len(list(Path(folder).glob(
                "extraction_chunks/*/invalid_invoice_response_*.txt"))), 8)
            self.assertEqual(json.loads((Path(folder) / "extracted_invoice.json").read_text()), result)
        self.assertEqual(accepted, [block for page in json.loads(layout)["pages"]
                                    for block in page["blocks"]])
        invoice = result["invoices"][0]
        self.assertEqual(len(invoice["item_level"]), 8)
        self.assertTrue(all(item["sku_code"] == "DUP" and item["gross_value"] == 100
                            for item in invoice["item_level"]))
        self.assertIsNone(invoice["order_level"]["net_weight"])
        self.assertEqual(generate.call_count, 16)

    def test_persistent_invalid_chunks_stop_at_indivisible_row(self):
        plain, layout = document(pages=2, rows=4)
        invalid = json.dumps(fragment(items=[{"net_weight": 23}]))
        generate = Mock(side_effect=[ContextBudgetError("split")] + [invalid] * 6)
        with TemporaryDirectory() as folder, self.assertLogs("AI.utils", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "invalid invoice schema after 2 attempts.*net_weight"):
                format_json(cleaned_text=plain, layout_text=layout, generate=generate, output_path=folder)
            self.assertFalse((Path(folder) / "extracted_invoice.json").exists())
        self.assertEqual(generate.call_count, 7)

    def test_unchunked_schema_type_error_also_gets_one_source_based_retry(self):
        invalid = {"invoices": [{"item_level": [{"number_of_units_for_this_sku": "ten"}]}]}
        valid = {"invoices": [{"item_level": [{"number_of_units_for_this_sku": 10}]}]}
        generate = Mock(side_effect=[json.dumps(invalid), json.dumps(valid)])
        result = format_json(cleaned_text="Source quantity 10", generate=generate)
        self.assertEqual(result["invoices"][0]["item_level"][0]["number_of_units_for_this_sku"], 10)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("Source quantity 10", generate.call_args.args[0][1]["content"])

    def test_generation_failure_during_schema_retry_is_not_masked(self):
        generate = Mock(side_effect=['{"invoices": []}', RuntimeError("CUDA failure")])
        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
            format_json(cleaned_text="Source", generate=generate)
        self.assertEqual(generate.call_count, 2)
