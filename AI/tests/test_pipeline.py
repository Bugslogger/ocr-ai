"""Regression tests using tiny local PDFs and fake model/database responses."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace, ModuleType
from unittest import TestCase, main
from unittest.mock import Mock, patch
import json
import subprocess
import sys

from AI.main import doc_pipeline
from AI.schema import ORDER_FIELDS, ITEM_FIELDS, NUMBER_FIELDS
from AI.tariff import (CatalogRetriever, WeaviateRetriever, classify_item,
                       iso2_from_country, validate_selection)
from AI.utils import clean_ocr_text, extract_doc, format_json, qwen_generate


def invoice(sku="DUPLICATE"):
    item = {key: None for key in ITEM_FIELDS + NUMBER_FIELDS}
    item.update(sku_code=sku, product_description="insulated electric cable", country_of_manufacture="Finland")
    order = {key: None for key in ORDER_FIELDS}
    order.update(receiver_country="Dominican Republic", total_amount=None)
    return {"order_level": order, "item_level": [item],
            "shipment": {"invoice_number": None, "invoice_date": None, "currency": None}}


ROWS = [
    {"hscode": "8544", "description": "Insulated electric conductors"},
    {"hscode": "854442", "description": "Electric cable with connectors"},
    {"hscode": "854449", "description": "Other electric cable"},
]
NATIONAL = [{"jurisdiction": "DO", "hs6": "854442", "national_code": "8544.42.00.00", "title": "Cable"}]


def make_pdf(path):
    import pymupdf as fitz
    with fitz.open() as document:
        for _ in range(2):
            page = document.new_page(width=72, height=72)
            page.insert_text((5, 20), "Invoice", fontsize=8)
        document.save(path)


class CountryMapTests(TestCase):
    def load(self, payload):
        with patch("AI.tariff.urlopen", return_value=BytesIO(json.dumps(payload).encode())):
            return WeaviateRetriever.load_country_map()

    def test_api_iso_suffixes_and_trade_bloc_annotations(self):
        mapping = self.load({"statuscode": 1, "countries": [
            {"country_name": "Afghanistan - AF", "country_code": "2"},
            {"country_name": "Finland - FI (EU)", "country_code": "74"},
            {"country_name": "Benin - BJ (UEMOA)", "country_code": "24"},
            {"country_name": "United States - US", "country_code": 1},
            {"country_name": "Australia", "country_code": "14"},
        ]})
        self.assertEqual(mapping, {"AF": "2", "FI": "74", "BJ": "24", "US": "1", "AU": "14"})

    def test_standard_schedule_wins_regardless_of_row_order(self):
        rows = [
            {"country_name": "Swiss 11 digits - CH", "country_code": "2208"},
            {"country_name": "Swiss 8 digits - CH", "country_code": "2207"},
            {"country_name": "Switzerland - CH", "country_code": "207"},
            {"country_name": "Israel - IL", "country_code": "107"},
            {"country_name": "Israel (11 digits) - IL", "country_code": "1107"},
        ]
        for ordered in [rows, list(reversed(rows))]:
            self.assertEqual(self.load({"countries": ordered}), {"CH": "207", "IL": "107"})

    def test_ambiguous_standard_schedules_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "ambiguous database IDs for FI"):
            self.load({"countries": [
                {"country_name": "Finland - FI (EU)", "country_code": "74"},
                {"country_name": "Finland - FI", "country_code": "999"},
            ]})

    def test_invalid_inactive_and_mismatched_rows_are_ignored(self):
        mapping = self.load({"countries": [
            None, "invalid", {},
            {"country_name": "Finland - FI", "country_code": "74", "isactive": False},
            {"country_name": "Finland - AU", "country_code": "wrong"},
            {"country_name": "Invalid - ZZ", "country_code": "999"},
            {"country_name": "Finland - FI", "country_code": True},
            {"country_name": "Finland - FI", "country_code": " "},
            {"country_name": "Australia - AU", "country_code": "14"},
        ]})
        self.assertEqual(mapping, {"AU": "14"})

    def test_empty_or_invalid_api_shape_reports_clear_error(self):
        for payload in [[], {}, {"countries": None}]:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "countries array"):
                self.load(payload)
        with self.assertRaisesRegex(ValueError, "no usable ISO-to-database mappings"):
            self.load({"countries": []})


class TariffTests(TestCase):
    def test_country_lookup_is_exact(self):
        for value, expected in [("Finland", "FI"), ("Dominican Republic", "DO"),
                                ("USA", "US"), ("AU", "AU"), ("zzz", None), (None, None)]:
            self.assertEqual(iso2_from_country(value), expected)

    def test_parent_description_contributes_to_retrieval(self):
        retriever = CatalogRetriever([
            {"hscode": "8544", "description": "Insulated electric conductors"},
            {"hscode": "854449", "description": "Other"},
        ])
        matches = retriever.retrieve(invoice()["item_level"][0], invoice()["order_level"])
        self.assertEqual(matches["hs_candidates"][0]["hs_code"], "854449")
        self.assertEqual(matches["origin_iso"], "FI")
        self.assertEqual(matches["destination_iso"], "DO")

    def test_national_mismatch_cannot_be_restored(self):
        result = validate_selection(
            {"hs6": "854449", "destination_national_code": "8544.42.00.00"},
            "PART-42", [{"hs_code": "854442"}, {"hs_code": "854449"}], NATIONAL, [],
        )
        self.assertEqual(result[:3], ("854449", None, None))
        self.assertTrue(result[3])

    def test_no_hs6_means_no_national_selection(self):
        result = validate_selection({"hs6": None, "destination_national_code": "8544.42.00.00"},
                                    "PART", [], NATIONAL, [])
        self.assertEqual(result[:3], (None, None, None))
        self.assertTrue(result[3])

    def test_outside_candidates_and_letter_codes_rejected(self):
        for value in ["999999", "ABC854442", "8544", "null"]:
            self.assertTrue(validate_selection({"hs6": value}, "SKU", [{"hs_code": "854442"}], [], [])[3])

    def test_numeric_sku_rejected_but_alphanumeric_sku_allowed(self):
        args = ([{"hs_code": "854442"}], [], [])
        self.assertTrue(validate_selection({"hs6": "854442"}, "854442", *args)[3])
        self.assertFalse(validate_selection({"hs6": "854442"}, "PART854442", *args)[3])

    def test_overlay_validated_at_load(self):
        with self.assertRaises(ValueError):
            CatalogRetriever(ROWS, [{**NATIONAL[0], "hs6": "854449"}])

    def test_absent_overlay_preserves_hs6_fallback(self):
        inv = invoice()
        matches = CatalogRetriever(ROWS).retrieve(inv["item_level"][0], inv["order_level"])
        generate = Mock(return_value=json.dumps({"hs6": "854442", "needs_review": False, "confidence": "HIGH"}))
        result = classify_item(inv["item_level"][0], inv["order_level"], matches, generate)
        self.assertEqual(result["hs_code"], "854442")
        self.assertIsNone(result["destination_country_hts_code"])
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["confidence"], "MEDIUM")

    def test_empty_description_does_not_call_model(self):
        inv = invoice()
        inv["item_level"][0]["product_description"] = None
        matches = CatalogRetriever(ROWS).retrieve(inv["item_level"][0], inv["order_level"])
        generate = Mock()
        result = classify_item(inv["item_level"][0], inv["order_level"], matches, generate)
        generate.assert_not_called()
        self.assertTrue(result["needs_review"])

    def test_model_failure_is_reviewable(self):
        inv = invoice()
        matches = CatalogRetriever(ROWS).retrieve(inv["item_level"][0], inv["order_level"])
        result = classify_item(inv["item_level"][0], inv["order_level"], matches, Mock(return_value="not JSON"))
        self.assertTrue(result["needs_review"])
        self.assertIsNone(result["hs_code"])
        self.assertIn("Classification failed", result["validation_note"])

    def test_invoice_codes_preserved_as_evidence(self):
        inv = invoice()
        item = inv["item_level"][0]
        item["hs_code"] = "8544.49"
        matches = CatalogRetriever(ROWS).retrieve(item, inv["order_level"])
        result = classify_item(item, inv["order_level"], matches, Mock(return_value='{"hs6":"854442","needs_review":false,"confidence":"MEDIUM"}'))
        self.assertEqual(result["hs_code"], "8544.49")
        self.assertEqual(result["invoice_tariff_codes"]["hs_code"], "8544.49")
        self.assertTrue(result["needs_review"])

    def test_same_country_retrieved_for_both_roles(self):
        inv = invoice()
        inv["item_level"][0]["country_of_manufacture"] = "DO"
        matches = CatalogRetriever(ROWS, NATIONAL).retrieve(inv["item_level"][0], inv["order_level"])
        self.assertEqual(matches["origin_candidates"], matches["destination_candidates"])
        self.assertEqual(len(matches["origin_candidates"]), 1)


class ExternalBoundaryTests(TestCase):
    def test_weaviate_country_scope_and_same_country_roles(self):
        query = Mock()
        query.bm25.return_value.objects = [SimpleNamespace(
            uuid="record-id", properties={"code": "8544.42.00.00", "description": "Electric cable"}
        )]
        client = Mock()
        client.collections.get.return_value.query = query
        fake_query = ModuleType("weaviate.classes.query")
        fake_query.Filter = Mock()
        fake_query.Filter.by_property.return_value.equal.side_effect = lambda value: ("country_code", value)
        fake_query.MetadataQuery = Mock()
        inv = invoice()
        inv["item_level"][0]["country_of_manufacture"] = "Dominican Republic"
        with patch.dict(sys.modules, {"weaviate.classes.query": fake_query}):
            matches = WeaviateRetriever(client, {"DO": "database-country-id"}).retrieve(inv["item_level"][0], inv["order_level"])
        query.bm25.assert_called_once()
        self.assertEqual(query.bm25.call_args.kwargs["filters"], ("country_code", "database-country-id"))
        self.assertEqual(matches["origin_candidates"], matches["destination_candidates"])
        self.assertEqual(matches["hs_candidates"][0]["hs_code"], "854442")

    def test_qwen_context_budget_and_truncation(self):
        import torch
        tokenizer = Mock()
        tokenizer.eos_token_id = 9
        class Encoded(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, device):
                return self

        inputs = Encoded(input_ids=torch.tensor([[1, 2, 3]]))
        tokenizer.return_value = inputs
        model = Mock()
        model.config.max_position_embeddings = 6
        with self.assertRaisesRegex(ValueError, "exceeds model context"):
            qwen_generate([], tokenizer, model, max_new_tokens=4)
        model.generate.assert_not_called()
        model.config.max_position_embeddings = 20
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            qwen_generate([], tokenizer, model, max_new_tokens=4)
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 9]])
        tokenizer.decode.return_value = '{"ok":true}'
        self.assertEqual(qwen_generate([], tokenizer, model, max_new_tokens=4), '{"ok":true}')

    def test_ocr_loader_fails_before_download_on_cpu(self):
        import torch
        from AI import model as model_module
        # Transformers need not be installed to check the CPU failure path.
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = Mock()
        fake_transformers.AutoTokenizer = Mock()
        with patch.dict(sys.modules, {"transformers": fake_transformers}), patch.dict(model_module._models, {}, clear=True), patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                model_module.download_ocr()
            fake_transformers.AutoModel.from_pretrained.assert_not_called()
            fake_transformers.AutoTokenizer.from_pretrained.assert_not_called()


class PipelineTests(TestCase):
    def test_weaviate_connection_closed_if_country_mapping_fails(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            client = Mock()
            with patch("AI.db_connect.connect_vdb", return_value={"weaviate_client": client}), patch(
                "AI.main.WeaviateRetriever.load_country_map", side_effect=ConnectionError("country service unavailable")
            ):
                with self.assertRaisesRegex(ConnectionError, "country service unavailable"):
                    doc_pipeline(pdf, Path(folder) / "output", backend="weaviate")
            client.close.assert_called_once()

    def test_server_adapter_remains_ocr_only(self):
        from server.services.processor import DocumentProcessor

        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.infer_multi.return_value = ("<PAGE>Invoice data", 2)
            output = Path(folder) / "output"
            with patch("AI.model.download_ocr", return_value={"ocr_model": model, "ocr_tokenizer": None}), patch(
                "AI.model.download_qwen"
            ) as load_qwen:
                result = DocumentProcessor().process(pdf, output, lambda stage: None)
            load_qwen.assert_not_called()
            self.assertEqual(Path(result["plain_text_path"]).read_text(), "Invoice data")

    def test_import_and_help_have_no_model_or_network_side_effects(self):
        code = "import sys; import AI.main; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'weaviate' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)
        subprocess.run([sys.executable, "-m", "AI.main", "--help"], check=True, stdout=subprocess.DEVNULL)

    def test_schema_failure_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Invalid invoice JSON"):
            format_json(cleaned_text="Invoice", generate=Mock(return_value='{"invoices":[]}'))

    def test_ocr_returns_text_and_cleans_temporary_pages(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            output = Path(folder) / "output"
            model = Mock()
            model.infer_multi.return_value = ("<PAGE>invoice data", 8)
            text = extract_doc(pdf, output, {"ocr_model": model, "ocr_tokenizer": object()})
            self.assertEqual(text, "<PAGE>invoice data")
            self.assertEqual(len(model.infer_multi.call_args.kwargs["image_files"]), 2)
            self.assertFalse(model.infer_multi.call_args.kwargs["save_results"])
            self.assertFalse(list(output.glob("pages_*")))

    def test_failed_ocr_cleans_temporary_pages(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.infer_multi.side_effect = RuntimeError("inference failed")
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                extract_doc(pdf, folder, {"ocr_model": model, "ocr_tokenizer": None})
            self.assertFalse(list(Path(folder).glob("pages_*")))

    def test_table_cell_boundaries_preserved(self):
        self.assertEqual(clean_ocr_text("<tr><td>sku</td><td>42</td></tr>"), "sku\t42")

    def test_all_invoices_and_duplicate_or_missing_skus(self):
        parsed = {"invoices": [invoice(), invoice()]}
        parsed["invoices"][1]["item_level"].append(deepcopy(invoice(None)["item_level"][0]))
        selections = [json.dumps({"hs6": code, "destination_national_code": national,
                                  "needs_review": False, "confidence": "MEDIUM"})
                      for code, national in [("854442", "8544.42.00.00"), ("854449", None), ("854442", None)]]
        generate = Mock(side_effect=[json.dumps(parsed), *selections])
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            output = Path(folder) / "output"
            ocr = Mock()
            ocr.infer_multi.return_value = ("<PAGE>Invoice one\n<PAGE>Invoice two", 10)
            stages = []
            result = doc_pipeline(pdf, output, models={"ocr": {"ocr_model": ocr, "ocr_tokenizer": None}},
                                  generate=generate, retriever=CatalogRetriever(ROWS, NATIONAL), progress=stages.append)
            self.assertEqual(result["invoices"][0]["item_level"][0]["hs6_code"], "854442")
            self.assertEqual(result["invoices"][1]["item_level"][0]["hs6_code"], "854449")
            self.assertEqual(result["invoices"][1]["item_level"][1]["hs6_code"], "854442")
            matches = json.loads((output / "db_matches.json").read_text())
            self.assertEqual([(m["invoice_index"],m["item_index"]) for m in matches], [(0,0),(1,0),(1,1)])
            self.assertEqual(generate.call_count, 4)
            self.assertEqual(stages[-1], "completed")
            self.assertEqual(json.loads((output / "final_classification.json").read_text()), result)
            # Extraction evidence must remain separate from classification mutations.
            self.assertNotIn("hs6_code", json.loads((output / "extracted_invoice.json").read_text())["invoices"][0]["item_level"][0])

    def test_missing_pdf_fails_before_loading_models(self):
        with patch("AI.main.download_models") as load:
            with self.assertRaises(FileNotFoundError):
                doc_pipeline("/missing/document.pdf")
            load.assert_not_called()


if __name__ == "__main__":
    main()
