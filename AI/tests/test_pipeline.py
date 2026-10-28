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
from AI.utils import clean_ocr_text, extract_doc, format_json, qwen_generate, _ocr_call, pdf_to_images


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
    def test_three_country_invoice_uses_manufacturing_origin_not_export(self):
        inv = invoice()
        inv["order_level"].update(origin_country="United Kingdom", shipper_country="United Kingdom",
                                  destination_country="Australia", receiver_country="Australia")
        item = inv["item_level"][0]
        item["country_of_manufacture"] = "China"
        overlay = [{**NATIONAL[0], "jurisdiction": iso} for iso in ("CN", "GB", "AU")]
        matches = CatalogRetriever(ROWS, overlay).retrieve(item, inv["order_level"])
        self.assertEqual(matches["origin_iso"], "CN")
        self.assertEqual(matches["destination_iso"], "AU")
        self.assertEqual({row["jurisdiction"] for row in matches["origin_candidates"]}, {"CN"})
        item["country_of_manufacture"] = None
        matches = CatalogRetriever(ROWS, overlay).retrieve(item, inv["order_level"])
        self.assertIsNone(matches["origin_iso"])
        self.assertEqual(matches["origin_candidates"], [])

    def test_nz_check_letter_is_validated_by_country_and_exact_membership(self):
        rows = [{"jurisdiction": "NZ", "hs6": "320190", "national_code": "3201900011G"}]
        candidates = [{"hs_code": "320190"}]
        result = validate_selection({"hs6": "320190", "destination_national_code": "3201.90.0011g"},
                                    "SKU", candidates, rows, [])
        self.assertEqual(result[:3], ("320190", "3201900011G", None))
        wrong_letter = validate_selection({"hs6": "320190", "destination_national_code": "3201900011H"},
                                         "SKU", candidates, rows, [])
        self.assertIsNone(wrong_letter[1])
        wrong_country = validate_selection({"hs6": "320190", "destination_national_code": "3201900011G"},
                                          "SKU", candidates, [{**rows[0], "jurisdiction": "AU"}], [])
        self.assertIsNone(wrong_country[1])

    def test_rejected_hs6_does_not_keep_model_success_explanation(self):
        inv = invoice()
        item = inv["item_level"][0]
        item.update(hs_code="3201", product_description="UV printing ink")
        matches = {"hs_candidates": [{"hs_code": "320110", "description": "Quebracho extract"}],
                   "origin_candidates": [{"jurisdiction": "AU", "hs6": "960810", "national_code": "9608100000"}],
                   "destination_candidates": [], "origin_iso": "AU", "destination_iso": "NZ", "source": "test"}
        generate = Mock(return_value=json.dumps({"hs6": "960810", "confidence": "MEDIUM", "needs_review": False,
                                                "classification_reason": "The product matches disposable liquid ink"}))
        result = classify_item(item, inv["order_level"], matches, generate)
        self.assertEqual(result["hs_code"], "3201")
        self.assertIsNone(result["hs6_code"])
        self.assertEqual(result["confidence"], "LOW")
        self.assertIn("rejected", result["classification_reason"])
        self.assertNotIn("product matches", result["classification_reason"])
        prompt = json.loads(generate.call_args.args[0][1]["content"])
        self.assertEqual(prompt["origin_candidates"], [])

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
        self.assertEqual(result["hs6_code"], "854442")
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
        self.assertEqual(result["hs_code"], "854449")
        self.assertEqual(result["invoice_tariff_codes"]["hs_code"], "8544.49")
        self.assertTrue(result["needs_review"])

    def test_invoice_hs_is_four_or_six_digits_with_original_evidence(self):
        inv = invoice()
        item = inv["item_level"][0]
        matches = CatalogRetriever(ROWS).retrieve(item, inv["order_level"])
        for printed, heading in [("8708299000", "870829"), ("0101.21.0000", "010121"),
                                 ("8544", "8544"), ("854442", "854442")]:
            with self.subTest(printed=printed):
                item["hs_code"] = printed
                result = classify_item(item, inv["order_level"], matches, Mock(return_value="invalid"))
                self.assertEqual(result["hs_code"], heading)
                self.assertEqual(result["invoice_tariff_codes"]["hs_code"], printed)

    def test_invoice_origin_hts_wins_even_when_model_fails_or_disagrees(self):
        inv = invoice()
        item = inv["item_level"][0]
        item["manufacturing_country_hts_code"] = "8544.49.0000"
        overlay = [{**NATIONAL[0], "jurisdiction": "FI"}]
        matches = CatalogRetriever(ROWS, overlay).retrieve(item, inv["order_level"])
        for response in ["invalid", json.dumps({"hs6": "854442", "origin_national_code": "8544.42.00.00",
                                               "needs_review": False, "confidence": "HIGH"})]:
            with self.subTest(response=response):
                result = classify_item(item, inv["order_level"], matches, Mock(return_value=response))
                self.assertEqual(result["manufacturing_country_hts_code"], "8544.49.0000")
                self.assertEqual(result["manufacturing_country_hts_source"], "invoice")
                self.assertTrue(result["needs_review"])
                if result["hs6_code"]:
                    self.assertIn("invoice manufacturing-country HTS conflicts", result["validation_note"])

    def test_missing_or_invalid_origin_hts_uses_validated_country_candidate(self):
        inv = invoice()
        item = inv["item_level"][0]
        matches = CatalogRetriever(ROWS, [{**NATIONAL[0], "jurisdiction": "FI"}]).retrieve(item, inv["order_level"])
        response = json.dumps({"hs6": "854442", "origin_national_code": "8544.42.00.00",
                               "needs_review": False, "confidence": "MEDIUM"})
        for value in [None, "UNITED KINGDOM", "GB", "8544", item["sku_code"]]:
            with self.subTest(value=value):
                item["manufacturing_country_hts_code"] = value
                result = classify_item(item, inv["order_level"], matches, Mock(return_value=response))
                self.assertEqual(result["manufacturing_country_hts_code"], "8544.42.00.00")
                self.assertEqual(result["manufacturing_country_hts_source"], matches["source"])

    def test_invalid_origin_evidence_and_rejected_candidate_leave_null(self):
        inv = invoice()
        item = inv["item_level"][0]
        item["manufacturing_country_hts_code"] = "UNITED KINGDOM"
        matches = CatalogRetriever(ROWS).retrieve(item, inv["order_level"])
        response = json.dumps({"hs6": "854442", "origin_national_code": "7616999001",
                               "needs_review": False, "confidence": "MEDIUM"})
        result = classify_item(item, inv["order_level"], matches, Mock(return_value=response))
        self.assertIsNone(result["manufacturing_country_hts_code"])
        self.assertIsNone(result["manufacturing_country_hts_source"])
        self.assertTrue(result["needs_review"])

    def test_invoice_hs_prevents_unrelated_origin_fallback(self):
        inv = invoice()
        item = inv["item_level"][0]
        item.update(hs_code="8708299000", manufacturing_country_hts_code="UNITED KINGDOM")
        matches = CatalogRetriever(ROWS, [{**NATIONAL[0], "jurisdiction": "FI"}]).retrieve(item, inv["order_level"])
        generate = Mock(return_value=json.dumps({"hs6": "854442", "origin_national_code": "8544.42.00.00",
                                                 "needs_review": False, "confidence": "HIGH"}))
        result = classify_item(item, inv["order_level"], matches, generate)
        generate.assert_not_called()
        self.assertEqual(result["hs_code"], "870829")
        self.assertIsNone(result["hs6_code"])
        self.assertIsNone(result["manufacturing_country_hts_code"])
        self.assertIn("No retrieved HS6 matches", result["validation_note"])

    def test_same_country_retrieved_for_both_roles(self):
        inv = invoice()
        inv["item_level"][0]["country_of_manufacture"] = "DO"
        matches = CatalogRetriever(ROWS, NATIONAL).retrieve(inv["item_level"][0], inv["order_level"])
        self.assertEqual(matches["origin_candidates"], matches["destination_candidates"])
        self.assertEqual(len(matches["origin_candidates"]), 1)


class ExternalBoundaryTests(TestCase):
    def test_weaviate_three_countries_queries_product_origin_and_destination_only(self):
        query = Mock()
        query.bm25.return_value.objects = []
        query.fetch_objects.return_value.objects = []
        client = Mock()
        client.collections.get.return_value.query = query
        matches = WeaviateRetriever(client, {"CN": "41", "GB": "225", "AU": "14"}).retrieve(
            {"product_description": "Electric cable", "country_of_manufacture": "China", "hs_code": "8544"},
            {"origin_country": "United Kingdom", "shipper_country": "United Kingdom", "destination_country": "Australia"})
        self.assertEqual(matches["origin_iso"], "CN")
        self.assertEqual({call.kwargs["filters"].value for call in query.bm25.call_args_list}, {"41", "14"})
        self.assertEqual({call.kwargs["filters"].filters[0].value for call in query.fetch_objects.call_args_list}, {"41", "14"})

    def test_ocr_generation_supplies_mask_pad_and_respects_configured_limit(self):
        import torch

        class Model:
            config = SimpleNamespace(sliding_window=128)
            def generate(self, **options):
                self.options = options
                return "generated"
            def infer(self, tokenizer):
                return self.generate(input_ids=torch.tensor([[0, 1, 7]]), max_length=15000)
        model = Model()
        tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=1)
        self.assertEqual(_ocr_call(model, model.infer, tokenizer), "generated")
        self.assertEqual(model.options["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertEqual(model.options["pad_token_id"], 1)
        self.assertIsNone(model.options["max_new_tokens"])
        self.assertEqual(model.options["max_length"], 15000)
        self.assertNotIn("generate", vars(model))

    def test_ocr_generate_method_is_restored_after_exception(self):
        class Model:
            config = SimpleNamespace(sliding_window=128)
            def generate(self, **options):
                raise RuntimeError("Generation failed")
            def infer(self, tokenizer):
                self.config.sliding_window = None
                self.config._ring_window = 128
                return self.generate(max_length=10)
        model = Model()
        with self.assertRaisesRegex(RuntimeError, "Generation failed"):
            _ocr_call(model, model.infer, SimpleNamespace(eos_token_id=1))
        self.assertNotIn("generate", vars(model))
        self.assertEqual(model.config.sliding_window, 128)
        self.assertFalse(hasattr(model.config, "_ring_window"))

    def test_nz_database_rows_keep_check_letters_and_parent_context(self):
        query = Mock()
        query.bm25.return_value.objects = []
        query.fetch_objects.return_value.objects = [
            SimpleNamespace(uuid="parent", properties={"code": "3201", "description": "Vegetable tanning extracts", "is_leaf": False}),
            SimpleNamespace(uuid="national", properties={"code": "3201900011G", "description": "Other", "is_leaf": True}),
            SimpleNamespace(uuid="sku", properties={"code": "PART3201900011G", "description": "Invalid"}),
        ]
        client = Mock()
        client.collections.get.return_value.query = query
        matches = WeaviateRetriever(client, {"NZ": "156"}).retrieve(
            {"hs_code": "3201", "country_of_manufacture": "NZ"}, {"receiver_country": "NZ"})
        self.assertEqual(matches["origin_candidates"][0]["national_code"], "3201900011G")
        self.assertEqual(len(matches["origin_candidates"]), 1)
        self.assertEqual(matches["hs_candidates"][0]["parent_description"], "Vegetable tanning extracts")
        result = classify_item({"product_description": "Other vegetable tanning extract", "hs_code": "3201"},
                               {"receiver_country": "NZ"}, matches, Mock(return_value=json.dumps({
                                   "hs6": "320190", "origin_national_code": "3201900011G",
                                   "destination_national_code": "3201900011G", "confidence": "MEDIUM", "needs_review": False})))
        self.assertEqual(result["destination_country_hts_code"], "3201900011G")

    def test_weaviate_country_scope_and_same_country_roles(self):
        query = Mock()
        query.bm25.return_value.objects = [SimpleNamespace(
            uuid="record-id", properties={"code": "8544.42.00.00", "description": "Electric cable"}
        )]
        query.fetch_objects.return_value.objects = []
        client = Mock()
        client.collections.get.return_value.query = query
        fake_query = ModuleType("weaviate.classes.query")
        fake_query.Filter = Mock()
        class Filters(tuple):
            def __and__(self, other):
                return ("and", self, other)
        fake_query.Filter.by_property.return_value.equal.side_effect = lambda value: Filters(("country_code", value))
        fake_query.Filter.any_of.side_effect = lambda values: Filters(("any", values))
        fake_query.MetadataQuery = Mock()
        inv = invoice()
        inv["item_level"][0]["country_of_manufacture"] = "Dominican Republic"
        with patch.dict(sys.modules, {"weaviate.classes.query": fake_query}):
            matches = WeaviateRetriever(client, {"DO": "database-country-id"}).retrieve(inv["item_level"][0], inv["order_level"])
        query.bm25.assert_called_once()
        self.assertEqual(query.bm25.call_args.kwargs["filters"], ("country_code", "database-country-id"))
        self.assertEqual(matches["origin_candidates"], matches["destination_candidates"])
        self.assertEqual(matches["hs_candidates"][0]["hs_code"], "854442")
        query.fetch_objects.assert_called_once()

    def test_weaviate_code_lookup_recovers_other_rows_for_both_countries(self):
        from weaviate.classes.query import Filter

        query = Mock()
        query.bm25.return_value.objects = [SimpleNamespace(
            uuid="irrelevant", properties={"code": "7616999001", "description": "Aluminium radiator"})]
        query.fetch_objects.side_effect = [SimpleNamespace(objects=[
            SimpleNamespace(uuid="au", properties={"code": "8708299194", "description": "Other"}),
            SimpleNamespace(uuid="au-duplicate", properties={"code": "8708299194", "description": "Other"}),
            SimpleNamespace(uuid="deleted", properties={"code": "8708299995", "description": "Other", "isdelete": True}),
            SimpleNamespace(uuid="heading", properties={"code": "8708299", "description": "Other", "is_leaf": False}),
        ]), SimpleNamespace(objects=[
            SimpleNamespace(uuid="gb", properties={"code": "8708299000", "description": "Other"}),
        ])]
        client = Mock()
        client.collections.get.return_value.query = query
        matches = WeaviateRetriever(client, {"GB": "225", "AU": "14"}).retrieve(
            {"product_description": "Radiator Grille", "country_of_manufacture": "GB", "hs_code": "8708299000"},
            {"destination_country": "Australia", "receiver_country": None})
        self.assertEqual(query.fetch_objects.call_count, 2)
        self.assertEqual(matches["origin_candidates"][-1]["national_code"], "8708299000")
        self.assertEqual([r["national_code"] for r in matches["destination_candidates"] if r["hs6"] == "870829"], ["8708299194"])
        self.assertIn("870829", [r["hs_code"] for r in matches["hs_candidates"]])
        for call, country_id in zip(query.fetch_objects.call_args_list, ["14", "225"]):
            expected = Filter.by_property("country_code").equal(country_id) & Filter.any_of([
                Filter.by_property("code").like("8708.29*"), Filter.by_property("code").like("870829*")])
            actual = call.kwargs["filters"]
            self.assertEqual(actual.filters[0], expected.filters[0])
            self.assertEqual(actual.filters[1].filters, expected.filters[1].filters)

    def test_weaviate_missing_invoice_code_expands_discovered_hs6(self):
        query = Mock()
        query.bm25.return_value.objects = [SimpleNamespace(
            uuid="hs6", properties={"code": "854442", "description": "Electric cable"})]
        query.fetch_objects.return_value.objects = [SimpleNamespace(
            uuid="national", properties={"code": "8544420000", "description": "Other"})]
        client = Mock()
        client.collections.get.return_value.query = query
        matches = WeaviateRetriever(client, {"FI": "74"}).retrieve(
            {"product_description": "Electric cable", "country_of_manufacture": "FI"},
            {"receiver_country": "FI"})
        self.assertEqual(matches["origin_candidates"], matches["destination_candidates"])
        self.assertEqual(matches["origin_candidates"][0]["national_code"], "8544420000")
        query.fetch_objects.assert_called_once()

    def test_weaviate_skips_origin_country_when_invoice_origin_code_exists(self):
        query = Mock()
        query.bm25.return_value.objects = []
        query.fetch_objects.return_value.objects = []
        client = Mock()
        client.collections.get.return_value.query = query
        WeaviateRetriever(client, {"GB": "225", "AU": "14"}).retrieve(
            {"product_description": "Grille", "country_of_manufacture": "GB",
             "manufacturing_country_hts_code": "8708299000"}, {"receiver_country": "AU"})
        query.bm25.assert_called_once()
        self.assertEqual(query.bm25.call_args.kwargs["filters"].value, "14")
        query.fetch_objects.assert_called_once()
        self.assertEqual(query.fetch_objects.call_args.kwargs["filters"].filters[0].value, "14")

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
    def test_pdf_decompression_warning_reopens_before_returning_images(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            with patch("pymupdf.TOOLS.mupdf_warnings", side_effect=["", "zlib error: incorrect header check", "", "", ""]), self.assertLogs(
                "AI.utils", level="WARNING"
            ):
                paths = pdf_to_images(pdf, dpi=72, output_dir=Path(folder) / "pages")
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_pdf_with_broken_deflate_stream_is_rejected_before_ocr(self):
        import pymupdf

        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "corrupt.pdf"
            with pymupdf.open() as doc:
                page = doc.new_page()
                page.insert_text((50, 50), "Invoice")
                xref = page.get_contents()[0]
                doc.update_stream(xref, b"this is not a zlib stream", compress=False)
                doc.xref_set_key(xref, "Filter", "/FlateDecode")
                doc.save(pdf)
            model = Mock()
            with self.assertLogs("AI.utils", level="WARNING"), self.assertRaisesRegex(ValueError, "compressed-stream decoding error"):
                extract_doc(pdf, Path(folder) / "out", {"ocr_model": model, "ocr_tokenizer": None})
            model.infer_multi.assert_not_called()
            self.assertFalse(list((Path(folder) / "out").glob("pages_*")))

    def test_expanded_order_fields_are_extracted_without_changing_values(self):
        order = {
            "company_name": "Acme Corporation",
            "company_address": "123 Main St, Springfield, IL 62704, United States",
            "invoice_number": "INV-2026-001", "invoice_date": "2026-05-20",
            "purchase_order_number": "PO-4789", "currency": "USD",
            "payment_terms": "Net 30", "inco_terms": "FOB", "tax_id": "ERO/GST/FED",
            "tracking_number": "1Z999AA10123456784", "shipper_company": "Acme Logistics",
            "shipper_email": "shipments@acme.com", "shipper_phone_number": "+1 555-0100",
            "shipper_country": "United States", "shipper_state": "IL",
            "shipper_city": "Springfield", "shipper_zipcode": "62704",
            "shipper_address1": "123 Main St", "shipper_address2": "Suite 400",
            "receiver_company": "Global Imports Ltd", "receiver_firstname": "Jane",
            "receiver_lastname": "Doe", "receiver_phone_number": "+44 20 7946 0000",
            "receiver_country": "United Kingdom", "receiver_state": "Greater London",
            "receiver_city": "London", "receiver_zipcode": "SW1A 1AA",
            "receiver_address1": "456 Industrial Way", "receiver_address2": "Unit 12",
            "receiver_email": "jane.doe@globalimports.co.uk", "gross_weight": 420,
            "net_weight": 400, "package_count": 12, "total_quantity": 50,
            "total_amount": 12500.0, "shipping_and_handling": None,
            "total_item_value": None, "dimensions_cm": "40 x 30 x 31",
            "origin_phone_code": "+1", "destination_phone_code": "+44",
            "origin_country": "United States", "destination_country": "United Kingdom",
        }
        payload = {"invoices": [{**invoice(), "order_level": order}]}
        generate = Mock(return_value=json.dumps(payload))
        parsed = format_json(cleaned_text="Invoice", generate=generate)
        self.assertEqual(parsed["invoices"][0]["order_level"], order)
        self.assertEqual(list(order), ORDER_FIELDS)
        # The extraction model receives the expanded contract automatically.
        system = generate.call_args.args[0][0]["content"]
        contract = json.loads(system.split("JSON Schema:\n", 1)[1])
        required = contract["properties"]["invoices"]["items"]["properties"]["order_level"]["required"]
        self.assertEqual(required, [])

    def test_omitted_fields_default_to_null_and_are_saved(self):
        payload = {"invoices": [{"order_level": {"invoice_number": "INV-1"},
                                 "item_level": [{"product_description": "Cable"}],
                                 "shipment": {}}]}
        with TemporaryDirectory() as folder:
            parsed = format_json(cleaned_text="Invoice", output_path=folder,
                                 generate=Mock(return_value=json.dumps(payload)))
            inv = parsed["invoices"][0]
            self.assertEqual(inv["order_level"], {
                **dict.fromkeys(ORDER_FIELDS), "invoice_number": "INV-1"})
            self.assertEqual(inv["item_level"][0], {
                **dict.fromkeys(ITEM_FIELDS + NUMBER_FIELDS), "product_description": "Cable"})
            self.assertEqual(inv["shipment"], dict.fromkeys(("invoice_number", "invoice_date", "currency")))
            self.assertEqual(json.loads((Path(folder) / "extracted_invoice.json").read_text()), parsed)

    def test_missing_order_and_shipment_groups_default_to_null_fields(self):
        parsed = format_json(cleaned_text="Invoice", generate=Mock(return_value=json.dumps(
            {"invoices": [{"item_level": [{}]}]})))
        self.assertEqual(parsed["invoices"][0]["order_level"], dict.fromkeys(ORDER_FIELDS))
        self.assertIsNone(parsed["invoices"][0]["shipment"]["invoice_number"])

    def test_order_numbers_allow_fractions_and_missing_values(self):
        payload = {"invoices": [invoice()]}
        payload["invoices"][0]["order_level"].update(
            gross_weight=420.5, net_weight=400.25, package_count=12,
            total_quantity=50.5, total_amount=12500.75,
            shipping_and_handling=10.5, total_item_value=12490.25,
        )
        self.assertEqual(format_json(cleaned_text="Invoice", generate=Mock(return_value=json.dumps(payload))), payload)
        payload["invoices"][0]["order_level"] = {key: None for key in ORDER_FIELDS}
        self.assertEqual(format_json(cleaned_text="Invoice", generate=Mock(return_value=json.dumps(payload))), payload)

    def test_invalid_order_field_types_fail_extraction(self):
        for field, value in (("gross_weight", "420"), ("net_weight", "400"),
                             ("package_count", 1.5), ("total_quantity", "50"),
                             ("shipping_and_handling", "10"), ("total_item_value", True),
                             ("dimensions_cm", 40), ("origin_phone_code", 1),
                             ("receiver_email", 123)):
            with self.subTest(field=field):
                payload = {"invoices": [invoice()]}
                payload["invoices"][0]["order_level"][field] = value
                with self.assertRaisesRegex(ValueError, "Invalid invoice JSON"):
                    format_json(cleaned_text="Invoice", generate=Mock(return_value=json.dumps(payload)))

    def test_weaviate_connection_closed_if_country_mapping_fails(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            client = Mock()
            with patch("AI.main.extract_doc"), patch("AI.main.clean_extracted_text", return_value="Invoice"), patch(
                "AI.main.format_json", return_value={"invoices": [invoice()]}
            ), patch("AI.db_connect.connect_vdb", return_value={"weaviate_client": client}), patch(
                "AI.main.WeaviateRetriever.load_country_map", side_effect=ConnectionError("country service unavailable")
            ):
                with self.assertRaisesRegex(ConnectionError, "country service unavailable"):
                    doc_pipeline(pdf, Path(folder) / "output", backend="weaviate", models={"ocr": {}}, generate=Mock())
            client.close.assert_called_once()

    def test_complete_invoice_codes_skip_db_catalog_and_classification(self):
        inv = invoice()
        inv["item_level"][0].update(hs_code="8544", manufacturing_country_hts_code="8544490000",
                                    destination_country_hts_code="8544490010")
        generate = Mock(return_value=json.dumps({"invoices": [inv]}))
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            with patch("AI.main.extract_doc"), patch("AI.main.clean_extracted_text", return_value="Invoice"), patch(
                "AI.db_connect.connect_vdb", side_effect=AssertionError("DB must not be contacted")
            ) as connect, patch("AI.main.CatalogRetriever.load") as load_catalog:
                result = doc_pipeline(pdf, Path(folder) / "out", backend="weaviate", models={"ocr": {}}, generate=generate)
            connect.assert_not_called()
            load_catalog.assert_not_called()
            generate.assert_called_once()  # Extraction only; no classification.
            item = result["invoices"][0]["item_level"][0]
            self.assertEqual(item["hs_code"], "8544")
            self.assertEqual(item["manufacturing_country_hts_code"], "8544490000")
            self.assertEqual(item["destination_country_hts_code"], "8544490010")
            self.assertEqual(item["hts_extraction_source"], "invoice")
            self.assertNotIn("unavailable", item["validation_note"])

    def test_nz_invoice_codes_skip_retrieval_with_check_letter(self):
        from AI.utils import fetch_hts
        inv = invoice()
        inv["order_level"]["receiver_country"] = "NZ"
        item = inv["item_level"][0]
        item.update(manufacturing_country_hts_code="3201900029", destination_country_hts_code="3201900011G")
        retriever = Mock()
        matches = fetch_hts({"invoices": [inv]}, retriever)
        retriever.retrieve.assert_not_called()
        generate = Mock()
        result = classify_item(item, inv["order_level"], matches[0], generate)
        generate.assert_not_called()
        self.assertEqual(result["destination_country_hts_code"], "3201900011G")

    def test_mixed_invoice_retrieves_only_items_with_missing_codes(self):
        from AI.utils import fetch_hts
        complete = invoice()["item_level"][0]
        complete.update(manufacturing_country_hts_code="8544490000", destination_country_hts_code="8544490010")
        missing = invoice()["item_level"][0]
        inv = invoice()
        inv["item_level"] = [complete, missing]
        retriever = Mock()
        retriever.retrieve.return_value = CatalogRetriever(ROWS).retrieve(missing, inv["order_level"])
        matches = fetch_hts({"invoices": [inv]}, retriever)
        retriever.retrieve.assert_called_once_with(missing, inv["order_level"])
        self.assertEqual(matches[0]["source"], "invoice")
        self.assertEqual(matches[1]["item_index"], 1)

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
            model.infer.assert_not_called()
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

    def test_empty_multi_page_ocr_retries_every_page_in_order(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.infer_multi.return_value = ("<PAGE>\n", 1)
            model.infer.side_effect = ["Invoice first page", "Invoice second page"]
            with self.assertLogs("AI.utils", level="WARNING"):
                raw = extract_doc(pdf, folder, {"ocr_model": model, "ocr_tokenizer": None})
            self.assertEqual(raw, "Invoice first page\n<PAGE>\nInvoice second page")
            self.assertEqual(model.infer_multi.call_args.kwargs["prompt"], "<image>Document parsing.")
            self.assertEqual(model.infer.call_count, 2)
            calls = model.infer.call_args_list
            self.assertTrue(calls[0].kwargs["eval_mode"])
            self.assertTrue(calls[0].kwargs["crop_mode"])
            self.assertTrue(calls[0].kwargs["image_file"].endswith("page_0001.png"))
            self.assertTrue(calls[1].kwargs["image_file"].endswith("page_0002.png"))
            self.assertEqual((Path(folder) / "detailed_invoice_output.txt").read_text(), raw)
            self.assertFalse(list(Path(folder).glob("pages_*")))

    def test_empty_retry_reports_page_and_does_not_persist_partial_invoice(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.infer_multi.return_value = ("", 0)
            model.infer.side_effect = ["Page one", ""]
            with self.assertLogs("AI.utils", level="WARNING"), self.assertRaisesRegex(RuntimeError, "page 2/2"):
                extract_doc(pdf, folder, {"ocr_model": model, "ocr_tokenizer": None})
            self.assertFalse((Path(folder) / "detailed_invoice_output.txt").exists())
            self.assertFalse(list(Path(folder).glob("pages_*")))

    def test_unsupported_ocr_return_type_is_explicit(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.infer_multi.return_value = {"text": "Unsupported"}
            with self.assertRaisesRegex(RuntimeError, "Unsupported OCR return type: dict"):
                extract_doc(pdf, folder, {"ocr_model": model, "ocr_tokenizer": None})
            model.infer.assert_not_called()
            self.assertFalse(list(Path(folder).glob("pages_*")))

    def test_ocr_restores_settings_after_failure_before_next_document(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.config = SimpleNamespace(sliding_window=128)
            def infer(*args, **kwargs):
                self.assertEqual(model.config.sliding_window, 128)
                self.assertFalse(hasattr(model.config, "_ring_window"))
                model.config.sliding_window = None
                model.config._ring_window = 128
                if model.infer_multi.call_count == 1:
                    raise RuntimeError("First request failed during generation")
                return "Second invoice", 2
            model.infer_multi.side_effect = infer
            bundle = {"ocr_model": model, "ocr_tokenizer": None}
            with self.assertRaisesRegex(RuntimeError, "First request failed"):
                extract_doc(pdf, Path(folder) / "first", bundle)
            self.assertEqual(model.config.sliding_window, 128)
            self.assertFalse(hasattr(model.config, "_ring_window"))
            self.assertEqual(extract_doc(pdf, Path(folder) / "second", bundle), "Second invoice")
            self.assertEqual(model.config.sliding_window, 128)
            self.assertFalse(hasattr(model.config, "_ring_window"))

    def test_ocr_commentary_retries_and_is_never_passed_to_extraction(self):
        commentary = ('The Ground Truth image displays a single solid horizontal line. '
                      'According to Rule 2, the provided OCR result is incorrect.')
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.infer_multi.return_value = (commentary, 40)
            model.infer.side_effect = ["Invoice first page", "Invoice second page"]
            with self.assertLogs("AI.utils", level="WARNING"):
                raw = extract_doc(pdf, folder, {"ocr_model": model, "ocr_tokenizer": None})
            self.assertNotIn("Ground Truth", raw)
            self.assertIn("Invoice first page", raw)
            self.assertEqual(model.infer.call_count, 2)
            model.infer.side_effect = None
            model.infer.return_value = commentary
            with self.assertLogs("AI.utils", level="WARNING"), self.assertRaisesRegex(RuntimeError, "evaluation commentary"):
                extract_doc(pdf, Path(folder) / "bad", {"ocr_model": model, "ocr_tokenizer": None})
            self.assertFalse((Path(folder) / "bad" / "detailed_invoice_output.txt").exists())

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
