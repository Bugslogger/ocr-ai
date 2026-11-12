"""Regression tests using tiny local PDFs and fake model/database responses."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
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
from AI.utils import clean_ocr_text, extract_doc, format_json, qwen_generate, pdf_to_images


def save_fake_extraction(output, text):
    (output / "extracted_text.txt").write_text(text, encoding="utf-8")
    (output / "layout_aware_text.txt").write_text('{"pages":[]}', encoding="utf-8")


def ocr_result(text):
    return {"rec_texts": [text], "rec_boxes": [[0, 0, 200, 20]]}


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


class WorkerConcurrencyTests(TestCase):
    def test_workers_cache_separate_ocr_predictors(self):
        from AI import model

        paddleocr = ModuleType("paddleocr")
        paddleocr.PaddleOCR = Mock(side_effect=[Mock(), Mock()])
        hub = ModuleType("huggingface_hub")
        hub.snapshot_download = Mock(return_value="/fake/model")
        with patch.dict(sys.modules, {"paddleocr": paddleocr, "huggingface_hub": hub}), patch.dict(
            model._models, {}, clear=True
        ):
            bundles = []
            for _ in range(2):
                with model.model_context():
                    bundle = model.download_ocr()
                    self.assertIs(bundle, model.download_ocr())
                    self.assertIn("ocr", model.loaded_model_kinds())
                    bundles.append(bundle)
            self.assertIsNot(bundles[0]["ocr_model"], bundles[1]["ocr_model"])
            self.assertNotIn("ocr", model._models)
            self.assertEqual(paddleocr.PaddleOCR.call_count, 2)

    def test_real_pipeline_and_nested_product_stage_overlap_between_workers(self):
        from AI.model import model_context, cached_models, pipeline_lock, ocr_lock
        from AI.product_pipeline import identify_products

        overlap = Barrier(2)

        def identify(item, generate):
            overlap.wait(timeout=3)
            return {"search_query": "cable"}

        def extract(*args, **kwargs):
            # Exercise the nested product lock while the invoice lock is held.
            identify_products({"invoices": [invoice()]}, generate=Mock())
            save_fake_extraction(args[1], "invoice")

        def run(pdf, output):
            with model_context():
                cache = cached_models()
                cache["qwen"] = {"worker": str(output)}
                result = doc_pipeline(pdf, output, models={}, generate=Mock())
                return result, cache, pipeline_lock(), ocr_lock()

        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            pdf.touch()
            with patch("AI.product_pipeline._identify", side_effect=identify), patch(
                "AI.main.extract_doc", side_effect=extract
            ), patch(
                "AI.main.format_json", return_value={"invoices": []}
            ), patch("AI.main.fetch_hts", return_value=[]), patch(
                "AI.main.update_invoice", return_value={"invoices": []}
            ), ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(run, pdf, Path(folder) / "first")
                second = pool.submit(run, pdf, Path(folder) / "second")
                results = [first.result(timeout=5), second.result(timeout=5)]
            self.assertEqual(results[0][0], {"invoices": []})
            self.assertEqual(results[1][0], {"invoices": []})
            for left, right in zip(results[0][1:], results[1][1:]):
                self.assertIsNot(left, right)


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
        inv["order_level"].update(origin_country="China", export_country="United Kingdom", shipper_country="United Kingdom",
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
        self.assertEqual(result["destination_country_hts_code"], "854442")
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

    def test_invalid_origin_evidence_and_rejected_candidate_use_validated_hs6(self):
        inv = invoice()
        item = inv["item_level"][0]
        item["manufacturing_country_hts_code"] = "UNITED KINGDOM"
        matches = CatalogRetriever(ROWS).retrieve(item, inv["order_level"])
        response = json.dumps({"hs6": "854442", "origin_national_code": "7616999001",
                               "needs_review": False, "confidence": "MEDIUM"})
        result = classify_item(item, inv["order_level"], matches, Mock(return_value=response))
        self.assertEqual(result["manufacturing_country_hts_code"], "854442")
        self.assertEqual(result["manufacturing_country_hts_source"], "hs6_fallback")
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
        self.assertEqual(result["manufacturing_country_hts_code"], "870829")
        self.assertEqual(result["manufacturing_country_hts_source"], "hs6_fallback")
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
            {"origin_country": "China", "export_country": "United Kingdom", "shipper_country": "United Kingdom", "destination_country": "Australia"})
        self.assertEqual(matches["origin_iso"], "CN")
        self.assertEqual({call.kwargs["filters"].value for call in query.bm25.call_args_list}, {"41", "14"})
        self.assertEqual({call.kwargs["filters"].filters[0].value for call in query.fetch_objects.call_args_list}, {"41", "14"})


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
        with self.assertRaisesRegex(ValueError, "Document input is too large"):
            qwen_generate([], tokenizer, model, max_new_tokens=4)
        model.generate.assert_not_called()
        model.config.max_position_embeddings = 32768
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            qwen_generate([], tokenizer, model, max_new_tokens=4)
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 9]])
        tokenizer.decode.return_value = '{"ok":true}'
        self.assertEqual(qwen_generate([], tokenizer, model, max_new_tokens=4), '{"ok":true}')
        tokenizer.apply_chat_template.assert_called_with(
            [], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def test_qwen_greedy_generation_clears_inherited_sampling_flags(self):
        import torch
        from transformers import GenerationConfig

        class Encoded(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, device):
                return self

        tokenizer = Mock()
        tokenizer.eos_token_id = 9
        tokenizer.return_value = Encoded(input_ids=torch.tensor([[1, 2, 3]]))
        tokenizer.decode.return_value = '{"ok":true}'
        model = Mock()
        model.device = torch.device("cpu")
        model.config.max_position_embeddings = 32768

        def generate(**kwargs):
            # Simulate generate() merging call options with pretrained defaults.
            config = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
            config.update(**{key: value for key, value in kwargs.items() if key != "input_ids"})
            config.validate(strict=True)
            self.assertEqual(config.do_sample, sampling)
            if sampling:
                self.assertEqual((config.temperature, config.top_p, config.top_k), (0.5, 0.8, 20))
            else:
                self.assertEqual((config.temperature, config.top_p, config.top_k), (None, None, None))
            return torch.tensor([[1, 2, 3, 9]])

        model.generate.side_effect = generate
        for sampling, temperature in ((False, 0.0), (True, 0.5)):
            with self.subTest(sampling=sampling):
                self.assertEqual(qwen_generate([], tokenizer, model, max_new_tokens=4,
                                               temperature=temperature), '{"ok":true}')

    def test_qwen_8b_is_quantized_on_one_gpu_and_cached(self):
        import torch
        from AI import model as module
        from AI.config import Qruntime
        from AI.model_config import QWEN_QUANTIZATION_CONFIG
        transformers = ModuleType("transformers")
        transformers.AutoModelForCausalLM = Mock()
        transformers.AutoTokenizer = Mock()
        transformers.BitsAndBytesConfig = Mock()
        with patch.dict(sys.modules, {"transformers": transformers}), patch.dict(module._models, {}, clear=True), patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.is_bf16_supported", return_value=False), patch.dict(
            QWEN_QUANTIZATION_CONFIG, {"load_in_4bit": True}
        ), patch.dict(Qruntime, {"device": "cuda:0", "compute_dtype": "auto"}):
            first = module.download_qwen()
            self.assertIs(first, module.download_qwen())
        options = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.args[0], "Qwen/Qwen2.5-14B-Instruct-1M")
        self.assertEqual(options["device_map"], {"": "cuda:0"})
        self.assertEqual(options["attn_implementation"], "sdpa")
        transformers.BitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_ocr_loader_uses_exact_models_and_caches_pipeline(self):
        from AI import model as module
        from AI.config import Oconfig
        paddleocr = ModuleType("paddleocr")
        paddleocr.PaddleOCR = Mock()
        with patch.dict(sys.modules, {"paddleocr": paddleocr}), patch.dict(module._models, {}, clear=True), patch(
            "huggingface_hub.snapshot_download", side_effect=["/det", "/rec"]
        ) as download, patch.dict(Oconfig, {"device": "gpu:0"}):
            first = module.download_ocr()
            self.assertIs(first, module.download_ocr())
            self.assertEqual(download.call_count, 2)
        options = paddleocr.PaddleOCR.call_args.kwargs
        self.assertEqual(options["text_detection_model_name"], "PP-OCRv5_mobile_det")
        self.assertEqual(options["text_recognition_model_name"], "en_PP-OCRv5_mobile_rec")
        self.assertEqual(options["device"], "gpu:0")
        self.assertFalse(options["use_doc_orientation_classify"])
        self.assertFalse(options["use_doc_unwarping"])
        self.assertFalse(options["use_textline_orientation"])
        self.assertEqual(first, {"ocr_model": paddleocr.PaddleOCR.return_value})

    def test_qwen_full_precision_gpu_loading_honors_disabled_quantization(self):
        from AI import model as module
        from AI.model_config import QWEN_QUANTIZATION_CONFIG, Qruntime
        transformers = ModuleType("transformers")
        transformers.AutoModelForCausalLM = Mock()
        transformers.AutoTokenizer = Mock()
        transformers.BitsAndBytesConfig = Mock()
        with patch.dict(sys.modules, {"transformers": transformers}), patch.dict(module._models, {}, clear=True), patch(
            "torch.cuda.is_available", return_value=True
        ), patch.dict(QWEN_QUANTIZATION_CONFIG, {"load_in_4bit": False, "load_in_8bit": False}), patch.dict(
            Qruntime, {"device": "cuda:0", "compute_dtype": "bfloat16"}
        ):
            module.download_qwen()
        transformers.BitsAndBytesConfig.assert_not_called()
        options = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertNotIn("quantization_config", options)
        self.assertEqual(options["device_map"], {"": "cuda:0"})


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
                extract_doc(pdf, Path(folder) / "out", {"ocr_model": model})
            model.predict.assert_not_called()
            self.assertFalse(list((Path(folder) / "out").glob("pages_*")))

    def test_origin_and_export_have_separate_extraction_fields(self):
        from AI.schema import COUNTRY_DESCRIPTIONS

        inv = invoice()
        inv["order_level"].update(origin_country="China", export_country="United Kingdom",
                                  destination_country="Australia")
        inv["item_level"][0]["country_of_manufacture"] = "China"
        generate = Mock(return_value=json.dumps({"invoices": [inv]}))
        result = format_json(
            cleaned_text="Country of Export: United Kingdom\nCountry of Origin: China\nShip To: Australia",
            generate=generate,
        )["invoices"][0]
        self.assertEqual(result["order_level"]["origin_country"], "China")
        self.assertEqual(result["order_level"]["export_country"], "United Kingdom")
        self.assertEqual(result["item_level"][0]["country_of_manufacture"], "China")
        system = generate.call_args.args[0][0]["content"]
        contract = json.loads(system.split("JSON Schema:\n", 1)[1])
        fields = contract["properties"]["invoices"]["items"]["properties"]["order_level"]["properties"]
        for field in ("origin_country", "export_country"):
            self.assertEqual(fields[field]["description"], COUNTRY_DESCRIPTIONS[field])

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
            "origin_country": "United States", "destination_country": "United Kingdom", "export_country": "United States",
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
            with patch("AI.main.extract_doc", side_effect=lambda pdf, output, *a, **kw: save_fake_extraction(output, "Invoice")), patch(
                "AI.main.format_json", return_value={"invoices": [invoice()]}
            ), patch("AI.db_connect.connect_vdb", return_value={"weaviate_client": client}), patch(
                "AI.main.WeaviateRetriever.load_country_map", side_effect=ConnectionError("country service unavailable")
            ), patch("AI.main.CatalogRetriever.load", return_value=CatalogRetriever(ROWS)
            ):
                with self.assertRaisesRegex(ConnectionError, "country service unavailable"):
                    doc_pipeline(pdf, Path(folder) / "output", backend="weaviate", models={"ocr": {}}, generate=Mock())
            client.close.assert_called_once()

    def test_complete_invoice_codes_skip_db_catalog_and_classification(self):
        extracted = "  Invoice <part> &amp;\t£100\n\nSecond page  "
        inv = invoice()
        inv["item_level"][0].update(hs_code="8544", manufacturing_country_hts_code="8544490000",
                                    destination_country_hts_code="8544490010")
        generate = Mock(return_value=json.dumps({"invoices": [inv]}))
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            with patch("AI.main.extract_doc", side_effect=lambda pdf, output, *a, **kw: save_fake_extraction(output, extracted)), patch(
                "AI.db_connect.connect_vdb", side_effect=AssertionError("DB must not be contacted")
            ) as connect, patch("AI.main.CatalogRetriever.load") as load_catalog:
                result = doc_pipeline(pdf, Path(folder) / "out", backend="weaviate", models={"ocr": {}}, generate=generate)
            connect.assert_not_called()
            load_catalog.assert_not_called()
            generate.assert_called_once()  # Extraction only; no classification.
            self.assertEqual(generate.call_args.args[0][1]["content"], "OCR Text:\n\n" + extracted
                             + '\n\nLayout-aware OCR (JSON):\n{"pages":[]}')
            self.assertFalse((Path(folder) / "out" / "plain_invoice_text.txt").exists())
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
            model.predict.return_value = [ocr_result("Invoice data")]
            output = Path(folder) / "output"
            with patch("AI.model.download_ocr", return_value={"ocr_model": model}), patch(
                "AI.model.download_qwen"
            ) as load_qwen:
                result = DocumentProcessor().process(pdf, output, lambda stage: None)
            load_qwen.assert_not_called()
            self.assertEqual(Path(result["plain_text_path"]).read_text(), "Invoice data\n\nInvoice data")

    def test_import_and_help_have_no_model_or_network_side_effects(self):
        code = "import sys; import AI.main; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'weaviate' not in sys.modules; assert 'paddleocr' not in sys.modules; assert 'paddle' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)
        subprocess.run([sys.executable, "-m", "AI.main", "--help"], check=True, stdout=subprocess.DEVNULL)

    def test_schema_failure_is_explicit(self):
        generate = Mock(return_value='{"invoices":[]}')
        with self.assertRaisesRegex(ValueError, "Invalid invoice JSON"):
            format_json(cleaned_text="Invoice", generate=generate)
        generate.assert_called_once()

    def test_invalid_json_regenerates_complete_invoice_and_keeps_diagnostics(self):
        payload = {"invoices": [invoice("FIRST"), invoice("SECOND")]}
        valid = json.dumps(payload)
        malformed = valid.replace('"total_amount": null', '"total_amount": ', 1)
        generate = Mock(side_effect=[malformed, valid])
        with TemporaryDirectory() as folder, self.assertLogs("AI.utils", level="WARNING"):
            result = format_json(cleaned_text="Original invoice source", output_path=folder,
                                 system_prompt="Custom extraction instructions", generate=generate)
            self.assertEqual(result, payload)
            self.assertEqual((Path(folder) / "invalid_invoice_response_1.txt").read_text(), malformed)
            self.assertEqual(json.loads((Path(folder) / "extracted_invoice.json").read_text()), payload)
        self.assertEqual(generate.call_count, 2)
        retry = generate.call_args.args[0]
        self.assertEqual(retry[0]["content"], "Custom extraction instructions")
        self.assertIn("Original invoice source", retry[1]["content"])
        self.assertIn("Expecting value", retry[1]["content"])
        self.assertEqual(generate.call_args.kwargs, generate.call_args_list[0].kwargs)

    def test_repeated_invalid_json_fails_without_saving_invoice(self):
        generate = Mock(side_effect=['{"invoices": ', '```json\n{"invoices": [}\n```'])
        with TemporaryDirectory() as folder, self.assertLogs("AI.utils", level="WARNING"):
            with self.assertRaisesRegex(ValueError, "invalid JSON after 2 attempts") as raised:
                format_json(cleaned_text="Invoice", output_path=folder, generate=generate)
            self.assertIsInstance(raised.exception.__cause__, json.JSONDecodeError)
            self.assertEqual((Path(folder) / "invalid_invoice_response_1.txt").read_text(), '{"invoices": ')
            self.assertEqual((Path(folder) / "invalid_invoice_response_2.txt").read_text(),
                             '```json\n{"invoices": [}\n```')
            self.assertFalse((Path(folder) / "extracted_invoice.json").exists())
        self.assertEqual(generate.call_count, 2)

    def test_truncated_generation_is_not_retried(self):
        generate = Mock(side_effect=RuntimeError("Qwen output reached the token limit"))
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            format_json(cleaned_text="Invoice", generate=generate)
        generate.assert_called_once()

    def test_ocr_returns_text_and_cleans_temporary_pages(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            output = Path(folder) / "output"
            model = Mock()
            model.predict.side_effect = [[ocr_result("Invoice one")], [ocr_result("Invoice two")]]
            text = extract_doc(pdf, output, {"ocr_model": model})
            self.assertEqual(text, "Invoice one\n<PAGE>\nInvoice two")
            self.assertEqual(model.predict.call_count, 2)
            paths = [c.kwargs["input"] for c in model.predict.call_args_list]
            self.assertTrue(paths[0].endswith("page_0001.png"))
            self.assertTrue(paths[1].endswith("page_0002.png"))
            self.assertFalse(any(Path(path).exists() for path in paths))
            self.assertEqual((output / "detailed_invoice_output.txt").read_text(), text)

    def test_failed_ocr_cleans_temporary_pages(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.predict.side_effect = RuntimeError("inference failed")
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                extract_doc(pdf, folder, {"ocr_model": model})
            self.assertFalse(list(Path(folder).glob("pages_*")))


    def test_empty_page_reports_page_and_does_not_persist_partial_invoice(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.predict.side_effect = [[ocr_result("Page one")], [{"rec_texts": [], "rec_boxes": []}]]
            with self.assertRaisesRegex(RuntimeError, "page 2/2"):
                extract_doc(pdf, folder, {"ocr_model": model})
            self.assertFalse((Path(folder) / "detailed_invoice_output.txt").exists())
            self.assertFalse(list(Path(folder).glob("pages_*")))

    def test_unsupported_ocr_return_type_is_explicit(self):
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            model = Mock()
            model.predict.return_value = [{"text": "Unsupported"}]
            with self.assertRaisesRegex(RuntimeError, "matching rec_texts and rec_boxes"):
                extract_doc(pdf, folder, {"ocr_model": model})
            self.assertFalse(list(Path(folder).glob("pages_*")))


    def test_table_cell_boundaries_preserved(self):
        self.assertEqual(clean_ocr_text("<tr><td>sku</td><td>42</td></tr>"), "sku\t42")

    def test_all_invoices_and_duplicate_or_missing_skus(self):
        parsed = {"invoices": [invoice(), invoice()]}
        parsed["invoices"][1]["item_level"].append(deepcopy(invoice(None)["item_level"][0]))
        selections = [json.dumps({"hs6": code, "destination_national_code": national,
                                  "needs_review": False, "confidence": "MEDIUM"})
                      for code, national in [("854442", "8544.42.00.00"), ("854449", None), ("854442", None)]]
        profiles = [json.dumps({"product_type": "insulated electric cable", "needs_review": False})] * 3
        generate = Mock(side_effect=[json.dumps(parsed), *profiles, *selections])
        with TemporaryDirectory() as folder:
            pdf = Path(folder) / "invoice.pdf"
            make_pdf(pdf)
            output = Path(folder) / "output"
            ocr = Mock()
            ocr.predict.return_value = [ocr_result("Invoice text")]
            stages = []
            result = doc_pipeline(pdf, output, models={"ocr": {"ocr_model": ocr}},
                                  generate=generate, retriever=CatalogRetriever(ROWS, NATIONAL), progress=stages.append)
            self.assertEqual(result["invoices"][0]["item_level"][0]["hs6_code"], "854442")
            self.assertEqual(result["invoices"][1]["item_level"][0]["hs6_code"], "854449")
            self.assertEqual(result["invoices"][1]["item_level"][1]["hs6_code"], "854442")
            matches = json.loads((output / "db_matches.json").read_text())
            self.assertEqual([(m["invoice_index"],m["item_index"]) for m in matches], [(0,0),(1,0),(1,1)])
            self.assertEqual(generate.call_count, 7)
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
