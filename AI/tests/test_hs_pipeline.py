"""Shared HS catalog -> validated HS6 -> country/prefix national retrieval."""
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import pymupdf

from AI.hs_pipeline import CatalogWeaviateRetriever
from AI.main import doc_pipeline
from AI.tariff import CatalogRetriever, WeaviateRetriever, classify_item


ROWS = [
    {"hscode": "32", "description": "Dyes and inks"},
    {"hscode": "3215", "description": "Printing writing drawing inks"},
    {"hscode": "321511", "description": "Printing ink black"},
    {"hscode": "321519", "description": "Printing ink other than black"},
    {"hscode": "321590", "description": "Writing drawing other ink"},
]
ITEM = {"sku_code": "INK-1", "product_description": "Magenta printing ink",
        "country_of_manufacture": "JP"}
ORDER = {"origin_country": "JP", "export_country": "AU", "shipper_country": "AU",
         "destination_country": "NZ"}


def selection(hs6="321519", **changes):
    return {"hs6": hs6, "confidence": "MEDIUM", "needs_review": False,
            "classification_reason": "Non-black printing ink", **changes}


def record(code, **props):
    return SimpleNamespace(uuid=code, properties={"code": code, "description": "Other", **props})


def national_client():
    client = Mock()
    query = client.collections.get.return_value.query
    query.fetch_objects.return_value.objects = []
    return client, query


class SharedCatalogTests(TestCase):
    def setUp(self):
        CatalogRetriever.clear_cache()
        self.addCleanup(CatalogRetriever.clear_cache)

    def test_concurrent_workers_and_repeated_invoices_read_catalog_once(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "hs.csv"
            path.write_text("hscode,description\n0101,Live horses\n010121,Pure bred live horses\n")
            reads = []
            original = Path.read_text

            def read(p, *args, **kwargs):
                reads.append(p)
                return original(p, *args, **kwargs)

            with patch.object(Path, "read_text", read), ThreadPoolExecutor(max_workers=4) as workers:
                catalogs = list(workers.map(lambda _: CatalogRetriever.load(path), range(12)))
                for catalog in catalogs:
                    catalog.retrieve({"product_description": "live horses"}, {})
            self.assertEqual(reads, [path.resolve()])
            self.assertTrue(all(catalog is catalogs[0] for catalog in catalogs))
            self.assertIn("010121", catalogs[0].by_code)

    def test_different_paths_are_isolated_and_reload_is_explicit(self):
        with TemporaryDirectory() as folder:
            a, b = Path(folder) / "a.csv", Path(folder) / "b.csv"
            a.write_text("hscode,description\n321519,Printing ink\n")
            b.write_text("hscode,description\n854442,Electric cable\n")
            first = CatalogRetriever.load(a)
            self.assertIsNot(first, CatalogRetriever.load(b))
            a.write_text("hscode,description\n321590,Writing ink\n")
            self.assertIs(CatalogRetriever.load(a), first)
            CatalogRetriever.clear_cache()
            self.assertIn("321590", CatalogRetriever.load(a).by_code)

    def test_aggregate_rows_are_not_selectable(self):
        catalog = CatalogRetriever([*ROWS, {"hscode": "999999", "section": "TOTAL",
                                            "description": "All commodities"}])
        self.assertNotIn("999999", catalog.by_code)


class HSPipelineTests(TestCase):
    def make(self, response=None, country_map=None):
        client, query = national_client()
        national = WeaviateRetriever(client, country_map if country_map is not None else {"JP": "jp-id", "NZ": "nz-id"})
        generate = Mock(return_value=json.dumps(response if response is not None else selection()))
        retriever = CatalogWeaviateRetriever(CatalogRetriever(ROWS), national, generate)
        return retriever, generate, query

    def test_catalog_selection_precedes_country_and_hs6_only_lookup(self):
        retriever, generate, query = self.make()

        def fetch(**options):
            self.assertEqual(generate.call_count, 1)
            filters = options["filters"].filters
            country = filters[0].value
            self.assertIn(country, {"jp-id", "nz-id"})
            self.assertEqual({f.value for f in filters[1].filters}, {"321519*", "3215.19*"})
            return SimpleNamespace(objects=[record("3215190000" if country == "jp-id" else "3215190000A")])

        query.fetch_objects.side_effect = fetch
        original = deepcopy(ITEM)
        matches = retriever.retrieve(ITEM, ORDER)
        self.assertEqual(ITEM, original)
        self.assertEqual([r["hs_code"] for r in matches["hs_candidates"]], ["321519"])
        self.assertEqual(matches["origin_iso"], "JP")
        self.assertEqual(matches["destination_iso"], "NZ")
        self.assertEqual(query.fetch_objects.call_count, 2)
        query.bm25.assert_not_called()
        payload = json.loads(generate.call_args.args[0][1]["content"])
        self.assertIn("321519", {r["hs_code"] for r in payload["hs_candidates"]})

    def test_no_national_rows_retains_hs6_without_second_model_call(self):
        retriever, generate, _ = self.make()
        matches = retriever.retrieve(ITEM, ORDER)
        result = classify_item(ITEM, ORDER, matches, generate)
        self.assertEqual(result["hs6_code"], "321519")
        self.assertEqual(result["manufacturing_country_hts_code"], "321519")
        self.assertEqual(result["destination_country_hts_code"], "321519")
        self.assertTrue(result["needs_review"])
        generate.assert_called_once()

    def test_rejected_hs6_or_model_failure_never_queries_database(self):
        for response in (selection("999999"), selection("3215"), selection(None), selection("INK-1")):
            with self.subTest(response=response):
                retriever, generate, query = self.make(response)
                matches = retriever.retrieve(ITEM, ORDER)
                query.fetch_objects.assert_not_called()
                result = classify_item(ITEM, ORDER, matches, generate)
                self.assertIsNone(result["hs6_code"])
                self.assertTrue(result["needs_review"])
        retriever, generate, query = self.make()
        generate.side_effect = RuntimeError("inference failed")
        self.assertIsNone(retriever.retrieve(ITEM, ORDER)["hs_selection"]["hs6"])
        query.fetch_objects.assert_not_called()

    def test_printed_heading_constrains_selection_and_same_country_is_queried_once(self):
        retriever, generate, query = self.make(country_map={"JP": "jp-id"})
        matches = retriever.retrieve({**ITEM, "hs_code": "3215"}, {"destination_country": "JP"})
        payload = json.loads(generate.call_args.args[0][1]["content"])
        self.assertTrue(all(row["hs_code"].startswith("3215") for row in payload["hs_candidates"]))
        query.fetch_objects.assert_called_once()
        self.assertEqual(matches["origin_candidates"], matches["destination_candidates"])

    def test_unknown_countries_do_not_use_export_or_shipper_as_fallback(self):
        retriever, _, query = self.make()
        matches = retriever.retrieve({**ITEM, "country_of_manufacture": None}, {"export_country": "JP", "shipper_country": "JP"})
        self.assertEqual(matches["hs_selection"]["hs6"], "321519")
        query.fetch_objects.assert_not_called()

    def test_national_lookup_rejects_unrelated_inactive_deleted_and_non_leaf_rows(self):
        retriever, _, query = self.make(country_map={"NZ": "nz-id"})
        query.fetch_objects.return_value.objects = [
            record("3215.19.00.00A"), record("3215190000A"), record("3215900000A"),
            record("3215190001A", isactive=False), record("3215190002A", isdelete=True),
            record("3215190003A", is_leaf=False), record("321519"),
        ]
        matches = retriever.retrieve({**ITEM, "country_of_manufacture": None}, ORDER)
        self.assertEqual([r["national_code"] for r in matches["destination_candidates"]], ["3215190000A"])

    def test_country_specific_rows_are_paginated(self):
        retriever, _, query = self.make(country_map={"JP": "jp-id"})
        query.fetch_objects.side_effect = [SimpleNamespace(objects=[record(f"321519{i:04d}") for i in range(100)]),
                                          SimpleNamespace(objects=[record("3215199999")])]
        matches = retriever.retrieve(ITEM, {})
        self.assertEqual(len(matches["origin_candidates"]), 101)
        self.assertEqual([call.kwargs["offset"] for call in query.fetch_objects.call_args_list], [0, 100])

    def test_valid_printed_origin_code_skips_that_country_and_restricts_hs6(self):
        retriever, generate, query = self.make()
        retriever.retrieve({**ITEM, "manufacturing_country_hts_code": "3215190000"}, ORDER)
        query.fetch_objects.assert_called_once()
        self.assertEqual(query.fetch_objects.call_args.kwargs["filters"].filters[0].value, "nz-id")
        candidates = json.loads(generate.call_args.args[0][1]["content"])["hs_candidates"]
        self.assertEqual([row["hs_code"] for row in candidates], ["321519"])

    def test_invalid_national_selection_does_not_erase_catalog_hs6(self):
        retriever, generate, query = self.make(country_map={"JP": "jp-id"})
        query.fetch_objects.return_value.objects = [record("3215190000")]
        matches = retriever.retrieve(ITEM, ORDER)
        generate.return_value = json.dumps({"hs6": "321590", "origin_national_code": "3215900000",
                                           "destination_national_code": None, "confidence": "HIGH", "needs_review": False})
        result = classify_item(ITEM, ORDER, matches, generate)
        self.assertEqual(result["hs6_code"], "321519")
        self.assertEqual(result["manufacturing_country_hts_code"], "321519")
        self.assertTrue(result["needs_review"])

    def test_national_model_selects_only_extensions_of_locked_hs6(self):
        retriever, generate, query = self.make()
        query.fetch_objects.side_effect = [SimpleNamespace(objects=[record("3215190000")]),
                                          SimpleNamespace(objects=[record("3215190000A")])]
        matches = retriever.retrieve(ITEM, ORDER)
        generate.return_value = json.dumps({"hs6": "321519", "origin_national_code": "3215190000",
                                           "destination_national_code": "3215190000A", "confidence": "HIGH",
                                           "needs_review": False, "classification_reason": "Selected supplied national rows"})
        result = classify_item(ITEM, ORDER, matches, generate)
        self.assertEqual(result["hs6_code"], "321519")
        self.assertEqual(result["manufacturing_country_hts_code"], "3215190000")
        self.assertEqual(result["destination_country_hts_code"], "3215190000A")
        self.assertFalse(result["needs_review"])
        self.assertEqual(result["confidence"], "MEDIUM")
        national_prompt = json.loads(generate.call_args.args[0][1]["content"])
        self.assertEqual([r["hs_code"] for r in national_prompt["hs_candidates"]], ["321519"])

    def test_hs_uncertainty_is_not_erased_by_national_selection(self):
        retriever, generate, query = self.make(selection(needs_review=True, confidence="LOW"))
        query.fetch_objects.return_value.objects = [record("3215190000")]
        matches = retriever.retrieve(ITEM, ORDER)
        generate.return_value = json.dumps({"hs6": "321519", "origin_national_code": "3215190000",
                                           "destination_national_code": None, "confidence": "HIGH", "needs_review": False})
        result = classify_item(ITEM, ORDER, matches, generate)
        self.assertEqual(result["confidence"], "LOW")
        self.assertTrue(result["needs_review"])

    def test_real_pipeline_uses_hybrid_and_closes_connection(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            pdf = root / "invoice.pdf"
            with pymupdf.open() as document:
                page = document.new_page()
                page.insert_text((20, 40), "Invoice magenta printing ink quantity 15 total 975 origin Japan delivery New Zealand")
                document.save(pdf)
            client, query = national_client()
            payload = {"invoices": [{"order_level": ORDER, "item_level": [ITEM]}]}
            generate = Mock(side_effect=[json.dumps(payload),
                                         json.dumps({"product_type": "printing ink", "search_terms": ["printing ink"], "needs_review": False}),
                                         json.dumps(selection())])
            with patch("AI.main.CatalogRetriever.load", return_value=CatalogRetriever(ROWS)) as load, patch(
                "AI.db_connect.connect_vdb", return_value={"weaviate_client": client}
            ), patch("AI.main.WeaviateRetriever.load_country_map", return_value={"JP": "jp-id", "NZ": "nz-id"}):
                result = doc_pipeline(pdf, root / "out", backend="weaviate", models={}, generate=generate)
            load.assert_called_once()
            client.close.assert_called_once()
            self.assertEqual(generate.call_count, 3)
            self.assertEqual(result["invoices"][0]["item_level"][0]["hs6_code"], "321519")
            self.assertEqual(result["invoices"][0]["order_level"]["origin_country"], "JP")
            self.assertTrue((root / "out" / "db_matches.json").exists())
