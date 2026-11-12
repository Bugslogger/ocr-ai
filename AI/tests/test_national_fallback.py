"""International fallbacks must not masquerade as validated national extensions."""
import json
from unittest import TestCase
from unittest.mock import Mock

from AI.tariff import classify_item


class NationalFallbackTests(TestCase):
    def matches(self, **changes):
        return {"hs_candidates": [{"hs_code": "321519", "description": "Non-black printing ink",
                                    "parent_description": "Printing ink"}],
                "origin_candidates": [], "destination_candidates": [],
                "origin_iso": None, "destination_iso": None, "source": "catalog + DB",
                "hs_selection": {"hs6": "321519", "confidence": "HIGH", "needs_review": False,
                                 "classification_reason": "Non-black printing ink"}, **changes}

    def test_missing_countries_return_hs6_in_both_fields_with_provenance(self):
        generate = Mock()
        result = classify_item({"product_description": "Magenta printing ink"}, {}, self.matches(), generate)
        for field in ("manufacturing_country_hts_code", "destination_country_hts_code", "destination_tariff_code"):
            self.assertEqual(result[field], "321519")
        self.assertEqual(result["manufacturing_country_hts_source"], "hs6_fallback")
        self.assertEqual(result["destination_country_hts_source"], "hs6_fallback")
        self.assertTrue(result["needs_review"])
        self.assertIn("international HS6 fallback", result["validation_note"])
        generate.assert_not_called()

    def test_valid_printed_heading_returns_hs4_without_inventing_digits(self):
        matches = self.matches(hs_candidates=[], hs_selection=None)
        result = classify_item({"hs_code": "3215", "product_description": "Ink"}, {}, matches, Mock())
        self.assertEqual(result["manufacturing_country_hts_code"], "3215")
        self.assertEqual(result["destination_country_hts_code"], "3215")
        self.assertEqual(result["destination_country_hts_source"], "hs4_fallback")
        self.assertIsNone(result["hs6_code"])

    def test_no_supported_hs_or_sku_as_hs_keeps_null(self):
        for item in ({"product_description": "Unknown"}, {"sku_code": "321519", "hs_code": "321519"}):
            with self.subTest(item=item):
                result = classify_item(item, {}, self.matches(hs_candidates=[], hs_selection=None), Mock())
                self.assertIsNone(result["manufacturing_country_hts_code"])
                self.assertIsNone(result["destination_country_hts_code"])

    def test_other_candidate_is_selected_and_missing_other_country_uses_hs6(self):
        matches = self.matches(destination_iso="NZ", destination_candidates=[{
            "jurisdiction": "NZ", "hs6": "321519", "national_code": "3215190000A", "title": "Others"}])
        generate = Mock(return_value=json.dumps({"hs6": "321519", "origin_national_code": None,
            "destination_national_code": "3215190000A", "confidence": "MEDIUM", "needs_review": False,
            "classification_reason": "Residual national category under non-black printing inks"}))
        result = classify_item({"product_description": "Magenta printing ink"}, {"destination_country": "NZ"}, matches, generate)
        self.assertEqual(result["destination_country_hts_code"], "3215190000A")
        self.assertEqual(result["destination_country_hts_source"], "catalog + DB")
        self.assertEqual(result["manufacturing_country_hts_code"], "321519")
        self.assertTrue(result["needs_review"])
        data = json.loads(generate.call_args.args[0][1]["content"])
        self.assertEqual(data["destination_candidates"][0]["title"], "Others")

    def test_valid_printed_national_code_is_not_shortened_by_fallback(self):
        matches = self.matches(origin_iso="JP", destination_iso=None)
        result = classify_item({"product_description": "Ink", "country_of_manufacture": "JP",
                               "manufacturing_country_hts_code": "3215190000"}, {}, matches, Mock())
        self.assertEqual(result["manufacturing_country_hts_code"], "3215190000")
        self.assertEqual(result["manufacturing_country_hts_source"], "invoice")
        self.assertEqual(result["destination_country_hts_code"], "321519")

    def test_invalid_national_code_uses_hs6_instead_of_unrelated_code(self):
        matches = self.matches(destination_iso="NZ", destination_candidates=[{
            "jurisdiction": "NZ", "hs6": "321519", "national_code": "3215190000A", "title": "Other"}])
        generate = Mock(return_value=json.dumps({"hs6": "321519", "destination_national_code": "9999990000",
            "origin_national_code": None, "confidence": "HIGH", "needs_review": False}))
        result = classify_item({"product_description": "Ink"}, {"destination_country": "NZ"}, matches, generate)
        self.assertEqual(result["destination_country_hts_code"], "321519")
        self.assertEqual(result["destination_country_hts_source"], "hs6_fallback")
        self.assertTrue(result["needs_review"])
