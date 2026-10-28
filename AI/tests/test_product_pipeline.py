"""Product identification tests with injected model responses."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock, patch

from AI.product_pipeline import identify_products


class ProductPipelineTests(TestCase):
    def test_multiple_invoices_duplicate_skus_and_persistence(self):
        item = {"sku_code": "same", "product_description": "Insulated copper cable"}
        data = {"invoices": [{"item_level": [dict(item), dict(item)]}, {"item_level": [dict(item)]}]}
        original = deepcopy(data)
        generate = Mock(return_value=json.dumps({
            "product_type": "Insulated cable", "material": "copper",
            "search_terms": ["insulated cable", "copper", "copper"], "needs_review": False}))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "products.json"
            result = identify_products(data, path, generate=generate)
            self.assertEqual(json.loads(path.read_text()), result)
        self.assertEqual(data, original)
        self.assertEqual([(p["invoice_index"], p["item_index"]) for p in result["products"]],
                         [(0, 0), (0, 1), (1, 0)])
        profile = result["products"][0]["identification"]
        self.assertEqual(profile["search_query"], "insulated cable copper")
        self.assertIsNone(profile["construction"])
        self.assertEqual(generate.call_count, 3)

    def test_missing_descriptions_skip_models(self):
        with patch("AI.product_pipeline.download_qwen") as load:
            result = identify_products({"invoices": [{"item_level": [{"sku_code": "12345"}]}]})
        load.assert_not_called()
        profile = result["products"][0]["identification"]
        self.assertIsNone(profile["product_type"])
        self.assertIsNone(profile["search_query"])
        self.assertTrue(profile["needs_review"])

    def test_failed_item_continues_with_original_description(self):
        for bad in ("not json", '{"needs_review": "false"}', '{"needs_review":false,"hs_code":"123456"}'):
            with self.subTest(bad=bad):
                generate = Mock(side_effect=[bad, '{"product_type":"Cable","needs_review":false}'])
                data = {"invoices": [{"item_level": [{"product_description": "Wire"},
                                                       {"simplified_description": "Cable"}]}]}
                result = identify_products(data, generate=generate)
                first, second = [p["identification"] for p in result["products"]]
                self.assertTrue(first["needs_review"])
                self.assertEqual(first["search_query"], "Wire")
                self.assertIn("failed", first["error"])
                self.assertEqual(second["product_type"], "Cable")

    def test_unknown_identity_and_missing_information_force_review(self):
        for response in ({"needs_review": False},
                         {"product_type": "Cable", "needs_review": False,
                          "missing_information": ["Does it have connectors?"]}):
            result = identify_products({"invoices": [{"item_level": [{"product_description": "Cable"}]}]},
                                       generate=Mock(return_value=json.dumps(response)))
            self.assertTrue(result["products"][0]["identification"]["needs_review"])


if __name__ == "__main__":
    main()
