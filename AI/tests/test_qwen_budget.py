"""Context budgeting without GPU inference or truncating source inputs."""
import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from AI.model_config import Qruntime
from AI.utils import qwen_generate


class QwenBudgetTests(TestCase):
    def setUp(self):
        self.enterContext(patch.dict(Qruntime, {"max_context_tokens": 32768,
                                              "min_output_tokens": 2048,
                                              "context_safety_margin": 512}))

    def run_generation(self, input_tokens, configured=10000, model_limit=32768, output_tokens=None):
        # Only the tensor shape is needed: keep large boundary tests lightweight.
        class Encoded(dict):
            input_ids = SimpleNamespace(shape=(1, input_tokens))

            def to(self, device):
                return self

        inputs = Encoded(input_ids=object())
        tokenizer = Mock(eos_token_id=9)
        tokenizer.apply_chat_template.return_value = "complete formatted prompt"
        tokenizer.return_value = inputs
        tokenizer.decode.return_value = '{"ok":true}'
        generated = Mock()
        generated.__getitem__ = Mock(return_value=[9] if output_tokens is None else [1] * output_tokens)
        model = Mock()
        model.config.max_position_embeddings = model_limit
        model.generate.return_value = [generated]
        self.model, self.tokenizer, self.inputs = model, tokenizer, inputs
        self.messages = [{"role": "user", "content": "Complete OCR text and layout metadata"}]
        return qwen_generate(self.messages, tokenizer, model, max_new_tokens=configured)

    def test_normal_one_and_two_page_sized_inputs_keep_configured_budget(self):
        for input_tokens in (2000, 8000):
            with self.subTest(input_tokens=input_tokens):
                self.assertEqual(self.run_generation(input_tokens), '{"ok":true}')
                self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 10000)
                self.tokenizer.apply_chat_template.assert_called_once()
                self.assertEqual(self.tokenizer.apply_chat_template.call_args.args[0], self.messages)
                self.tokenizer.assert_called_once_with(["complete formatted prompt"], return_tensors="pt")
                self.assertIs(self.model.generate.call_args.kwargs["input_ids"], self.inputs["input_ids"])

    def test_reported_prompt_succeeds_with_new_default_and_clamps_old_override(self):
        for configured, expected in ((10000, 10000), (15000, 14099)):
            with self.subTest(configured=configured):
                self.run_generation(18157, configured)
                self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], expected)

    def test_large_prompt_reduces_budget_and_logs_all_budget_fields(self):
        with self.assertLogs("AI.performance", level="INFO") as logs:
            self.run_generation(25000)
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 7256)
        records = [json.loads(line.split("performance ", 1)[1]) for line in logs.output]
        budget = next(record for record in records if record["stage"] == "qwen_context_budget")
        for key, expected in {"input_tokens": 25000, "configured_output_tokens": 10000,
                              "effective_output_tokens": 7256, "context_limit": 32768,
                              "safety_margin": 512}.items():
            self.assertEqual(budget[key], expected)

    def test_prompt_over_context_limit_fails_before_generation(self):
        with self.assertLogs("AI.performance", level="INFO") as logs:
            with self.assertRaisesRegex(ValueError, "Document input is too large.*Split the document"):
                self.run_generation(33000)
        self.model.generate.assert_not_called()
        self.assertTrue(any('"effective_output_tokens": 0' in line for line in logs.output))

    def test_insufficient_output_capacity_including_safety_margin(self):
        for input_tokens in (30209, 32000, 32768):
            with self.subTest(input_tokens=input_tokens):
                with self.assertRaisesRegex(ValueError, "at least 2048 are required"):
                    self.run_generation(input_tokens)
                self.model.generate.assert_not_called()

    def test_exact_minimum_capacity_is_accepted(self):
        self.run_generation(30208)
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 2048)

    def test_smaller_model_context_is_respected(self):
        self.run_generation(8000, model_limit=12000)
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 3488)

    def test_short_product_and_tariff_budgets_still_work(self):
        for configured in (700, 1000):
            with self.subTest(configured=configured):
                self.run_generation(2000, configured)
                self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], configured)

    def test_effective_budget_still_rejects_truncated_generation(self):
        with self.assertRaisesRegex(RuntimeError, "token limit"):
            self.run_generation(30208, output_tokens=2048)
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 2048)
