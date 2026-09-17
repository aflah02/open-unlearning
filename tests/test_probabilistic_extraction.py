import math
import unittest
from types import SimpleNamespace

import torch

from evals.metrics.memorization import (
    _operating_point_key,
    _probability_at_least_once,
    _sampling_log_probs,
    _select_target_span,
    probabilistic_extraction,
)
from evals.metrics.utils import tokenwise_logprobs, tokenwise_vocab_logprobs


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, logits, eos_token_id=None):
        self.logits = logits
        self.config = SimpleNamespace(eos_token_id=eos_token_id)

    def __call__(self, **batch):
        batch_size = batch["input_ids"].shape[0]
        return SimpleNamespace(logits=self.logits.expand(batch_size, -1, -1))


def stack_collator(instances):
    return {
        key: torch.stack([instance[key] for instance in instances])
        for key in instances[0]
    }


class SamplingLogProbTests(unittest.TestCase):
    def test_temperature_rescales_and_normalizes_distribution(self):
        log_probs = torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64))

        actual = _sampling_log_probs(log_probs, temperature=2.0).exp()

        expected = torch.tensor([[2 / 3, 1 / 3]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_top_k_truncates_and_renormalizes_distribution(self):
        log_probs = torch.log(torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64))

        actual = _sampling_log_probs(log_probs, top_k=2).exp()

        expected = torch.tensor([[2 / 3, 1 / 3, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_top_p_keeps_smallest_set_reaching_threshold(self):
        log_probs = torch.log(torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64))

        actual = _sampling_log_probs(log_probs, top_p=0.6).exp()

        expected = torch.tensor([[0.625, 0.375, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_top_k_assigns_zero_probability_to_excluded_target(self):
        log_probs = torch.log(torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64))

        actual = _sampling_log_probs(log_probs, top_k=1).exp()

        self.assertEqual(actual[0, 2].item(), 0.0)


class ExtractionProbabilityTests(unittest.TestCase):
    def test_probability_of_at_least_one_extraction(self):
        actual = _probability_at_least_once(math.log(0.1), num_queries=10)

        self.assertAlmostEqual(actual, 1 - 0.9**10)

    def test_impossible_target_remains_impossible(self):
        actual = _probability_at_least_once(-math.inf, num_queries=1_000_000)

        self.assertEqual(actual, 0.0)

    def test_near_certain_target_does_not_lose_precision(self):
        actual = _probability_at_least_once(-1e-20, num_queries=1)

        self.assertEqual(actual, 1.0)


class TargetSelectionTests(unittest.TestCase):
    def test_selects_suffix_after_content_prefix(self):
        log_probs = torch.arange(30).reshape(6, 5)
        labels = torch.arange(6)

        selected_log_probs, selected_labels = _select_target_span(
            log_probs, labels, prefix_length=2, suffix_length=3
        )

        torch.testing.assert_close(selected_log_probs, log_probs[2:5])
        torch.testing.assert_close(selected_labels, labels[2:5])

    def test_rejects_examples_shorter_than_fixed_suffix(self):
        result = _select_target_span(
            torch.zeros(4, 3),
            torch.arange(4),
            prefix_length=2,
            suffix_length=3,
        )

        self.assertIsNone(result)


class TokenBoundaryTests(unittest.TestCase):
    def test_keeps_final_labeled_token_when_it_is_not_eos(self):
        model = FakeModel(torch.zeros(1, 4, 5), eos_token_id=4)
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "labels": torch.tensor([[-100, 1, 2, 3]]),
        }

        log_probs, labels = tokenwise_vocab_logprobs(model, batch, return_labels=True)
        token_log_probs, token_labels = tokenwise_logprobs(
            model, batch, return_labels=True
        )

        self.assertEqual(log_probs[0].shape, (3, 5))
        torch.testing.assert_close(labels[0], torch.tensor([1, 2, 3]))
        self.assertEqual(token_log_probs[0].shape, (3,))
        torch.testing.assert_close(token_labels[0], torch.tensor([1, 2, 3]))

    def test_excludes_terminal_eos_only_when_present(self):
        model = FakeModel(torch.zeros(1, 4, 5), eos_token_id=4)
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 4]]),
            "labels": torch.tensor([[-100, 1, 2, 4]]),
        }

        log_probs, labels = tokenwise_vocab_logprobs(model, batch, return_labels=True)
        token_log_probs, token_labels = tokenwise_logprobs(
            model, batch, return_labels=True
        )

        self.assertEqual(log_probs[0].shape, (2, 5))
        torch.testing.assert_close(labels[0], torch.tensor([1, 2]))
        self.assertEqual(token_log_probs[0].shape, (2,))
        torch.testing.assert_close(token_labels[0], torch.tensor([1, 2]))


class ProbabilisticExtractionMetricTests(unittest.TestCase):
    def setUp(self):
        probs = torch.full((1, 5, 5), 0.2, dtype=torch.float64)
        probs[0, 2] = torch.tensor([0.1, 0.1, 0.1, 0.4, 0.3])
        probs[0, 3] = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        self.model = FakeModel(torch.log(probs), eos_token_id=None)
        self.data = [
            {
                "input_ids": torch.tensor([0, 1, 2, 3, 4]),
                "labels": torch.tensor([-100, 1, 2, 3, 4]),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "index": torch.tensor(0),
            }
        ]

    def evaluate(self, num_queries, probability_threshold):
        return probabilistic_extraction._metric_fn(
            self.model,
            data=self.data,
            collators=stack_collator,
            batch_size=1,
            num_queries=num_queries,
            probability_threshold=probability_threshold,
            temperature=1.0,
            top_k=None,
            top_p=None,
            prefix_length=2,
            suffix_length=2,
        )

    def test_scores_only_requested_suffix(self):
        result = self.evaluate(num_queries=10, probability_threshold=0.8)
        per_example = result["value_by_index"][0]

        self.assertAlmostEqual(per_example["single_query_probability"], 0.2)
        self.assertAlmostEqual(per_example["extraction_probability"], 1 - 0.8**10)
        self.assertEqual(per_example["num_target_tokens"], 2)
        self.assertTrue(per_example["is_extractable"])
        self.assertEqual(result["agg_value"], 1.0)

    def test_reports_full_operating_point_grid(self):
        result = self.evaluate(num_queries=[1, 10], probability_threshold=[0.5, 0.9])

        expected = {
            _operating_point_key(1, 0.5): 0.0,
            _operating_point_key(1, 0.9): 0.0,
            _operating_point_key(10, 0.5): 1.0,
            _operating_point_key(10, 0.9): 0.0,
        }
        self.assertEqual(result["agg_value"], expected)


if __name__ == "__main__":
    unittest.main()
