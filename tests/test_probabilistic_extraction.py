import math
import unittest

import torch

from evals.metrics.memorization import (
    _probability_at_least_once,
    _sampling_log_probs,
)


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


if __name__ == "__main__":
    unittest.main()
