from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from egtp_transition_policy import apply_egtp  # noqa: E402
from rq2_stability_metrics import edit_score, temporal_fragmentation_index  # noqa: E402
from tec_loss import temporal_error_cascade_loss, temporal_error_cascade_weights  # noqa: E402


def probabilities_for_labels(labels: list[int], confidence: float = 0.8) -> np.ndarray:
    probabilities = np.full((len(labels), 7), (1.0 - confidence) / 6.0)
    probabilities[np.arange(len(labels)), np.asarray(labels, dtype=int)] = confidence
    return probabilities


class EGTPTests(unittest.TestCase):
    def test_k_zero_accepts_clear_positive_challenger(self):
        probabilities = probabilities_for_labels([0, 1, 2, 3, 4, 5, 6])
        result = apply_egtp(probabilities, 0.0)
        np.testing.assert_array_equal(result.predictions, np.arange(1, 8))

    def test_negative_evidence_reduces_accumulation(self):
        probabilities = np.array(
            [
                [0.8, 0.2, 0, 0, 0, 0, 0],
                [0.4, 0.6, 0, 0, 0, 0, 0],
                [0.7, 0.3, 0, 0, 0, 0, 0],
                [0.4, 0.6, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        result = apply_egtp(probabilities, 10.0, return_trace=True)
        evidence = result.trace["challenger_evidence"].to_numpy()
        self.assertGreater(evidence[0], 0.0)
        self.assertLess(evidence[1], evidence[0])

    def test_each_video_initialises_independently(self):
        first = apply_egtp(probabilities_for_labels([0, 1, 1]), 0.8)
        second = apply_egtp(probabilities_for_labels([4, 5, 5]), 0.8)
        self.assertEqual(first.predictions[0], 1)
        self.assertEqual(second.predictions[0], 5)


class TECTests(unittest.TestCase):
    def test_front_loaded_error_run_weights(self):
        prediction = torch.tensor([0, 1, 1, 1, 0, 2, 2])
        target = torch.zeros(7, dtype=torch.long)
        weights = temporal_error_cascade_weights(
            prediction,
            target,
            alpha=19,
            sigma=1.5,
            onset_window=8,
        )
        self.assertEqual(float(weights[0]), 1.0)
        self.assertGreater(float(weights[1]), float(weights[2]))
        self.assertGreater(float(weights[2]), float(weights[3]))
        self.assertEqual(float(weights[4]), 1.0)
        self.assertGreater(float(weights[5]), float(weights[6]))

    def test_zero_alpha_is_standard_cross_entropy(self):
        torch.manual_seed(3)
        logits = torch.randn(20, 7)
        targets = torch.randint(0, 7, (20,))
        loss, weights = temporal_error_cascade_loss(logits, targets, alpha=0.0)
        reference = torch.nn.functional.cross_entropy(logits, targets)
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))
        self.assertTrue(torch.allclose(loss, reference))


class StabilityMetricTests(unittest.TestCase):
    def test_perfect_sequences(self):
        sequence = np.array([1, 1, 2, 2, 3])
        self.assertEqual(edit_score(sequence, sequence), 100.0)
        self.assertEqual(temporal_fragmentation_index(sequence, sequence), 0.0)

    def test_fragmentation_increases_tfi(self):
        truth = np.array([1, 1, 1, 2, 2, 2])
        stable = np.array([1, 1, 1, 1, 1, 1])
        fragmented = np.array([1, 2, 1, 2, 1, 2])
        self.assertGreater(
            temporal_fragmentation_index(truth, fragmented),
            temporal_fragmentation_index(truth, stable),
        )


if __name__ == "__main__":
    unittest.main()
