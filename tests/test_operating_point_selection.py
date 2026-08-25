from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from operating_point_selection import (  # noqa: E402
    annotate_feasibility,
    assess_stable_entropy_gain,
    highest_common_feasibility_layer,
    select_final_family,
    select_variant_operating_points,
)
SEEDS = (0, 1, 2)


def raw_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "training_seed": seed,
                "macro_f1": 0.50,
                "boundary_recall_tol10": 0.50,
                "boundary_f1_tol10": 0.10,
                "predicted_boundary_count": 100,
            }
            for seed in SEEDS
        ]
    )


def variant_rows(variant: str, states: list[tuple[float, float, float, float, int]]) -> list[dict]:
    rows = []
    for threshold, macro, recall, f1, count in states:
        for seed in SEEDS:
            rows.append(
                {
                    "variant_id": variant,
                    "training_seed": seed,
                    "A": threshold,
                    "macro_f1": macro,
                    "boundary_recall_tol10": recall,
                    "boundary_f1_tol10": f1 + seed * 0.001,
                    "predicted_boundary_count": count,
                }
            )
    return rows


STRICT = [(0.0, 0.50, 0.50, 0.10, 100), (1.0, 0.51, 0.51, 0.20, 90)]
BASIC = [(0.0, 0.50, 0.50, 0.10, 100), (1.0, 0.51, 0.45, 0.20, 90)]
FAILURE = [(0.0, 0.50, 0.50, 0.10, 100), (1.0, 0.49, 0.45, 0.20, 90)]


def selections(first_states, second_states, first="full_calibrated", second="logratio_calibrated"):
    metrics = pd.DataFrame(variant_rows(first, first_states) + variant_rows(second, second_states))
    _, summary = annotate_feasibility(metrics, raw_table())
    first_selection = select_variant_operating_points(summary[summary["variant_id"].eq(first)])
    second_selection = select_variant_operating_points(summary[summary["variant_id"].eq(second)])
    return metrics, first_selection, second_selection


class OperatingPointSelectionBranchTests(unittest.TestCase):
    def test_both_strict_compare_strict_and_still_compute_basic_points(self):
        _, first, second = selections(STRICT, STRICT)
        common = highest_common_feasibility_layer(first, second)
        self.assertEqual(common["layer"], "strict")
        self.assertEqual(common["primary_point_type"], "A_strict")
        for selection in (first, second):
            self.assertEqual(selection["selection_status"], "strict_feasible")
            self.assertIn("A_strict", selection["points"])
            self.assertIn("A_F1", selection["points"])
            self.assertIn("A_recall", selection["points"])

    def test_one_strict_both_basic_compare_f1_at_basic_layer(self):
        _, first, second = selections(STRICT, BASIC)
        common = highest_common_feasibility_layer(first, second)
        self.assertEqual(common["layer"], "basic")
        self.assertEqual(common["primary_point_type"], "A_F1")
        self.assertEqual(common["supplementary_point_type"], "A_recall")

    def test_both_basic_without_strict(self):
        _, first, second = selections(BASIC, BASIC)
        self.assertEqual(first["selection_status"], "basic_only")
        self.assertEqual(second["selection_status"], "basic_only")
        self.assertEqual(highest_common_feasibility_layer(first, second)["layer"], "basic")

    def test_one_basic_empty_disables_stable_gain(self):
        metrics, first, second = selections(BASIC, FAILURE)
        common = highest_common_feasibility_layer(first, second)
        self.assertEqual(common["layer"], "none")
        assessment = assess_stable_entropy_gain(metrics, first, second)
        self.assertFalse(assessment["stable_gain"])
        self.assertEqual(assessment["assessment_status"], "not_assessed_no_common_valid_layer")
        final = select_final_family(first, second, assessment)
        self.assertEqual(final["selected_family"], "full_calibrated")
        self.assertFalse(final["entropy_stable_superiority_claim_allowed"])

    def test_both_basic_empty_selects_no_successful_gate(self):
        metrics, first, second = selections(FAILURE, FAILURE)
        for selection in (first, second):
            self.assertEqual(selection["selection_status"], "constraint_failure")
            self.assertFalse(selection["has_effective_operating_point"])
            self.assertTrue(selection["points"]["A_descriptive_F1"]["descriptive_only"])
            self.assertTrue(selection["points"]["A_descriptive_recall"]["descriptive_only"])
        assessment = assess_stable_entropy_gain(metrics, first, second)
        final = select_final_family(first, second, assessment)
        self.assertIsNone(final["selected_family"])

    def test_calibrated_uncalibrated_layer_mismatch_uses_basic(self):
        _, calibrated, uncalibrated = selections(
            STRICT,
            BASIC,
            first="full_calibrated",
            second="full_uncalibrated",
        )
        common = highest_common_feasibility_layer(calibrated, uncalibrated)
        self.assertEqual(common["layer"], "basic")
        self.assertEqual(common["primary_point_type"], "A_F1")
        self.assertNotEqual(common["primary_point_type"], "A_strict")

if __name__ == "__main__":
    unittest.main()
