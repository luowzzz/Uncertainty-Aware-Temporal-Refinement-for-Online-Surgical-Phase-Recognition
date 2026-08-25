"""Pure validation-based operating-point selection rules.

The functions operate on already-computed validation metrics.  They contain no
file access, no model inference, and no test-set logic, which makes every
feasibility branch directly unit-testable. The generic column name ``A`` is
used internally for a candidate threshold; EGTP supplies its candidate ``k``
values through that column.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_METRICS = (
    "macro_f1",
    "boundary_recall_tol10",
    "boundary_f1_tol10",
    "predicted_boundary_count",
)


def _validate_tables(
    candidate_metrics: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    expected_seeds: tuple[int, ...],
) -> None:
    candidate_required = {"variant_id", "training_seed", "A", *REQUIRED_METRICS}
    raw_required = {"training_seed", *REQUIRED_METRICS}
    if not candidate_required.issubset(candidate_metrics.columns):
        raise ValueError(f"candidate metrics lack {sorted(candidate_required - set(candidate_metrics.columns))}")
    if not raw_required.issubset(raw_metrics.columns):
        raise ValueError(f"raw metrics lack {sorted(raw_required - set(raw_metrics.columns))}")
    if set(raw_metrics["training_seed"].astype(int)) != set(expected_seeds):
        raise ValueError("raw metrics do not contain exactly the expected seeds")
    if candidate_metrics.duplicated(["variant_id", "training_seed", "A"]).any():
        raise ValueError("candidate metrics contain duplicate variant/seed/A rows")
    if not np.isfinite(candidate_metrics[["A", *REQUIRED_METRICS]].to_numpy(dtype=float)).all():
        raise ValueError("candidate metrics contain non-finite values")


def annotate_feasibility(
    candidate_metrics: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    *,
    expected_seeds: tuple[int, ...] = (0, 1, 2),
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_tables(candidate_metrics, raw_metrics, expected_seeds)
    raw = raw_metrics[["training_seed", *REQUIRED_METRICS]].rename(
        columns={metric: f"raw_{metric}" for metric in REQUIRED_METRICS}
    )
    annotated = candidate_metrics.merge(raw, on="training_seed", validate="many_to_one")
    annotated["seed_macro_non_decrease"] = (
        annotated["macro_f1"] >= annotated["raw_macro_f1"] - tolerance
    )
    annotated["seed_recall_non_decrease"] = (
        annotated["boundary_recall_tol10"]
        >= annotated["raw_boundary_recall_tol10"] - tolerance
    )
    annotated["seed_boundary_count_reduction"] = (
        annotated["predicted_boundary_count"]
        < annotated["raw_predicted_boundary_count"] - tolerance
    )
    annotated["seed_basic_feasible"] = (
        annotated["seed_macro_non_decrease"]
        & annotated["seed_boundary_count_reduction"]
    )
    annotated["seed_strict_feasible"] = (
        annotated["seed_basic_feasible"]
        & annotated["seed_recall_non_decrease"]
    )

    rows = []
    keys = ["variant_id", "A"]
    for (variant_id, threshold), group in annotated.groupby(keys, sort=True):
        seeds = tuple(sorted(group["training_seed"].astype(int).unique()))
        if seeds != tuple(sorted(expected_seeds)):
            raise ValueError(f"{variant_id} A={threshold} does not contain every seed")
        row: dict[str, Any] = {
            "variant_id": variant_id,
            "A": float(threshold),
            "all_seed_basic_feasible": bool(group["seed_basic_feasible"].all()),
            "all_seed_strict_feasible": bool(group["seed_strict_feasible"].all()),
        }
        for metric in REQUIRED_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return annotated, pd.DataFrame(rows)


def _best_point(
    summary: pd.DataFrame,
    mask: pd.Series,
    metric: str,
    *,
    tolerance: float,
) -> dict[str, Any] | None:
    candidates = summary[mask].copy()
    if candidates.empty:
        return None
    best_value = float(candidates[metric].max())
    tied = candidates[candidates[metric] >= best_value - tolerance]
    selected = tied.sort_values("A", ascending=True).iloc[0]
    return {
        "A": float(selected["A"]),
        "selection_metric": metric,
        "selection_metric_value": float(selected[metric]),
        "descriptive_only": False,
    }


def select_variant_operating_points(
    variant_summary: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    variants = variant_summary["variant_id"].unique()
    if len(variants) != 1:
        raise ValueError("select_variant_operating_points expects exactly one variant")
    variant_id = str(variants[0])
    strict_mask = variant_summary["all_seed_strict_feasible"].astype(bool)
    basic_mask = variant_summary["all_seed_basic_feasible"].astype(bool)
    positive_mask = variant_summary["A"] > tolerance
    has_strict = bool(strict_mask.any())
    has_basic = bool(basic_mask.any())
    result: dict[str, Any] = {
        "variant_id": variant_id,
        "has_strict_set": has_strict,
        "has_basic_set": has_basic,
        "has_effective_operating_point": has_basic,
        "selection_status": (
            "strict_feasible" if has_strict else "basic_only" if has_basic else "constraint_failure"
        ),
        "points": {},
    }
    if has_strict:
        result["points"]["A_strict"] = _best_point(
            variant_summary,
            strict_mask,
            "boundary_f1_tol10_mean",
            tolerance=tolerance,
        )
    if has_basic:
        result["points"]["A_F1"] = _best_point(
            variant_summary,
            basic_mask & positive_mask,
            "boundary_f1_tol10_mean",
            tolerance=tolerance,
        )
        result["points"]["A_recall"] = _best_point(
            variant_summary,
            basic_mask & positive_mask,
            "boundary_recall_tol10_mean",
            tolerance=tolerance,
        )
        if result["points"]["A_F1"] is None or result["points"]["A_recall"] is None:
            raise ValueError("basic set must contain at least one positive A because count reduction is strict")
    else:
        descriptive_f1 = _best_point(
            variant_summary,
            positive_mask,
            "boundary_f1_tol10_mean",
            tolerance=tolerance,
        )
        descriptive_recall = _best_point(
            variant_summary,
            positive_mask,
            "boundary_recall_tol10_mean",
            tolerance=tolerance,
        )
        for name, point in (
            ("A_descriptive_F1", descriptive_f1),
            ("A_descriptive_recall", descriptive_recall),
        ):
            if point is None:
                raise ValueError("descriptive points require at least one A>0")
            point["descriptive_only"] = True
            result["points"][name] = point
    return result


def highest_common_feasibility_layer(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    if first["has_strict_set"] and second["has_strict_set"]:
        return {
            "layer": "strict",
            "primary_point_type": "A_strict",
            "supplementary_point_type": None,
            "stable_gain_assessment_allowed": True,
        }
    if first["has_basic_set"] and second["has_basic_set"]:
        return {
            "layer": "basic",
            "primary_point_type": "A_F1",
            "supplementary_point_type": "A_recall",
            "stable_gain_assessment_allowed": True,
        }
    return {
        "layer": "none",
        "primary_point_type": None,
        "supplementary_point_type": None,
        "stable_gain_assessment_allowed": False,
    }


def metrics_at_point(
    candidate_metrics: pd.DataFrame,
    selection: dict[str, Any],
    point_type: str,
    *,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    point = selection["points"].get(point_type)
    if point is None:
        raise ValueError(f"{selection['variant_id']} has no {point_type}")
    selected = candidate_metrics[
        candidate_metrics["variant_id"].eq(selection["variant_id"])
        & np.isclose(candidate_metrics["A"], point["A"], atol=tolerance, rtol=0.0)
    ].copy()
    if selected.empty:
        raise ValueError("selected A is absent from candidate metrics")
    return selected.sort_values("training_seed").reset_index(drop=True)


def assess_stable_entropy_gain(
    candidate_metrics: pd.DataFrame,
    full_selection: dict[str, Any],
    logratio_selection: dict[str, Any],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    common = highest_common_feasibility_layer(full_selection, logratio_selection)
    result: dict[str, Any] = {"common_feasibility": common}
    if not common["stable_gain_assessment_allowed"]:
        result.update(assessment_status="not_assessed_no_common_valid_layer", stable_gain=False)
        return result
    point_type = str(common["primary_point_type"])
    full = metrics_at_point(candidate_metrics, full_selection, point_type, tolerance=tolerance)
    simple = metrics_at_point(candidate_metrics, logratio_selection, point_type, tolerance=tolerance)
    merged = full.merge(simple, on="training_seed", suffixes=("_full", "_logratio"), validate="one_to_one")
    f1_delta = merged["boundary_f1_tol10_full"] - merged["boundary_f1_tol10_logratio"]
    macro_delta = merged["macro_f1_full"] - merged["macro_f1_logratio"]
    recall_delta = merged["boundary_recall_tol10_full"] - merged["boundary_recall_tol10_logratio"]
    requirements = {
        "mean_boundary_f1_higher": bool(f1_delta.mean() > tolerance),
        "boundary_f1_positive_in_all_seeds": bool((f1_delta > tolerance).all()),
        "macro_f1_not_lower_in_any_seed": bool((macro_delta >= -tolerance).all()),
        "boundary_recall_not_lower_in_any_seed": bool((recall_delta >= -tolerance).all()),
    }
    result.update(
        assessment_status="assessed",
        point_type=point_type,
        full_A=float(full["A"].iloc[0]),
        logratio_A=float(simple["A"].iloc[0]),
        requirements=requirements,
        stable_gain=bool(all(requirements.values())),
        per_seed_boundary_f1_delta={str(int(seed)): float(delta) for seed, delta in zip(merged["training_seed"], f1_delta)},
        per_seed_macro_f1_delta={str(int(seed)): float(delta) for seed, delta in zip(merged["training_seed"], macro_delta)},
        per_seed_boundary_recall_delta={str(int(seed)): float(delta) for seed, delta in zip(merged["training_seed"], recall_delta)},
    )
    return result


def select_final_family(
    full_selection: dict[str, Any],
    logratio_selection: dict[str, Any],
    entropy_assessment: dict[str, Any],
) -> dict[str, Any]:
    full_valid = bool(full_selection["has_effective_operating_point"])
    logratio_valid = bool(logratio_selection["has_effective_operating_point"])
    stable_gain = bool(entropy_assessment.get("stable_gain", False))
    if full_valid and stable_gain:
        return {
            "selected_family": "full_calibrated",
            "selection_reason": "entropy_weighting_stable_gain_at_common_valid_layer",
            "entropy_stable_superiority_claim_allowed": True,
        }
    if logratio_valid:
        return {
            "selected_family": "logratio_calibrated",
            "selection_reason": "parsimony_without_stable_entropy_gain",
            "entropy_stable_superiority_claim_allowed": False,
        }
    if full_valid:
        return {
            "selected_family": "full_calibrated",
            "selection_reason": "only_full_has_effective_operating_point",
            "entropy_stable_superiority_claim_allowed": False,
        }
    return {
        "selected_family": None,
        "selection_reason": "both_families_constraint_failure",
        "entropy_stable_superiority_claim_allowed": False,
    }
