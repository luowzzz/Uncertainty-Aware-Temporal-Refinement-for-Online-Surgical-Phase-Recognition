"""Shared, exact empirical risk-coverage metrics.

Lower scores are assumed to indicate more reliable predictions.  The empirical
curve contains every achievable retained-sample count from 1 to N.  When
multiple samples have exactly the same score, expected cumulative error within
that tie group is used, which makes AURC invariant to input row order.

The reported AURC is the right-Riemann empirical integral over coverage
(0, 1], equivalently the mean risk across retained counts k=1..N.  This avoids
the inconsistent truncated integrations previously used by separate scripts.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_KEY_COVERAGES = (0.10, 0.30, 0.50, 0.60, 0.80, 0.90, 1.00)


def _validated_arrays(
    errors: Iterable[int | float],
    scores: Iterable[int | float],
) -> tuple[np.ndarray, np.ndarray]:
    error_array = np.asarray(list(errors), dtype=float)
    score_array = np.asarray(list(scores), dtype=float)
    if error_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("errors and scores must be one-dimensional")
    if len(error_array) != len(score_array) or len(error_array) == 0:
        raise ValueError("errors and scores must have equal non-zero length")
    if not np.isfinite(error_array).all() or not np.isfinite(score_array).all():
        raise ValueError("errors and scores must be finite")
    if not np.isin(error_array, [0.0, 1.0]).all():
        raise ValueError("errors must contain only binary values 0 and 1")
    return error_array, score_array


def empirical_risk_coverage(
    errors: Iterable[int | float],
    scores: Iterable[int | float],
) -> pd.DataFrame:
    error_array, score_array = _validated_arrays(errors, scores)
    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    sorted_errors = error_array[order]
    n_samples = len(sorted_errors)

    expected_cumulative_errors = np.zeros(n_samples, dtype=float)
    score_thresholds = np.zeros(n_samples, dtype=float)
    prior_count = 0
    prior_errors = 0.0
    group_start = 0
    while group_start < n_samples:
        group_end = group_start + 1
        while (
            group_end < n_samples
            and sorted_scores[group_end] == sorted_scores[group_start]
        ):
            group_end += 1
        group_size = group_end - group_start
        group_errors = float(sorted_errors[group_start:group_end].sum())
        for offset in range(1, group_size + 1):
            index = group_start + offset - 1
            expected_cumulative_errors[index] = (
                prior_errors + offset * group_errors / group_size
            )
            score_thresholds[index] = sorted_scores[group_start]
        prior_count += group_size
        prior_errors += group_errors
        group_start = group_end

    kept = np.arange(1, n_samples + 1, dtype=int)
    coverage = kept / float(n_samples)
    risk = expected_cumulative_errors / kept
    full_risk = float(error_array.mean())
    return pd.DataFrame(
        {
            "coverage": coverage,
            "kept_samples": kept,
            "abstained_samples": n_samples - kept,
            "score_threshold": score_thresholds,
            "expected_errors_kept": expected_cumulative_errors,
            "risk_error_rate": risk,
            "accuracy_on_kept": 1.0 - risk,
            "full_coverage_error_rate": full_risk,
            "risk_reduction_vs_full": full_risk - risk,
        }
    )


def exact_aurc(curve: pd.DataFrame) -> dict[str, float | int]:
    required = {"coverage", "risk_error_rate", "full_coverage_error_rate"}
    missing = sorted(required - set(curve.columns))
    if missing or curve.empty:
        raise ValueError(f"Invalid risk-coverage curve; missing={missing}")
    ordered = curve.sort_values("coverage")
    risks = ordered["risk_error_rate"].to_numpy(dtype=float)
    full_risk = float(ordered["full_coverage_error_rate"].iloc[-1])
    aurc = float(risks.mean())

    n_samples = len(ordered)
    n_errors = int(round(full_risk * n_samples))
    oracle_errors = np.concatenate(
        [
            np.zeros(n_samples - n_errors, dtype=float),
            np.ones(n_errors, dtype=float),
        ]
    )
    oracle_cumulative = np.cumsum(oracle_errors)
    oracle_risk = oracle_cumulative / np.arange(1, n_samples + 1)
    oracle_aurc = float(oracle_risk.mean())
    return {
        "n_samples": n_samples,
        "n_errors": n_errors,
        "aurc_lower_better": aurc,
        "oracle_aurc": oracle_aurc,
        "excess_aurc": aurc - oracle_aurc,
        "random_ranking_expected_aurc": full_risk,
        "full_coverage_error_rate": full_risk,
    }


def key_coverage_points(
    curve: pd.DataFrame,
    requested_coverages: Iterable[float] = DEFAULT_KEY_COVERAGES,
) -> pd.DataFrame:
    if curve.empty:
        raise ValueError("risk-coverage curve is empty")
    ordered = curve.sort_values("kept_samples").reset_index(drop=True)
    n_samples = len(ordered)
    rows = []
    for requested in requested_coverages:
        requested = float(requested)
        if not 0.0 < requested <= 1.0:
            raise ValueError("requested coverages must be in (0, 1]")
        kept = max(1, min(n_samples, int(round(n_samples * requested))))
        row = ordered.iloc[kept - 1].to_dict()
        row["requested_coverage"] = requested
        row["coverage"] = requested
        row["achieved_coverage"] = kept / float(n_samples)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_score_frame(
    frame: pd.DataFrame,
    *,
    split_col: str,
    error_col: str,
    score_columns: Iterable[str],
    key_coverages: Iterable[float] = DEFAULT_KEY_COVERAGES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_frames = []
    key_frames = []
    aurc_rows = []
    for split, split_frame in frame.groupby(split_col, sort=False):
        for score in score_columns:
            curve = empirical_risk_coverage(
                split_frame[error_col].to_numpy(),
                split_frame[score].to_numpy(),
            )
            curve.insert(0, "score", score)
            curve.insert(0, "split", split)
            full_frames.append(curve)

            key = key_coverage_points(curve, key_coverages)
            key_frames.append(key)

            aurc_rows.append(
                {
                    "split": split,
                    "score": score,
                    **exact_aurc(curve),
                }
            )
    return (
        pd.concat(full_frames, ignore_index=True),
        pd.concat(key_frames, ignore_index=True),
        pd.DataFrame(aurc_rows),
    )
