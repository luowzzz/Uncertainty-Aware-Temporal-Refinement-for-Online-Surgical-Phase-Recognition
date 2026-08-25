"""Shared event-level boundary extraction and one-to-one matching utilities.

The project historically used nearest-pair greedy matching.  That strategy is
kept as ``greedy_one_to_one`` for exact legacy reproducibility.  The preferred
strategy for new validation analyses is ``optimal_ordered_one_to_one``:

1. maximise the number of matched boundary events within the tolerance;
2. among maximum-cardinality matchings, minimise total absolute timing error;
3. preserve temporal order.

For one-dimensional temporal events, an optimal non-crossing matching exists,
so the dynamic programme below provides the desired maximum-cardinality,
minimum-error assignment without an external optimisation dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


LEGACY_GREEDY = "greedy_one_to_one"
OPTIMAL_ORDERED = "optimal_ordered_one_to_one"
SUPPORTED_MATCHING_STRATEGIES = (LEGACY_GREEDY, OPTIMAL_ORDERED)


@dataclass(frozen=True)
class BoundaryMatch:
    gt_time: int
    pred_time: int

    @property
    def absolute_error(self) -> int:
        return abs(self.pred_time - self.gt_time)


def extract_boundaries(phases: Iterable[int], times: Iterable[int] | None = None) -> list[int]:
    phase_array = np.asarray(list(phases), dtype=int)
    if times is None:
        time_array = np.arange(len(phase_array), dtype=int)
    else:
        time_array = np.asarray(list(times), dtype=int)
    if phase_array.ndim != 1 or time_array.ndim != 1:
        raise ValueError("phases and times must be one-dimensional")
    if len(phase_array) != len(time_array):
        raise ValueError("phases and times must have the same length")
    if len(phase_array) <= 1:
        return []
    boundary_indices = np.flatnonzero(phase_array[1:] != phase_array[:-1]) + 1
    return time_array[boundary_indices].astype(int).tolist()


def _normalise_events(events: Iterable[int], name: str) -> list[int]:
    result = [int(value) for value in events]
    if any(result[index] > result[index + 1] for index in range(len(result) - 1)):
        raise ValueError(f"{name} boundaries must be sorted in non-decreasing time order")
    return result


def _greedy_pairs(
    gt_boundaries: list[int],
    pred_boundaries: list[int],
    tolerance: int,
) -> list[BoundaryMatch]:
    candidates: list[tuple[int, int, int]] = []
    for gt_index, gt_time in enumerate(gt_boundaries):
        for pred_index, pred_time in enumerate(pred_boundaries):
            distance = abs(pred_time - gt_time)
            if distance <= tolerance:
                candidates.append((distance, gt_index, pred_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[BoundaryMatch] = []
    for _, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append(
            BoundaryMatch(
                gt_time=gt_boundaries[gt_index],
                pred_time=pred_boundaries[pred_index],
            )
        )
    return sorted(matches, key=lambda match: (match.gt_time, match.pred_time))


def _is_better(
    candidate_matches: int,
    candidate_cost: int,
    current_matches: int,
    current_cost: int,
) -> bool:
    return candidate_matches > current_matches or (
        candidate_matches == current_matches and candidate_cost < current_cost
    )


def _optimal_ordered_pairs(
    gt_boundaries: list[int],
    pred_boundaries: list[int],
    tolerance: int,
) -> list[BoundaryMatch]:
    n_gt = len(gt_boundaries)
    n_pred = len(pred_boundaries)
    match_count = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int32)
    total_cost = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int64)
    # 0 = skip GT, 1 = skip prediction, 2 = match.
    decision = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int8)

    for gt_index in range(1, n_gt + 1):
        for pred_index in range(1, n_pred + 1):
            best_matches = int(match_count[gt_index - 1, pred_index])
            best_cost = int(total_cost[gt_index - 1, pred_index])
            best_decision = 0

            skip_pred_matches = int(match_count[gt_index, pred_index - 1])
            skip_pred_cost = int(total_cost[gt_index, pred_index - 1])
            if _is_better(
                skip_pred_matches,
                skip_pred_cost,
                best_matches,
                best_cost,
            ):
                best_matches = skip_pred_matches
                best_cost = skip_pred_cost
                best_decision = 1

            distance = abs(
                gt_boundaries[gt_index - 1] - pred_boundaries[pred_index - 1]
            )
            if distance <= tolerance:
                matched_count = int(match_count[gt_index - 1, pred_index - 1]) + 1
                matched_cost = int(total_cost[gt_index - 1, pred_index - 1]) + distance
                if _is_better(
                    matched_count,
                    matched_cost,
                    best_matches,
                    best_cost,
                ) or (
                    matched_count == best_matches
                    and matched_cost == best_cost
                    and best_decision != 2
                ):
                    best_matches = matched_count
                    best_cost = matched_cost
                    best_decision = 2

            match_count[gt_index, pred_index] = best_matches
            total_cost[gt_index, pred_index] = best_cost
            decision[gt_index, pred_index] = best_decision

    matches: list[BoundaryMatch] = []
    gt_index = n_gt
    pred_index = n_pred
    while gt_index > 0 and pred_index > 0:
        action = int(decision[gt_index, pred_index])
        if action == 2:
            matches.append(
                BoundaryMatch(
                    gt_time=gt_boundaries[gt_index - 1],
                    pred_time=pred_boundaries[pred_index - 1],
                )
            )
            gt_index -= 1
            pred_index -= 1
        elif action == 1:
            pred_index -= 1
        else:
            gt_index -= 1
    matches.reverse()
    return matches


def match_boundaries(
    gt_boundaries: Iterable[int],
    pred_boundaries: Iterable[int],
    tolerance: int,
    *,
    strategy: str = OPTIMAL_ORDERED,
) -> dict[str, object]:
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative integer")
    if strategy not in SUPPORTED_MATCHING_STRATEGIES:
        raise ValueError(
            f"Unsupported boundary matching strategy {strategy!r}; "
            f"expected one of {SUPPORTED_MATCHING_STRATEGIES}"
        )

    gt = _normalise_events(gt_boundaries, "ground-truth")
    pred = _normalise_events(pred_boundaries, "predicted")
    if strategy == LEGACY_GREEDY:
        matches = _greedy_pairs(gt, pred, tolerance)
    else:
        matches = _optimal_ordered_pairs(gt, pred, tolerance)

    matched_errors = [match.absolute_error for match in matches]
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    mean_error = float(np.mean(matched_errors)) if matched_errors else float("nan")
    return {
        "strategy": strategy,
        "tolerance": tolerance,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_error": mean_error,
        "mean_abs_error_sec": mean_error,
        "matched_errors": matched_errors,
        "matched_pairs": [
            {
                "gt_time": match.gt_time,
                "pred_time": match.pred_time,
                "absolute_error": match.absolute_error,
            }
            for match in matches
        ],
    }


def greedy_boundary_match(
    gt_boundaries: Iterable[int],
    pred_boundaries: Iterable[int],
    tolerance: int,
) -> dict[str, object]:
    """Exact legacy wrapper retained for traceable reproduction."""
    return match_boundaries(
        gt_boundaries,
        pred_boundaries,
        tolerance,
        strategy=LEGACY_GREEDY,
    )


def optimal_boundary_match(
    gt_boundaries: Iterable[int],
    pred_boundaries: Iterable[int],
    tolerance: int,
) -> dict[str, object]:
    return match_boundaries(
        gt_boundaries,
        pred_boundaries,
        tolerance,
        strategy=OPTIMAL_ORDERED,
    )
