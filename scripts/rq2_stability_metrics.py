"""Shared RQ2 phase, boundary, and temporal-stability metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from boundary_metrics import OPTIMAL_ORDERED, extract_boundaries, match_boundaries


PHASES = tuple(range(1, 8))


def _compressed(sequence: Iterable[int]) -> list[int]:
    values = [int(value) for value in sequence]
    if not values:
        return []
    return [values[0], *[values[i] for i in range(1, len(values)) if values[i] != values[i - 1]]]


def edit_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Return the conventional normalised segmental edit score in [0, 100]."""
    first = _compressed(truth)
    second = _compressed(prediction)
    if not first and not second:
        return 100.0
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    distance = previous[-1]
    return 100.0 * (1.0 - distance / max(len(first), len(second), 1))


def temporal_fragmentation_index(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute TFI exactly as equation (7) of Liu et al. (2026)."""
    truth = np.asarray(truth, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("truth and prediction must be aligned one-dimensional arrays")
    if not len(truth):
        raise ValueError("a sequence must be non-empty")
    gt_segments = 1 + int(np.sum(truth[1:] != truth[:-1]))
    starts = np.r_[0, np.flatnonzero(prediction[1:] != prediction[:-1]) + 1]
    ends = np.r_[starts[1:], len(prediction)]
    disagreement_sum = 0.0
    for start, end in zip(starts, ends):
        disagreement_sum += float(np.mean(prediction[start:end] != truth[start:end]))
    return disagreement_sum / gt_segments


def boundary_event_table(
    truth: np.ndarray,
    prediction: np.ndarray,
    times: np.ndarray,
    tolerance: int,
    *,
    strategy: str = OPTIMAL_ORDERED,
) -> pd.DataFrame:
    gt = extract_boundaries(truth, times)
    pred = extract_boundaries(prediction, times)
    match = match_boundaries(gt, pred, tolerance, strategy=strategy)
    matched_gt = {int(item["gt_time"]) for item in match["matched_pairs"]}
    matched_pred = {int(item["pred_time"]) for item in match["matched_pairs"]}
    rows: list[dict[str, float | int | str]] = []
    for item in match["matched_pairs"]:
        signed = int(item["pred_time"]) - int(item["gt_time"])
        rows.append(
            {
                "event_type": "matched",
                "gt_time_sec": int(item["gt_time"]),
                "pred_time_sec": int(item["pred_time"]),
                "signed_delay_sec": signed,
                "absolute_error_sec": abs(signed),
            }
        )
    for value in gt:
        if int(value) not in matched_gt:
            rows.append(
                {
                    "event_type": "missed",
                    "gt_time_sec": int(value),
                    "pred_time_sec": np.nan,
                    "signed_delay_sec": np.nan,
                    "absolute_error_sec": np.nan,
                }
            )
    for value in pred:
        if int(value) not in matched_pred:
            rows.append(
                {
                    "event_type": "extra",
                    "gt_time_sec": np.nan,
                    "pred_time_sec": int(value),
                    "signed_delay_sec": np.nan,
                    "absolute_error_sec": np.nan,
                }
            )
    return pd.DataFrame(rows)


def evaluate_videos(
    videos: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, np.ndarray],
    *,
    tolerances: tuple[int, ...] = (5, 10),
    strategy: str = OPTIMAL_ORDERED,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Evaluate one training seed across a split of videos."""
    all_truth: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    video_rows: list[dict[str, float | int | str]] = []
    event_frames: list[pd.DataFrame] = []
    totals = {
        tolerance: {"tp": 0, "fp": 0, "fn": 0, "offsets": []}
        for tolerance in tolerances
    }
    for video_id in sorted(videos):
        video = videos[video_id]
        truth = np.asarray(video["truth"], dtype=int)
        prediction = np.asarray(predictions[video_id], dtype=int)
        times = np.asarray(video["times"], dtype=int)
        if len(truth) != len(prediction) or len(truth) != len(times):
            raise ValueError(f"unaligned arrays for video {video_id}")
        gt_boundaries = extract_boundaries(truth, times)
        pred_boundaries = extract_boundaries(prediction, times)
        row: dict[str, float | int | str] = {
            "video_id": str(video_id).zfill(2),
            "n_frames": len(truth),
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(
                f1_score(
                    truth,
                    prediction,
                    labels=list(PHASES),
                    average="macro",
                    zero_division=0,
                )
            ),
            "predicted_boundary_count": len(pred_boundaries),
            "ground_truth_boundary_count": len(gt_boundaries),
            "edit_score": edit_score(truth, prediction),
            "tfi": temporal_fragmentation_index(truth, prediction),
        }
        for tolerance in tolerances:
            result = match_boundaries(
                gt_boundaries,
                pred_boundaries,
                tolerance,
                strategy=strategy,
            )
            offsets = [
                int(item["pred_time"]) - int(item["gt_time"])
                for item in result["matched_pairs"]
            ]
            totals[tolerance]["tp"] += int(result["tp"])
            totals[tolerance]["fp"] += int(result["fp"])
            totals[tolerance]["fn"] += int(result["fn"])
            totals[tolerance]["offsets"].extend(offsets)
            row.update(
                {
                    f"boundary_precision_tol{tolerance}": float(result["precision"]),
                    f"boundary_recall_tol{tolerance}": float(result["recall"]),
                    f"boundary_f1_tol{tolerance}": float(result["f1"]),
                    f"matched_signed_delay_tol{tolerance}": (
                        float(np.mean(offsets)) if offsets else np.nan
                    ),
                    f"matched_abs_error_tol{tolerance}": (
                        float(np.mean(np.abs(offsets))) if offsets else np.nan
                    ),
                    f"tp_tol{tolerance}": int(result["tp"]),
                    f"fp_tol{tolerance}": int(result["fp"]),
                    f"fn_tol{tolerance}": int(result["fn"]),
                }
            )
            event_frame = boundary_event_table(
                truth,
                prediction,
                times,
                tolerance,
                strategy=strategy,
            )
            event_frame.insert(0, "tolerance_sec", tolerance)
            event_frame.insert(0, "video_id", str(video_id).zfill(2))
            event_frames.append(event_frame)
        video_rows.append(row)
        all_truth.append(truth)
        all_prediction.append(prediction)

    truth = np.concatenate(all_truth)
    prediction = np.concatenate(all_prediction)
    aggregate: dict[str, float] = {
        "n_frames": float(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=list(PHASES),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                prediction,
                labels=list(PHASES),
                average="weighted",
                zero_division=0,
            )
        ),
        "predicted_boundary_count": float(
            sum(int(row["predicted_boundary_count"]) for row in video_rows)
        ),
        "ground_truth_boundary_count": float(
            sum(int(row["ground_truth_boundary_count"]) for row in video_rows)
        ),
        "mean_video_edit_score": float(
            np.mean([float(row["edit_score"]) for row in video_rows])
        ),
        "mean_video_tfi": float(np.mean([float(row["tfi"]) for row in video_rows])),
    }
    for tolerance, values in totals.items():
        tp, fp, fn = int(values["tp"]), int(values["fp"]), int(values["fn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        offsets = np.asarray(values["offsets"], dtype=float)
        aggregate.update(
            {
                f"boundary_precision_tol{tolerance}": precision,
                f"boundary_recall_tol{tolerance}": recall,
                f"boundary_f1_tol{tolerance}": f1,
                f"matched_signed_delay_tol{tolerance}": (
                    float(offsets.mean()) if len(offsets) else np.nan
                ),
                f"matched_abs_error_tol{tolerance}": (
                    float(np.abs(offsets).mean()) if len(offsets) else np.nan
                ),
                f"tp_tol{tolerance}": float(tp),
                f"fp_tol{tolerance}": float(fp),
                f"fn_tol{tolerance}": float(fn),
            }
        )
    return (
        aggregate,
        pd.DataFrame(video_rows),
        pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(),
    )


__all__ = [
    "PHASES",
    "boundary_event_table",
    "edit_score",
    "evaluate_videos",
    "temporal_fragmentation_index",
]
