"""Consolidate the final RQ1 evidence without training or model inference.

This script reads only existing deterministic predictions, validation MC
convergence outputs, final test MC outputs, and final raw-logit calibration
outputs. It recomputes the frozen metrics under one definition and writes a
versioned evidence bundle. It intentionally does not compute near/far AURC.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from reproducibility_utils import PROJECT_ROOT, ensure_fresh_output_dir, sha256_file
from selective_prediction_metrics import empirical_risk_coverage, exact_aurc


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rq1_final_evidence_v1.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "rq1_final_evidence_v1"
WINDOWS = (5, 10, 20)
SPLITS = ("val", "test")
SEEDS = (0, 1, 2)
PROBABILITY_COLUMNS = [f"prob_phase_{index}" for index in range(1, 8)]
PRIMARY_SCORES = {
    "deterministic_raw": ("one_minus_confidence", "normalised_entropy"),
    "mc_dropout_t30": ("mc_entropy", "mc_mutual_info"),
}


def source_record(path: str | Path, role: str, **metadata: Any) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "role": role,
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
        **metadata,
    }


def safe_detection_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or np.unique(labels).size < 2:
        return np.nan, np.nan
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def normalised_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / np.log(probabilities.shape[1])


def standardise_video_time(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["video_id"] = output["video_id"].astype(str).str.zfill(2)
    output["t_sec"] = output["t_sec"].astype(np.int64)
    return output.sort_values(["video_id", "t_sec"]).reset_index(drop=True)


def load_deterministic_frames(
    sources: list[dict[str, Any]],
) -> dict[tuple[int, str, str], pd.DataFrame]:
    frames: dict[tuple[int, str, str], pd.DataFrame] = {}
    expected_videos = {
        "val": [f"{video:02d}" for video in range(11, 15)],
        "test": [f"{video:02d}" for video in range(15, 22)],
    }
    for training_seed in SEEDS:
        run_dir = PROJECT_ROOT / "outputs" / f"v2_lstm_online_resnet18_seed{training_seed:02d}"
        checkpoint = run_dir / "checkpoints" / "best.pt"
        sources.append(
            source_record(
                checkpoint,
                "selected_baseline_checkpoint",
                training_seed=training_seed,
            )
        )
        for split in SPLITS:
            path = run_dir / "model_outputs" / f"per_second_outputs_{split}.csv"
            sources.append(
                source_record(
                    path,
                    "deterministic_frame_scores",
                    training_seed=training_seed,
                    split=split,
                )
            )
            raw = standardise_video_time(pd.read_csv(path, dtype={"video_id": str}))
            required = {
                "true_label_idx",
                "pred_label_idx",
                "dist_to_gt_boundary_sec",
                *PROBABILITY_COLUMNS,
            }
            missing = sorted(required - set(raw.columns))
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            if sorted(raw["video_id"].unique()) != expected_videos[split]:
                raise ValueError(f"unexpected {split} videos in {path}")
            probabilities = raw[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-6, rtol=0.0):
                raise ValueError(f"probability rows do not sum to one in {path}")
            labels = raw["true_label_idx"].to_numpy(dtype=np.int64)
            predictions = raw["pred_label_idx"].to_numpy(dtype=np.int64)
            frame = raw[["video_id", "t_sec"]].copy()
            frame["training_seed"] = training_seed
            frame["split"] = split
            frame["prediction_source"] = "deterministic_raw"
            frame["true_label_idx"] = labels
            frame["pred_label_idx"] = predictions
            frame["error"] = (predictions != labels).astype(np.int64)
            frame["dist_to_gt_boundary"] = raw[
                "dist_to_gt_boundary_sec"
            ].to_numpy(dtype=np.float64)
            frame["one_minus_confidence"] = 1.0 - probabilities.max(axis=1)
            frame["normalised_entropy"] = normalised_entropy(probabilities)
            frames[(training_seed, split, "deterministic_raw")] = frame
    return frames


def load_mc_validation_frames(
    deterministic_frames: dict[tuple[int, str, str], pd.DataFrame],
    sources: list[dict[str, Any]],
) -> dict[tuple[int, str, str], pd.DataFrame]:
    convergence_dir = PROJECT_ROOT / "outputs" / "mc_dropout_convergence_validation_v3"
    frame_path = convergence_dir / "validation_frame_scores.csv.gz"
    manifest_path = convergence_dir / "run_manifest.json"
    sources.extend(
        [
            source_record(manifest_path, "mc_validation_convergence_manifest"),
            source_record(frame_path, "mc_validation_nested_frame_scores"),
            source_record(
                convergence_dir / "validation_convergence_per_seed.csv",
                "mc_validation_convergence_per_seed",
            ),
            source_record(
                convergence_dir / "validation_convergence_summary.csv",
                "mc_validation_convergence_summary",
            ),
            source_record(
                convergence_dir / "selected_mc_config.json",
                "mc_validation_selected_config",
            ),
        ]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_files_read") is not False:
        raise ValueError("MC validation convergence did not preserve test isolation")
    if manifest.get("selected_T") != 30 or manifest.get("inference_seed") != 0:
        raise ValueError("unexpected MC validation T or inference seed")
    all_mc = pd.read_csv(frame_path, dtype={"video_id": str})
    all_mc = all_mc[all_mc["T"].eq(30)].copy()
    if set(all_mc["training_seed"].astype(int)) != set(SEEDS):
        raise ValueError("MC validation frame scores do not contain all three seeds")
    frames: dict[tuple[int, str, str], pd.DataFrame] = {}
    for training_seed in SEEDS:
        mc = standardise_video_time(
            all_mc[all_mc["training_seed"].eq(training_seed)].copy()
        )
        deterministic = deterministic_frames[(training_seed, "val", "deterministic_raw")]
        distance = deterministic[
            ["video_id", "t_sec", "true_label_idx", "dist_to_gt_boundary"]
        ].rename(columns={"true_label_idx": "deterministic_true_label_idx"})
        merged = mc.merge(distance, on=["video_id", "t_sec"], how="inner", validate="one_to_one")
        if len(merged) != len(mc) or len(merged) != len(deterministic):
            raise ValueError(f"MC validation alignment failed for seed {training_seed}")
        if not np.array_equal(
            merged["true_label_idx"].to_numpy(dtype=np.int64),
            merged["deterministic_true_label_idx"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"MC validation labels disagree for seed {training_seed}")
        if not np.array_equal(
            merged["error"].to_numpy(dtype=np.int64),
            (
                merged["pred_label_idx"].to_numpy(dtype=np.int64)
                != merged["true_label_idx"].to_numpy(dtype=np.int64)
            ).astype(np.int64),
        ):
            raise ValueError(f"MC validation error column disagrees for seed {training_seed}")
        frame = merged[
            [
                "video_id",
                "t_sec",
                "true_label_idx",
                "pred_label_idx",
                "error",
                "dist_to_gt_boundary",
                "mc_entropy",
                "mc_mutual_information",
            ]
        ].rename(columns={"mc_mutual_information": "mc_mutual_info"})
        frame.insert(0, "prediction_source", "mc_dropout_t30")
        frame.insert(0, "split", "val")
        frame.insert(0, "training_seed", training_seed)
        frames[(training_seed, "val", "mc_dropout_t30")] = frame
    return frames


def load_mc_test_frames(
    deterministic_frames: dict[tuple[int, str, str], pd.DataFrame],
    sources: list[dict[str, Any]],
) -> dict[tuple[int, str, str], pd.DataFrame]:
    bundle_manifest = PROJECT_ROOT / "outputs" / "mc_dropout_three_seed_protocol_v4" / "run_manifest.json"
    sources.append(source_record(bundle_manifest, "complete_mc_test_protocol_manifest"))
    frames: dict[tuple[int, str, str], pd.DataFrame] = {}
    for training_seed in SEEDS:
        run_dir = PROJECT_ROOT / "outputs" / f"rq1_mc_dropout_t30_seed{training_seed:02d}_v4"
        manifest_path = run_dir / "run_manifest.json"
        frame_path = run_dir / "mc_dropout_frame_scores.csv"
        sources.extend(
            [
                source_record(
                    manifest_path,
                    "mc_test_manifest",
                    training_seed=training_seed,
                ),
                source_record(
                    frame_path,
                    "mc_test_frame_scores",
                    training_seed=training_seed,
                    split="test",
                ),
            ]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocol = manifest["protocol"]
        if (
            protocol.get("mc_passes") != 30
            or protocol.get("inference_seed") != 0
            or protocol.get("rng_scope") != "reset_once_at_the_start_of_each_split"
            or protocol.get("splits") != ["test"]
        ):
            raise ValueError(f"seed {training_seed} does not follow final MC test protocol")
        mc = standardise_video_time(pd.read_csv(frame_path, dtype={"video_id": str}))
        deterministic = deterministic_frames[(training_seed, "test", "deterministic_raw")]
        comparison = deterministic[["video_id", "t_sec", "true_label_idx"]].merge(
            mc[["video_id", "t_sec", "true_label_idx"]],
            on=["video_id", "t_sec"],
            suffixes=("_deterministic", "_mc"),
            validate="one_to_one",
        )
        if len(comparison) != len(mc) or not np.array_equal(
            comparison["true_label_idx_deterministic"].to_numpy(dtype=np.int64),
            comparison["true_label_idx_mc"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"MC test alignment failed for seed {training_seed}")
        frame = mc[
            [
                "video_id",
                "t_sec",
                "true_label_idx",
                "pred_label_idx",
                "error",
                "dist_to_gt_boundary",
                "mc_entropy",
                "mc_mutual_info",
            ]
        ].copy()
        frame.insert(0, "prediction_source", "mc_dropout_t30")
        frame.insert(0, "split", "test")
        frame.insert(0, "training_seed", training_seed)
        frames[(training_seed, "test", "mc_dropout_t30")] = frame
    return frames


def phase_metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["true_label_idx"].to_numpy(dtype=np.int64)
    predictions = frame["pred_label_idx"].to_numpy(dtype=np.int64)
    return {
        "training_seed": int(frame["training_seed"].iloc[0]),
        "split": str(frame["split"].iloc[0]),
        "prediction_source": str(frame["prediction_source"].iloc[0]),
        "n_frames": len(frame),
        "n_errors": int((predictions != labels).sum()),
        "error_rate": float((predictions != labels).mean()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(7)),
                average="macro",
                zero_division=0,
            )
        ),
    }


def evaluate_frames(
    frames: dict[tuple[int, str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for key in sorted(frames):
        frame = frames[key]
        training_seed, split, prediction_source = key
        phase_rows.append(phase_metric_row(frame))
        errors = frame["error"].to_numpy(dtype=np.int64)
        distances = frame["dist_to_gt_boundary"].to_numpy(dtype=np.float64)
        for score in PRIMARY_SCORES[prediction_source]:
            scores = frame[score].to_numpy(dtype=np.float64)
            auroc, aupr = safe_detection_metrics(errors, scores)
            aurc = exact_aurc(empirical_risk_coverage(errors, scores))
            overall_rows.append(
                {
                    "training_seed": training_seed,
                    "split": split,
                    "prediction_source": prediction_source,
                    "score": score,
                    "n_frames": len(frame),
                    "n_errors": int(errors.sum()),
                    "error_rate": float(errors.mean()),
                    "auroc": auroc,
                    "aupr": aupr,
                    "exact_aurc": float(aurc["aurc_lower_better"]),
                    "oracle_aurc": float(aurc["oracle_aurc"]),
                    "excess_aurc": float(aurc["excess_aurc"]),
                }
            )
            for window in WINDOWS:
                near = distances <= window
                for region, member in (
                    ("near_boundary", near),
                    ("far_from_boundary", ~near),
                ):
                    region_auroc, region_aupr = safe_detection_metrics(
                        errors[member],
                        scores[member],
                    )
                    conditional_rows.append(
                        {
                            "training_seed": training_seed,
                            "split": split,
                            "prediction_source": prediction_source,
                            "score": score,
                            "region": region,
                            "window_sec": window,
                            "n_frames": int(member.sum()),
                            "n_errors": int(errors[member].sum()),
                            "error_rate": float(errors[member].mean()),
                            "mean_uncertainty": float(scores[member].mean()),
                            "auroc": region_auroc,
                            "aupr": region_aupr,
                        }
                    )
                boundary_positive = near.astype(np.int64)
                boundary_auroc, boundary_aupr = safe_detection_metrics(
                    boundary_positive,
                    scores,
                )
                boundary_rows.append(
                    {
                        "training_seed": training_seed,
                        "split": split,
                        "prediction_source": prediction_source,
                        "score": score,
                        "window_sec": window,
                        "n_frames": len(frame),
                        "n_near_boundary": int(boundary_positive.sum()),
                        "near_boundary_prevalence": float(boundary_positive.mean()),
                        "auroc": boundary_auroc,
                        "aupr": boundary_aupr,
                    }
                )
    return (
        pd.DataFrame(phase_rows),
        pd.DataFrame(overall_rows),
        pd.DataFrame(conditional_rows),
        pd.DataFrame(boundary_rows),
    )


def aggregate(
    frame: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_training_seeds"] = int(group["training_seed"].nunique())
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def build_master_table(
    overall_summary: pd.DataFrame,
    conditional_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_metric(
        section: str,
        row: pd.Series,
        metric: str,
        direction: str,
        region: str = "overall",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "section": section,
                "split": row.get("split", ""),
                "prediction_source": row.get("prediction_source", ""),
                "score_or_state": row.get("score", row.get("calibration_state", "")),
                "region": region,
                "window_sec": row.get("window_sec", np.nan),
                "metric": metric,
                "direction": direction,
                "value_mean": row.get(f"{metric}_mean", np.nan),
                "value_std": row.get(f"{metric}_std", np.nan),
                "n_training_seeds": row.get("n_training_seeds", 3),
                "n_frames_mean": row.get("n_frames_mean", np.nan),
                "positive_rate_mean": row.get(
                    "error_rate_mean",
                    row.get("near_boundary_prevalence_mean", np.nan),
                ),
                "notes": notes,
            }
        )

    for _, row in overall_summary.iterrows():
        for metric, direction in (("auroc", "higher"), ("aupr", "higher"), ("exact_aurc", "lower")):
            append_metric("overall_error_detection", row, metric, direction)
    for _, row in conditional_summary.iterrows():
        for metric in ("auroc", "aupr"):
            append_metric(
                "conditional_error_detection",
                row,
                metric,
                "higher",
                region=str(row["region"]),
                notes="AUPR depends on the error prevalence inside this region.",
            )
    for _, row in boundary_summary.iterrows():
        for metric in ("auroc", "aupr"):
            append_metric(
                "boundary_proximity_detection",
                row,
                metric,
                "higher",
                region="near_boundary_is_positive",
                notes="Positive means within the inclusive ground-truth boundary window.",
            )

    convergence_metrics = [
        ("all_seeds_stable", "required_true"),
        ("prediction_disagreement_max", "lower"),
        ("mc_entropy_spearman_min", "higher"),
        ("mc_mutual_information_spearman_min", "higher"),
        ("error_auroc_abs_diff_max", "lower"),
        ("error_aupr_abs_diff_max", "lower"),
        ("aurc_abs_diff_max", "lower"),
        ("inference_seconds_mean", "lower"),
    ]
    for _, row in convergence.iterrows():
        for metric, direction in convergence_metrics:
            value = float(bool(row[metric])) if metric == "all_seeds_stable" else float(row[metric])
            rows.append(
                {
                    "section": "mc_validation_convergence",
                    "split": "val",
                    "prediction_source": "mc_dropout",
                    "score_or_state": f"T={int(row['T'])}",
                    "region": "overall",
                    "window_sec": np.nan,
                    "metric": metric,
                    "direction": direction,
                    "value_mean": value,
                    "value_std": row.get("inference_seconds_std", np.nan)
                    if metric == "inference_seconds_mean"
                    else np.nan,
                    "n_training_seeds": int(row["n_training_seeds"]),
                    "n_frames_mean": float(row["n_frames_mean"]),
                    "positive_rate_mean": np.nan,
                    "notes": "T=30 is the smallest candidate stable for all three seeds."
                    if int(row["T"]) == 30
                    else "",
                }
            )

    calibration_metrics = [
        ("nll", "lower"),
        ("brier", "lower"),
        ("ece_15_bins", "lower"),
        ("accuracy", "unchanged"),
        ("macro_f1", "unchanged"),
    ]
    for _, row in calibration.iterrows():
        for metric, direction in calibration_metrics:
            append_metric(
                "temperature_scaling_calibration",
                row,
                metric,
                direction,
                notes="Temperature fitted per seed on validation raw logits; test evaluation only.",
            )
    return pd.DataFrame(rows)


def build_legacy_comparison(
    final_overall: pd.DataFrame,
    final_conditional: pd.DataFrame,
    final_boundary: pd.DataFrame,
    final_calibration: pd.DataFrame,
    sources: list[dict[str, Any]],
) -> pd.DataFrame:
    legacy_dir = PROJECT_ROOT / "outputs" / "multiseed_rq1_frozen_consistency_v2"
    legacy_detection_path = legacy_dir / "per_seed_error_detection.csv"
    legacy_risk_path = legacy_dir / "per_seed_risk_coverage.csv"
    legacy_boundary_path = legacy_dir / "per_seed_boundary_proximity_detection.csv"
    legacy_calibration_path = (
        PROJECT_ROOT / "outputs" / "calibration_multiseed_v1" / "per_seed_calibration_metrics.csv"
    )
    for path, role in (
        (legacy_dir / "run_manifest.json", "legacy_unified_rq1_manifest"),
        (legacy_detection_path, "legacy_unified_error_detection"),
        (legacy_risk_path, "legacy_unified_risk_coverage"),
        (legacy_boundary_path, "legacy_unified_boundary_proximity"),
        (legacy_calibration_path, "legacy_calibration_metrics"),
    ):
        sources.append(source_record(path, role))

    allowed = {
        (source, score)
        for source, scores in PRIMARY_SCORES.items()
        for score in scores
    }
    rows: list[dict[str, Any]] = []

    def append_comparison(
        section: str,
        metric: str,
        keys: dict[str, Any],
        legacy_value: float,
        final_value: float,
    ) -> None:
        difference = float(final_value - legacy_value)
        rows.append(
            {
                "section": section,
                **keys,
                "metric": metric,
                "legacy_value": float(legacy_value),
                "final_value": float(final_value),
                "signed_difference_final_minus_legacy": difference,
                "absolute_difference": abs(difference),
                "exactly_equal": bool(final_value == legacy_value),
            }
        )

    old_detection = pd.read_csv(legacy_detection_path)
    old_overall = old_detection[old_detection["region"].eq("overall")].copy()
    old_overall = old_overall[
        old_overall.apply(
            lambda row: (row["prediction_source"], row["score"]) in allowed,
            axis=1,
        )
    ]
    merged = old_overall.merge(
        final_overall,
        on=["training_seed", "split", "prediction_source", "score"],
        suffixes=("_legacy", "_final"),
        validate="one_to_one",
    )
    for row in merged.itertuples():
        keys = {
            "training_seed": row.training_seed,
            "split": row.split,
            "prediction_source": row.prediction_source,
            "score_or_state": row.score,
            "region": "overall",
            "window_sec": np.nan,
        }
        for metric in ("auroc", "aupr"):
            append_comparison(
                "overall_error_detection",
                metric,
                keys,
                getattr(row, f"{metric}_legacy"),
                getattr(row, f"{metric}_final"),
            )

    old_risk = pd.read_csv(legacy_risk_path)
    old_risk = old_risk[
        old_risk.apply(
            lambda row: (row["prediction_source"], row["score"]) in allowed,
            axis=1,
        )
    ]
    merged = old_risk.merge(
        final_overall,
        on=["training_seed", "split", "prediction_source", "score"],
        suffixes=("_legacy", "_final"),
        validate="one_to_one",
    )
    for row in merged.itertuples():
        append_comparison(
            "overall_error_detection",
            "exact_aurc",
            {
                "training_seed": row.training_seed,
                "split": row.split,
                "prediction_source": row.prediction_source,
                "score_or_state": row.score,
                "region": "overall",
                "window_sec": np.nan,
            },
            row.aurc,
            row.exact_aurc,
        )

    old_conditional = old_detection[old_detection["region"].ne("overall")].copy()
    old_conditional = old_conditional[
        old_conditional.apply(
            lambda row: (row["prediction_source"], row["score"]) in allowed,
            axis=1,
        )
    ]
    merged = old_conditional.merge(
        final_conditional,
        on=[
            "training_seed",
            "split",
            "prediction_source",
            "score",
            "region",
            "window_sec",
        ],
        suffixes=("_legacy", "_final"),
        validate="one_to_one",
    )
    for row in merged.itertuples():
        keys = {
            "training_seed": row.training_seed,
            "split": row.split,
            "prediction_source": row.prediction_source,
            "score_or_state": row.score,
            "region": row.region,
            "window_sec": row.window_sec,
        }
        for metric in ("auroc", "aupr"):
            append_comparison(
                "conditional_error_detection",
                metric,
                keys,
                getattr(row, f"{metric}_legacy"),
                getattr(row, f"{metric}_final"),
            )

    old_boundary = pd.read_csv(legacy_boundary_path)
    old_boundary = old_boundary[
        old_boundary.apply(
            lambda row: (row["prediction_source"], row["score"]) in allowed,
            axis=1,
        )
    ]
    merged = old_boundary.merge(
        final_boundary,
        on=["training_seed", "split", "prediction_source", "score", "window_sec"],
        suffixes=("_legacy", "_final"),
        validate="one_to_one",
    )
    for row in merged.itertuples():
        keys = {
            "training_seed": row.training_seed,
            "split": row.split,
            "prediction_source": row.prediction_source,
            "score_or_state": row.score,
            "region": "near_boundary_is_positive",
            "window_sec": row.window_sec,
        }
        for metric in ("auroc", "aupr"):
            append_comparison(
                "boundary_proximity_detection",
                metric,
                keys,
                getattr(row, f"{metric}_legacy"),
                getattr(row, f"{metric}_final"),
            )

    old_calibration = pd.read_csv(legacy_calibration_path)
    old_calibration = old_calibration[old_calibration["split"].eq("test")]
    merged = old_calibration.merge(
        final_calibration,
        on=["training_seed", "split", "calibration_state"],
        suffixes=("_legacy", "_final"),
        validate="one_to_one",
    )
    for row in merged.itertuples():
        keys = {
            "training_seed": row.training_seed,
            "split": row.split,
            "prediction_source": "deterministic_temperature_scaling",
            "score_or_state": row.calibration_state,
            "region": "overall",
            "window_sec": np.nan,
        }
        for metric in ("nll", "brier", "ece_15_bins", "accuracy"):
            append_comparison(
                "temperature_scaling_calibration",
                metric,
                keys,
                getattr(row, f"{metric}_legacy"),
                getattr(row, f"{metric}_final"),
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("config_id") != "rq1_final_evidence_v1":
        raise ValueError("unexpected final RQ1 evidence configuration")
    if config["explicit_exclusions"].get("near_far_aurc") is None:
        raise ValueError("final config must explicitly exclude near/far AURC")
    out_dir = ensure_fresh_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = [
        source_record(config_path, "final_evidence_configuration"),
    ]
    for path in (
        PROJECT_ROOT / "configs" / "rq1_evaluation_v4.json",
        PROJECT_ROOT / "configs" / "baseline_training_v1.json",
        PROJECT_ROOT / "configs" / "mc_dropout_evaluation_v4.json",
        PROJECT_ROOT / "configs" / "temperature_scaling_calibration_v2.json",
        PROJECT_ROOT / "configs" / "selective_prediction_evaluation_v2.json",
    ):
        sources.append(source_record(path, "authoritative_configuration"))

    deterministic_frames = load_deterministic_frames(sources)
    all_frames = dict(deterministic_frames)
    all_frames.update(load_mc_validation_frames(deterministic_frames, sources))
    all_frames.update(load_mc_test_frames(deterministic_frames, sources))
    expected_keys = {
        (seed, split, source)
        for seed in SEEDS
        for split in SPLITS
        for source in PRIMARY_SCORES
    }
    if set(all_frames) != expected_keys:
        raise ValueError(f"final RQ1 frame sources incomplete: {sorted(expected_keys - set(all_frames))}")

    phase, overall, conditional, boundary = evaluate_frames(all_frames)
    phase_summary = aggregate(
        phase,
        ["split", "prediction_source"],
        ["n_frames", "n_errors", "error_rate", "accuracy", "macro_f1"],
    )
    overall_summary = aggregate(
        overall,
        ["split", "prediction_source", "score"],
        [
            "n_frames",
            "n_errors",
            "error_rate",
            "auroc",
            "aupr",
            "exact_aurc",
            "oracle_aurc",
            "excess_aurc",
        ],
    )
    conditional_summary = aggregate(
        conditional,
        ["split", "prediction_source", "score", "region", "window_sec"],
        ["n_frames", "n_errors", "error_rate", "mean_uncertainty", "auroc", "aupr"],
    )
    boundary_summary = aggregate(
        boundary,
        ["split", "prediction_source", "score", "window_sec"],
        ["n_frames", "n_near_boundary", "near_boundary_prevalence", "auroc", "aupr"],
    )

    convergence_dir = PROJECT_ROOT / "outputs" / "mc_dropout_convergence_validation_v3"
    convergence_source = pd.read_csv(convergence_dir / "validation_convergence_summary.csv")
    convergence = convergence_source[
        [
            "T",
            "n_frames_mean",
            "inference_seconds_mean",
            "inference_seconds_std",
            "all_seeds_stable",
            "prediction_disagreement_max",
            "mc_entropy_spearman_min",
            "mc_mutual_information_spearman_min",
            "error_auroc_abs_diff_max",
            "error_aupr_abs_diff_max",
            "aurc_abs_diff_max",
        ]
    ].copy()
    convergence.insert(1, "n_training_seeds", 3)
    if convergence.loc[convergence["all_seeds_stable"], "T"].astype(int).tolist() != [30, 50]:
        raise ValueError("current convergence evidence no longer selects T=30 as the minimum stable T")

    calibration_dir = PROJECT_ROOT / "outputs" / "temperature_scaling_three_seed_protocol_v2"
    calibration_per_seed_path = calibration_dir / "test_calibration_metrics_per_seed.csv"
    calibration_summary_path = calibration_dir / "test_calibration_mean_std.csv"
    temperature_path = calibration_dir / "temperature_per_seed.csv"
    calibration_manifest_path = calibration_dir / "run_manifest.json"
    for path, role in (
        (calibration_manifest_path, "complete_calibration_protocol_manifest"),
        (calibration_dir / "calibration_protocol.json", "frozen_calibration_protocol"),
        (calibration_per_seed_path, "final_calibration_per_seed"),
        (calibration_summary_path, "final_calibration_summary"),
        (temperature_path, "validation_fitted_temperatures"),
    ):
        sources.append(source_record(path, role))
    calibration_manifest = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
    if (
        calibration_manifest.get("test_used_for_temperature_fitting_or_selection") is not False
        or calibration_manifest.get("rq2_gate_inputs_changed") is not False
    ):
        raise ValueError("final calibration protocol violates RQ1/RQ2 separation")
    calibration_per_seed = pd.read_csv(calibration_per_seed_path)
    calibration_summary = pd.read_csv(calibration_summary_path)
    temperatures = pd.read_csv(temperature_path)

    legacy_comparison = build_legacy_comparison(
        overall,
        conditional,
        boundary,
        calibration_per_seed,
        sources,
    )
    master = build_master_table(
        overall_summary,
        conditional_summary,
        boundary_summary,
        convergence,
        calibration_summary,
    )

    outputs: dict[str, pd.DataFrame] = {
        "per_seed_phase_metrics.csv": phase,
        "phase_metrics_summary_mean_std.csv": phase_summary,
        "per_seed_overall_error_detection.csv": overall,
        "overall_error_detection_summary_mean_std.csv": overall_summary,
        "per_seed_conditional_error_detection.csv": conditional,
        "conditional_error_detection_summary_mean_std.csv": conditional_summary,
        "per_seed_boundary_proximity_detection.csv": boundary,
        "boundary_proximity_detection_summary_mean_std.csv": boundary_summary,
        "mc_validation_convergence_evidence.csv": convergence,
        "calibration_temperature_per_seed.csv": temperatures,
        "calibration_test_metrics_per_seed.csv": calibration_per_seed,
        "calibration_test_summary_mean_std.csv": calibration_summary,
        "legacy_unified_rq1_comparison.csv": legacy_comparison,
        "rq1_final_evidence_table.csv": master,
    }
    for filename, table in outputs.items():
        table.to_csv(out_dir / filename, index=False)

    authority = {
        "schema_version": 1,
        "authority_id": "rq1_final_evidence_v1",
        "status": "authoritative_after_final_consolidation",
        "evaluation_configuration": "configs/rq1_evaluation_v4.json",
        "final_evidence_configuration": "configs/rq1_final_evidence_v1.json",
        "deterministic": {
            "runs": [
                f"outputs/v2_lstm_online_resnet18_seed{seed:02d}" for seed in SEEDS
            ],
            "roles": ["validation", "test"],
        },
        "mc_validation": {
            "source": "outputs/mc_dropout_convergence_validation_v3",
            "T": 30,
            "role": "validation-only convergence and selected-T validation metrics",
        },
        "mc_test": {
            "source": "outputs/mc_dropout_three_seed_protocol_v4",
            "seed_runs": [
                f"outputs/rq1_mc_dropout_t30_seed{seed:02d}_v4" for seed in SEEDS
            ],
            "T": 30,
            "role": "test consistency evidence",
        },
        "calibration": {
            "source": "outputs/temperature_scaling_three_seed_protocol_v2",
            "input": "original raw logits",
            "ece_bins": 15,
            "rq2_effect": "none",
        },
        "legacy_not_authoritative": [
            "outputs/multiseed_rq1_unified_v2",
            "outputs/multiseed_rq1_frozen_v2",
            "outputs/multiseed_rq1_frozen_consistency_v2",
            "outputs/calibration_multiseed_v1",
            "outputs/rq1_mc_dropout_t30_seed00_v1",
            "outputs/rq1_mc_dropout_t30_seed01_v1",
            "outputs/rq1_mc_dropout_t30_seed02_v1"
        ],
        "excluded_from_final_evidence": {
            "near_far_aurc": True,
            "ece_10_20": "legacy sensitivity only",
            "new_training_or_inference": True,
        },
        "evidence_status": config["evidence_status"],
    }
    authority_path = out_dir / "artifact_authority.json"
    authority_path.write_text(json.dumps(authority, indent=2), encoding="utf-8")

    def lookup(
        table: pd.DataFrame,
        split: str,
        source: str,
        score: str,
        **filters: Any,
    ) -> pd.Series:
        selected = table[
            table["split"].eq(split)
            & table["prediction_source"].eq(source)
            & table["score"].eq(score)
        ]
        for column, value in filters.items():
            selected = selected[selected[column].eq(value)]
        if len(selected) != 1:
            raise ValueError(f"headline lookup is not unique: {split}, {source}, {score}, {filters}")
        return selected.iloc[0]

    overall_headlines = [
        ("Deterministic 1-confidence", "deterministic_raw", "one_minus_confidence"),
        ("Deterministic entropy", "deterministic_raw", "normalised_entropy"),
        ("MC entropy", "mc_dropout_t30", "mc_entropy"),
        ("MC mutual information", "mc_dropout_t30", "mc_mutual_info"),
    ]
    readme = [
        "# Final RQ1 Evidence v1",
        "",
        "This bundle is the authoritative consolidation of the completed core RQ1 experiments.",
        "It performs no training and no model inference.",
        "",
        "## Authoritative inputs",
        "",
        "- Deterministic validation/test: the three validation-selected baseline checkpoints.",
        "- MC validation: T=30 rows from `mc_dropout_convergence_validation_v3`.",
        "- MC test: the three split-isolated v4 runs collected in `mc_dropout_three_seed_protocol_v4`.",
        "- Calibration: the raw-logit `temperature_scaling_three_seed_protocol_v2` bundle.",
        "- Earlier unified RQ1 and v1 MC/calibration outputs remain provenance records, not current authority.",
        "",
        "## Overall test error detection (mean ± sample SD)",
        "",
    ]
    for label, source, score in overall_headlines:
        row = lookup(overall_summary, "test", source, score)
        readme.append(
            f"- {label}: AUROC {row.auroc_mean:.6f} ± {row.auroc_std:.6f}; "
            f"AUPR {row.aupr_mean:.6f} ± {row.aupr_std:.6f}; "
            f"Exact AURC {row.exact_aurc_mean:.6f} ± {row.exact_aurc_std:.6f}."
        )
    readme.extend(
        [
            "",
            "## Conditional error detection at ±10 seconds",
            "",
        ]
    )
    for label, source, score in overall_headlines:
        near = lookup(
            conditional_summary,
            "test",
            source,
            score,
            region="near_boundary",
            window_sec=10,
        )
        far = lookup(
            conditional_summary,
            "test",
            source,
            score,
            region="far_from_boundary",
            window_sec=10,
        )
        readme.append(
            f"- {label}: near AUROC {near.auroc_mean:.6f} ± {near.auroc_std:.6f} "
            f"versus far {far.auroc_mean:.6f} ± {far.auroc_std:.6f}; "
            f"near error rate {near.error_rate_mean:.6f} versus far {far.error_rate_mean:.6f}."
        )
    raw_cal = calibration_summary[calibration_summary["calibration_state"].eq("raw")].iloc[0]
    scaled_cal = calibration_summary[
        calibration_summary["calibration_state"].eq("temperature_scaled")
    ].iloc[0]
    legacy_mc = legacy_comparison[
        legacy_comparison["prediction_source"].eq("mc_dropout_t30")
    ]
    readme.extend(
        [
            "",
            "## MC convergence and calibration",
            "",
            "- T=30 is the smallest candidate that satisfies every frozen convergence criterion for all three training seeds; T=10 and T=20 do not.",
            f"- Test NLL: {raw_cal.nll_mean:.6f} ± {raw_cal.nll_std:.6f} -> {scaled_cal.nll_mean:.6f} ± {scaled_cal.nll_std:.6f}.",
            f"- Test Brier: {raw_cal.brier_mean:.6f} ± {raw_cal.brier_std:.6f} -> {scaled_cal.brier_mean:.6f} ± {scaled_cal.brier_std:.6f}.",
            f"- Test ECE-15: {raw_cal.ece_15_bins_mean:.6f} ± {raw_cal.ece_15_bins_std:.6f} -> {scaled_cal.ece_15_bins_mean:.6f} ± {scaled_cal.ece_15_bins_std:.6f}.",
            "- Accuracy and macro-F1 are unchanged by Temperature Scaling.",
            "- ECE-10/20 are legacy sensitivity records and are not part of this frozen final protocol.",
            "",
            "## Legacy audit",
            "",
            f"- Old v1 versus current MC primary metrics are not bitwise equivalent: maximum absolute AUROC difference {legacy_mc.loc[legacy_mc['metric'].eq('auroc'), 'absolute_difference'].max():.6f}; maximum AUPR difference {legacy_mc.loc[legacy_mc['metric'].eq('aupr'), 'absolute_difference'].max():.6f}.",
            "- The old unified tables are therefore preserved but not relabelled as final evidence.",
            "",
            "## Frozen conclusion",
            "",
            "Uncertainty identifies errors overall. Transition regions have higher error rates and higher uncertainty, but conditional error discrimination near transitions is only modest. AUPR comparisons across near/far regions must account for their different error prevalence.",
            "",
            "Near/far AURC is intentionally not reported because the dissertation does not currently claim selective rejection specifically within transition regions.",
            "",
            "The results are three-seed iterative-development evidence rather than a fully independent confirmatory test because the test split had been inspected earlier in the project.",
        ]
    )
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(readme) + "\n", encoding="utf-8")

    output_paths = [
        *(out_dir / filename for filename in outputs),
        authority_path,
        readme_path,
    ]
    script_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "selective_prediction_metrics.py",
        PROJECT_ROOT / "scripts" / "reproducibility_utils.py",
    ]
    manifest = {
        "schema_version": 1,
        "analysis": "final_rq1_evidence_consolidation",
        "status": "complete_and_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": config["evidence_status"],
        "operation": {
            "training_run": False,
            "model_inference_run": False,
            "new_uncertainty_method_added": False,
            "test_used_for_selection": False,
            "description": "Read-only metric recomputation and evidence consolidation from frozen existing outputs."
        },
        "configuration": source_record(config_path, "final_evidence_configuration"),
        "script": source_record(Path(__file__).resolve(), "final_evidence_script"),
        "code_dependencies": [source_record(path, "code_dependency") for path in script_paths[1:]],
        "authoritative_inputs": sources,
        "protocol": {
            "training_seeds": list(SEEDS),
            "splits": list(SPLITS),
            "primary_scores": PRIMARY_SCORES,
            "overall_metrics": ["auroc", "aupr", "exact_aurc"],
            "conditional_metrics": ["auroc", "aupr"],
            "conditional_windows_seconds": list(WINDOWS),
            "near_far_aurc_included": False,
            "boundary_proximity_metrics": ["auroc", "aupr"],
            "calibration_ece_bins": 15,
            "ece_10_20_final_claim": False,
            "aggregation": "mean and sample standard deviation across training seeds; ddof=1",
            "deterministic_and_mc_pairing": "not strictly paired because MC uses MC-averaged predictions and may have a different error set",
            "test_exposure": "Test had already been inspected during iterative development; final evidence is not confirmatory."
        },
        "legacy_audit": {
            "comparison_rows": len(legacy_comparison),
            "old_mc_max_absolute_auroc_difference": float(
                legacy_mc.loc[legacy_mc["metric"].eq("auroc"), "absolute_difference"].max()
            ),
            "old_mc_max_absolute_aupr_difference": float(
                legacy_mc.loc[legacy_mc["metric"].eq("aupr"), "absolute_difference"].max()
            ),
            "old_outputs_reclassified_as_provenance_only": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "outputs": [source_record(path, "final_evidence_output") for path in output_paths],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved final frozen RQ1 evidence to: {out_dir}")


if __name__ == "__main__":
    main()
