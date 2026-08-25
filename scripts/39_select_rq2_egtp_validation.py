"""Evaluate and freeze the new RQ2 EGTP family using validation only."""

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

from egtp_transition_policy import apply_egtp
from reproducibility_utils import ensure_fresh_output_dir, sha256_file
from operating_point_selection import annotate_feasibility, select_variant_operating_points
from rq2_stability_metrics import evaluate_videos
from temperature_scaling_logits import (
    fit_temperature_from_logits,
    load_raw_logit_split,
    softmax_logits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "rq2_egtp_validation_protocol_v1.json"
BASELINE_RUNS = {
    seed: PROJECT_ROOT / "outputs" / f"v2_lstm_online_resnet18_seed{seed:02d}"
    for seed in (0, 1, 2)
}
TEC_RUNS = {
    seed: PROJECT_ROOT / "outputs" / f"rq2_tec_lstm_seed{seed:02d}_v1"
    for seed in (0, 1, 2)
}


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_id") != "rq2_egtp_validation_protocol_v1":
        raise ValueError("unexpected EGTP validation protocol")
    if payload.get("status") != "frozen_before_new_validation_metric_evaluation":
        raise ValueError("EGTP validation protocol is not frozen")
    if payload["data"].get("validation_only_until_final_freeze") is not True:
        raise ValueError("validation-only protection is missing")
    return payload


def split_arrays(
    frame: pd.DataFrame,
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    videos: dict[str, dict[str, np.ndarray]] = {}
    for video_id, group in frame.groupby("video_id", sort=True):
        indices = group.index.to_numpy(dtype=int)
        videos[str(video_id).zfill(2)] = {
            "logits": logits[indices],
            "truth": labels[indices].astype(int) + 1,
            "times": group["t_sec"].to_numpy(dtype=int),
        }
    return videos


def load_tec_validation(seed: int, expected_ids: list[str]) -> tuple[
    dict[str, dict[str, np.ndarray]],
    list[dict[str, Any]],
]:
    run_dir = TEC_RUNS[seed]
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["data"].get("test_files_accessed") is not False:
        raise ValueError("TEC run accessed test data")
    videos: dict[str, dict[str, np.ndarray]] = {}
    sources: list[dict[str, Any]] = [
        {
            "role": "tec_training_manifest",
            "training_seed": seed,
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        }
    ]
    for video_id in expected_ids:
        path = run_dir / "validation_predictions" / f"val_video_{video_id}_predictions.npz"
        with np.load(path, allow_pickle=False) as payload:
            observed_id = str(payload["video_id"].item()).zfill(2)
            logits = np.asarray(payload["logits"], dtype=np.float64)
            labels = np.asarray(payload["true_label_idx"], dtype=np.int64)
        if observed_id != video_id:
            raise ValueError("TEC validation video id mismatch")
        videos[video_id] = {
            "logits": logits,
            "truth": labels + 1,
            "times": np.arange(len(labels), dtype=int),
        }
        sources.append(
            {
                "role": "tec_validation_logits",
                "training_seed": seed,
                "video_id": video_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return videos, sources


def causal_persistence(raw: np.ndarray, min_run: int = 5) -> np.ndarray:
    raw = np.asarray(raw, dtype=int)
    output = np.empty_like(raw)
    current = int(raw[0])
    candidate: int | None = None
    count = 0
    output[0] = current
    for index in range(1, len(raw)):
        value = int(raw[index])
        if value == current:
            candidate = None
            count = 0
        else:
            if value == candidate:
                count += 1
            else:
                candidate = value
                count = 1
            if count >= min_run:
                current = value
                candidate = None
                count = 0
        output[index] = current
    return output


def predictions_for(
    videos: dict[str, dict[str, np.ndarray]],
    *,
    temperature: float,
    k: float | None,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for video_id, video in videos.items():
        probabilities = softmax_logits(video["logits"], temperature)
        if k is None:
            predictions[video_id] = probabilities.argmax(axis=1) + 1
        else:
            predictions[video_id] = apply_egtp(
                probabilities,
                k,
                epsilon=1e-8,
                initial_std=1.0,
                std_floor=1e-6,
                dynamic_normalisation=True,
            ).predictions
    return predictions


def evaluate_method(
    seed: int,
    method_id: str,
    variant_id: str,
    model_family: str,
    probability_source: str,
    k: float | None,
    videos: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    aggregate, per_video, events = evaluate_videos(videos, predictions)
    metadata = {
        "training_seed": seed,
        "method_id": method_id,
        "variant_id": variant_id,
        "model_family": model_family,
        "probability_source": probability_source,
        "A": np.nan if k is None else float(k),
        "k": np.nan if k is None else float(k),
    }
    aggregate = {**metadata, **aggregate}
    for key, value in reversed(list(metadata.items())):
        per_video.insert(0, key, value)
        events.insert(0, key, value)
    return aggregate, per_video, events


def mean_std(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metadata = set(group_columns) | {"training_seed"}
    metric_columns = [
        column
        for column in table.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(table[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in table.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_training_seeds"] = int(group["training_seed"].nunique())
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def point_metrics(
    candidate_metrics: pd.DataFrame,
    selection: dict[str, Any],
    point_type: str,
) -> dict[str, float]:
    point = selection["points"][point_type]
    table = candidate_metrics[
        candidate_metrics["variant_id"].eq(selection["variant_id"])
        & np.isclose(candidate_metrics["A"], float(point["A"]), atol=1e-12, rtol=0.0)
    ]
    return {
        "boundary_f1_tol10_mean": float(table["boundary_f1_tol10"].mean()),
        "macro_f1_mean": float(table["macro_f1"].mean()),
    }


def select_final_variant(
    candidate_metrics: pd.DataFrame,
    selections: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    strict = [
        (variant, selection)
        for variant, selection in selections.items()
        if selection["has_strict_set"]
    ]
    if strict:
        layer = "strict"
        point_type = "A_strict"
        eligible = strict
    else:
        basic = [
            (variant, selection)
            for variant, selection in selections.items()
            if selection["has_basic_set"]
        ]
        if not basic:
            return {
                "selection_status": "constraint_failure",
                "selected_variant": None,
                "selected_point_type": None,
                "selected_k": None,
            }
        layer = "basic"
        point_type = "A_F1"
        eligible = basic
    rows = []
    for variant, selection in eligible:
        metrics = point_metrics(candidate_metrics, selection, point_type)
        rows.append(
            {
                "variant_id": variant,
                "point_type": point_type,
                "k": float(selection["points"][point_type]["A"]),
                **metrics,
                "uncalibrated_preference": int(variant.endswith("_uncalibrated")),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "boundary_f1_tol10_mean",
            "macro_f1_mean",
            "uncalibrated_preference",
            "k",
        ],
        ascending=[False, False, False, True],
    )
    selected = ranking.iloc[0]
    return {
        "selection_status": f"{layer}_selected",
        "feasibility_layer": layer,
        "selected_variant": str(selected["variant_id"]),
        "selected_point_type": point_type,
        "selected_k": float(selected["k"]),
        "validation_boundary_f1_tol10_mean": float(
            selected["boundary_f1_tol10_mean"]
        ),
        "validation_macro_f1_mean": float(selected["macro_f1_mean"]),
        "ranking": ranking.drop(columns=["uncalibrated_preference"]).to_dict(
            orient="records"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rq2_egtp_validation_selection_v1",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    out_dir = ensure_fresh_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(value) for value in protocol["data"]["training_seeds"]]
    validation_ids = [
        str(value).zfill(2) for value in protocol["data"]["validation_video_ids"]
    ]
    k_candidates = [float(value) for value in protocol["egtp"]["candidate_k"]]
    temperatures: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_per_video: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []

    variants = {
        "baseline_uncalibrated": ("baseline", False),
        "baseline_calibrated": ("baseline", True),
        "tec_uncalibrated": ("tec", False),
        "tec_calibrated": ("tec", True),
    }
    raw_reference_rows: list[dict[str, Any]] = []

    for seed in seeds:
        frame, logits, labels, sources = load_raw_logit_split(
            seed,
            BASELINE_RUNS[seed],
            "val",
            validation_ids,
        )
        baseline_videos = split_arrays(frame.reset_index(drop=True), logits, labels)
        tec_videos, tec_sources = load_tec_validation(seed, validation_ids)
        source_records.extend(sources)
        source_records.extend(tec_sources)

        baseline_temperature = fit_temperature_from_logits(logits, labels)
        tec_logits = np.concatenate(
            [tec_videos[video_id]["logits"] for video_id in validation_ids],
            axis=0,
        )
        tec_labels = np.concatenate(
            [tec_videos[video_id]["truth"] - 1 for video_id in validation_ids],
            axis=0,
        )
        tec_temperature = fit_temperature_from_logits(tec_logits, tec_labels)
        temperatures.extend(
            [
                {
                    "training_seed": seed,
                    "model_family": "baseline",
                    **baseline_temperature,
                },
                {
                    "training_seed": seed,
                    "model_family": "tec",
                    **tec_temperature,
                },
            ]
        )

        raw_predictions = predictions_for(
            baseline_videos,
            temperature=1.0,
            k=None,
        )
        aggregate, per_video, events = evaluate_method(
            seed,
            "baseline_raw",
            "reference",
            "baseline",
            "uncalibrated",
            None,
            baseline_videos,
            raw_predictions,
        )
        raw_reference_rows.append(aggregate)
        all_metrics.append(aggregate)
        all_per_video.append(per_video)
        all_events.append(events)

        persistence_predictions = {
            video_id: causal_persistence(prediction, 5)
            for video_id, prediction in raw_predictions.items()
        }
        aggregate, per_video, events = evaluate_method(
            seed,
            "persistence_5",
            "comparator",
            "baseline",
            "uncalibrated",
            None,
            baseline_videos,
            persistence_predictions,
        )
        all_metrics.append(aggregate)
        all_per_video.append(per_video)
        all_events.append(events)

        for variant_id, (model_family, calibrated) in variants.items():
            videos = baseline_videos if model_family == "baseline" else tec_videos
            temperature = (
                float(
                    baseline_temperature["temperature"]
                    if model_family == "baseline"
                    else tec_temperature["temperature"]
                )
                if calibrated
                else 1.0
            )
            probability_source = "temperature_scaled" if calibrated else "uncalibrated"
            raw_model_predictions = predictions_for(
                videos,
                temperature=temperature,
                k=None,
            )
            aggregate, per_video, events = evaluate_method(
                seed,
                f"{variant_id}_raw",
                f"{variant_id}_raw",
                model_family,
                probability_source,
                None,
                videos,
                raw_model_predictions,
            )
            all_metrics.append(aggregate)
            all_per_video.append(per_video)
            all_events.append(events)
            for k in k_candidates:
                method_predictions = predictions_for(
                    videos,
                    temperature=temperature,
                    k=k,
                )
                aggregate, per_video, events = evaluate_method(
                    seed,
                    f"{variant_id}_k{k:.1f}",
                    variant_id,
                    model_family,
                    probability_source,
                    k,
                    videos,
                    method_predictions,
                )
                all_metrics.append(aggregate)
                all_per_video.append(per_video)
                all_events.append(events)

    metrics = pd.DataFrame(all_metrics)
    per_video = pd.concat(all_per_video, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True)
    raw_reference = pd.DataFrame(raw_reference_rows)
    candidate_metrics = metrics[metrics["variant_id"].isin(variants)].copy()
    annotated, feasibility = annotate_feasibility(
        candidate_metrics,
        raw_reference,
        expected_seeds=tuple(seeds),
    )
    selections: dict[str, dict[str, Any]] = {}
    for variant_id in variants:
        selections[variant_id] = select_variant_operating_points(
            feasibility[feasibility["variant_id"].eq(variant_id)].copy()
        )
    final = select_final_variant(candidate_metrics, selections)
    summary = mean_std(
        metrics,
        [
            "method_id",
            "variant_id",
            "model_family",
            "probability_source",
            "A",
            "k",
        ],
    )

    pd.DataFrame(temperatures).to_csv(
        out_dir / "temperature_per_seed_model.csv",
        index=False,
    )
    metrics.to_csv(out_dir / "validation_metrics_per_seed.csv", index=False)
    summary.to_csv(out_dir / "validation_metrics_mean_std.csv", index=False)
    per_video.to_csv(out_dir / "validation_metrics_by_seed_video.csv", index=False)
    events.to_csv(out_dir / "validation_boundary_events.csv", index=False)
    annotated.to_csv(out_dir / "validation_feasibility_per_seed.csv", index=False)
    feasibility.to_csv(out_dir / "validation_feasibility_summary.csv", index=False)
    paper_fixed = summary[
        summary["variant_id"].isin(variants)
        & np.isclose(summary["k"], 0.8, atol=1e-12, rtol=0.0)
    ]
    paper_fixed.to_csv(out_dir / "paper_fixed_k_0p8_summary.csv", index=False)

    selection_record = {
        "schema_version": 1,
        "config_id": "rq2_egtp_validation_selection_v1",
        "status": "frozen_after_validation_before_new_test_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "selection_split": "val",
        "validation_video_ids": validation_ids,
        "training_seeds": seeds,
        "variant_selections": selections,
        "final_selection": final,
        "paper_fixed_k": 0.8,
        "test_files_accessed": False,
        "test_metrics_used": False,
        "change_control": (
            "This record is immutable before test. New test metrics cannot alter "
            "the selected family, probability source, or k."
        ),
    }
    selection_path = (
        PROJECT_ROOT / "configs" / "rq2_egtp_validation_selection_v1.json"
    )
    if selection_path.exists():
        raise FileExistsError(selection_path)
    selection_path.write_text(
        json.dumps(selection_record, indent=2),
        encoding="utf-8",
    )

    output_files = sorted(path for path in out_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "analysis": "rq2_egtp_validation_selection_and_single_tec_optimisation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": {
            "split": "val",
            "test_files_accessed": False,
            "test_metrics_computed": False,
            "model_training_in_this_script": False,
            "saved_validation_logits_used": True,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "protocol": selection_record["protocol"],
        "selection_record": {
            "path": str(selection_path.resolve()),
            "sha256": sha256_file(selection_path),
        },
        "inputs": source_records,
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_files
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2))
    print(f"Selection record: {selection_path}")


if __name__ == "__main__":
    main()
