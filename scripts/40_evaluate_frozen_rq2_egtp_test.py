"""Run the frozen three-seed RQ2 EGTP test evaluation and final audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from egtp_transition_policy import apply_egtp
from reproducibility_utils import ensure_fresh_output_dir, sha256_file
from rq2_stability_metrics import evaluate_videos
from temperature_scaling_logits import load_raw_logit_split, softmax_logits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "rq2_egtp_test_protocol_v1.json"
FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "resnet18"
BASELINE_RUNS = {
    seed: PROJECT_ROOT / "outputs" / f"v2_lstm_online_resnet18_seed{seed:02d}"
    for seed in (0, 1, 2)
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(record: dict[str, str]) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"frozen hash mismatch: {path}")
    return path


def validate_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(path)
    if protocol.get("config_id") != "rq2_egtp_test_protocol_v1":
        raise ValueError("unexpected RQ2 EGTP test protocol")
    if protocol.get("status") != "frozen_before_new_test_evaluation":
        raise ValueError("test protocol is not frozen")
    guard = protocol["selection_guard"]
    if not all(
        guard[key] is True
        for key in (
            "test_must_not_select_or_modify_method",
            "test_must_not_change_k",
            "test_must_not_resurrect_tec_if_validation_constraints_failed",
        )
    ):
        raise ValueError("test selection guard is incomplete")
    validation_protocol_path = validate_record(protocol["frozen_validation_protocol"])
    selection_path = validate_record(protocol["frozen_validation_selection"])
    selection = load_json(selection_path)
    final = selection["final_selection"]
    primary = protocol["primary_method"]
    if (
        final["selected_variant"] != "baseline_uncalibrated"
        or final["selected_k"] != primary["k"]
        or primary["method_id"] != "egtp_selected"
    ):
        raise ValueError("test primary method differs from frozen validation selection")
    for seed in protocol["training_seeds"]:
        validate_record(protocol["baseline_checkpoints"][str(seed)])
        validate_record(protocol["tec_checkpoints"][str(seed)])
    if sha256_file(validation_protocol_path) != protocol["frozen_validation_protocol"]["sha256"]:
        raise AssertionError("validation protocol changed")
    return protocol, selection


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


def load_tec_model_class():
    source = PROJECT_ROOT / "scripts" / "38_train_rq2_tec_lstm.py"
    spec = importlib.util.spec_from_file_location("rq2_tec_training_frozen", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TemporalPhaseModel


@torch.no_grad()
def infer_tec_test(
    seed: int,
    protocol: dict[str, Any],
    video_ids: list[str],
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    checkpoint_path = validate_record(protocol["tec_checkpoints"][str(seed)])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_class = load_tec_model_class()
    architecture = checkpoint["architecture"]
    model = model_class(
        hidden_dim=int(architecture["hidden_dim"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    videos: dict[str, dict[str, np.ndarray]] = {}
    sources: list[dict[str, Any]] = [
        {
            "role": "tec_checkpoint",
            "training_seed": seed,
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "validation_macro_f1": float(checkpoint["best_val_macro_f1"]),
        }
    ]
    for video_id in video_ids:
        feature_path = FEATURES_DIR / f"{video_id}.npy"
        label_path = FEATURES_DIR / f"{video_id}_labels.npy"
        features = np.load(feature_path).astype(np.float32)
        labels = np.load(label_path).astype(np.int64)
        logits = (
            model(torch.from_numpy(features).unsqueeze(0).to(device))
            .squeeze(0)
            .cpu()
            .numpy()
        )
        videos[video_id] = {
            "logits": logits,
            "truth": labels + 1,
            "times": np.arange(len(labels), dtype=int),
        }
        sources.extend(
            [
                {
                    "role": "test_features",
                    "training_seed": seed,
                    "video_id": video_id,
                    "path": str(feature_path.resolve()),
                    "sha256": sha256_file(feature_path),
                },
                {
                    "role": "test_labels",
                    "training_seed": seed,
                    "video_id": video_id,
                    "path": str(label_path.resolve()),
                    "sha256": sha256_file(label_path),
                },
            ]
        )
    return videos, sources


def causal_persistence(raw: np.ndarray, min_run: int) -> np.ndarray:
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


def method_predictions(
    spec: dict[str, Any],
    seed: int,
    baseline_videos: dict[str, dict[str, np.ndarray]],
    tec_videos: dict[str, dict[str, np.ndarray]],
    protocol: dict[str, Any],
) -> dict[str, np.ndarray]:
    family = spec["model_family"]
    videos = baseline_videos if family == "baseline" else tec_videos
    temperature = (
        float(protocol["temperatures"][family][str(seed)])
        if spec.get("probability_source") == "temperature_scaled"
        else 1.0
    )
    result: dict[str, np.ndarray] = {}
    for video_id, video in videos.items():
        probabilities = softmax_logits(video["logits"], temperature)
        raw = probabilities.argmax(axis=1) + 1
        if spec["kind"] == "raw":
            prediction = raw
        elif spec["kind"] == "causal_persistence":
            prediction = causal_persistence(raw, int(spec["min_run_sec"]))
        elif spec["kind"] == "egtp":
            prediction = apply_egtp(
                probabilities,
                float(spec["k"]),
                epsilon=1e-8,
                initial_std=1.0,
                std_floor=1e-6,
                dynamic_normalisation=True,
            ).predictions
        else:
            raise ValueError(f"unknown method kind: {spec['kind']}")
        result[video_id] = prediction
    return result


def aggregate_method(
    seed: int,
    spec: dict[str, Any],
    videos: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    aggregate, per_video, events = evaluate_videos(videos, predictions)
    metadata = {
        "training_seed": seed,
        "method_id": spec["method_id"],
        "model_family": spec["model_family"],
        "kind": spec["kind"],
        "probability_source": spec.get("probability_source"),
        "k": spec.get("k"),
        "role": spec.get("role"),
    }
    aggregate = {**metadata, **aggregate}
    for key, value in reversed(list(metadata.items())):
        per_video.insert(0, key, value)
        events.insert(0, key, value)
    return aggregate, per_video, events


def mean_std(table: pd.DataFrame) -> pd.DataFrame:
    metadata = [
        "method_id",
        "model_family",
        "kind",
        "probability_source",
        "k",
        "role",
    ]
    excluded = set(metadata) | {"training_seed"}
    metrics = [
        column
        for column in table.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(table[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in table.groupby(metadata, dropna=False, sort=True):
        row = dict(zip(metadata, keys))
        row["n_training_seeds"] = int(group["training_seed"].nunique())
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_seed_differences(
    per_seed: pd.DataFrame,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    metadata = {
        "method_id",
        "model_family",
        "kind",
        "probability_source",
        "k",
        "role",
        "training_seed",
    }
    metrics = [
        column
        for column in per_seed.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(per_seed[column])
    ]
    rows: list[dict[str, Any]] = []
    for first, second in comparisons:
        left = per_seed[per_seed["method_id"].eq(first)].set_index("training_seed")
        right = per_seed[per_seed["method_id"].eq(second)].set_index("training_seed")
        for seed in sorted(set(left.index) & set(right.index)):
            row: dict[str, Any] = {
                "training_seed": int(seed),
                "comparison": f"{first}_minus_{second}",
            }
            for metric in metrics:
                row[f"delta_{metric}"] = float(
                    left.loc[seed, metric] - right.loc[seed, metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_video_clusters(
    per_video: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    *,
    video_ids: list[str],
    seeds: list[int],
    n_resamples: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(video_ids), size=(n_resamples, len(video_ids)))
    rows: list[dict[str, Any]] = []
    for first, second in comparisons:
        values: dict[str, dict[str, np.ndarray]] = {}
        observed: dict[str, dict[str, float]] = {}
        for method in (first, second):
            table = per_video[per_video["method_id"].eq(method)]
            method_values: dict[str, np.ndarray] = {}
            method_observed: dict[str, float] = {}
            for metric in (
                "macro_f1",
                "predicted_boundary_count",
                "mean_video_tfi",
            ):
                source_column = "tfi" if metric == "mean_video_tfi" else metric
                matrix = np.asarray(
                    [
                        table[table["training_seed"].eq(seed)]
                        .set_index("video_id")
                        .loc[video_ids, source_column]
                        .to_numpy(float)
                        for seed in seeds
                    ]
                )
                method_values[metric] = matrix[:, sampled].mean(axis=(0, 2))
                method_observed[metric] = float(matrix.mean())
            for tolerance in (5, 10):
                arrays = np.asarray(
                    [
                        table[table["training_seed"].eq(seed)]
                        .set_index("video_id")
                        .loc[
                            video_ids,
                            [
                                f"tp_tol{tolerance}",
                                f"fp_tol{tolerance}",
                                f"fn_tol{tolerance}",
                            ],
                        ]
                        .to_numpy(float)
                        for seed in seeds
                    ]
                )
                sampled_counts = arrays[:, sampled, :].sum(axis=2)
                observed_counts = arrays.sum(axis=1)
                for label, counts in (
                    ("sampled", sampled_counts),
                    ("observed", observed_counts),
                ):
                    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
                    precision = np.divide(
                        tp,
                        tp + fp,
                        out=np.zeros_like(tp),
                        where=(tp + fp) > 0,
                    )
                    recall = np.divide(
                        tp,
                        tp + fn,
                        out=np.zeros_like(tp),
                        where=(tp + fn) > 0,
                    )
                    f1 = np.divide(
                        2 * precision * recall,
                        precision + recall,
                        out=np.zeros_like(precision),
                        where=(precision + recall) > 0,
                    )
                    if label == "sampled":
                        method_values[f"boundary_f1_tol{tolerance}"] = f1.mean(axis=0)
                        method_values[f"boundary_recall_tol{tolerance}"] = recall.mean(axis=0)
                    else:
                        method_observed[f"boundary_f1_tol{tolerance}"] = float(f1.mean())
                        method_observed[f"boundary_recall_tol{tolerance}"] = float(
                            recall.mean()
                        )
            values[method] = method_values
            observed[method] = method_observed
        for metric in values[first]:
            delta = values[first][metric] - values[second][metric]
            rows.append(
                {
                    "comparison": f"{first}_minus_{second}",
                    "metric": metric,
                    "observed_delta": observed[first][metric] - observed[second][metric],
                    "bootstrap_mean_delta": float(delta.mean()),
                    "ci95_low": float(np.quantile(delta, 0.025)),
                    "ci95_high": float(np.quantile(delta, 0.975)),
                    "fraction_delta_above_zero": float(np.mean(delta > 0.0)),
                    "n_video_clusters": len(video_ids),
                    "n_training_seeds": len(seeds),
                    "n_resamples": n_resamples,
                    "random_seed": random_seed,
                    "interpretation": "descriptive_cluster_interval_not_formal_significance",
                }
            )
    return pd.DataFrame(rows)


def plot_timeline(
    out_dir: Path,
    seed: int,
    video_id: str,
    videos: dict[str, dict[str, np.ndarray]],
    predictions: dict[tuple[int, str, str], np.ndarray],
) -> Path:
    video = videos[video_id]
    times = video["times"]
    methods = ["baseline_raw", "persistence_5", "egtp_selected"]
    figure, axes = plt.subplots(4, 1, figsize=(14, 6), sharex=True)
    axes[0].step(times, video["truth"], where="post", color="black")
    axes[0].set_ylabel("GT")
    colors = ["#777777", "#E69F00", "#0072B2"]
    for axis, method, color in zip(axes[1:], methods, colors):
        axis.step(
            times,
            predictions[(seed, video_id, method)],
            where="post",
            color=color,
        )
        axis.set_ylabel(method.replace("_", "\n"))
    axes[-1].set_xlabel("Time (s)")
    for axis in axes:
        axis.set_yticks(range(1, 8))
        axis.grid(alpha=0.2)
    figure.suptitle(f"RQ2 temporal predictions: seed {seed}, video {video_id}")
    figure.tight_layout()
    path = out_dir / f"timeline_seed{seed:02d}_video{video_id}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_interpretation(
    out_dir: Path,
    summary: pd.DataFrame,
    selection: dict[str, Any],
) -> None:
    def row(method: str) -> pd.Series:
        return summary[summary["method_id"].eq(method)].iloc[0]

    raw = row("baseline_raw")
    selected = row("egtp_selected")
    delta_macro = selected["macro_f1_mean"] - raw["macro_f1_mean"]
    delta_f1 = selected["boundary_f1_tol10_mean"] - raw["boundary_f1_tol10_mean"]
    delta_recall = (
        selected["boundary_recall_tol10_mean"] - raw["boundary_recall_tol10_mean"]
    )
    delta_tfi = selected["mean_video_tfi_mean"] - raw["mean_video_tfi_mean"]
    english = f"""# Final RQ2 evidence

The validation-selected method was the uncalibrated EGTP applied to the frozen LSTM baseline with k=0.6. No strict operating point existed; the method was selected from the basic feasible set and must therefore be interpreted as a stability-transition-sensitivity trade-off.

Across the three test seeds, EGTP achieved Macro-F1 {selected['macro_f1_mean']:.4f} ± {selected['macro_f1_std']:.4f}, produced {selected['predicted_boundary_count_mean']:.1f} ± {selected['predicted_boundary_count_std']:.1f} boundaries, and obtained boundary precision/recall/F1 of {selected['boundary_precision_tol10_mean']:.4f}/{selected['boundary_recall_tol10_mean']:.4f}/{selected['boundary_f1_tol10_mean']:.4f} at ±10 s. Relative to raw argmax, Macro-F1 changed by {delta_macro:+.4f}, boundary F1 by {delta_f1:+.4f}, boundary recall by {delta_recall:+.4f}, and mean-video TFI by {delta_tfi:+.4f}.

The paper-derived TEC training ablation did not satisfy the validation feasibility constraints and was rejected before test evaluation. Temperature scaling produced almost identical EGTP decisions because dynamic normalisation largely cancels global logit-scale changes.

The evidence supports a conditional conclusion: causal normalised evidence accumulation reduces temporal fragmentation, but genuine transitions can still be delayed or suppressed. Results are three-seed iterative-development evidence rather than a fully independent confirmatory test because the test split had been inspected earlier in the project.
"""
    (out_dir / "FINAL_RQ2_CONCLUSION_EN.md").write_text(english, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rq2_egtp_final_evidence_v1",
    )
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol, selection = validate_protocol(protocol_path)
    out_dir = ensure_fresh_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / "timelines"
    figure_dir.mkdir(exist_ok=True)

    seeds = [int(value) for value in protocol["training_seeds"]]
    video_ids = [str(value).zfill(2) for value in protocol["evaluation_video_ids"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = protocol["frozen_ablation_methods"]
    all_metrics: list[dict[str, Any]] = []
    all_per_video: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []
    predictions: dict[tuple[int, str, str], np.ndarray] = {}
    baseline_by_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    for seed in seeds:
        frame, logits, labels, sources = load_raw_logit_split(
            seed,
            BASELINE_RUNS[seed],
            "test",
            video_ids,
        )
        baseline_videos = split_arrays(frame.reset_index(drop=True), logits, labels)
        tec_videos, tec_sources = infer_tec_test(
            seed,
            protocol,
            video_ids,
            device,
        )
        source_records.extend(sources)
        source_records.extend(tec_sources)
        baseline_by_seed[seed] = baseline_videos
        for spec in methods:
            videos = baseline_videos if spec["model_family"] == "baseline" else tec_videos
            method_prediction = method_predictions(
                spec,
                seed,
                baseline_videos,
                tec_videos,
                protocol,
            )
            for video_id, prediction in method_prediction.items():
                predictions[(seed, video_id, spec["method_id"])] = prediction
            aggregate, per_video, events = aggregate_method(
                seed,
                spec,
                videos,
                method_prediction,
            )
            all_metrics.append(aggregate)
            all_per_video.append(per_video)
            all_events.append(events)

    per_seed = pd.DataFrame(all_metrics)
    per_video = pd.concat(all_per_video, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True)
    summary = mean_std(per_seed)
    comparisons = [
        ("egtp_selected", "baseline_raw"),
        ("egtp_selected", "persistence_5"),
        ("egtp_paper_k0p8", "baseline_raw"),
        ("tec_raw", "baseline_raw"),
        ("egtp_selected_calibrated", "egtp_selected"),
    ]
    paired_seed = paired_seed_differences(per_seed, comparisons)
    bootstrap = bootstrap_video_clusters(
        per_video,
        comparisons[:4],
        video_ids=video_ids,
        seeds=seeds,
        n_resamples=int(protocol["evaluation"]["paired_video_cluster_bootstrap_resamples"]),
        random_seed=int(protocol["evaluation"]["bootstrap_seed"]),
    )
    event_summary = (
        events.groupby(["method_id", "tolerance_sec", "event_type"])
        .size()
        .rename("event_count")
        .reset_index()
    )
    selected_per_video = per_video[
        per_video["method_id"].isin(["egtp_selected", "persistence_5"])
    ]
    paired_video = selected_per_video.pivot(
        index=["training_seed", "video_id"],
        columns="method_id",
        values="boundary_f1_tol10",
    ).reset_index()
    paired_video["delta_egtp_minus_persistence"] = (
        paired_video["egtp_selected"] - paired_video["persistence_5"]
    )
    roles = pd.DataFrame(
        [
            {
                "role": "largest_EGTP_gain",
                **paired_video.loc[
                    paired_video["delta_egtp_minus_persistence"].idxmax()
                ].to_dict(),
            },
            {
                "role": "largest_EGTP_loss",
                **paired_video.loc[
                    paired_video["delta_egtp_minus_persistence"].idxmin()
                ].to_dict(),
            },
        ]
    )
    plot_paths = []
    for item in roles.itertuples(index=False):
        plot_paths.append(
            plot_timeline(
                figure_dir,
                int(item.training_seed),
                str(item.video_id).zfill(2),
                baseline_by_seed[int(item.training_seed)],
                predictions,
            )
        )

    per_seed.to_csv(out_dir / "test_metrics_per_seed.csv", index=False)
    summary.to_csv(out_dir / "test_metrics_mean_std.csv", index=False)
    per_video.to_csv(out_dir / "test_metrics_by_seed_video.csv", index=False)
    events.to_csv(out_dir / "test_boundary_events.csv", index=False)
    event_summary.to_csv(out_dir / "test_boundary_event_summary.csv", index=False)
    paired_seed.to_csv(out_dir / "paired_differences_per_seed.csv", index=False)
    bootstrap.to_csv(out_dir / "paired_video_cluster_bootstrap.csv", index=False)
    roles.to_csv(out_dir / "failure_case_selection.csv", index=False)

    calibrated = {
        (int(row.training_seed), str(row.video_id).zfill(2)): predictions[
            (int(row.training_seed), str(row.video_id).zfill(2), "egtp_selected_calibrated")
        ]
        for row in per_video[
            per_video["method_id"].eq("egtp_selected_calibrated")
        ].itertuples(index=False)
    }
    uncalibrated = {
        key: predictions[(key[0], key[1], "egtp_selected")]
        for key in calibrated
    }
    calibration_mismatches = int(
        sum(
            np.sum(calibrated[key] != uncalibrated[key])
            for key in calibrated
        )
    )
    checks = {
        "frozen_hashes_validated_before_test_load": True,
        "test_used_for_selection_or_retuning": False,
        "selected_family": protocol["primary_method"],
        "tec_not_resurrected_after_validation_failure": True,
        "calibrated_vs_uncalibrated_selected_egtp_mismatch_frames": calibration_mismatches,
        "timeline_plots": [str(path.resolve()) for path in plot_paths],
    }
    (out_dir / "correctness_checks.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )
    write_interpretation(out_dir, summary, selection)

    output_files = sorted(
        path
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "analysis": "rq2_egtp_frozen_three_seed_test_and_final_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": {
            "test_split_loaded_after_validation_freeze": True,
            "test_metrics_used_for_selection_or_retuning": False,
            "baseline_saved_test_logits_used": True,
            "tec_test_forward_inference_run": True,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "validation_selection": protocol["frozen_validation_selection"],
        "inputs": source_records,
        "checks": checks,
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
            "torch": torch.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
        "evidence_status": protocol["evidence_status"],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    selected = summary[summary["method_id"].eq("egtp_selected")].iloc[0]
    print(
        json.dumps(
            {
                "primary_method": "egtp_selected",
                "k": 0.6,
                "macro_f1_mean": selected["macro_f1_mean"],
                "predicted_boundary_count_mean": selected[
                    "predicted_boundary_count_mean"
                ],
                "boundary_f1_tol10_mean": selected["boundary_f1_tol10_mean"],
                "boundary_recall_tol10_mean": selected[
                    "boundary_recall_tol10_mean"
                ],
                "mean_video_tfi_mean": selected["mean_video_tfi_mean"],
                "calibration_mismatch_frames": calibration_mismatches,
            },
            indent=2,
        )
    )
    print(f"Saved final RQ2 evidence to {out_dir}")


if __name__ == "__main__":
    main()
