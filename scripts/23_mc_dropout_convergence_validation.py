"""Validation-only convergence analysis for MC Dropout pass count T.

For each trained baseline seed, the script generates 50 stochastic forward
passes once and evaluates nested prefixes T={10,20,30,50}.  Using nested
prefixes removes avoidable Monte Carlo differences between T settings.  The
test split is never loaded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from reproducibility_utils import (
    ensure_fresh_output_dir,
    set_inference_seed,
    sha256_file,
)
from selective_prediction_metrics import (
    empirical_risk_coverage,
    exact_aurc,
    key_coverage_points,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "resnet18"
DEFAULT_SEED_RUNS = {
    0: PROJECT_ROOT / "outputs" / "v2_lstm_online_resnet18_seed00",
    1: PROJECT_ROOT / "outputs" / "v2_lstm_online_resnet18_seed01",
    2: PROJECT_ROOT / "outputs" / "v2_lstm_online_resnet18_seed02",
}
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "mc_dropout_convergence_validation_v3"
VALIDATION_VIDEO_IDS = ("11", "12", "13", "14")
T_VALUES = (10, 20, 30, 50)
INFERENCE_SEED = 0

# Practical stability tolerances for choosing the smallest T.  These are
# project-specific operating criteria rather than universal statistical laws.
# Every criterion must hold for every training seed relative to T=50.
STABILITY_CRITERIA = {
    "prediction_disagreement_rate_max": 0.01,
    "mc_entropy_spearman_min": 0.995,
    "mc_mutual_information_spearman_min": 0.98,
    "error_auroc_abs_difference_max": 0.01,
    "error_aupr_abs_difference_max": 0.01,
    "aurc_abs_difference_max": 0.005,
}


def load_numbered_script(module_name: str, filename: str):
    source = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MC_RUNTIME = load_numbered_script("mc_runtime_for_convergence", "08_eval_mc_dropout.py")


def parse_seed_run(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text.removeprefix("seed"))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "Use SEED=PATH, for example 0=outputs/v2_lstm_online_resnet18_seed00"
        ) from error
    return seed, Path(path_text).expanduser().resolve()


@torch.no_grad()
def stochastic_probability_samples(
    model,
    features: np.ndarray,
    device: torch.device,
    passes: int,
    timing_passes: tuple[int, ...],
) -> tuple[np.ndarray, dict[int, float]]:
    tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
    MC_RUNTIME.enable_dropout_only(model)
    samples = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    cumulative_seconds: dict[int, float] = {}
    for pass_index in range(1, passes + 1):
        logits = model(tensor).squeeze(0)
        samples.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if pass_index in timing_passes:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cumulative_seconds[pass_index] = time.perf_counter() - start
    return np.stack(samples, axis=0), cumulative_seconds


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=-1)


def prefix_scores(samples: np.ndarray, passes: int) -> dict[str, np.ndarray]:
    prefix = samples[:passes]
    mean_probabilities = prefix.mean(axis=0)
    predictive_entropy = entropy(mean_probabilities)
    expected_entropy = entropy(prefix).mean(axis=0)
    return {
        "mean_probabilities": mean_probabilities,
        "prediction": mean_probabilities.argmax(axis=-1),
        "mc_entropy": predictive_entropy,
        "mc_mutual_information": predictive_entropy - expected_entropy,
    }


def risk_metrics(errors: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    curve = empirical_risk_coverage(errors, scores)
    aurc = exact_aurc(curve)
    key = key_coverage_points(curve, [0.5, 0.8, 1.0]).set_index("coverage")
    return {
        "aurc": float(aurc["aurc_lower_better"]),
        "excess_aurc": float(aurc["excess_aurc"]),
        "risk_at_50": float(key.loc[0.5, "risk_error_rate"]),
        "risk_at_80": float(key.loc[0.8, "risk_error_rate"]),
        "risk_at_100": float(key.loc[1.0, "risk_error_rate"]),
    }


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = pd.Series(first).rank(method="average")
    second_rank = pd.Series(second).rank(method="average")
    return float(first_rank.corr(second_rank, method="pearson"))


def evaluate_training_seed(
    training_seed: int,
    run_dir: Path,
    features_dir: Path,
    device: torch.device,
    inference_seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    set_inference_seed(inference_seed)
    model = MC_RUNTIME.load_model(checkpoint, device)

    truth_by_video: dict[str, np.ndarray] = {}
    scores_by_t: dict[int, dict[str, list[np.ndarray]]] = {
        passes: {
            "prediction": [],
            "mc_entropy": [],
            "mc_mutual_information": [],
        }
        for passes in T_VALUES
    }
    frame_rows: list[dict] = []
    runtime_rows: list[dict] = []
    inference_seconds_by_t = {passes: 0.0 for passes in T_VALUES}
    for video_id in VALIDATION_VIDEO_IDS:
        feature_path = features_dir / f"{video_id}.npy"
        label_path = features_dir / f"{video_id}_labels.npy"
        if not feature_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing validation feature or label for video {video_id}")
        features = np.load(feature_path).astype(np.float32)
        truth = np.load(label_path).astype(np.int64)
        truth_by_video[video_id] = truth
        samples, cumulative_seconds = stochastic_probability_samples(
            model,
            features,
            device,
            max(T_VALUES),
            T_VALUES,
        )
        for passes in T_VALUES:
            inference_seconds_by_t[passes] += cumulative_seconds[passes]
            runtime_rows.append(
                {
                    "training_seed": training_seed,
                    "inference_seed": inference_seed,
                    "split": "val",
                    "video_id": video_id,
                    "n_frames": len(truth),
                    "T": passes,
                    "cumulative_inference_seconds": cumulative_seconds[passes],
                    "milliseconds_per_frame_pass": (
                        1000.0 * cumulative_seconds[passes] / (len(truth) * passes)
                    ),
                }
            )
            scores = prefix_scores(samples, passes)
            for name in scores_by_t[passes]:
                scores_by_t[passes][name].append(scores[name])
            for frame_index in range(len(truth)):
                frame_rows.append(
                    {
                        "training_seed": training_seed,
                        "inference_seed": inference_seed,
                        "split": "val",
                        "video_id": video_id,
                        "t_sec": frame_index,
                        "T": passes,
                        "true_label_idx": int(truth[frame_index]),
                        "pred_label_idx": int(scores["prediction"][frame_index]),
                        "error": int(scores["prediction"][frame_index] != truth[frame_index]),
                        "mc_entropy": float(scores["mc_entropy"][frame_index]),
                        "mc_mutual_information": float(
                            scores["mc_mutual_information"][frame_index]
                        ),
                    }
                )
        print(f"training seed {training_seed}: validation video {video_id} complete")

    y_true = np.concatenate([truth_by_video[video] for video in VALIDATION_VIDEO_IDS])
    combined = {
        passes: {
            name: np.concatenate(values)
            for name, values in score_groups.items()
        }
        for passes, score_groups in scores_by_t.items()
    }
    metric_rows: list[dict] = []
    for passes in T_VALUES:
        prediction = combined[passes]["prediction"]
        errors = (prediction != y_true).astype(int)
        row = {
            "training_seed": training_seed,
            "inference_seed": inference_seed,
            "split": "val",
            "T": passes,
            "n_frames": len(y_true),
            "inference_seconds": inference_seconds_by_t[passes],
            "milliseconds_per_frame_pass": (
                1000.0
                * inference_seconds_by_t[passes]
                / (len(y_true) * passes)
            ),
            "accuracy": accuracy_score(y_true, prediction),
            "macro_f1": f1_score(
                y_true,
                prediction,
                labels=list(range(7)),
                average="macro",
                zero_division=0,
            ),
            "prediction_error_rate": errors.mean(),
        }
        for score_name in ("mc_entropy", "mc_mutual_information"):
            score = combined[passes][score_name]
            row[f"{score_name}_error_auroc"] = roc_auc_score(errors, score)
            row[f"{score_name}_error_aupr"] = average_precision_score(errors, score)
            for metric, value in risk_metrics(errors, score).items():
                row[f"{score_name}_{metric}"] = value
        metric_rows.append(row)

    reference_t = max(T_VALUES)
    distance_rows: list[dict] = []
    reference_prediction = combined[reference_t]["prediction"]
    for passes in T_VALUES:
        row = {
            "training_seed": training_seed,
            "inference_seed": inference_seed,
            "split": "val",
            "T": passes,
            "reference_T": reference_t,
            "prediction_disagreement_rate_vs_T50": float(
                np.mean(combined[passes]["prediction"] != reference_prediction)
            ),
        }
        for score_name in ("mc_entropy", "mc_mutual_information"):
            current = combined[passes][score_name]
            reference = combined[reference_t][score_name]
            row[f"{score_name}_mae_vs_T50"] = float(np.mean(np.abs(current - reference)))
            row[f"{score_name}_max_abs_vs_T50"] = float(
                np.max(np.abs(current - reference))
            )
            row[f"{score_name}_spearman_vs_T50"] = rank_correlation(current, reference)
        distance_rows.append(row)
    return metric_rows, distance_rows, frame_rows, runtime_rows


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    identifier = ["inference_seed", "split", "T"]
    metrics = [
        column
        for column in per_seed.columns
        if column not in {"training_seed", *identifier}
        and pd.api.types.is_numeric_dtype(per_seed[column])
    ]
    rows = []
    for key, group in per_seed.groupby(identifier, sort=True):
        row = dict(zip(identifier, key))
        row["n_training_seeds"] = len(group)
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def assess_stability(
    per_seed: pd.DataFrame,
    distance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Apply fixed practical criteria against the nested T=50 reference."""
    reference = per_seed[per_seed["T"].eq(max(T_VALUES))].set_index("training_seed")
    rows = []
    for metric in per_seed.itertuples(index=False):
        training_seed = int(metric.training_seed)
        passes = int(metric.T)
        reference_row = reference.loc[training_seed]
        distance_row = distance[
            distance["training_seed"].eq(training_seed)
            & distance["T"].eq(passes)
        ].iloc[0]
        values = {
            "prediction_disagreement_rate_vs_T50": float(
                distance_row["prediction_disagreement_rate_vs_T50"]
            ),
            "mc_entropy_spearman_vs_T50": float(
                distance_row["mc_entropy_spearman_vs_T50"]
            ),
            "mc_mutual_information_spearman_vs_T50": float(
                distance_row["mc_mutual_information_spearman_vs_T50"]
            ),
            "mc_entropy_error_auroc_abs_diff_vs_T50": abs(
                float(metric.mc_entropy_error_auroc)
                - float(reference_row["mc_entropy_error_auroc"])
            ),
            "mc_entropy_error_aupr_abs_diff_vs_T50": abs(
                float(metric.mc_entropy_error_aupr)
                - float(reference_row["mc_entropy_error_aupr"])
            ),
            "mc_entropy_aurc_abs_diff_vs_T50": abs(
                float(metric.mc_entropy_aurc)
                - float(reference_row["mc_entropy_aurc"])
            ),
            "mc_mutual_information_error_auroc_abs_diff_vs_T50": abs(
                float(metric.mc_mutual_information_error_auroc)
                - float(reference_row["mc_mutual_information_error_auroc"])
            ),
            "mc_mutual_information_error_aupr_abs_diff_vs_T50": abs(
                float(metric.mc_mutual_information_error_aupr)
                - float(reference_row["mc_mutual_information_error_aupr"])
            ),
            "mc_mutual_information_aurc_abs_diff_vs_T50": abs(
                float(metric.mc_mutual_information_aurc)
                - float(reference_row["mc_mutual_information_aurc"])
            ),
        }
        checks = {
            "prediction_disagreement_pass": values[
                "prediction_disagreement_rate_vs_T50"
            ]
            <= STABILITY_CRITERIA["prediction_disagreement_rate_max"],
            "mc_entropy_ranking_pass": values["mc_entropy_spearman_vs_T50"]
            >= STABILITY_CRITERIA["mc_entropy_spearman_min"],
            "mc_mutual_information_ranking_pass": values[
                "mc_mutual_information_spearman_vs_T50"
            ]
            >= STABILITY_CRITERIA["mc_mutual_information_spearman_min"],
        }
        for score_name in ("mc_entropy", "mc_mutual_information"):
            checks[f"{score_name}_auroc_pass"] = values[
                f"{score_name}_error_auroc_abs_diff_vs_T50"
            ] <= STABILITY_CRITERIA["error_auroc_abs_difference_max"]
            checks[f"{score_name}_aupr_pass"] = values[
                f"{score_name}_error_aupr_abs_diff_vs_T50"
            ] <= STABILITY_CRITERIA["error_aupr_abs_difference_max"]
            checks[f"{score_name}_aurc_pass"] = values[
                f"{score_name}_aurc_abs_diff_vs_T50"
            ] <= STABILITY_CRITERIA["aurc_abs_difference_max"]
        failed = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "training_seed": training_seed,
                "inference_seed": int(metric.inference_seed),
                "split": metric.split,
                "T": passes,
                **values,
                **checks,
                "all_stability_criteria_pass": not failed,
                "failed_criteria": ";".join(failed),
            }
        )
    per_seed_stability = pd.DataFrame(rows)
    summary_rows = []
    for passes, group in per_seed_stability.groupby("T", sort=True):
        summary_rows.append(
            {
                "T": int(passes),
                "n_training_seeds": group["training_seed"].nunique(),
                "n_seeds_passing_all_criteria": int(
                    group["all_stability_criteria_pass"].sum()
                ),
                "all_seeds_stable": bool(
                    group["all_stability_criteria_pass"].all()
                ),
                "prediction_disagreement_max": group[
                    "prediction_disagreement_rate_vs_T50"
                ].max(),
                "mc_entropy_spearman_min": group[
                    "mc_entropy_spearman_vs_T50"
                ].min(),
                "mc_mutual_information_spearman_min": group[
                    "mc_mutual_information_spearman_vs_T50"
                ].min(),
                "error_auroc_abs_diff_max": group[
                    [
                        "mc_entropy_error_auroc_abs_diff_vs_T50",
                        "mc_mutual_information_error_auroc_abs_diff_vs_T50",
                    ]
                ].to_numpy().max(),
                "error_aupr_abs_diff_max": group[
                    [
                        "mc_entropy_error_aupr_abs_diff_vs_T50",
                        "mc_mutual_information_error_aupr_abs_diff_vs_T50",
                    ]
                ].to_numpy().max(),
                "aurc_abs_diff_max": group[
                    [
                        "mc_entropy_aurc_abs_diff_vs_T50",
                        "mc_mutual_information_aurc_abs_diff_vs_T50",
                    ]
                ].to_numpy().max(),
            }
        )
    stability_summary = pd.DataFrame(summary_rows)
    stable_candidates = stability_summary[
        stability_summary["all_seeds_stable"]
    ]["T"].astype(int)
    if stable_candidates.empty:
        raise RuntimeError("No T candidate satisfies all stability criteria")
    selected_t = int(stable_candidates.min())
    return per_seed_stability, stability_summary, selected_t


def plot_metrics(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("mc_entropy_error_auroc_mean", "MC entropy error AUROC"),
        ("mc_mutual_information_error_auroc_mean", "MC MI error AUROC"),
        ("mc_entropy_aurc_mean", "MC entropy AURC"),
        ("mc_mutual_information_aurc_mean", "MC MI AURC"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))
    for axis, (metric, label) in zip(axes.flat, metrics):
        error_metric = metric.removesuffix("_mean") + "_std"
        axis.errorbar(
            summary["T"],
            summary[metric],
            yerr=summary[error_metric],
            marker="o",
            capsize=3,
        )
        axis.set_title(label)
        axis.set_xlabel("MC Dropout passes T")
        axis.grid(alpha=0.25)
    figure.suptitle("Validation-only MC Dropout convergence (mean +/- SD over training seeds)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def plot_distance(distance: pd.DataFrame, out_path: Path) -> None:
    summary = (
        distance.groupby("T", as_index=False)
        .agg(
            prediction_disagreement=(
                "prediction_disagreement_rate_vs_T50",
                "mean",
            ),
            entropy_mae=("mc_entropy_mae_vs_T50", "mean"),
            mi_mae=("mc_mutual_information_mae_vs_T50", "mean"),
        )
        .sort_values("T")
    )
    figure, axis = plt.subplots(figsize=(8.3, 5.4))
    axis.plot(
        summary["T"],
        summary["prediction_disagreement"],
        marker="o",
        label="Prediction disagreement vs T=50",
    )
    axis.plot(
        summary["T"],
        summary["entropy_mae"],
        marker="s",
        label="MC entropy MAE vs T=50",
    )
    axis.plot(
        summary["T"],
        summary["mi_mae"],
        marker="^",
        label="MC MI MAE vs T=50",
    )
    axis.set_xlabel("MC Dropout passes T")
    axis.set_ylabel("Difference from T=50 reference")
    axis.set_title("Validation convergence toward T=50")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def write_notes(
    out_dir: Path,
    summary: pd.DataFrame,
    distance: pd.DataFrame,
    stability_summary: pd.DataFrame,
    selected_t: int,
) -> None:
    t30 = summary[summary["T"] == 30].iloc[0]
    t50 = summary[summary["T"] == 50].iloc[0]
    distance30 = distance[distance["T"] == 30]
    text = f"""# Validation-Only MC Dropout Convergence

- Training seeds: 0, 1 and 2.
- Inference seed: {INFERENCE_SEED}.
- Nested pass counts: {list(T_VALUES)}.
- Reference for numerical differences: T=50.
- Test files read: no.

## Frozen practical stability rule

The smallest T must satisfy every criterion for every training seed relative
to the nested T=50 reference.  The project-specific tolerances are stored in
`selected_mc_config.json`; they cover prediction disagreement, entropy and MI
rank correlation, error-AUROC/AUPR differences, and Exact AURC difference.

- Selected minimum stable T: {selected_t}.
- Stable candidates across all three seeds: {
    stability_summary.loc[stability_summary['all_seeds_stable'], 'T'].astype(int).tolist()
  }.

## T=30 versus T=50

- MC entropy error-AUROC mean: {t30['mc_entropy_error_auroc_mean']:.6f}
  versus {t50['mc_entropy_error_auroc_mean']:.6f}.
- MC mutual-information error-AUROC mean:
  {t30['mc_mutual_information_error_auroc_mean']:.6f}
  versus {t50['mc_mutual_information_error_auroc_mean']:.6f}.
- MC entropy AURC mean: {t30['mc_entropy_aurc_mean']:.6f}
  versus {t50['mc_entropy_aurc_mean']:.6f}.
- Mean prediction disagreement rate between T=30 and T=50:
  {distance30['prediction_disagreement_rate_vs_T50'].mean():.6f}.
- Mean MC entropy rank correlation between T=30 and T=50:
  {distance30['mc_entropy_spearman_vs_T50'].mean():.6f}.
- Mean MC mutual-information rank correlation between T=30 and T=50:
  {distance30['mc_mutual_information_spearman_vs_T50'].mean():.6f}.

The thresholds are explicit project operating tolerances, not universal laws.
They were formalised during reproducibility hardening after the initial
exploratory analysis, so the evidence remains development-stage rather than
confirmatory.
"""
    (out_dir / "analysis_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-run",
        action="append",
        type=parse_seed_run,
        help="Training run as SEED=PATH; repeat for each seed.",
    )
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--inference-seed", type=int, default=INFERENCE_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    seed_runs = dict(args.seed_run) if args.seed_run else DEFAULT_SEED_RUNS
    if len(seed_runs) < 2:
        raise ValueError("MC convergence analysis requires at least two training seeds")
    features_dir = args.features_dir.expanduser().resolve()
    out_dir = ensure_fresh_output_dir(args.out_dir, args.allow_overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metric_rows: list[dict] = []
    distance_rows: list[dict] = []
    frame_rows: list[dict] = []
    runtime_rows: list[dict] = []
    sources = []
    for training_seed, run_dir in sorted(seed_runs.items()):
        run_dir = run_dir.expanduser().resolve()
        checkpoint = run_dir / "checkpoints" / "best.pt"
        checkpoint_payload = torch.load(checkpoint, map_location="cpu")
        checkpoint_args = checkpoint_payload.get("args", {})
        checkpoint_training_seed = int(checkpoint_args.get("seed", training_seed))
        if checkpoint_training_seed != training_seed:
            raise ValueError(
                f"Checkpoint training seed {checkpoint_training_seed} does not "
                f"match requested seed {training_seed}: {checkpoint}"
            )
        metrics, distances, frames, runtimes = evaluate_training_seed(
            training_seed,
            run_dir,
            features_dir,
            device,
            args.inference_seed,
        )
        metric_rows.extend(metrics)
        distance_rows.extend(distances)
        frame_rows.extend(frames)
        runtime_rows.extend(runtimes)
        sources.append(
            {
                "training_seed": training_seed,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_epoch": int(checkpoint_payload["epoch"]),
                "checkpoint_validation_macro_f1": float(
                    checkpoint_payload["best_val_macro_f1"]
                ),
                "checkpoint_selection_split": "val",
                "checkpoint_selection_metric": "macro_f1",
            }
        )

    per_seed = pd.DataFrame(metric_rows)
    distance = pd.DataFrame(distance_rows)
    frame_scores = pd.DataFrame(frame_rows)
    runtime_by_video = pd.DataFrame(runtime_rows)
    stability_per_seed, stability_summary, selected_t = assess_stability(
        per_seed,
        distance,
    )
    convergence_per_seed = (
        per_seed.merge(
            distance,
            on=["training_seed", "inference_seed", "split", "T"],
            validate="one_to_one",
        )
        .merge(
            stability_per_seed,
            on=["training_seed", "inference_seed", "split", "T"],
            validate="one_to_one",
        )
    )
    summary = summarise(convergence_per_seed)
    summary = summary.merge(stability_summary, on="T", validate="one_to_one")
    convergence_per_seed.to_csv(
        out_dir / "validation_convergence_per_seed.csv",
        index=False,
    )
    summary.to_csv(out_dir / "validation_convergence_summary.csv", index=False)
    stability_per_seed.to_csv(
        out_dir / "validation_stability_checks_per_seed.csv",
        index=False,
    )
    runtime_by_video.to_csv(
        out_dir / "validation_inference_runtime_by_video.csv",
        index=False,
    )
    selected_config = {
        "schema_version": 1,
        "config_id": "selected_mc_config_validation_v3",
        "status": "selected_on_validation_only",
        "selected_T": selected_t,
        "training_seeds": sorted(seed_runs),
        "inference_seed": args.inference_seed,
        "selection_split": "val",
        "validation_videos": list(VALIDATION_VIDEO_IDS),
        "test_files_read": False,
        "reference_T": max(T_VALUES),
        "nested_prefix_candidates": list(T_VALUES),
        "selection_rule": (
            "Select the smallest T for which every stability criterion passes "
            "for every training seed relative to the nested T=50 reference."
        ),
        "stability_criteria": STABILITY_CRITERIA,
        "stable_candidates": stability_summary.loc[
            stability_summary["all_seeds_stable"], "T"
        ].astype(int).tolist(),
        "error_definition": (
            "Incorrect MC-averaged argmax prediction is positive."
        ),
        "score_definitions": {
            "mc_predictive_entropy": "Entropy of the MC-mean probability.",
            "mc_mutual_information": (
                "Entropy of the MC-mean probability minus mean entropy of "
                "the stochastic probability samples."
            ),
        },
        "evidence_status": "validation-selected iterative-development evidence",
    }
    (out_dir / "selected_mc_config.json").write_text(
        json.dumps(selected_config, indent=2),
        encoding="utf-8",
    )
    # Compatibility exports retain the earlier filenames inside this new v2
    # directory; the v1 directory is never overwritten.
    per_seed.to_csv(out_dir / "per_seed_t_metrics.csv", index=False)
    summarise(per_seed).to_csv(out_dir / "mean_std_t_metrics.csv", index=False)
    distance.to_csv(out_dir / "distance_to_t50.csv", index=False)
    frame_scores.to_csv(
        out_dir / "validation_frame_scores.csv.gz",
        index=False,
        compression="gzip",
    )
    plot_metrics(summary, figure_dir / "mc_convergence_metrics.png")
    plot_distance(distance, figure_dir / "distance_to_t50.png")
    write_notes(out_dir, summary, distance, stability_summary, selected_t)

    feature_sources = []
    for video_id in VALIDATION_VIDEO_IDS:
        for suffix in (".npy", "_labels.npy"):
            path = features_dir / f"{video_id}{suffix}"
            feature_sources.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    elapsed_seconds = time.perf_counter() - run_started
    manifest = {
        "schema_version": 2,
        "analysis": "mc_dropout_convergence_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_split": "val_only",
        "test_files_read": False,
        "training_runs": sources,
        "feature_sources": feature_sources,
        "T_values": T_VALUES,
        "nested_prefix_design": True,
        "selected_T": selected_t,
        "stability_criteria": STABILITY_CRITERIA,
        "inference_seed": args.inference_seed,
        "dropout_mode": "model_eval_with_nn_dropout_modules_enabled",
        "error_definition": "Incorrect MC-averaged argmax prediction is positive.",
        "risk_coverage_config": "configs/selective_prediction_evaluation_v2.json",
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "code_dependencies": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in (
                PROJECT_ROOT / "scripts" / "08_eval_mc_dropout.py",
                PROJECT_ROOT / "scripts" / "selective_prediction_metrics.py",
                PROJECT_ROOT / "scripts" / "reproducibility_utils.py",
            )
        ],
        "runtime": {
            "wall_clock_seconds": elapsed_seconds,
            "timing_definition": (
                "Cumulative stochastic forward-pass time, including softmax "
                "and CPU transfer, measured with CUDA synchronisation at each T."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "outputs": [
            str((out_dir / filename).resolve())
            for filename in (
                "validation_convergence_per_seed.csv",
                "validation_convergence_summary.csv",
                "validation_stability_checks_per_seed.csv",
                "validation_inference_runtime_by_video.csv",
                "selected_mc_config.json",
                "validation_frame_scores.csv.gz",
                "analysis_notes.md",
            )
        ],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved validation-only MC convergence analysis to: {out_dir}")


if __name__ == "__main__":
    main()
