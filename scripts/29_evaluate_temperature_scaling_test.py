"""Evaluate validation-frozen temperatures on test and finalise calibration v2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproducibility_utils import PROJECT_ROOT, sha256_file
from temperature_scaling_logits import (
    load_raw_logit_split,
    metrics_from_logits,
    reliability_from_logits,
    softmax_logits,
    validate_checkpoint,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "temperature_scaling_three_seed_protocol_v2"
METRICS = ["nll", "brier", "ece_15_bins", "accuracy", "macro_f1", "mean_confidence"]


def source_with_hash(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def plot_reliability(
    reliability: pd.DataFrame,
    output_path: Path,
    training_seed: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True, sharey=True)
    states = (("raw", "Before calibration"), ("temperature_scaled", "After calibration"))
    for axis, (state, title) in zip(axes, states):
        subset = reliability[
            reliability["calibration_state"].eq(state)
            & reliability["count"].gt(0)
        ]
        axis.plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1)
        axis.plot(
            subset["mean_confidence"],
            subset["accuracy"],
            marker="o",
            color="#2C6EAA" if state == "raw" else "#D1495B",
            linewidth=1.7,
        )
        axis.set_title(title)
        axis.set_xlabel("Mean confidence")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Empirical accuracy")
    figure.suptitle(f"Seed {training_seed:02d} test reliability (15 equal-width bins)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state, group in metrics.groupby("calibration_state", sort=False):
        row: dict[str, object] = {
            "split": "test",
            "calibration_state": state,
            "n_training_seeds": group["training_seed"].nunique(),
            "aggregation": "mean_and_sample_standard_deviation_across_seed_level_pooled_metrics",
            "ddof": 1,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    out_dir = args.out_dir.expanduser().resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError("run validation fitting before test evaluation")
    protected_outputs = [
        out_dir / "test_calibration_metrics_per_seed.csv",
        out_dir / "test_calibration_mean_std.csv",
        out_dir / "run_manifest_seed00.json",
    ]
    if any(path.exists() for path in protected_outputs):
        raise FileExistsError(
            "refusing to overwrite existing test calibration outputs; choose a new output directory"
        )

    protocol_path = out_dir / "calibration_protocol.json"
    temperature_path = out_dir / "temperature_per_seed.csv"
    validation_manifest_path = out_dir / "run_manifest_validation_fit.json"
    for path in (protocol_path, temperature_path, validation_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol_hash_before_test = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "temperature_frozen_after_validation_fit_before_test_evaluation":
        raise ValueError("temperature protocol was not frozen by the validation-only stage")
    if validation_manifest["data_flow_checks"].get("test_files_read") is not False:
        raise ValueError("validation fitting manifest does not prove test isolation")
    if validation_manifest["protocol_output"]["sha256"] != protocol_hash_before_test:
        raise ValueError("frozen protocol changed after validation fitting")

    fits = pd.read_csv(temperature_path)
    if set(fits["training_seed"].astype(int)) != set(protocol["training_seeds"]):
        raise ValueError("temperature table does not contain exactly the frozen seeds")
    fit_by_seed = fits.set_index("training_seed")
    seeds = [int(seed) for seed in protocol["training_seeds"]]
    test_spec = protocol["data"]["test_split"]
    video_ids = [str(video).zfill(2) for video in test_spec["video_ids"]]

    metric_rows: list[dict[str, object]] = []
    per_video_rows: list[dict[str, object]] = []
    reliability_rows: list[pd.DataFrame] = []
    check_rows: list[dict[str, object]] = []
    all_sources: list[dict[str, object]] = []
    checkpoint_records: dict[int, dict[str, object]] = {}

    for training_seed in seeds:
        checkpoint = validate_checkpoint(protocol, training_seed)
        checkpoint_records[training_seed] = checkpoint
        run_dir = Path(checkpoint["checkpoint_path"]).parents[1]
        frame, logits, labels, sources = load_raw_logit_split(
            training_seed,
            run_dir,
            split="test",
            video_ids=video_ids,
        )
        all_sources.extend(sources)
        temperature = float(fit_by_seed.loc[training_seed, "temperature"])
        if temperature <= 0 or not np.isfinite(temperature):
            raise ValueError(f"invalid frozen temperature for seed {training_seed}")
        if not np.isclose(
            temperature,
            float(protocol["fitted_temperatures_by_training_seed"][str(training_seed)]),
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(f"temperature/protocol mismatch for seed {training_seed}")

        raw_probabilities = softmax_logits(logits)
        scaled_probabilities = softmax_logits(logits, temperature)
        raw_prediction = raw_probabilities.argmax(axis=1)
        scaled_prediction = scaled_probabilities.argmax(axis=1)
        seed_metrics: dict[str, dict[str, float]] = {}
        seed_reliability: list[pd.DataFrame] = []
        for state, state_temperature in (("raw", 1.0), ("temperature_scaled", temperature)):
            state_metrics = metrics_from_logits(
                logits,
                labels,
                state_temperature,
                ece_bins=15,
            )
            seed_metrics[state] = state_metrics
            metric_rows.append(
                {
                    **checkpoint,
                    "split": "test",
                    "calibration_state": state,
                    "temperature_fitted_on_validation": temperature,
                    "temperature_applied": state_temperature,
                    "n_frames": len(labels),
                    **state_metrics,
                }
            )
            reliability = reliability_from_logits(
                logits,
                labels,
                state_temperature,
                n_bins=15,
            )
            reliability.insert(0, "calibration_state", state)
            reliability.insert(0, "split", "test")
            reliability.insert(0, "training_seed", training_seed)
            reliability_rows.append(reliability)
            seed_reliability.append(reliability)

            for video_id in video_ids:
                member = frame["video_id"].eq(video_id).to_numpy()
                video_metrics = metrics_from_logits(
                    logits[member],
                    labels[member],
                    state_temperature,
                    ece_bins=15,
                )
                per_video_rows.append(
                    {
                        "training_seed": training_seed,
                        "split": "test",
                        "video_id": video_id,
                        "calibration_state": state,
                        "temperature_fitted_on_validation": temperature,
                        "temperature_applied": state_temperature,
                        "n_frames": int(member.sum()),
                        **video_metrics,
                    }
                )

        accuracy_equal = np.isclose(
            seed_metrics["raw"]["accuracy"],
            seed_metrics["temperature_scaled"]["accuracy"],
            atol=0.0,
            rtol=0.0,
        )
        macro_f1_equal = np.isclose(
            seed_metrics["raw"]["macro_f1"],
            seed_metrics["temperature_scaled"]["macro_f1"],
            atol=0.0,
            rtol=0.0,
        )
        checks = {
            "training_seed": training_seed,
            "raw_vs_scaled_argmax_identical_every_second": bool(
                np.array_equal(raw_prediction, scaled_prediction)
            ),
            "accuracy_exactly_identical": bool(accuracy_equal),
            "macro_f1_exactly_identical": bool(macro_f1_equal),
            "raw_probability_max_sum_error": float(
                np.abs(raw_probabilities.sum(axis=1) - 1.0).max()
            ),
            "scaled_probability_max_sum_error": float(
                np.abs(scaled_probabilities.sum(axis=1) - 1.0).max()
            ),
            "frame_logit_label_lengths_aligned": bool(
                len(frame) == len(logits) == len(labels)
            ),
            "validation_nll_non_increasing": bool(
                fit_by_seed.loc[training_seed, "validation_nll_after"]
                <= fit_by_seed.loc[training_seed, "validation_nll_before"] + 1e-12
            ),
            "test_used_for_temperature_selection": False,
        }
        if not all(
            checks[key]
            for key in (
                "raw_vs_scaled_argmax_identical_every_second",
                "accuracy_exactly_identical",
                "macro_f1_exactly_identical",
                "frame_logit_label_lengths_aligned",
                "validation_nll_non_increasing",
            )
        ):
            raise AssertionError(f"test correctness check failed: {checks}")
        check_rows.append(checks)
        plot_reliability(
            pd.concat(seed_reliability, ignore_index=True),
            out_dir / f"reliability_diagram_seed{training_seed:02d}.png",
            training_seed,
        )

    if any(record["split"] != "test" for record in all_sources):
        raise AssertionError("test stage read a non-test raw-logit source")
    if sha256_file(protocol_path) != protocol_hash_before_test:
        raise AssertionError("frozen protocol changed during test evaluation")

    metrics = pd.DataFrame(metric_rows)
    per_video = pd.DataFrame(per_video_rows)
    reliability = pd.concat(reliability_rows, ignore_index=True)
    checks = pd.DataFrame(check_rows)
    summary = summarise(metrics)
    metrics.to_csv(out_dir / "test_calibration_metrics_per_seed.csv", index=False)
    summary.to_csv(out_dir / "test_calibration_mean_std.csv", index=False)
    per_video.to_csv(out_dir / "test_calibration_metrics_per_video.csv", index=False)
    reliability.to_csv(out_dir / "test_reliability_bins_15.csv", index=False)
    checks.to_csv(out_dir / "test_correctness_checks.csv", index=False)

    comparison = metrics.pivot(index="training_seed", columns="calibration_state", values=METRICS)
    improvement_by_metric = {
        metric: bool(
            (
                comparison[(metric, "temperature_scaled")]
                < comparison[(metric, "raw")]
            ).all()
        )
        for metric in ("nll", "brier", "ece_15_bins")
    }
    all_calibration_metrics_improved = all(improvement_by_metric.values())
    conclusion = (
        "Temperature scaling improved NLL, multiclass Brier score and 15-bin ECE "
        "for every seed without changing phase recognition accuracy."
        if all_calibration_metrics_improved
        else "Temperature scaling did not improve every required calibration metric for every seed; individual metric changes must be reported."
    )
    raw_summary = summary[summary["calibration_state"].eq("raw")].iloc[0]
    calibrated_summary = summary[
        summary["calibration_state"].eq("temperature_scaled")
    ].iloc[0]
    readme_lines = [
        "# Temperature Scaling Calibration v2",
        "",
        "## Frozen data flow",
        "",
        "- One positive scalar temperature was fitted separately for each training seed.",
        "- Fitting read only validation videos 11-14 and minimised pooled per-second multiclass NLL.",
        "- The test stage read videos 15-21 only after the temperature file and protocol were frozen.",
        "- Inputs are original logits saved by each validation-selected best checkpoint; no probabilities were transformed back into logits.",
        "- Calibration is an RQ1 reliability analysis and does not enter the frozen RQ2 gate.",
        "",
        "## Metric definitions",
        "",
        "- NLL: `-(1/N) * sum_i log(p_i,y_i)`; lower is better.",
        "- Multiclass Brier: `(1/N) * sum_i sum_c (p_i,c - 1[y_i=c])^2`; lower is better.",
        "- ECE-15: top-label ECE with 15 equal-width confidence bins on [0,1], weighted by bin frequency; confidence 1 is included in the final bin.",
        "- Accuracy: pooled per-second argmax accuracy.",
        "- Macro-F1: unweighted mean of seven class F1 values (`zero_division=0`).",
        "- Each seed pools all seconds in its split. Three-seed summaries are the arithmetic mean and sample standard deviation (`ddof=1`) of seed-level metrics.",
        "- Per-video metrics use the same definitions within each video; they are diagnostic and are not used to fit temperature.",
        "",
        "## Validation-fitted temperatures",
        "",
    ]
    for row in fits.itertuples():
        readme_lines.append(
            f"- Seed {int(row.training_seed):02d}: T={row.temperature:.6f}; validation NLL {row.validation_nll_before:.6f} -> {row.validation_nll_after:.6f}."
        )
    readme_lines.extend(
        [
            "",
            "## Three-seed test summary",
            "",
            f"- NLL: {raw_summary['nll_mean']:.6f} ± {raw_summary['nll_std']:.6f} -> {calibrated_summary['nll_mean']:.6f} ± {calibrated_summary['nll_std']:.6f}.",
            f"- Multiclass Brier: {raw_summary['brier_mean']:.6f} ± {raw_summary['brier_std']:.6f} -> {calibrated_summary['brier_mean']:.6f} ± {calibrated_summary['brier_std']:.6f}.",
            f"- ECE-15: {raw_summary['ece_15_bins_mean']:.6f} ± {raw_summary['ece_15_bins_std']:.6f} -> {calibrated_summary['ece_15_bins_mean']:.6f} ± {calibrated_summary['ece_15_bins_std']:.6f}.",
            f"- Accuracy: {raw_summary['accuracy_mean']:.6f} ± {raw_summary['accuracy_std']:.6f} before and after calibration.",
            f"- Macro-F1: {raw_summary['macro_f1_mean']:.6f} ± {raw_summary['macro_f1_std']:.6f} before and after calibration.",
            "",
            "## Correctness and interpretation",
            "",
            f"- {conclusion}",
            "- Every second retained the same argmax class; accuracy and macro-F1 are exactly unchanged within each seed.",
            "- Probability rows sum to one within numerical precision, and the saved video/time/label order was cross-checked against feature metadata and deterministic per-second tables.",
            "- The validation fitting manifest records that no test file or label was read before temperatures were frozen.",
            "- These results are three-seed iterative-development evidence, not a fully independent confirmatory test, because the test split had been inspected earlier in the project.",
            "",
            "## Reproduction commands",
            "",
            "```powershell",
            "python scripts\\28_fit_temperature_scaling_validation.py",
            "python scripts\\29_evaluate_temperature_scaling_test.py",
            "```",
        ]
    )
    readme_path = out_dir / "README_calibration.md"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    shared_output_names = [
        "calibration_protocol.json",
        "temperature_per_seed.csv",
        "test_calibration_metrics_per_seed.csv",
        "test_calibration_mean_std.csv",
        "test_calibration_metrics_per_video.csv",
        "test_reliability_bins_15.csv",
        "test_correctness_checks.csv",
        "README_calibration.md",
    ]
    script_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "temperature_scaling_logits.py",
        PROJECT_ROOT / "scripts" / "calibration_metrics.py",
        PROJECT_ROOT / "scripts" / "reproducibility_utils.py",
    ]
    validation_fit_reference = source_with_hash(validation_manifest_path)
    for training_seed in seeds:
        seed_sources = [
            record for record in all_sources if int(record["training_seed"]) == training_seed
        ]
        seed_figure = out_dir / f"reliability_diagram_seed{training_seed:02d}.png"
        manifest = {
            "schema_version": 2,
            "stage": "test_evaluation_of_validation_frozen_temperature",
            "training_seed": training_seed,
            "temperature": float(fit_by_seed.loc[training_seed, "temperature"]),
            "evidence_status": protocol["evidence_status"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "checkpoint": checkpoint_records[training_seed],
            "frozen_protocol": source_with_hash(protocol_path),
            "validation_fit_manifest": validation_fit_reference,
            "test_inputs": seed_sources,
            "data_flow_checks": {
                "temperature_fit_split": "val",
                "temperature_test_split": "test",
                "test_video_ids": video_ids,
                "test_used_for_temperature_fitting_or_selection": False,
                "probability_to_logit_reconstruction_used": False,
                "raw_logit_label_video_time_alignment_verified": True,
                "rq2_gate_inputs_changed": False,
            },
            "correctness": checks[
                checks["training_seed"].eq(training_seed)
            ].iloc[0].to_dict(),
            "code": [source_with_hash(path) for path in script_paths],
            "outputs": [
                *[source_with_hash(out_dir / name) for name in shared_output_names],
                source_with_hash(seed_figure),
            ],
        }
        (out_dir / f"run_manifest_seed{training_seed:02d}.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    final_manifest = {
        "schema_version": 2,
        "analysis": "three_seed_raw_logit_temperature_scaling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": time.perf_counter() - started,
        "evidence_status": protocol["evidence_status"],
        "all_required_calibration_metrics_improved_for_every_seed": all_calibration_metrics_improved,
        "improvement_by_metric_for_every_seed": improvement_by_metric,
        "temperature_protocol_hash_before_and_after_test_identical": True,
        "test_used_for_temperature_fitting_or_selection": False,
        "rq2_gate_inputs_changed": False,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
        "artifacts": [
            source_with_hash(out_dir / name)
            for name in [
                *shared_output_names,
                *[f"reliability_diagram_seed{seed:02d}.png" for seed in seeds],
                *[f"run_manifest_seed{seed:02d}.json" for seed in seeds],
                "run_manifest_validation_fit.json",
            ]
        ],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(final_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved final calibration protocol to: {out_dir}")


if __name__ == "__main__":
    main()
