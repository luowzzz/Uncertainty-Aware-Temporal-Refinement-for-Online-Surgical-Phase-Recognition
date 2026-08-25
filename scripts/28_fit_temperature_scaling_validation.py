"""Fit and freeze one raw-logit temperature per seed using validation only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from reproducibility_utils import PROJECT_ROOT, ensure_fresh_output_dir, sha256_file
from temperature_scaling_logits import (
    fit_temperature_from_logits,
    load_protocol,
    load_raw_logit_split,
    metrics_from_logits,
    reliability_from_logits,
    softmax_logits,
    validate_checkpoint,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "temperature_scaling_calibration_v2.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "temperature_scaling_three_seed_protocol_v2"


def source_with_hash(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    config_path, config = load_protocol(args.config)
    out_dir = ensure_fresh_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config["training_seeds"]]
    val_spec = config["data"]["validation_split"]
    if val_spec["name"] != config["fit"]["split"]:
        raise ValueError("validation split and fit split disagree in config")
    video_ids = [str(video).zfill(2) for video in val_spec["video_ids"]]
    lower, upper = (float(value) for value in config["fit"]["temperature_bounds"])
    xatol = float(config["fit"]["optimizer_xatol"])
    maxiter = int(config["fit"]["optimizer_maxiter"])

    fit_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    reliability_rows: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    fitted_temperatures: dict[str, float] = {}
    fit_check_rows: list[dict[str, object]] = []

    for training_seed in seeds:
        checkpoint = validate_checkpoint(config, training_seed)
        checkpoint_records.append(checkpoint)
        run_dir = Path(checkpoint["checkpoint_path"]).parents[1]
        frame, logits, labels, seed_sources = load_raw_logit_split(
            training_seed,
            run_dir,
            split="val",
            video_ids=video_ids,
        )
        sources.extend(seed_sources)
        fit = fit_temperature_from_logits(
            logits,
            labels,
            lower=lower,
            upper=upper,
            xatol=xatol,
            maxiter=maxiter,
        )
        temperature = float(fit["temperature"])
        if not lower < temperature < upper:
            raise AssertionError("fitted temperature is not strictly inside the frozen bounds")
        fitted_temperatures[str(training_seed)] = temperature
        fit_rows.append(
            {
                **checkpoint,
                "fit_split": "val",
                "fit_video_ids": ",".join(video_ids),
                "n_validation_frames": len(labels),
                **fit,
            }
        )

        raw_prediction = logits.argmax(axis=1)
        scaled_prediction = (logits / temperature).argmax(axis=1)
        raw_probabilities = softmax_logits(logits)
        scaled_probabilities = softmax_logits(logits, temperature)
        checks = {
            "training_seed": training_seed,
            "split": "val",
            "raw_vs_scaled_argmax_identical": bool(
                np.array_equal(raw_prediction, scaled_prediction)
            ),
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
                float(fit["validation_nll_after"])
                <= float(fit["validation_nll_before"]) + 1e-12
            ),
        }
        if not all(
            checks[key]
            for key in (
                "raw_vs_scaled_argmax_identical",
                "frame_logit_label_lengths_aligned",
                "validation_nll_non_increasing",
            )
        ):
            raise AssertionError(f"validation correctness check failed: {checks}")
        fit_check_rows.append(checks)

        for state, state_temperature in (("raw", 1.0), ("temperature_scaled", temperature)):
            metric_rows.append(
                {
                    "training_seed": training_seed,
                    "split": "val",
                    "calibration_state": state,
                    "temperature": state_temperature,
                    "n_frames": len(labels),
                    **metrics_from_logits(logits, labels, state_temperature, ece_bins=15),
                }
            )
            reliability = reliability_from_logits(
                logits,
                labels,
                state_temperature,
                n_bins=15,
            )
            reliability.insert(0, "calibration_state", state)
            reliability.insert(0, "split", "val")
            reliability.insert(0, "training_seed", training_seed)
            reliability_rows.append(reliability)

    if any(record["split"] != "val" for record in sources):
        raise AssertionError("validation-fit stage read a non-validation source")
    if any("test" in str(record["path"]).lower() for record in sources):
        raise AssertionError("validation-fit stage unexpectedly read a test path")

    fits = pd.DataFrame(fit_rows)
    metrics = pd.DataFrame(metric_rows)
    reliability = pd.concat(reliability_rows, ignore_index=True)
    checks = pd.DataFrame(fit_check_rows)
    fits.to_csv(out_dir / "temperature_per_seed.csv", index=False)
    metrics.to_csv(out_dir / "validation_calibration_metrics_per_seed.csv", index=False)
    reliability.to_csv(out_dir / "validation_reliability_bins_15.csv", index=False)
    checks.to_csv(out_dir / "validation_fit_correctness_checks.csv", index=False)

    frozen_protocol = {
        **config,
        "status": "temperature_frozen_after_validation_fit_before_test_evaluation",
        "source_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "fitted_temperatures_by_training_seed": fitted_temperatures,
        "temperature_selection": {
            "split": "val",
            "objective": "pooled per-second multiclass negative log likelihood",
            "test_files_read": False,
            "test_labels_used": False,
            "frozen_before_test_stage": True,
        },
    }
    protocol_path = out_dir / "calibration_protocol.json"
    protocol_path.write_text(json.dumps(frozen_protocol, indent=2), encoding="utf-8")

    output_names = [
        "temperature_per_seed.csv",
        "validation_calibration_metrics_per_seed.csv",
        "validation_reliability_bins_15.csv",
        "validation_fit_correctness_checks.csv",
        "calibration_protocol.json",
    ]
    script_paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts" / "temperature_scaling_logits.py",
        PROJECT_ROOT / "scripts" / "calibration_metrics.py",
        PROJECT_ROOT / "scripts" / "reproducibility_utils.py",
    ]
    manifest = {
        "schema_version": 2,
        "stage": "validation_only_temperature_fit",
        "evidence_status": config["evidence_status"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": time.perf_counter() - started,
        "command": " ".join(sys.argv),
        "code": [source_with_hash(path) for path in script_paths],
        "config": source_with_hash(config_path),
        "protocol_output": source_with_hash(protocol_path),
        "checkpoint_records": checkpoint_records,
        "inputs": sources,
        "data_flow_checks": {
            "fit_split": "val",
            "fit_video_ids": video_ids,
            "test_files_read": False,
            "test_labels_used": False,
            "input_roles_read": sorted({str(record["role"]) for record in sources}),
            "logit_source": "original raw logits from selected-checkpoint prediction NPZ files",
            "probability_to_logit_reconstruction_used": False,
        },
        "determinism": {
            "random_sampling_used": False,
            "optimizer": "deterministic bounded scalar optimisation",
            "data_order": "ascending video_id then ascending t_sec",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "outputs": [source_with_hash(out_dir / name) for name in output_names],
    }
    (out_dir / "run_manifest_validation_fit.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Frozen validation-fitted temperatures in: {out_dir}")


if __name__ == "__main__":
    main()
