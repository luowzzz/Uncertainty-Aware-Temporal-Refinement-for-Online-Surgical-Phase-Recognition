"""Strict raw-logit utilities for the frozen temperature-scaling protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from calibration_metrics import (
    calibration_metrics,
    multiclass_nll,
    reliability_table,
)
from reproducibility_utils import PROJECT_ROOT, sha256_file


TEMPERATURE_EPSILON = 1e-12
N_CLASSES = 7


def validate_logits(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != N_CLASSES:
        raise ValueError(f"logits must have shape [n_samples, {N_CLASSES}]")
    if labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("labels must be one-dimensional and aligned with logits")
    if not len(labels):
        raise ValueError("at least one logit/label pair is required")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    if labels.min() < 0 or labels.max() >= N_CLASSES:
        raise ValueError("labels fall outside the seven phase classes")
    return logits, labels


def softmax_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != N_CLASSES:
        raise ValueError(f"logits must have shape [n_samples, {N_CLASSES}]")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    scaled = logits / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("softmax probability rows do not sum to one")
    return probabilities


def fit_temperature_from_logits(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    lower: float = 0.05,
    upper: float = 20.0,
    xatol: float = 1e-8,
    maxiter: int = 500,
) -> dict[str, float | bool | int | str]:
    logits, labels = validate_logits(validation_logits, validation_labels)
    if not 0 < lower < upper:
        raise ValueError("temperature bounds must satisfy 0 < lower < upper")

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return multiclass_nll(softmax_logits(logits, temperature), labels)

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )
    temperature = float(np.exp(result.x))
    before = multiclass_nll(softmax_logits(logits), labels)
    after = multiclass_nll(softmax_logits(logits, temperature), labels)
    if not result.success:
        raise RuntimeError(f"temperature optimisation failed: {result.message}")
    if after > before + 1e-12:
        raise AssertionError("validation NLL increased after temperature fitting")
    return {
        "temperature": temperature,
        "validation_nll_before": before,
        "validation_nll_after": after,
        "success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_evaluations": int(result.nfev),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "xatol": float(xatol),
        "maxiter": int(maxiter),
    }


def metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    ece_bins: int = 15,
) -> dict[str, float]:
    logits, labels = validate_logits(logits, labels)
    probabilities = softmax_logits(logits, temperature)
    prediction = probabilities.argmax(axis=1)
    metrics = calibration_metrics(probabilities, labels, ece_bins=(ece_bins,))
    metrics["macro_f1"] = float(
        f1_score(
            labels,
            prediction,
            labels=list(range(N_CLASSES)),
            average="macro",
            zero_division=0,
        )
    )
    return metrics


def reliability_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    n_bins: int = 15,
) -> pd.DataFrame:
    logits, labels = validate_logits(logits, labels)
    return reliability_table(softmax_logits(logits, temperature), labels, n_bins=n_bins)


def load_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("config_id") != "temperature_scaling_calibration_v2":
        raise ValueError(f"unexpected calibration config: {payload.get('config_id')!r}")
    if payload.get("fit", {}).get("test_used_for_fitting_or_selection") is not False:
        raise ValueError("calibration config must prohibit test-informed fitting")
    if payload.get("separation_from_rq2", {}).get(
        "calibrated_probabilities_enter_frozen_gate"
    ) is not False:
        raise ValueError("calibration config must remain separate from RQ2")
    return source, payload


def validate_checkpoint(config: dict[str, Any], training_seed: int) -> dict[str, Any]:
    expected = config["checkpoints"][str(training_seed)]
    checkpoint = (PROJECT_ROOT / expected["path"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_hash = sha256_file(checkpoint)
    if observed_hash.lower() != str(expected["sha256"]).lower():
        raise ValueError(f"checkpoint hash mismatch for seed {training_seed}")
    return {
        "training_seed": training_seed,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": observed_hash,
        "checkpoint_epoch": int(expected["epoch"]),
        "checkpoint_validation_macro_f1": float(expected["validation_macro_f1"]),
        "checkpoint_selection_split": "val",
        "checkpoint_selection_metric": "macro_f1",
    }


def _source_record(path: Path, role: str, **extra: Any) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        **extra,
    }


def load_raw_logit_split(
    training_seed: int,
    run_dir: str | Path,
    split: str,
    video_ids: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load original saved logits and audit their label/video/time alignment."""
    run_dir = Path(run_dir).expanduser().resolve()
    feature_dir = PROJECT_ROOT / "data" / "features" / "resnet18"
    per_second_path = run_dir / "model_outputs" / f"per_second_outputs_{split}.csv"
    per_second = pd.read_csv(per_second_path, dtype={"video_id": str})
    per_second["video_id"] = per_second["video_id"].str.zfill(2)
    if set(per_second["split"].astype(str)) != {split}:
        raise ValueError(f"{per_second_path} contains rows outside {split!r}")
    sources = [
        _source_record(
            per_second_path,
            "deterministic_per_second_cross_check",
            training_seed=training_seed,
            split=split,
        )
    ]
    frames: list[pd.DataFrame] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    probability_columns = [f"prob_phase_{index}" for index in range(1, N_CLASSES + 1)]

    for video_id in video_ids:
        video_id = str(video_id).zfill(2)
        prediction_path = run_dir / "predictions" / f"{split}_video_{video_id}_predictions.npz"
        label_path = feature_dir / f"{video_id}_labels.npy"
        meta_path = feature_dir / f"{video_id}_meta.csv"
        for path in (prediction_path, label_path, meta_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(prediction_path, allow_pickle=False) as saved:
            required = {"video_id", "logits", "probs", "pred_label_idx", "true_label_idx"}
            if not required.issubset(saved.files):
                raise ValueError(f"{prediction_path} is missing {sorted(required - set(saved.files))}")
            saved_video_id = str(saved["video_id"].item()).zfill(2)
            logits = np.asarray(saved["logits"], dtype=np.float64)
            labels = np.asarray(saved["true_label_idx"], dtype=np.int64)
            saved_probabilities = np.asarray(saved["probs"], dtype=np.float64)
            saved_prediction = np.asarray(saved["pred_label_idx"], dtype=np.int64)
        validate_logits(logits, labels)
        if saved_video_id != video_id:
            raise ValueError(f"video id mismatch inside {prediction_path}")
        probabilities = softmax_logits(logits)
        if not np.allclose(probabilities, saved_probabilities, atol=2e-7, rtol=1e-6):
            raise ValueError(f"saved probabilities do not match raw logits in {prediction_path}")
        if not np.array_equal(logits.argmax(axis=1), saved_prediction):
            raise ValueError(f"saved predictions do not match raw logits in {prediction_path}")

        feature_labels = np.load(label_path).astype(np.int64)
        meta = pd.read_csv(meta_path, dtype={"video_id": str})
        meta["video_id"] = meta["video_id"].str.zfill(2)
        expected_time = np.arange(len(labels), dtype=np.int64)
        if not np.array_equal(labels, feature_labels):
            raise ValueError(f"label mismatch between logits and feature labels for video {video_id}")
        if len(meta) != len(labels):
            raise ValueError(f"metadata length mismatch for video {video_id}")
        if set(meta["video_id"]) != {video_id} or set(meta["split"].astype(str)) != {split}:
            raise ValueError(f"metadata video/split mismatch for video {video_id}")
        if not np.array_equal(meta["t_sec"].to_numpy(dtype=np.int64), expected_time):
            raise ValueError(f"metadata time order mismatch for video {video_id}")
        if not np.array_equal(meta["feature_row"].to_numpy(dtype=np.int64), expected_time):
            raise ValueError(f"metadata feature order mismatch for video {video_id}")
        if not np.array_equal(meta["label_idx"].to_numpy(dtype=np.int64), labels):
            raise ValueError(f"metadata label order mismatch for video {video_id}")

        cross_check = per_second[per_second["video_id"].eq(video_id)].sort_values("t_sec")
        if len(cross_check) != len(labels):
            raise ValueError(f"per-second table length mismatch for video {video_id}")
        if not np.array_equal(cross_check["t_sec"].to_numpy(dtype=np.int64), expected_time):
            raise ValueError(f"per-second table time order mismatch for video {video_id}")
        if not np.array_equal(cross_check["true_label_idx"].to_numpy(dtype=np.int64), labels):
            raise ValueError(f"per-second table label mismatch for video {video_id}")
        if not np.array_equal(cross_check["pred_label_idx"].to_numpy(dtype=np.int64), saved_prediction):
            raise ValueError(f"per-second table prediction mismatch for video {video_id}")
        if not np.allclose(
            cross_check[probability_columns].to_numpy(dtype=np.float64),
            saved_probabilities,
            atol=2e-7,
            rtol=1e-6,
        ):
            raise ValueError(f"per-second probability mismatch for video {video_id}")

        frames.append(
            pd.DataFrame(
                {
                    "training_seed": training_seed,
                    "split": split,
                    "video_id": video_id,
                    "t_sec": expected_time,
                    "true_label_idx": labels,
                }
            )
        )
        all_logits.append(logits)
        all_labels.append(labels)
        sources.extend(
            [
                _source_record(
                    prediction_path,
                    "original_raw_logits_npz",
                    training_seed=training_seed,
                    split=split,
                    video_id=video_id,
                ),
                _source_record(
                    label_path,
                    "feature_labels",
                    training_seed=training_seed,
                    split=split,
                    video_id=video_id,
                ),
                _source_record(
                    meta_path,
                    "feature_metadata",
                    training_seed=training_seed,
                    split=split,
                    video_id=video_id,
                ),
            ]
        )

    frame_table = pd.concat(frames, ignore_index=True)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if len(frame_table) != len(logits):
        raise AssertionError("frame provenance and logits became misaligned")
    return frame_table, logits, labels, sources
