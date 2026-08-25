"""Shared multiclass calibration metrics and temperature scaling.

Temperature scaling is fitted on validation probabilities only.  Existing
softmax probabilities are sufficient because ``softmax(log(p) / T)`` is
equivalent to scaling the original logits up to their irrelevant additive
constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


PROBABILITY_EPSILON = 1e-12


def validate_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [n_samples, n_classes]")
    if labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels must be one-dimensional and aligned with probabilities")
    if len(probabilities) == 0:
        raise ValueError("calibration metrics require at least one sample")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if (probabilities < 0).any():
        raise ValueError("probabilities cannot be negative")
    row_sums = probabilities.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("every probability row must have positive mass")
    probabilities = probabilities / row_sums[:, None]
    if labels.min() < 0 or labels.max() >= probabilities.shape[1]:
        raise ValueError("labels fall outside the probability columns")
    return probabilities, labels


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.log(np.clip(probabilities, PROBABILITY_EPSILON, 1.0))
    scaled_logits = log_probabilities / float(temperature)
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled_logits)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def multiclass_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities, labels = validate_probabilities(probabilities, labels)
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true_probabilities, PROBABILITY_EPSILON, 1.0)).mean())


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities, labels = validate_probabilities(probabilities, labels)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.square(probabilities - one_hot).sum(axis=1).mean())


def reliability_table(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> pd.DataFrame:
    probabilities, labels = validate_probabilities(probabilities, labels)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    # Right-closed final bin includes confidence exactly equal to one.
    bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for index in range(n_bins):
        member = bin_index == index
        count = int(member.sum())
        lower = index / n_bins
        upper = (index + 1) / n_bins
        mean_confidence = float(confidence[member].mean()) if count else np.nan
        accuracy = float(correct[member].mean()) if count else np.nan
        gap = abs(accuracy - mean_confidence) if count else np.nan
        rows.append(
            {
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "fraction": count / len(labels),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    table = reliability_table(probabilities, labels, n_bins=n_bins)
    populated = table["count"].gt(0)
    return float(
        (table.loc[populated, "fraction"] * table.loc[populated, "absolute_gap"]).sum()
    )


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    ece_bins: tuple[int, ...] = (10, 15, 20),
) -> dict[str, float]:
    probabilities, labels = validate_probabilities(probabilities, labels)
    prediction = probabilities.argmax(axis=1)
    metrics = {
        "nll": multiclass_nll(probabilities, labels),
        "brier": multiclass_brier(probabilities, labels),
        "accuracy": float((prediction == labels).mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }
    for n_bins in ece_bins:
        metrics[f"ece_{n_bins}_bins"] = expected_calibration_error(
            probabilities,
            labels,
            n_bins=n_bins,
        )
    return metrics


def fit_temperature(
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    lower: float = 0.05,
    upper: float = 20.0,
) -> dict[str, float | bool | int | str]:
    probabilities, labels = validate_probabilities(
        validation_probabilities,
        validation_labels,
    )
    if not (0 < lower < upper):
        raise ValueError("temperature bounds must satisfy 0 < lower < upper")

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        return multiclass_nll(temperature_scale(probabilities, temperature), labels)

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    temperature = float(np.exp(result.x))
    return {
        "temperature": temperature,
        "validation_nll_before": multiclass_nll(probabilities, labels),
        "validation_nll_after": multiclass_nll(
            temperature_scale(probabilities, temperature),
            labels,
        ),
        "success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_evaluations": int(result.nfev),
        "lower_bound": lower,
        "upper_bound": upper,
    }
