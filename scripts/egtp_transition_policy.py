"""Evidence-Gated Transition Predictor (EGTP) for online phase recognition.

The implementation follows equations (3)-(6) of:

    Liu et al., "Stabilizing Temporal Inference Dynamics for Online Surgical
    Phase Recognition", arXiv:2605.16387v1, 2026.

The paper specifies a causal per-candidate log-probability evidence process,
one-sided hysteresis, and normalisation by ``sqrt(n) * running_std``.  It does
not specify the numerical treatment of the first observation or zero
variance.  This implementation therefore freezes the following minimal
convention:

* the first running standard deviation is 1.0;
* later population standard deviations use Welford's online update;
* a fixed numerical floor of 1e-6 prevents division by zero.

These are numerical conventions, not additional selection thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_EPSILON = 1e-8
DEFAULT_STD_FLOOR = 1e-6
DEFAULT_INITIAL_STD = 1.0


@dataclass(frozen=True)
class EGTPResult:
    predictions: np.ndarray
    trace: pd.DataFrame


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [time, classes]")
    if len(probabilities) == 0:
        raise ValueError("a video must contain at least one timestep")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("each probability row must sum to one")
    return probabilities


def _welford_update(
    count: int,
    mean: float,
    m2: float,
    value: float,
) -> tuple[int, float, float]:
    count += 1
    delta = value - mean
    mean += delta / count
    delta2 = value - mean
    m2 += delta * delta2
    return count, mean, m2


def apply_egtp(
    probabilities: np.ndarray,
    threshold_k: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
    std_floor: float = DEFAULT_STD_FLOOR,
    initial_std: float = DEFAULT_INITIAL_STD,
    dynamic_normalisation: bool = True,
    return_trace: bool = False,
) -> EGTPResult:
    """Apply causal EGTP independently to one video.

    Labels in the returned prediction array are one-based to match the
    project's phase and boundary evaluation tables.
    """
    probabilities = validate_probabilities(probabilities)
    if not np.isfinite(threshold_k) or threshold_k < 0.0:
        raise ValueError("threshold_k must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not np.isfinite(std_floor) or std_floor <= 0.0:
        raise ValueError("std_floor must be finite and positive")
    if not np.isfinite(initial_std) or initial_std <= 0.0:
        raise ValueError("initial_std must be finite and positive")

    n_steps, n_classes = probabilities.shape
    raw = probabilities.argmax(axis=1).astype(np.int64)
    output = np.empty(n_steps, dtype=np.int64)
    current = int(raw[0])
    output[0] = current

    evidence = np.zeros(n_classes, dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    means = np.zeros(n_classes, dtype=np.float64)
    m2 = np.zeros(n_classes, dtype=np.float64)
    trace_rows: list[dict[str, float | int | bool]] = []

    for time_index in range(1, n_steps):
        current_before = current
        z_scores = np.full(n_classes, -np.inf, dtype=np.float64)
        deltas = np.zeros(n_classes, dtype=np.float64)
        sigmas = np.full(n_classes, np.nan, dtype=np.float64)

        for candidate in range(n_classes):
            if candidate == current:
                continue
            delta = float(
                np.log(probabilities[time_index, candidate] + epsilon)
                - np.log(probabilities[time_index, current] + epsilon)
            )
            deltas[candidate] = delta
            evidence[candidate] = max(0.0, evidence[candidate] + delta)
            (
                counts[candidate],
                means[candidate],
                m2[candidate],
            ) = _welford_update(
                int(counts[candidate]),
                float(means[candidate]),
                float(m2[candidate]),
                delta,
            )
            if counts[candidate] == 1:
                sigma = float(initial_std)
            else:
                sigma = float(np.sqrt(max(m2[candidate] / counts[candidate], 0.0)))
            sigma = max(sigma, float(std_floor))
            sigmas[candidate] = sigma
            if dynamic_normalisation:
                denominator = np.sqrt(float(counts[candidate])) * sigma
                z_scores[candidate] = evidence[candidate] / denominator
            else:
                z_scores[candidate] = evidence[candidate]

        challenger = int(np.argmax(z_scores))
        best_score = float(z_scores[challenger])
        accepted = bool(best_score > threshold_k)
        if accepted:
            current = challenger
            evidence.fill(0.0)
            counts.fill(0)
            means.fill(0.0)
            m2.fill(0.0)
        output[time_index] = current

        if return_trace:
            trace_rows.append(
                {
                    "t_sec": time_index,
                    "raw_phase": int(raw[time_index] + 1),
                    "current_phase_before": int(current_before + 1),
                    "challenger_phase": int(challenger + 1),
                    "challenger_delta": float(deltas[challenger]),
                    "challenger_evidence": float(
                        0.0 if accepted else evidence[challenger]
                    ),
                    "challenger_sigma": float(sigmas[challenger]),
                    "challenger_score": best_score,
                    "threshold_k": float(threshold_k),
                    "accepted": accepted,
                    "output_phase": int(current + 1),
                }
            )

    return EGTPResult(
        predictions=output + 1,
        trace=pd.DataFrame(trace_rows),
    )


__all__ = [
    "DEFAULT_EPSILON",
    "DEFAULT_INITIAL_STD",
    "DEFAULT_STD_FLOOR",
    "EGTPResult",
    "apply_egtp",
    "validate_probabilities",
]
