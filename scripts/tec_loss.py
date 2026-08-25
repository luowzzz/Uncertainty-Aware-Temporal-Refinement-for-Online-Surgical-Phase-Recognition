"""Temporal Error-Cascade (TEC) loss used for the RQ2 training ablation.

The frame weights follow equations (1)-(2) of Liu et al.,
"Stabilizing Temporal Inference Dynamics for Online Surgical Phase
Recognition", arXiv:2605.16387v1 (2026).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def temporal_error_cascade_weights(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 19.0,
    sigma: float = 1.5,
    onset_window: int = 8,
) -> torch.Tensor:
    """Return non-differentiable per-frame TEC weights.

    ``predictions`` and ``targets`` must be one-dimensional aligned class
    indices.  Each maximal error run receives a front-loaded Gaussian
    upweighting over its first ``onset_window`` frames.
    """
    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("predictions and targets must be one-dimensional")
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be aligned")
    if alpha < 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    if isinstance(onset_window, bool) or onset_window <= 0:
        raise ValueError("onset_window must be a positive integer")

    error = predictions.detach().ne(targets.detach())
    weights = torch.ones(
        len(targets),
        dtype=torch.float32,
        device=targets.device,
    )
    index = 0
    while index < len(error):
        if not bool(error[index]):
            index += 1
            continue
        start = index
        while index < len(error) and bool(error[index]):
            index += 1
        length = index - start
        count = min(length, int(onset_window))
        offsets = torch.arange(count, dtype=torch.float32, device=targets.device)
        weights[start : start + count] += float(alpha) * torch.exp(
            -(offsets**2) / (2.0 * float(sigma) ** 2)
        )
    return weights


def temporal_error_cascade_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    alpha: float = 19.0,
    sigma: float = 1.5,
    onset_window: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute class-balanced per-frame CE multiplied by TEC onset weights."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [time, classes]")
    if targets.ndim != 1 or len(targets) != len(logits):
        raise ValueError("targets must be aligned with logits")
    predictions = logits.detach().argmax(dim=-1)
    weights = temporal_error_cascade_weights(
        predictions,
        targets,
        alpha=alpha,
        sigma=sigma,
        onset_window=onset_window,
    )
    per_frame = F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
    )
    return torch.mean(weights * per_frame), weights


__all__ = [
    "temporal_error_cascade_loss",
    "temporal_error_cascade_weights",
]
