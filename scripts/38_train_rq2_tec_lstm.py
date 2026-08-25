"""Train the three RQ2 LSTM copies with the paper-derived TEC loss.

This script intentionally reads only training videos 01-10 and validation
videos 11-14.  It never loads test videos.  Test inference is performed only
after the RQ2 validation protocol has been frozen.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from reproducibility_utils import ensure_fresh_output_dir, sha256_file
from tec_loss import temporal_error_cascade_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features" / "resnet18"
N_CLASSES = 7


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def video_ids(split: str) -> list[str]:
    if split == "train":
        return [f"{value:02d}" for value in range(1, 11)]
    if split == "val":
        return [f"{value:02d}" for value in range(11, 15)]
    raise ValueError("TEC training script is restricted to train and val")


def load_video(video_id: str) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(FEATURES_DIR / f"{video_id}.npy").astype(np.float32)
    labels = np.load(FEATURES_DIR / f"{video_id}_labels.npy").astype(np.int64)
    if len(features) != len(labels):
        raise ValueError(f"feature/label length mismatch for video {video_id}")
    return features, labels


def class_weights() -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate([load_video(item)[1] for item in video_ids("train")])
    counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1.0))
    weights /= weights.mean()
    return weights, counts


class TemporalPhaseModel(nn.Module):
    """Architecture-identical copy of the frozen baseline LSTM."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_CLASSES),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(values)
        hidden, _ = self.lstm(projected)
        return self.classifier(hidden)


def run_epoch(
    model: TemporalPhaseModel,
    split_ids: list[str],
    class_weight_tensor: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    alpha: float,
    sigma: float,
    onset_window: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    ids = split_ids.copy()
    if training:
        random.shuffle(ids)
    losses: list[float] = []
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    weight_means: list[float] = []
    for video_id in ids:
        features, labels = load_video(video_id)
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        y = torch.from_numpy(labels).to(device)
        with torch.set_grad_enabled(training):
            logits = model(x).squeeze(0)
            loss, tec_weights = temporal_error_cascade_loss(
                logits,
                y,
                class_weights=class_weight_tensor,
                alpha=alpha,
                sigma=sigma,
                onset_window=onset_window,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.item()))
        weight_means.append(float(tec_weights.mean().item()))
        truths.append(labels)
        predictions.append(logits.argmax(dim=-1).detach().cpu().numpy())
    truth = np.concatenate(truths)
    prediction = np.concatenate(predictions)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                prediction,
                labels=list(range(N_CLASSES)),
                average="weighted",
                zero_division=0,
            )
        ),
        "mean_tec_weight": float(np.mean(weight_means)),
    }


@torch.no_grad()
def save_validation_predictions(
    model: TemporalPhaseModel,
    out_dir: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    prediction_dir = out_dir / "validation_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for video_id in video_ids("val"):
        features, labels = load_video(video_id)
        logits = (
            model(torch.from_numpy(features).unsqueeze(0).to(device))
            .squeeze(0)
            .cpu()
            .numpy()
        )
        path = prediction_dir / f"val_video_{video_id}_predictions.npz"
        np.savez_compressed(
            path,
            video_id=video_id,
            logits=logits,
            true_label_idx=labels,
        )
        records.append(
            {
                "video_id": video_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "n_frames": len(labels),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--tec-alpha", type=float, default=19.0)
    parser.add_argument("--tec-sigma", type=float, default=1.5)
    parser.add_argument("--tec-onset-window", type=int, default=8)
    args = parser.parse_args()

    out_dir = ensure_fresh_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights, counts = class_weights()
    class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    model = TemporalPhaseModel(args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_val_macro_f1 = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model,
            video_ids("train"),
            class_weight_tensor,
            optimizer,
            device,
            alpha=args.tec_alpha,
            sigma=args.tec_sigma,
            onset_window=args.tec_onset_window,
        )
        validation = run_epoch(
            model,
            video_ids("val"),
            class_weight_tensor,
            None,
            device,
            alpha=args.tec_alpha,
            sigma=args.tec_sigma,
            onset_window=args.tec_onset_window,
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train.items()},
                **{f"val_{key}": value for key, value in validation.items()},
            }
        )
        print(
            f"seed={args.seed} epoch={epoch:02d} "
            f"train_f1={train['macro_f1']:.4f} "
            f"val_f1={validation['macro_f1']:.4f} "
            f"val_loss={validation['loss']:.4f}"
        )
        if validation["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = validation["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "training_seed": args.seed,
                    "epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "architecture": {
                        "input_dim": 512,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": 1,
                        "dropout": args.dropout,
                        "bidirectional": False,
                        "num_classes": N_CLASSES,
                    },
                    "training": {
                        "epochs": args.epochs,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "loss": "class_weighted_temporal_error_cascade",
                        "tec_alpha": args.tec_alpha,
                        "tec_sigma": args.tec_sigma,
                        "tec_onset_window": args.tec_onset_window,
                    },
                },
                out_dir / "checkpoints" / "best.pt",
            )

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    checkpoint_path = out_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_records = save_validation_predictions(model, out_dir, device)
    manifest = {
        "schema_version": 1,
        "analysis": "rq2_tec_lstm_training_validation_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seed": args.seed,
        "data": {
            "train_video_ids": video_ids("train"),
            "validation_video_ids": video_ids("val"),
            "test_files_accessed": False,
            "feature_directory": str(FEATURES_DIR.resolve()),
        },
        "architecture_matches_frozen_baseline": True,
        "class_counts": counts.astype(int).tolist(),
        "class_weights": weights.astype(float).tolist(),
        "tec": {
            "paper": "arXiv:2605.16387v1",
            "alpha": args.tec_alpha,
            "sigma": args.tec_sigma,
            "onset_window": args.tec_onset_window,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "epoch": best_epoch,
            "validation_macro_f1": best_val_macro_f1,
            "selection_split": "val",
            "selection_metric": "macro_f1",
        },
        "validation_predictions": validation_records,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "training_seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_macro_f1,
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "test_files_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
