import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PHASE_NAMES = {
    0: "Preparation",
    1: "Dividing Ligament and Peritoneum",
    2: "Dividing Uterine Vessels and Ligament",
    3: "Transecting the Vagina",
    4: "Specimen Removal",
    5: "Suturing",
    6: "Washing",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_split_video_ids(split: str):
    if split == "train":
        return [f"{i:02d}" for i in range(1, 11)]
    if split == "val":
        return [f"{i:02d}" for i in range(11, 15)]
    if split == "test":
        return [f"{i:02d}" for i in range(15, 22)]
    raise ValueError(split)


def load_video_features(features_dir: Path, video_id: str):
    x = np.load(features_dir / f"{video_id}.npy").astype(np.float32)
    y = np.load(features_dir / f"{video_id}_labels.npy").astype(np.int64)
    if len(x) != len(y):
        raise ValueError(f"Length mismatch for video {video_id}: x={len(x)}, y={len(y)}")
    return x, y


def compute_class_weights(features_dir: Path, train_ids):
    labels = []
    for vid in train_ids:
        _, y = load_video_features(features_dir, vid)
        labels.append(y)
    labels = np.concatenate(labels)
    counts = np.bincount(labels, minlength=7).astype(np.float32)
    total = counts.sum()
    weights = total / (7.0 * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return weights, counts


class TemporalPhaseModel(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        num_layers=1,
        num_classes=7,
        dropout=0.3,
        bidirectional=False,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        direction_factor = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * direction_factor, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        h, _ = self.lstm(x)
        return self.classifier(h)


def run_epoch(model, features_dir, video_ids, criterion, optimizer, device, train=True):
    if train:
        model.train()
        random.shuffle(video_ids)
    else:
        model.eval()

    losses = []
    all_true = []
    all_pred = []

    for vid in tqdm(video_ids, desc="train" if train else "eval", leave=False):
        x_np, y_np = load_video_features(features_dir, vid)
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        y = torch.from_numpy(y_np).unsqueeze(0).to(device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        pred = logits.argmax(dim=-1).squeeze(0).detach().cpu().numpy()
        losses.append(float(loss.item()))
        all_true.append(y_np)
        all_pred.append(pred)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
    }


@torch.no_grad()
def predict_split(model, features_dir, video_ids, device, out_dir: Path, split: str):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_true = []
    all_pred = []
    all_video = []
    all_t = []

    for vid in tqdm(video_ids, desc=f"predict {split}"):
        x_np, y_np = load_video_features(features_dir, vid)
        x = torch.from_numpy(x_np).unsqueeze(0).to(device)
        logits = model(x).squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = probs.argmax(axis=-1)
        confidence = probs.max(axis=-1)

        np.savez_compressed(
            out_dir / f"{split}_video_{vid}_predictions.npz",
            video_id=vid,
            logits=logits.cpu().numpy(),
            probs=probs,
            pred_label_idx=pred,
            true_label_idx=y_np,
        )

        t = np.arange(len(y_np))
        pd.DataFrame({
            "video_id": vid,
            "t_sec": t,
            "true_label_idx": y_np,
            "true_phase": y_np + 1,
            "pred_label_idx": pred,
            "pred_phase": pred + 1,
            "confidence": confidence,
        }).to_csv(out_dir / f"{split}_video_{vid}_timeline.csv", index=False)

        all_true.append(y_np)
        all_pred.append(pred)
        all_video.extend([vid] * len(y_np))
        all_t.extend(t.tolist())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    pd.DataFrame({
        "video_id": all_video,
        "t_sec": all_t,
        "true_label_idx": all_true,
        "true_phase": all_true + 1,
        "pred_label_idx": all_pred,
        "pred_phase": all_pred + 1,
    }).to_csv(out_dir / f"{split}_all_predictions.csv", index=False)

    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(7)),
        target_names=[PHASE_NAMES[i] for i in range(7)],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_true, all_pred, labels=list(range(7)))

    with open(out_dir / f"{split}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(cm).to_csv(out_dir / f"{split}_confusion_matrix.csv", index=False)

    return {
        "split": split,
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
    }


def write_experiment_rationale(out_dir: Path, args, train_counts):
    rationale = {
        "experiment": "ResNet18 features plus temporal LSTM phase-recognition baseline",
        "why_resnet18_features": (
            "CNN/ResNet visual feature extraction followed by temporal modelling is a standard "
            "surgical phase-recognition baseline family, motivated by SV-RCNet and summarised "
            "in later work such as LoViT. ResNet18 is used as a lightweight frozen feature "
            "extractor for reproducibility and computational practicality on AutoLaparo-T1."
        ),
        "why_unidirectional_lstm": (
            "The main v2 baseline uses a unidirectional LSTM because online surgical workflow "
            "recognition should not use future frames at inference time. BiLSTM can still be "
            "used as an offline comparison, but it is not the preferred baseline for online-style "
            "reliability analysis."
        ),
        "evaluation_plan": [
            "phase recognition: accuracy, macro-F1, weighted-F1",
            "boundary behaviour: tolerance-based boundary precision/recall/F1",
            "reliability: entropy, confidence, near/far boundary error, risk-coverage",
        ],
        "args": vars(args),
        "train_class_counts_label_idx_0_to_6": train_counts.astype(int).tolist(),
    }
    with open(out_dir / "experiment_rationale.json", "w", encoding="utf-8") as f:
        json.dump(rationale, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/features/resnet18")
    parser.add_argument("--out_dir", type=str, default="outputs/v2_lstm_online_resnet18")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--no_class_weights", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ids = get_split_video_ids("train")
    val_ids = get_split_video_ids("val")
    test_ids = get_split_video_ids("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Bidirectional LSTM: {args.bidirectional}")

    class_weights, train_counts = compute_class_weights(features_dir, train_ids)
    print("Train class counts, label_idx 0-6:")
    print(train_counts.astype(int))
    print("Class weights:")
    print(class_weights)
    write_experiment_rationale(out_dir, args, train_counts)

    model = TemporalPhaseModel(
        input_dim=512,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=7,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, features_dir, train_ids.copy(), criterion, optimizer, device, train=True
        )
        val_metrics = run_epoch(
            model, features_dir, val_ids.copy(), criterion, optimizer, device, train=False
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} macroF1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} macroF1 {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "best_val_macro_f1": best_val_f1,
                    "epoch": epoch,
                    "model_class": "TemporalPhaseModel",
                },
                ckpt_dir / "best.pt",
            )
            print(f"  Saved best checkpoint: val macro-F1={best_val_f1:.4f}")

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    print()
    print("Loading best checkpoint...")
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    final_metrics = []
    for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        metrics = predict_split(model, features_dir, ids, device, pred_dir, split)
        final_metrics.append(metrics)
        print(
            f"{split}: acc={metrics['accuracy']:.4f}, "
            f"macroF1={metrics['macro_f1']:.4f}, weightedF1={metrics['weighted_f1']:.4f}"
        )

    pd.DataFrame(final_metrics).to_csv(out_dir / "final_metrics.csv", index=False)

    print()
    print("Saved:")
    print(out_dir / "experiment_rationale.json")
    print(out_dir / "training_history.csv")
    print(out_dir / "final_metrics.csv")
    print(pred_dir)


if __name__ == "__main__":
    main()
