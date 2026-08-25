import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from reproducibility_utils import ensure_fresh_output_dir, set_inference_seed, sha256_file
from selective_prediction_metrics import (
    DEFAULT_KEY_COVERAGES,
    empirical_risk_coverage,
    exact_aurc,
    key_coverage_points,
)


class TemporalPhaseModel(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        num_layers=1,
        num_classes=7,
        dropout=0.3,
        bidirectional=True,
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
        logits = self.classifier(h)
        return logits


def get_split_video_ids(split):
    if split == "train":
        return [f"{i:02d}" for i in range(1, 11)]
    if split == "val":
        return [f"{i:02d}" for i in range(11, 15)]
    if split == "test":
        return [f"{i:02d}" for i in range(15, 22)]
    raise ValueError(split)


def entropy(probs, eps=1e-12):
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def extract_boundaries(labels):
    labels = np.asarray(labels)
    if len(labels) <= 1:
        return np.array([], dtype=int)
    return np.where(labels[1:] != labels[:-1])[0] + 1


def distance_to_nearest_boundary(n, boundaries):
    if len(boundaries) == 0:
        return np.full(n, np.inf)

    t = np.arange(n)
    dist = np.full(n, np.inf)
    for b in boundaries:
        dist = np.minimum(dist, np.abs(t - b))
    return dist


def enable_dropout_only(model):
    """
    Keep model in eval mode, but activate dropout modules.
    This avoids changing LSTM/other eval behavior while enabling MC Dropout.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    args = ckpt.get("args", {})
    hidden_dim = int(args.get("hidden_dim", 256))
    num_layers = int(args.get("num_layers", 1))
    dropout = float(args.get("dropout", 0.3))
    # Old checkpoints from 03_train_bilstm_baseline.py did not store this flag
    # and were bidirectional by construction.
    bidirectional = bool(args.get("bidirectional", True))

    model = TemporalPhaseModel(
        input_dim=512,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=7,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model


@torch.no_grad()
def mc_predict_video(model, x_np, device, T):
    x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(device)

    prob_samples = []

    enable_dropout_only(model)

    for _ in range(T):
        logits = model(x).squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        prob_samples.append(probs)

    prob_samples = np.stack(prob_samples, axis=0)  # [T, N, C]

    mean_probs = prob_samples.mean(axis=0)
    pred = mean_probs.argmax(axis=-1)
    confidence = mean_probs.max(axis=-1)

    predictive_entropy = entropy(mean_probs)
    expected_entropy = entropy(prob_samples).mean(axis=0)
    mutual_info = predictive_entropy - expected_entropy

    sampled_preds = prob_samples.argmax(axis=-1)  # [T, N]
    variation_ratio = []

    for i in range(sampled_preds.shape[1]):
        counts = np.bincount(sampled_preds[:, i], minlength=7)
        variation_ratio.append(1.0 - counts.max() / float(T))

    variation_ratio = np.asarray(variation_ratio, dtype=np.float32)

    return {
        "mean_probs": mean_probs,
        "pred": pred,
        "confidence": confidence,
        "mc_entropy": predictive_entropy,
        "mc_expected_entropy": expected_entropy,
        "mc_mutual_info": mutual_info,
        "mc_variation_ratio": variation_ratio,
        "mc_1_minus_conf": 1.0 - confidence,
    }


def evaluate_splits(model, features_dir, splits, device, T, out_dir, inference_seed):
    rows = []
    split_metrics = []
    runtime_rows = []

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        # Make each split reproducible in isolation.  Test predictions must not
        # depend on whether validation happened to be evaluated first.
        set_inference_seed(inference_seed)
        video_ids = get_split_video_ids(split)

        split_true = []
        split_pred = []

        for vid in video_ids:
            x_np = np.load(features_dir / f"{vid}.npy").astype(np.float32)
            y_np = np.load(features_dir / f"{vid}_labels.npy").astype(np.int64)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            result = mc_predict_video(model, x_np, device, T)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_started
            runtime_rows.append(
                {
                    "split": split,
                    "video_id": vid,
                    "n_frames": len(y_np),
                    "T": T,
                    "inference_seconds": inference_seconds,
                    "milliseconds_per_frame_pass": (
                        1000.0 * inference_seconds / (len(y_np) * T)
                    ),
                }
            )

            pred = result["pred"]
            mean_probs = result["mean_probs"]

            np.savez_compressed(
                pred_dir / f"{split}_video_{vid}_mc_predictions.npz",
                video_id=vid,
                true_label_idx=y_np,
                pred_label_idx=pred,
                mean_probs=mean_probs,
                confidence=result["confidence"],
                mc_entropy=result["mc_entropy"],
                mc_expected_entropy=result["mc_expected_entropy"],
                mc_mutual_info=result["mc_mutual_info"],
                mc_variation_ratio=result["mc_variation_ratio"],
                mc_1_minus_conf=result["mc_1_minus_conf"],
            )

            correct = (pred == y_np).astype(int)
            error = 1 - correct

            gt_boundaries = extract_boundaries(y_np)
            pred_boundaries = extract_boundaries(pred)

            dist_gt = distance_to_nearest_boundary(len(y_np), gt_boundaries)
            dist_pred = distance_to_nearest_boundary(len(y_np), pred_boundaries)

            for t in range(len(y_np)):
                rows.append({
                    "split": split,
                    "video_id": vid,
                    "t_sec": t,
                    "true_label_idx": int(y_np[t]),
                    "true_phase": int(y_np[t] + 1),
                    "pred_label_idx": int(pred[t]),
                    "pred_phase": int(pred[t] + 1),
                    "correct": int(correct[t]),
                    "error": int(error[t]),
                    "confidence": float(result["confidence"][t]),
                    "mc_1_minus_conf": float(result["mc_1_minus_conf"][t]),
                    "mc_entropy": float(result["mc_entropy"][t]),
                    "mc_expected_entropy": float(result["mc_expected_entropy"][t]),
                    "mc_mutual_info": float(result["mc_mutual_info"][t]),
                    "mc_variation_ratio": float(result["mc_variation_ratio"][t]),
                    "dist_to_gt_boundary": float(dist_gt[t]),
                    "dist_to_pred_boundary": float(dist_pred[t]),
                })

            split_true.append(y_np)
            split_pred.append(pred)

            print(f"{split} video {vid}: done")

        split_true = np.concatenate(split_true)
        split_pred = np.concatenate(split_pred)

        split_metrics.append({
            "split": split,
            "accuracy": float(accuracy_score(split_true, split_pred)),
            "macro_f1": float(f1_score(split_true, split_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(split_true, split_pred, average="weighted", zero_division=0)),
        })

    scores_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(split_metrics)
    runtime_df = pd.DataFrame(runtime_rows)

    scores_df.to_csv(out_dir / "mc_dropout_frame_scores.csv", index=False)
    metrics_df.to_csv(out_dir / "mc_dropout_phase_metrics.csv", index=False)
    runtime_df.to_csv(out_dir / "mc_dropout_runtime_by_video.csv", index=False)

    return scores_df, metrics_df, runtime_df


def per_video_metrics(df, score_cols):
    """Long-form classification, error-detection and Exact AURC by video."""
    rows = []
    for (split, video_id), video in df.groupby(["split", "video_id"], sort=True):
        labels = video["true_label_idx"].to_numpy(dtype=int)
        predictions = video["pred_label_idx"].to_numpy(dtype=int)
        errors = video["error"].to_numpy(dtype=int)
        for score in score_cols:
            values = video[score].to_numpy(dtype=float)
            if np.unique(errors).size == 2:
                auroc = float(roc_auc_score(errors, values))
                aupr = float(average_precision_score(errors, values))
            else:
                auroc = np.nan
                aupr = np.nan
            curve = empirical_risk_coverage(errors, values)
            aurc = exact_aurc(curve)
            rows.append(
                {
                    "split": split,
                    "video_id": str(video_id).zfill(2),
                    "n_frames": len(video),
                    "n_errors": int(errors.sum()),
                    "error_rate": float(errors.mean()),
                    "accuracy": float(accuracy_score(labels, predictions)),
                    "macro_f1": float(
                        f1_score(
                            labels,
                            predictions,
                            labels=list(range(7)),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "score": score,
                    "error_auroc": auroc,
                    "error_aupr": aupr,
                    "exact_aurc": float(aurc["aurc_lower_better"]),
                    "oracle_aurc": float(aurc["oracle_aurc"]),
                    "excess_aurc": float(aurc["excess_aurc"]),
                }
            )
    return pd.DataFrame(rows)


def near_far_analysis(df, windows, score_cols):
    rows = []

    for split, sdf in df.groupby("split"):
        for w in windows:
            near = sdf[sdf["dist_to_gt_boundary"] <= w]
            far = sdf[sdf["dist_to_gt_boundary"] > w]

            rows.append({
                "split": split,
                "window_sec": w,
                "near_n": len(near),
                "far_n": len(far),
                "near_error_rate": float(near["error"].mean()),
                "far_error_rate": float(far["error"].mean()),
                "error_gap_near_minus_far": float(near["error"].mean() - far["error"].mean()),
            })

            for score in score_cols:
                rows[-1][f"near_mean_{score}"] = float(near[score].mean())
                rows[-1][f"far_mean_{score}"] = float(far[score].mean())
                rows[-1][f"gap_near_minus_far_{score}"] = float(near[score].mean() - far[score].mean())

    return pd.DataFrame(rows)


def auc_analysis(df, windows, score_cols):
    rows = []

    for split, sdf in df.groupby("split"):
        for w in windows:
            boundary_near = (sdf["dist_to_gt_boundary"] <= w).astype(int).to_numpy()
            error = sdf["error"].astype(int).to_numpy()

            for score in score_cols:
                values = sdf[score].to_numpy()

                if len(np.unique(boundary_near)) == 2:
                    rows.append({
                        "split": split,
                        "window_sec": w,
                        "task": "detect_boundary_near",
                        "score": score,
                        "auroc": float(roc_auc_score(boundary_near, values)),
                        "aupr": float(average_precision_score(boundary_near, values)),
                    })

                if len(np.unique(error)) == 2:
                    rows.append({
                        "split": split,
                        "window_sec": w,
                        "task": "detect_error",
                        "score": score,
                        "auroc": float(roc_auc_score(error, values)),
                        "aupr": float(average_precision_score(error, values)),
                    })

    return pd.DataFrame(rows)


def risk_coverage(df, score_cols, coverages):
    full_frames = []
    key_frames = []
    aurc_rows = []
    for split, sdf in df.groupby("split"):
        for score in score_cols:
            curve = empirical_risk_coverage(
                sdf["error"].to_numpy(),
                sdf[score].to_numpy(),
            )
            curve.insert(0, "score", score)
            curve.insert(0, "split", split)
            curve["kept_frames"] = curve["kept_samples"]
            curve["accuracy"] = curve["accuracy_on_kept"]
            curve["full_risk"] = curve["full_coverage_error_rate"]
            full_frames.append(curve)

            key = key_coverage_points(curve, coverages)
            key["split"] = split
            key["score"] = score
            key["kept_frames"] = key["kept_samples"]
            key["accuracy"] = key["accuracy_on_kept"]
            key["full_risk"] = key["full_coverage_error_rate"]
            key_frames.append(key)
            aurc_rows.append(
                {
                    "split": split,
                    "score": score,
                    **exact_aurc(curve),
                }
            )

    return (
        pd.concat(full_frames, ignore_index=True),
        pd.concat(key_frames, ignore_index=True),
        pd.DataFrame(aurc_rows),
    )


def plot_test_risk_coverage(rc_df, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test = rc_df[rc_df["split"] == "test"].copy()

    plt.figure(figsize=(8, 6))

    for score in sorted(test["score"].unique()):
        g = test[test["score"] == score].sort_values("coverage")
        plt.plot(g["coverage"], g["risk_error_rate"], label=score)

    plt.xlabel("Coverage: fraction of frames kept")
    plt.ylabel("Risk: error rate on kept frames")
    plt.title("Test set: MC Dropout risk-coverage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "test_mc_dropout_risk_coverage.png", dpi=200)
    plt.close()


def plot_uncertainty_timeline(df, split, video_id, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(video_id).zfill(2)
    vdf = df[(df["split"] == split) & (df["video_id"] == video_id)].copy()

    if vdf.empty:
        print(f"[WARN] missing {split} video {video_id}")
        return

    vdf = vdf.sort_values("t_sec")

    t = vdf["t_sec"].to_numpy()
    true_phase = vdf["true_phase"].to_numpy()
    pred_phase = vdf["pred_phase"].to_numpy()

    boundaries = extract_boundaries(vdf["true_label_idx"].to_numpy())

    plt.figure(figsize=(18, 7))

    ax1 = plt.gca()
    ax1.plot(t, true_phase, label="GT phase", linewidth=1.8)
    ax1.plot(t, pred_phase, label="MC pred phase", linewidth=1.0, alpha=0.75)
    ax1.set_xlabel("Time / second")
    ax1.set_ylabel("Phase")
    ax1.set_yticks([1, 2, 3, 4, 5, 6, 7])

    for b in boundaries:
        ax1.axvline(b, linestyle="--", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(t, vdf["mc_entropy"], label="MC entropy", linewidth=1.0, alpha=0.65)
    ax2.plot(t, vdf["mc_mutual_info"], label="MC mutual information", linewidth=1.0, alpha=0.65)
    ax2.plot(t, vdf["mc_variation_ratio"], label="Variation ratio", linewidth=1.0, alpha=0.65)
    ax2.set_ylabel("Uncertainty score")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"{split} video {video_id}: MC Dropout uncertainty")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{split}_video_{video_id}_mc_uncertainty_timeline.png", dpi=200)
    plt.close()


def main():
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="data/features/resnet18")
    parser.add_argument("--checkpoint", type=str, default="outputs/baseline_bilstm/checkpoints/best.pt")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/baseline_bilstm/mc_dropout_reproducible_v1",
    )
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument(
        "--training_seed",
        type=int,
        default=None,
        help="Training seed; inferred from checkpoint and validated when provided.",
    )
    parser.add_argument(
        "--inference_seed",
        type=int,
        default=0,
        help="Fixed RNG seed for stochastic MC Dropout forward passes.",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument(
        "--mc_config",
        type=Path,
        default=None,
        help="Optional frozen MC configuration whose T and inference seed must match.",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Explicitly allow replacement of existing MC Dropout result files.",
    )
    args = parser.parse_args()

    if args.T < 2:
        raise ValueError("MC Dropout requires at least two stochastic passes")
    invalid_splits = sorted(set(args.splits) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"Unknown splits: {invalid_splits}")

    features_dir = Path(args.features_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {features_dir}")

    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    checkpoint_args = checkpoint_payload.get("args", {})
    inferred_training_seed = int(checkpoint_args.get("seed", -1))
    training_seed = (
        inferred_training_seed if args.training_seed is None else args.training_seed
    )
    if inferred_training_seed >= 0 and training_seed != inferred_training_seed:
        raise ValueError(
            f"Requested training seed {training_seed} does not match checkpoint "
            f"seed {inferred_training_seed}"
        )
    mc_config_record = None
    if args.mc_config is not None:
        mc_config_path = args.mc_config.expanduser().resolve()
        if not mc_config_path.is_file():
            raise FileNotFoundError(mc_config_path)
        mc_config = json.loads(mc_config_path.read_text(encoding="utf-8"))
        expected_t = int(
            mc_config.get("selected_T", mc_config.get("inference", {}).get("passes"))
        )
        expected_inference_seed = int(
            mc_config.get(
                "inference_seed",
                mc_config.get("inference", {}).get("inference_seed"),
            )
        )
        if args.T != expected_t or args.inference_seed != expected_inference_seed:
            raise ValueError(
                "MC command does not match frozen configuration: "
                f"expected T={expected_t}, inference_seed={expected_inference_seed}; "
                f"got T={args.T}, inference_seed={args.inference_seed}"
            )
        mc_config_record = {
            "path": str(mc_config_path),
            "sha256": sha256_file(mc_config_path),
            "config_id": mc_config.get("config_id"),
        }

    output_files = [
        out_dir / "mc_dropout_frame_scores.csv",
        out_dir / "mc_dropout_phase_metrics.csv",
        out_dir / "mc_dropout_near_far_boundary_table.csv",
        out_dir / "mc_dropout_auc_summary.csv",
        out_dir / "mc_dropout_risk_coverage.csv",
        out_dir / "mc_dropout_risk_coverage_full.csv",
        out_dir / "mc_dropout_risk_coverage_aurc.csv",
        out_dir / "mc_dropout_metrics_by_video.csv",
        out_dir / "mc_dropout_runtime_by_video.csv",
    ]
    out_dir = ensure_fresh_output_dir(out_dir, args.allow_overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_inference_seed(args.inference_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"MC samples T={args.T}")
    print(f"Inference seed={args.inference_seed}")

    model = load_model(checkpoint, device)

    scores_df, metrics_df, runtime_df = evaluate_splits(
        model=model,
        features_dir=features_dir,
        splits=args.splits,
        device=device,
        T=args.T,
        out_dir=out_dir,
        inference_seed=args.inference_seed,
    )

    score_cols = [
        "mc_entropy",
        "mc_1_minus_conf",
        "mc_mutual_info",
        "mc_variation_ratio",
    ]
    per_video_df = per_video_metrics(scores_df, score_cols)

    near_far_df = near_far_analysis(scores_df, args.windows, score_cols)
    auc_df = auc_analysis(scores_df, args.windows, score_cols)

    rc_full_df, rc_df, aurc_df = risk_coverage(
        scores_df,
        score_cols,
        DEFAULT_KEY_COVERAGES,
    )

    near_far_df.to_csv(out_dir / "mc_dropout_near_far_boundary_table.csv", index=False)
    auc_df.to_csv(out_dir / "mc_dropout_auc_summary.csv", index=False)
    rc_df.to_csv(out_dir / "mc_dropout_risk_coverage.csv", index=False)
    rc_full_df.to_csv(out_dir / "mc_dropout_risk_coverage_full.csv", index=False)
    aurc_df.to_csv(out_dir / "mc_dropout_risk_coverage_aurc.csv", index=False)
    per_video_df.to_csv(out_dir / "mc_dropout_metrics_by_video.csv", index=False)

    plot_test_risk_coverage(rc_full_df, out_dir)
    plot_uncertainty_timeline(scores_df, "test", "15", out_dir)

    elapsed_seconds = time.perf_counter() - run_started
    manifest = {
        "schema_version": 2,
        "analysis": "mc_dropout_evaluation",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "inputs": {
            "features_dir": str(features_dir),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "training_seed": training_seed,
            "checkpoint_epoch": int(checkpoint_payload["epoch"]),
            "checkpoint_validation_macro_f1": float(
                checkpoint_payload["best_val_macro_f1"]
            ),
            "checkpoint_selection_split": "val",
            "checkpoint_selection_metric": "macro_f1",
            "mc_config": mc_config_record,
        },
        "protocol": {
            "mc_passes": args.T,
            "inference_seed": args.inference_seed,
            "rng_scope": "reset_once_at_the_start_of_each_split",
            "splits": args.splits,
            "boundary_windows_seconds": args.windows,
            "dropout_mode": "model_eval_with_nn_dropout_modules_enabled",
            "error_definition": "Incorrect MC-averaged argmax prediction is positive.",
            "risk_coverage_definition": (
                "Exact empirical coverage k/N for k=1..N with expected risk "
                "within exact score ties."
            ),
            "aurc_definition": (
                "Mean empirical selective risk over all retained counts k=1..N."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "determinism_note": (
                "The RNG seed is fixed. Exact cross-platform or cross-version bitwise "
                "reproducibility is not guaranteed for all accelerator kernels."
            ),
        },
        "runtime": {
            "wall_clock_seconds": elapsed_seconds,
            "mc_inference_seconds": float(runtime_df["inference_seconds"].sum()),
            "timing_definition": (
                "Per-video T-pass inference including probability transfer and MC "
                "score construction, measured with CUDA synchronisation."
            ),
        },
        "code_dependencies": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in (
                Path(__file__).resolve().parent / "selective_prediction_metrics.py",
                Path(__file__).resolve().parent / "reproducibility_utils.py",
            )
        ],
        "outputs": [str(path) for path in output_files],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("MC Dropout phase metrics:")
    print(metrics_df.to_string(index=False))

    print()
    print("MC Dropout near/far boundary table:")
    print(near_far_df.to_string(index=False))

    print()
    print("MC Dropout AUC summary:")
    print(auc_df.to_string(index=False))

    print()
    print("MC Dropout risk-coverage key points:")
    key = rc_df[rc_df["coverage"].isin([1.0, 0.8, 0.6, 0.5])].copy()
    print(key[[
        "split",
        "score",
        "coverage",
        "risk_error_rate",
        "accuracy",
        "risk_reduction_vs_full",
    ]].to_string(index=False))

    print()
    print("Saved:")
    print(out_dir / "mc_dropout_frame_scores.csv")
    print(out_dir / "mc_dropout_phase_metrics.csv")
    print(out_dir / "mc_dropout_near_far_boundary_table.csv")
    print(out_dir / "mc_dropout_auc_summary.csv")
    print(out_dir / "mc_dropout_risk_coverage.csv")
    print(out_dir / "mc_dropout_metrics_by_video.csv")
    print(out_dir / "mc_dropout_runtime_by_video.csv")
    print(out_dir / "run_manifest.json")
    print(out_dir / "figures")


if __name__ == "__main__":
    main()
