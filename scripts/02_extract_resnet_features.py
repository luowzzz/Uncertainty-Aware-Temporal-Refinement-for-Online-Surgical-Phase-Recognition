import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class FrameDataset(Dataset):
    def __init__(self, rows, project_root: Path, transform):
        self.rows = rows.reset_index(drop=True)
        self.project_root = project_root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        img_path = self.project_root / row["frame_path"]

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        return x, int(row["label_idx"]), int(row["t_sec"])


def build_resnet18(device):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # remove final classifier, keep pooled 512-d feature
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    transform = weights.transforms()
    return model, transform


@torch.no_grad()
def extract_one_video(
    video_id: str,
    video_df: pd.DataFrame,
    project_root: Path,
    out_dir: Path,
    model,
    transform,
    device,
    batch_size: int,
    num_workers: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    out_feature_path = out_dir / f"{video_id}.npy"
    out_label_path = out_dir / f"{video_id}_labels.npy"
    out_meta_path = out_dir / f"{video_id}_meta.csv"

    dataset = FrameDataset(video_df, project_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    features = []
    labels = []
    times = []

    for x, y, t in tqdm(loader, desc=f"Features {video_id}", leave=False):
        x = x.to(device, non_blocking=True)
        feat = model(x)

        features.append(feat.cpu().numpy())
        labels.append(y.numpy())
        times.append(t.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    times = np.concatenate(times, axis=0)

    np.save(out_feature_path, features.astype(np.float32))
    np.save(out_label_path, labels.astype(np.int64))

    meta = video_df.copy()
    meta["feature_row"] = np.arange(len(meta))
    meta.to_csv(out_meta_path, index=False)

    return features.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/manifests/all.csv")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--out_dir", type=str, default="data/features/resnet18")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    manifest_path = project_root / args.manifest
    out_dir = project_root / args.out_dir

    df = pd.read_csv(manifest_path, dtype={"video_id": str})
    df["video_id"] = df["video_id"].str.zfill(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, transform = build_resnet18(device)

    video_ids = sorted(df["video_id"].unique().tolist())
    print(f"Found {len(video_ids)} videos in manifest")

    summary_rows = []

    for vid in video_ids:
        video_df = df[df["video_id"] == vid].sort_values("t_sec").reset_index(drop=True)

        out_feature_path = out_dir / f"{vid}.npy"
        if out_feature_path.exists() and not args.force:
            arr = np.load(out_feature_path, mmap_mode="r")
            print(f"Skip video {vid}: existing {out_feature_path}, shape={arr.shape}")
            summary_rows.append({
                "video_id": vid,
                "n_frames": arr.shape[0],
                "feature_dim": arr.shape[1],
                "status": "skipped_existing",
            })
            continue

        shape = extract_one_video(
            video_id=vid,
            video_df=video_df,
            project_root=project_root,
            out_dir=out_dir,
            model=model,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        print(f"Video {vid}: feature shape {shape}")

        summary_rows.append({
            "video_id": vid,
            "n_frames": shape[0],
            "feature_dim": shape[1],
            "status": "extracted",
        })

    summary = pd.DataFrame(summary_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "feature_summary.csv", index=False)

    print()
    print("Saved feature summary:")
    print(out_dir / "feature_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()