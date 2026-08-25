import argparse
import re
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def video_id_from_name(path: Path) -> str:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        raise ValueError(f"Cannot extract video id from: {path.name}")
    return f"{int(nums[-1]):02d}"


def get_split(video_id: str) -> str:
    vid = int(video_id)
    if 1 <= vid <= 10:
        return "train"
    if 11 <= vid <= 14:
        return "val"
    if 15 <= vid <= 21:
        return "test"
    return "unknown"


def read_label_file(label_path: Path) -> pd.DataFrame:
    df = pd.read_csv(label_path, sep=r"\s+", engine="python")
    if "Frame" not in df.columns or "Phase" not in df.columns:
        raise ValueError(f"Bad label format: {label_path}")
    df["Frame"] = df["Frame"].astype(int)
    df["Phase"] = df["Phase"].astype(int)
    return df


def extract_video_frames(
    video_id: str,
    video_path: Path,
    label_df: pd.DataFrame,
    frames_root: Path,
    image_size: int,
    jpeg_quality: int,
    force: bool,
):
    out_dir = frames_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_int = int(round(fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps_int <= 0:
        raise RuntimeError(f"Invalid FPS for {video_path}: {fps}")

    rows = []
    current_frame_idx = -1

    phases = label_df["Phase"].tolist()
    label_frames = label_df["Frame"].tolist()

    for i, (label_frame, phase) in enumerate(
        tqdm(
            list(zip(label_frames, phases)),
            desc=f"Extracting video {video_id}",
            leave=False,
        )
    ):
        # label frame is 1-based at 1 fps.
        # Frame 1 -> t=0 sec -> original frame 0.
        t_sec = int(label_frame) - 1
        target_orig_frame_idx = t_sec * fps_int

        img_name = f"{int(label_frame):06d}.jpg"
        img_path = out_dir / img_name

        if target_orig_frame_idx >= total_frames:
            print(
                f"[WARN] video {video_id}: target frame {target_orig_frame_idx} "
                f">= total frames {total_frames}. Skipping label frame {label_frame}."
            )
            continue

        if force or not img_path.exists():
            while current_frame_idx < target_orig_frame_idx:
                ok = cap.grab()
                if not ok:
                    raise RuntimeError(
                        f"Failed to grab frame {current_frame_idx + 1} "
                        f"from video {video_id}"
                    )
                current_frame_idx += 1

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Failed to retrieve target frame {target_orig_frame_idx} "
                    f"from video {video_id}"
                )

            if image_size > 0:
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)

            cv2.imwrite(
                str(img_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
        else:
            # Need to advance capture even if file exists, otherwise future retrieval breaks.
            while current_frame_idx < target_orig_frame_idx:
                ok = cap.grab()
                if not ok:
                    raise RuntimeError(
                        f"Failed to grab frame {current_frame_idx + 1} "
                        f"from video {video_id}"
                    )
                current_frame_idx += 1

        prev_phase = phases[i - 1] if i > 0 else phase
        boundary = int(phase != prev_phase)

        rel_path = img_path.as_posix()

        rows.append(
            {
                "video_id": video_id,
                "split": get_split(video_id),
                "label_frame": int(label_frame),
                "t_sec": int(t_sec),
                "phase": int(phase),
                "label": int(phase),
                "label_idx": int(phase) - 1,
                "boundary": boundary,
                "orig_frame_idx": int(target_orig_frame_idx),
                "frame_path": rel_path,
            }
        )

    cap.release()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="data")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    video_dir = data_root / "videos"
    label_dir = data_root / "labels"

    frames_root = out_root / "frames_1fps"
    manifest_dir = out_root / "manifests"

    frames_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_dir.glob("*.mp4"))
    label_files = sorted(label_dir.glob("*.txt"))

    videos = {video_id_from_name(p): p for p in video_files}
    labels = {video_id_from_name(p): p for p in label_files}

    all_ids = sorted(set(videos.keys()) | set(labels.keys()))

    all_rows = []

    print(f"Found {len(videos)} videos")
    print(f"Found {len(labels)} labels")
    print(f"Output frames root: {frames_root}")
    print(f"Output manifest root: {manifest_dir}")
    print()

    for vid in all_ids:
        if vid not in videos:
            print(f"[ERROR] Missing video for {vid}")
            continue
        if vid not in labels:
            print(f"[ERROR] Missing label for {vid}")
            continue

        label_df = read_label_file(labels[vid])
        rows = extract_video_frames(
            video_id=vid,
            video_path=videos[vid],
            label_df=label_df,
            frames_root=frames_root,
            image_size=args.image_size,
            jpeg_quality=args.jpeg_quality,
            force=args.force,
        )
        all_rows.extend(rows)

        print(
            f"Video {vid}: labels={len(label_df)}, extracted/manifest rows={len(rows)}, "
            f"split={get_split(vid)}"
        )

    manifest = pd.DataFrame(all_rows)
    manifest = manifest.sort_values(["video_id", "label_frame"]).reset_index(drop=True)

    manifest.to_csv(manifest_dir / "all.csv", index=False)

    for split in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split].copy()
        split_df.to_csv(manifest_dir / f"{split}.csv", index=False)

    print()
    print("Saved manifests:")
    print(f"  {manifest_dir / 'all.csv'}")
    print(f"  {manifest_dir / 'train.csv'}")
    print(f"  {manifest_dir / 'val.csv'}")
    print(f"  {manifest_dir / 'test.csv'}")

    print()
    print("Manifest summary:")
    print(manifest.groupby("split")["video_id"].nunique())
    print()
    print(manifest.groupby("split").size())

    print()
    print("Phase distribution:")
    print(manifest["phase"].value_counts().sort_index())

    print()
    print("Boundary count by split:")
    print(manifest.groupby("split")["boundary"].sum())


if __name__ == "__main__":
    main()