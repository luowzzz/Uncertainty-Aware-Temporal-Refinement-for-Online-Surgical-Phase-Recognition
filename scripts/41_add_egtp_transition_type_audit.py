"""Add a post-hoc transition-type audit to the frozen EGTP evidence.

This script does not train or run a neural network and does not select a method
or threshold. It deterministically replays Raw, Persistence-5 and the frozen
EGTP (k=0.6) from the already-saved baseline logits, verifies that the replayed
time matches equal the authoritative RQ2 event table, and then describes phase
type correctness conditional on those time matches.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from boundary_metrics import OPTIMAL_ORDERED, extract_boundaries, match_boundaries
from egtp_transition_policy import apply_egtp
from reproducibility_utils import PROJECT_ROOT, ensure_fresh_output_dir, sha256_file
from temperature_scaling_logits import load_raw_logit_split, softmax_logits


PROTOCOL = PROJECT_ROOT / "configs" / "rq2_egtp_transition_type_audit_v1.json"
OUT = PROJECT_ROOT / "outputs" / "rq2_egtp_transition_type_audit_v1"
BASELINE_RUNS = {
    seed: PROJECT_ROOT / "outputs" / f"v2_lstm_online_resnet18_seed{seed:02d}"
    for seed in (0, 1, 2)
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_record(record: dict[str, str]) -> Path:
    path = (PROJECT_ROOT / record["path"]).resolve()
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"frozen hash mismatch: {path}")
    return path


def causal_persistence(raw: np.ndarray, min_run: int = 5) -> np.ndarray:
    raw = np.asarray(raw, dtype=int)
    output = np.empty_like(raw)
    current = int(raw[0])
    candidate: int | None = None
    count = 0
    output[0] = current
    for index in range(1, len(raw)):
        value = int(raw[index])
        if value == current:
            candidate = None
            count = 0
        elif value == candidate:
            count += 1
        else:
            candidate = value
            count = 1
        if candidate is not None and count >= min_run:
            current = candidate
            candidate = None
            count = 0
        output[index] = current
    return output


def split_arrays(
    frame: pd.DataFrame,
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    videos: dict[str, dict[str, np.ndarray]] = {}
    for video_id, group in frame.groupby("video_id", sort=True):
        indices = group.index.to_numpy(dtype=int)
        videos[str(video_id).zfill(2)] = {
            "logits": logits[indices],
            "truth": labels[indices].astype(int) + 1,
            "times": group["t_sec"].to_numpy(dtype=int),
        }
    return videos


def method_predictions(video: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    probabilities = softmax_logits(video["logits"])
    raw = probabilities.argmax(axis=1).astype(int) + 1
    return {
        "baseline_raw": raw,
        "persistence_5": causal_persistence(raw, min_run=5),
        "egtp_selected": apply_egtp(
            probabilities,
            0.6,
            epsilon=1e-8,
            initial_std=1.0,
            std_floor=1e-6,
            dynamic_normalisation=True,
        ).predictions,
    }


def transition_at(sequence: np.ndarray, times: np.ndarray, boundary_time: int) -> tuple[int, int]:
    matches = np.flatnonzero(times == int(boundary_time))
    if len(matches) != 1 or int(matches[0]) == 0:
        raise ValueError(f"invalid boundary time {boundary_time}")
    index = int(matches[0])
    return int(sequence[index - 1]), int(sequence[index])


def audit_rows(protocol: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scope = protocol["scope"]
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for seed in scope["training_seeds"]:
        frame, logits, labels, seed_sources = load_raw_logit_split(
            int(seed), BASELINE_RUNS[int(seed)], "test", scope["test_video_ids"]
        )
        sources.extend(seed_sources)
        for video_id, video in split_arrays(frame, logits, labels).items():
            truth = video["truth"]
            times = video["times"]
            gt_boundaries = extract_boundaries(truth, times)
            for method_id, prediction in method_predictions(video).items():
                pred_boundaries = extract_boundaries(prediction, times)
                for tolerance in scope["tolerances_sec"]:
                    result = match_boundaries(
                        gt_boundaries,
                        pred_boundaries,
                        int(tolerance),
                        strategy=OPTIMAL_ORDERED,
                    )
                    for pair in result["matched_pairs"]:
                        gt_from, gt_to = transition_at(
                            truth, times, int(pair["gt_time"])
                        )
                        pred_from, pred_to = transition_at(
                            prediction, times, int(pair["pred_time"])
                        )
                        rows.append(
                            {
                                "training_seed": int(seed),
                                "video_id": video_id,
                                "method_id": method_id,
                                "tolerance_sec": int(tolerance),
                                "gt_time_sec": int(pair["gt_time"]),
                                "pred_time_sec": int(pair["pred_time"]),
                                "signed_delay_sec": int(pair["pred_time"])
                                - int(pair["gt_time"]),
                                "gt_from_phase": gt_from,
                                "gt_to_phase": gt_to,
                                "pred_from_phase": pred_from,
                                "pred_to_phase": pred_to,
                                "from_phase_correct": pred_from == gt_from,
                                "to_phase_correct": pred_to == gt_to,
                                "exact_transition_correct": (
                                    pred_from == gt_from and pred_to == gt_to
                                ),
                            }
                        )
    return pd.DataFrame(rows), sources


def summarise(detail: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = (
        detail.groupby(columns, sort=True)
        .agg(
            n_time_matched=("exact_transition_correct", "size"),
            from_phase_correct_count=("from_phase_correct", "sum"),
            to_phase_correct_count=("to_phase_correct", "sum"),
            exact_transition_correct_count=("exact_transition_correct", "sum"),
        )
        .reset_index()
    )
    for prefix in ("from_phase", "to_phase", "exact_transition"):
        result[f"{prefix}_correct_rate"] = (
            result[f"{prefix}_correct_count"] / result["n_time_matched"]
        )
    return result


def verify_against_authority(protocol: dict[str, Any], detail: pd.DataFrame) -> Path:
    source_events = resolve_record(protocol["frozen_source"]["boundary_events"])
    events = pd.read_csv(source_events, dtype={"video_id": str})
    events["video_id"] = events["video_id"].str.zfill(2)
    methods = set(protocol["scope"]["methods"])
    expected = (
        events[events["method_id"].isin(methods) & events["event_type"].eq("matched")]
        .groupby(["training_seed", "video_id", "method_id", "tolerance_sec"])
        .size()
        .rename("expected")
    )
    observed = (
        detail.groupby(["training_seed", "video_id", "method_id", "tolerance_sec"])
        .size()
        .rename("observed")
    )
    comparison = expected.to_frame().join(observed, how="outer").fillna(0).astype(int)
    if not comparison["expected"].equals(comparison["observed"]):
        raise AssertionError("replayed matched-boundary counts differ from authority")
    return source_events


def main() -> None:
    protocol = load_json(PROTOCOL)
    if protocol.get("status") != "frozen_post_hoc_secondary_descriptive_audit":
        raise ValueError("unexpected transition-type audit status")
    resolve_record(protocol["frozen_source"]["test_protocol"])
    source_manifest = resolve_record(
        protocol["frozen_source"]["final_evidence_manifest"]
    )
    detail, sources = audit_rows(protocol)
    source_events = verify_against_authority(protocol, detail)

    out = ensure_fresh_output_dir(OUT, allow_overwrite=False)
    out.mkdir(parents=True, exist_ok=True)
    per_seed = summarise(detail, ["training_seed", "method_id", "tolerance_sec"])
    pooled = summarise(detail, ["method_id", "tolerance_sec"])
    by_video = summarise(
        detail, ["training_seed", "video_id", "method_id", "tolerance_sec"]
    )
    mean_std_rows: list[dict[str, Any]] = []
    for (method_id, tolerance), group in per_seed.groupby(
        ["method_id", "tolerance_sec"], sort=True
    ):
        row: dict[str, Any] = {
            "method_id": method_id,
            "tolerance_sec": int(tolerance),
            "n_training_seeds": int(group["training_seed"].nunique()),
        }
        for metric in (
            "n_time_matched",
            "from_phase_correct_rate",
            "to_phase_correct_rate",
            "exact_transition_correct_rate",
        ):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        mean_std_rows.append(row)
    mean_std = pd.DataFrame(mean_std_rows)
    confusion = (
        detail.assign(
            gt_transition=detail["gt_from_phase"].astype(str)
            + "->"
            + detail["gt_to_phase"].astype(str),
            pred_transition=detail["pred_from_phase"].astype(str)
            + "->"
            + detail["pred_to_phase"].astype(str),
        )
        .groupby(
            ["method_id", "tolerance_sec", "gt_transition", "pred_transition"],
            sort=True,
        )
        .size()
        .rename("time_matched_pair_count")
        .reset_index()
    )

    detail.to_csv(out / "time_matched_transition_type_detail.csv", index=False)
    per_seed.to_csv(out / "transition_type_correctness_per_seed.csv", index=False)
    mean_std.to_csv(out / "transition_type_correctness_mean_std.csv", index=False)
    pooled.to_csv(out / "transition_type_correctness_pooled.csv", index=False)
    by_video.to_csv(out / "transition_type_correctness_by_seed_video.csv", index=False)
    confusion.to_csv(out / "time_matched_transition_type_confusion.csv", index=False)

    selected = pooled[
        pooled["method_id"].eq("egtp_selected")
        & pooled["tolerance_sec"].eq(10)
    ].iloc[0]
    readme = f"""# EGTP transition-type audit v1

This is a post-hoc, secondary, descriptive audit based on the already-frozen
test protocol. It does not change time-based TP/FP/FN, Boundary F1, method
selection, or k=0.6.

At +/-10 seconds, the validation-selected EGTP had
{int(selected['n_time_matched'])} time-matched boundary pairs pooled across the
three training seeds. Conditional on those matches, the from-phase, to-phase,
and exact transition-type correctness rates were
{selected['from_phase_correct_rate']:.3f},
{selected['to_phase_correct_rate']:.3f}, and
{selected['exact_transition_correct_rate']:.3f}, respectively.

These conditional rates must always be reported with Boundary Recall because
missed boundaries are absent from the denominator. They are not primary
performance metrics and do not support a confirmatory claim.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    output_files = sorted(
        path for path in out.iterdir() if path.is_file() and path.name != "run_manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "analysis": "rq2_egtp_post_hoc_transition_type_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": {
            "new_model_training": False,
            "new_model_inference": False,
            "deterministic_postprocessing_replay": True,
            "method_or_threshold_selection": False,
            "primary_boundary_matching_changed": False,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256_file(PROTOCOL)},
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "source_events": {
            "path": str(source_events),
            "sha256": sha256_file(source_events),
        },
        "saved_logit_sources": sources,
        "checks": {
            "replayed_time_match_counts_equal_authority": True,
            "primary_boundary_metrics_unchanged": True,
            "conditional_denominator_is_time_matched_pairs": True,
            "test_used_for_selection_or_retuning": False,
        },
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_files
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "evidence_status": "post_hoc_secondary_descriptive_three_seed_iterative_development_evidence",
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(pooled.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
