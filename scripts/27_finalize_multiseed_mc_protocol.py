"""Package the frozen three-seed MC Dropout protocol and test evidence.

This script never runs a model.  It validates the validation-selected MC
configuration and three already-generated test runs, computes a consistent
multi-seed/per-video aggregation, and writes the standard deliverables without
overwriting any source output.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from reproducibility_utils import PROJECT_ROOT, ensure_fresh_output_dir, sha256_file
from selective_prediction_metrics import empirical_risk_coverage, exact_aurc


DEFAULT_TEST_RUNS = {
    0: PROJECT_ROOT / "outputs" / "rq1_mc_dropout_t30_seed00_v4",
    1: PROJECT_ROOT / "outputs" / "rq1_mc_dropout_t30_seed01_v4",
    2: PROJECT_ROOT / "outputs" / "rq1_mc_dropout_t30_seed02_v4",
}
DEFAULT_CONVERGENCE_DIR = (
    PROJECT_ROOT / "outputs" / "mc_dropout_convergence_validation_v3"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mc_dropout_evaluation_v4.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "mc_dropout_three_seed_protocol_v4"
PRIMARY_SCORES = ("mc_entropy", "mc_mutual_info")


def parse_seed_run(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        return int(seed_text.removeprefix("seed")), Path(path_text).expanduser().resolve()
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "Use SEED=PATH, for example 0=outputs/rq1_mc_dropout_t30_seed00_v4"
        ) from error


def source_record(path: Path, role: str, training_seed: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "training_seed": training_seed,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def aggregate_mean_std(table: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in table.groupby(groups, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n_training_seeds"] = group["training_seed"].nunique()
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="append", type=parse_seed_run)
    parser.add_argument("--convergence-dir", type=Path, default=DEFAULT_CONVERGENCE_DIR)
    parser.add_argument("--mc-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    test_runs = dict(args.test_run) if args.test_run else DEFAULT_TEST_RUNS
    if sorted(test_runs) != [0, 1, 2]:
        raise ValueError("The frozen protocol requires exactly training seeds 0, 1 and 2")
    convergence_dir = args.convergence_dir.expanduser().resolve()
    config_path = args.mc_config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected_t = int(config["inference"]["passes"])
    inference_seed = int(config["inference"]["inference_seed"])
    if config.get("test_used_for_selection") is not False:
        raise ValueError("Frozen MC configuration must state test_used_for_selection=false")

    convergence_per_seed_path = convergence_dir / "validation_convergence_per_seed.csv"
    convergence_summary_path = convergence_dir / "validation_convergence_summary.csv"
    convergence_selected_path = convergence_dir / "selected_mc_config.json"
    convergence_manifest_path = convergence_dir / "run_manifest.json"
    convergence_selected = json.loads(
        convergence_selected_path.read_text(encoding="utf-8")
    )
    if int(convergence_selected["selected_T"]) != selected_t:
        raise ValueError("Authoritative MC config disagrees with validation selection")
    if convergence_selected.get("test_files_read") is not False:
        raise ValueError("Validation convergence record must state test_files_read=false")

    out_dir = ensure_fresh_output_dir(args.out_dir, args.allow_overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        source_record(config_path, "authoritative_mc_config"),
        source_record(convergence_per_seed_path, "validation_convergence_per_seed"),
        source_record(convergence_summary_path, "validation_convergence_summary"),
        source_record(convergence_selected_path, "validation_selected_mc_config"),
        source_record(convergence_manifest_path, "validation_run_manifest"),
    ]

    per_seed_rows = []
    per_video_frames = []
    run_manifests: dict[int, dict] = {}
    for training_seed, run_dir in sorted(test_runs.items()):
        run_dir = run_dir.expanduser().resolve()
        manifest_path = run_dir / "run_manifest.json"
        frame_path = run_dir / "mc_dropout_frame_scores.csv"
        per_video_path = run_dir / "mc_dropout_metrics_by_video.csv"
        runtime_path = run_dir / "mc_dropout_runtime_by_video.csv"
        for role, path in (
            ("test_run_manifest", manifest_path),
            ("test_frame_scores", frame_path),
            ("test_per_video_metrics", per_video_path),
            ("test_runtime", runtime_path),
        ):
            sources.append(source_record(path, role, training_seed))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_manifests[training_seed] = manifest
        inputs = manifest["inputs"]
        protocol = manifest["protocol"]
        if int(inputs["training_seed"]) != training_seed:
            raise ValueError(f"Run manifest training seed mismatch for seed {training_seed}")
        if int(protocol["mc_passes"]) != selected_t:
            raise ValueError(f"Run seed {training_seed} does not use selected T")
        if int(protocol["inference_seed"]) != inference_seed:
            raise ValueError(f"Run seed {training_seed} has wrong inference seed")
        if protocol.get("rng_scope") != "reset_once_at_the_start_of_each_split":
            raise ValueError(f"Run seed {training_seed} lacks split-isolated RNG")
        if protocol.get("splits") != ["test"]:
            raise ValueError(f"Run seed {training_seed} is not a test-only frozen run")
        expected_checkpoint = config["checkpoints"][str(training_seed)]
        if inputs["checkpoint_sha256"].lower() != expected_checkpoint["sha256"].lower():
            raise ValueError(f"Checkpoint hash mismatch for seed {training_seed}")
        if int(inputs["checkpoint_epoch"]) != int(expected_checkpoint["epoch"]):
            raise ValueError(f"Checkpoint epoch mismatch for seed {training_seed}")

        frame = pd.read_csv(frame_path)
        if set(frame["split"].astype(str)) != {"test"}:
            raise ValueError(f"Seed {training_seed} frame table contains non-test rows")
        errors = frame["error"].to_numpy(dtype=int)
        phase = pd.read_csv(run_dir / "mc_dropout_phase_metrics.csv").iloc[0]
        runtime = pd.read_csv(runtime_path)
        inference_seconds = float(runtime["inference_seconds"].sum())
        for score in PRIMARY_SCORES:
            values = frame[score].to_numpy(dtype=float)
            curve = empirical_risk_coverage(errors, values)
            aurc = exact_aurc(curve)
            per_seed_rows.append(
                {
                    "training_seed": training_seed,
                    "inference_seed": inference_seed,
                    "T": selected_t,
                    "split": "test",
                    "checkpoint_epoch": int(inputs["checkpoint_epoch"]),
                    "checkpoint_validation_macro_f1": float(
                        inputs["checkpoint_validation_macro_f1"]
                    ),
                    "checkpoint_sha256": inputs["checkpoint_sha256"],
                    "n_frames": len(frame),
                    "n_errors": int(errors.sum()),
                    "error_rate": float(errors.mean()),
                    "mc_accuracy": float(phase["accuracy"]),
                    "mc_macro_f1": float(phase["macro_f1"]),
                    "score": score,
                    "error_auroc": float(roc_auc_score(errors, values)),
                    "error_aupr": float(average_precision_score(errors, values)),
                    "exact_aurc": float(aurc["aurc_lower_better"]),
                    "oracle_aurc": float(aurc["oracle_aurc"]),
                    "excess_aurc": float(aurc["excess_aurc"]),
                    "test_inference_seconds": inference_seconds,
                    "milliseconds_per_frame_pass": (
                        1000.0 * inference_seconds / (len(frame) * selected_t)
                    ),
                }
            )
        per_video = pd.read_csv(per_video_path)
        per_video = per_video[per_video["score"].isin(PRIMARY_SCORES)].copy()
        per_video.insert(0, "training_seed", training_seed)
        per_video.insert(1, "inference_seed", inference_seed)
        per_video.insert(2, "T", selected_t)
        per_video_frames.append(per_video)

    per_seed = pd.DataFrame(per_seed_rows)
    mean_std = aggregate_mean_std(
        per_seed,
        ["split", "score", "T", "inference_seed"],
        [
            "n_frames",
            "n_errors",
            "error_rate",
            "mc_accuracy",
            "mc_macro_f1",
            "error_auroc",
            "error_aupr",
            "exact_aurc",
            "oracle_aurc",
            "excess_aurc",
            "test_inference_seconds",
            "milliseconds_per_frame_pass",
        ],
    )
    per_video = pd.concat(per_video_frames, ignore_index=True)
    per_video_summary = aggregate_mean_std(
        per_video,
        ["split", "video_id", "score", "T", "inference_seed"],
        [
            "n_frames",
            "n_errors",
            "error_rate",
            "accuracy",
            "macro_f1",
            "error_auroc",
            "error_aupr",
            "exact_aurc",
            "oracle_aurc",
            "excess_aurc",
        ],
    )

    # Standard deliverable names requested by the frozen protocol task.
    pd.read_csv(convergence_per_seed_path).to_csv(
        out_dir / "validation_convergence_per_seed.csv", index=False
    )
    pd.read_csv(convergence_summary_path).to_csv(
        out_dir / "validation_convergence_summary.csv", index=False
    )
    (out_dir / "selected_mc_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    per_seed.to_csv(out_dir / "test_mc_metrics_per_seed.csv", index=False)
    mean_std.to_csv(out_dir / "test_mc_metrics_mean_std.csv", index=False)
    per_video.to_csv(out_dir / "test_mc_metrics_per_video.csv", index=False)
    per_video_summary.to_csv(
        out_dir / "test_mc_metrics_per_video_mean_std.csv", index=False
    )
    for training_seed, manifest in run_manifests.items():
        (out_dir / f"run_manifest_seed{training_seed:02d}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    commands = [
        "# Three-Seed MC Dropout Reproducibility Commands",
        "",
        "Run from the repository root with the project Python environment.",
        "",
        "## Validation-only nested convergence",
        "",
        "```powershell",
        "python scripts\\23_mc_dropout_convergence_validation.py --out-dir outputs\\mc_dropout_convergence_validation_v3",
        "```",
        "",
        "## Frozen test consistency runs",
        "",
    ]
    for seed in range(3):
        commands.extend(
            [
                "```powershell",
                (
                    f"python scripts\\08_eval_mc_dropout.py --training_seed {seed} "
                    f"--checkpoint outputs\\v2_lstm_online_resnet18_seed{seed:02d}\\checkpoints\\best.pt "
                    f"--out_dir outputs\\rq1_mc_dropout_t30_seed{seed:02d}_v4 "
                    "--T 30 --inference_seed 0 --splits test --windows 5 10 20 "
                    "--mc_config configs\\mc_dropout_evaluation_v4.json"
                ),
                "```",
                "",
            ]
        )
    commands.extend(
        [
            "## Aggregation",
            "",
            "```powershell",
            "python scripts\\27_finalize_multiseed_mc_protocol.py",
            "```",
            "",
            "## Interpretation",
            "",
            "Incorrect MC-averaged predictions are positive errors. Deterministic and MC scores are not treated as strictly paired when their prediction error sets differ.",
            "",
            "The three-seed protocol improves reproducibility and robustness, but the results remain iterative-development evidence because the test set had previously been inspected.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(commands) + "\n", encoding="utf-8")

    output_names = [
        "validation_convergence_per_seed.csv",
        "validation_convergence_summary.csv",
        "selected_mc_config.json",
        "test_mc_metrics_per_seed.csv",
        "test_mc_metrics_mean_std.csv",
        "test_mc_metrics_per_video.csv",
        "test_mc_metrics_per_video_mean_std.csv",
        "run_manifest_seed00.json",
        "run_manifest_seed01.json",
        "run_manifest_seed02.json",
        "README.md",
    ]
    manifest = {
        "schema_version": 1,
        "analysis": "frozen_three_seed_mc_dropout_protocol",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "code_dependencies": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in (
                PROJECT_ROOT / "scripts" / "08_eval_mc_dropout.py",
                PROJECT_ROOT / "scripts" / "23_mc_dropout_convergence_validation.py",
                PROJECT_ROOT / "scripts" / "selective_prediction_metrics.py",
                PROJECT_ROOT / "scripts" / "reproducibility_utils.py",
            )
        ],
        "inputs": sources,
        "protocol": {
            "selected_T": selected_t,
            "training_seeds": [0, 1, 2],
            "inference_seed": inference_seed,
            "validation_selection_only": True,
            "test_used_for_selection": False,
            "rng_scope": "reset_once_at_the_start_of_each_split",
            "error_definition": "Incorrect MC-averaged argmax prediction is positive.",
            "aurc_definition": "Exact empirical mean selective risk over k=1..N with expected risk inside exact score ties.",
            "test_exposure": config["test_exposure_note"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "outputs": [
            {
                "path": str((out_dir / name).resolve()),
                "sha256": sha256_file(out_dir / name),
            }
            for name in output_names
        ],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved frozen three-seed MC protocol bundle to: {out_dir}")


if __name__ == "__main__":
    main()
