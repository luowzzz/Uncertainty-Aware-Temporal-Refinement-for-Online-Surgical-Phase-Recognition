"""Create the versioned RQ1 thesis bundle without rerunning an experiment.

The v1 output values are copied byte-for-byte. The only known v1 manifest
failure is the README hash, so v2 records that repair explicitly and recomputes
all output hashes without editing or deleting the legacy bundle.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reproducibility_utils import PROJECT_ROOT, ensure_fresh_output_dir, sha256_file


SOURCE = PROJECT_ROOT / "outputs" / "rq1_final_evidence_v1"
OUT = PROJECT_ROOT / "outputs" / "rq1_final_evidence_v2"
CONFIG = PROJECT_ROOT / "configs" / "rq1_final_evidence_v2.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    source_manifest_path = SOURCE / "run_manifest.json"
    source_manifest = load_json(source_manifest_path)
    mismatches: list[dict[str, str]] = []
    for record in source_manifest["outputs"]:
        path = Path(record["path"])
        actual = sha256_file(path)
        if actual != record["sha256"]:
            mismatches.append(
                {
                    "path": str(path.resolve()),
                    "recorded_sha256": record["sha256"],
                    "actual_sha256": actual,
                }
            )
    if len(mismatches) != 1 or Path(mismatches[0]["path"]).name != "README.md":
        raise AssertionError(f"unexpected v1 manifest mismatch set: {mismatches}")

    out = ensure_fresh_output_dir(OUT, allow_overwrite=False)
    out.mkdir(parents=True, exist_ok=True)
    for source in SOURCE.iterdir():
        if source.is_file() and source.name not in {
            "run_manifest.json",
            "artifact_authority.json",
            "README.md",
        }:
            shutil.copy2(source, out / source.name)

    authority = {
        "schema_version": 2,
        "authority_id": "rq1_final_evidence_v2",
        "status": "authoritative_thesis_evidence_index",
        "supersedes": "rq1_final_evidence_v1",
        "change_scope": "manifest repair only; all quantitative evidence files are byte-identical copies of v1",
        "configuration": {
            "path": str(CONFIG.resolve()),
            "sha256": sha256_file(CONFIG),
        },
        "source_bundle": {
            "path": str(SOURCE.resolve()),
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_sha256": sha256_file(source_manifest_path),
        },
        "authoritative_components": {
            "deterministic": [
                "outputs/v2_lstm_online_resnet18_seed00",
                "outputs/v2_lstm_online_resnet18_seed01",
                "outputs/v2_lstm_online_resnet18_seed02",
            ],
            "mc_validation": "outputs/mc_dropout_convergence_validation_v3",
            "mc_test": "outputs/mc_dropout_three_seed_protocol_v4",
            "calibration": "outputs/temperature_scaling_three_seed_protocol_v2",
        },
        "evidence_status": "three_seed_iterative_development_evidence_not_independent_confirmatory_test",
    }
    (out / "artifact_authority.json").write_text(
        json.dumps(authority, indent=2), encoding="utf-8"
    )
    readme = """# RQ1 final thesis evidence v2

This is the authoritative RQ1 thesis evidence bundle. All quantitative CSV
files are byte-identical copies of `rq1_final_evidence_v1`; no training,
inference, metric recomputation, parameter selection, or conclusion change was
performed.

Version 2 exists because the v1 `README.md` changed after its manifest was
written. The stale README hash caused one reproducibility test to fail even
though the quantitative evidence files remained intact. This bundle preserves
v1 and records the repair instead of overwriting history.

Core thesis evidence:

- overall error detection: AUROC, AUPR and Exact AURC;
- near/far conditional AUROC and AUPR, with +/-10 s as the primary window;
- T=30 MC Dropout selected by validation convergence;
- raw-logit Temperature Scaling fitted independently on validation for each
  training seed, reported using NLL, Brier score and ECE-15.

The test split had been inspected during earlier iterative development. These
results are therefore traceable three-seed iterative-development evidence, not
a fully independent confirmatory test.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    # Confirm that every copied quantitative artifact is byte-identical.
    copied_names = sorted(
        path.name
        for path in SOURCE.iterdir()
        if path.is_file()
        and path.name not in {"run_manifest.json", "artifact_authority.json", "README.md"}
    )
    for name in copied_names:
        if sha256_file(SOURCE / name) != sha256_file(out / name):
            raise AssertionError(f"copied evidence changed: {name}")

    output_files = sorted(
        path for path in out.iterdir() if path.is_file() and path.name != "run_manifest.json"
    )
    manifest = {
        "schema_version": 2,
        "analysis": "rq1_final_thesis_evidence_manifest_repair_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": {
            "training_run": False,
            "model_inference_run": False,
            "metric_recomputation": False,
            "test_used_for_selection": False,
            "quantitative_values_changed": False,
            "legacy_output_overwritten": False,
        },
        "configuration": {"path": str(CONFIG.resolve()), "sha256": sha256_file(CONFIG)},
        "source_bundle": {
            "path": str(SOURCE.resolve()),
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_sha256": sha256_file(source_manifest_path),
            "documented_mismatches": mismatches,
        },
        "authoritative_inputs": source_manifest["authoritative_inputs"],
        "protocol": source_manifest["protocol"],
        "checks": {
            "all_quantitative_files_byte_identical_to_v1": True,
            "v1_preserved": True,
            "v1_only_known_manifest_mismatch_is_README": True,
            "all_v2_output_hashes_current": True,
        },
        "outputs": [
            {
                "role": "final_evidence_output",
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in output_files
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "evidence_status": "three_seed_iterative_development_evidence_not_independent_confirmatory_test",
        "argv": sys.argv,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"source_mismatches": mismatches, "copied_files": copied_names}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
