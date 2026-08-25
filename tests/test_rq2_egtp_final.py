from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from reproducibility_utils import sha256_file  # noqa: E402


PUBLICLY_EXCLUDED_NARRATIVE_FILES = {"FINAL_RQ2_CONCLUSION_ZH.md"}


def public_record_path(record: dict[str, str]) -> Path:
    """Map an immutable historical output path to its public evidence copy."""
    path = Path(record["path"])
    if not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()
    parts = list(path.parts)
    if "outputs" not in parts:
        raise ValueError(f"No public mapping for historical path: {path}")
    index = parts.index("outputs")
    return (PROJECT_ROOT / "evidence" / "results" / Path(*parts[index + 1 :])).resolve()


class RQ2EGTPFinalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = PROJECT_ROOT / "evidence" / "results" / "rq2_egtp_final_evidence_v1"
        cls.protocol_path = PROJECT_ROOT / "configs" / "rq2_egtp_test_protocol_v1.json"
        cls.protocol = json.loads(cls.protocol_path.read_text(encoding="utf-8"))
        cls.per_seed = pd.read_csv(cls.bundle / "test_metrics_per_seed.csv")
        cls.events = pd.read_csv(cls.bundle / "test_boundary_events.csv")

    def test_protocol_matches_validation_selection(self):
        selection_record = self.protocol["frozen_validation_selection"]
        selection_path = public_record_path(selection_record)
        self.assertEqual(sha256_file(selection_path), selection_record["sha256"])
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        self.assertFalse(selection["test_files_accessed"])
        self.assertEqual(selection["final_selection"]["selected_variant"], "baseline_uncalibrated")
        self.assertEqual(selection["final_selection"]["selected_k"], 0.6)
        self.assertEqual(self.protocol["primary_method"]["k"], 0.6)

    def test_three_seeds_and_all_frozen_methods_are_present(self):
        expected = {item["method_id"] for item in self.protocol["frozen_ablation_methods"]}
        self.assertEqual(set(self.per_seed["training_seed"].astype(int)), {0, 1, 2})
        self.assertEqual(set(self.per_seed["method_id"]), expected)
        self.assertTrue((self.per_seed.groupby("method_id")["training_seed"].nunique() == 3).all())

    def test_boundary_event_accounting_is_conservative(self):
        for (seed, method, tolerance), table in self.events.groupby(
            ["training_seed", "method_id", "tolerance_sec"]
        ):
            counts = table["event_type"].value_counts()
            metrics = self.per_seed[
                self.per_seed["training_seed"].eq(seed)
                & self.per_seed["method_id"].eq(method)
            ].iloc[0]
            self.assertEqual(int(counts.get("matched", 0)), int(metrics[f"tp_tol{tolerance}"]))
            self.assertEqual(int(counts.get("extra", 0)), int(metrics[f"fp_tol{tolerance}"]))
            self.assertEqual(int(counts.get("missed", 0)), int(metrics[f"fn_tol{tolerance}"]))

    def test_manifest_outputs_are_byte_identical_to_frozen_records(self):
        manifest = json.loads((self.bundle / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["operation"]["test_metrics_used_for_selection_or_retuning"])
        excluded = set()
        for record in manifest["outputs"]:
            path = public_record_path(record)
            if path.name in PUBLICLY_EXCLUDED_NARRATIVE_FILES:
                excluded.add(path.name)
                self.assertFalse(path.exists())
                continue
            self.assertEqual(sha256_file(path), record["sha256"])
        self.assertEqual(excluded, PUBLICLY_EXCLUDED_NARRATIVE_FILES)


if __name__ == "__main__":
    unittest.main()
