from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from reproducibility_utils import sha256_file  # noqa: E402


def evidence_dir(name: str) -> Path:
    return PROJECT_ROOT / "evidence" / "results" / name


def mapped_output_path(record: dict[str, str]) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        return (PROJECT_ROOT / path).resolve()
    parts = list(path.parts)
    index = parts.index("outputs")
    return (PROJECT_ROOT / "evidence" / "results" / Path(*parts[index + 1 :])).resolve()


class ThesisEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.index_path = PROJECT_ROOT / "configs" / "project_evidence_index_v1.json"
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        protocol_record = self.index["project_protocol"]
        self.protocol_path = PROJECT_ROOT / protocol_record["path"]
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))

    def test_title_and_project_authority_are_consistent(self):
        title = "Uncertainty-Aware Temporal Refinement for Online Surgical Phase Recognition"
        self.assertEqual(self.index["thesis_title"], title)
        self.assertEqual(self.protocol["thesis_title"], title)
        self.assertEqual(sha256_file(self.protocol_path), self.index["project_protocol"]["sha256"])

    def test_rq1_bundle_is_a_manifest_only_repair(self):
        bundle = evidence_dir("rq1_final_evidence_v2")
        manifest = json.loads((bundle / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["operation"]["training_run"])
        self.assertFalse(manifest["operation"]["model_inference_run"])
        self.assertFalse(manifest["operation"]["metric_recomputation"])
        self.assertFalse(manifest["operation"]["quantitative_values_changed"])
        for record in manifest["outputs"]:
            self.assertEqual(sha256_file(mapped_output_path(record)), record["sha256"])

    def test_main_rq2_comparison_is_raw_persistence_and_egtp(self):
        methods = [item["method_id"] for item in self.protocol["rq2"]["main_comparison"]]
        self.assertEqual(methods, ["baseline_raw", "persistence_5", "egtp_selected"])
        self.assertEqual(self.protocol["rq2"]["main_comparison"][2]["k"], 0.6)

    def test_transition_type_audit_matches_primary_event_counts(self):
        detail = pd.read_csv(evidence_dir("rq2_egtp_transition_type_audit_v1") / "time_matched_transition_type_detail.csv")
        source = pd.read_csv(evidence_dir("rq2_egtp_final_evidence_v1") / "test_boundary_events.csv")
        methods = {"baseline_raw", "persistence_5", "egtp_selected"}
        expected = (
            source[source["method_id"].isin(methods) & source["event_type"].eq("matched")]
            .groupby(["training_seed", "method_id", "tolerance_sec"])
            .size()
            .sort_index()
        )
        observed = detail.groupby(["training_seed", "method_id", "tolerance_sec"]).size().sort_index()
        pd.testing.assert_series_equal(expected, observed, check_names=False)


if __name__ == "__main__":
    unittest.main()
