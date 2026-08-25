# Authoritative experiment configurations

`project_evidence_index_v1.json` is the current entry point for thesis writing.
It points to `frozen_protocol_v3.json`, which consolidates the completed RQ1
analysis and the current EGTP-based RQ2 without rewriting any earlier frozen
selection or test record.

## Current authorities

- `frozen_protocol_v3.json`: project-wide thesis scope, metric hierarchy,
  evidence limitations and change control;
- `baseline_training_v1.json`: frozen baseline architecture, training seeds
  and validation-only checkpoint selection;
- `rq1_final_evidence_v2.json`: current RQ1 authority. Version 2 repairs the
  evidence manifest only; its quantitative CSV files are byte-identical to v1;
- `rq2_egtp_validation_protocol_v1.json`: EGTP validation protocol frozen
  before validation metric evaluation;
- `rq2_egtp_validation_selection_v1.json`: validation selection of the
  uncalibrated baseline EGTP at k=0.6 from the Basic Set;
- `rq2_egtp_test_protocol_v1.json`: frozen three-seed RQ2 test protocol;
- `rq2_egtp_transition_type_audit_v1.json`: post-hoc secondary descriptive
  audit of transition type among time-matched boundaries.

## Thesis evidence directories

- RQ1: `outputs/rq1_final_evidence_v2`;
- RQ2 primary: `outputs/rq2_egtp_final_evidence_v1`;
- RQ2 transition-type appendix: `outputs/rq2_egtp_transition_type_audit_v1`.

## Legacy records

All v1/v2 custom Gate configurations and outputs remain read-only provenance.
They are not the final RQ2 method and must not be used for current thesis
numbers. In particular, `rq2_proposed_gate_v1.json`,
`rq2_gate_v2_test_protocol_v1.json`, `rq2_final_evidence_v1.json` and the old
research-log file `RQ2_FINAL_EVIDENCE_v1_ZH.md` refer to superseded project-
specific gates.

The old protocols are deliberately preserved so that the experiment history
remains auditable. They are superseded at the project-authority level, not
deleted or silently edited.

## Evidence status

The test split was inspected during earlier iterative development. Final
methods and parameters were selected on validation and were not changed from
the final test metrics, but the results must still be described as traceable
three-seed iterative-development evidence rather than a fully independent
confirmatory test.
