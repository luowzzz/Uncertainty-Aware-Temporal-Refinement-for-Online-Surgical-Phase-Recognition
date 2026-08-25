# Portable Experiment Configurations

This directory contains portable copies of the experiment configurations used in the dissertation. Historical machine-specific path prefixes were replaced with repository-relative paths so that the files can be read from the public repository. Hashes that depend on the copied configurations were updated after this path change.

The original files are preserved without modification in `evidence/frozen_records/configs/`. Quantitative result files were not changed.

## Main Configuration Records

| Area | Configuration | Purpose |
|---|---|---|
| Overall study protocol | `frozen_protocol_v3.json` | Records the fixed dataset split, training seeds and main experimental settings. |
| Evidence index | `project_evidence_index_v1.json` | Links the main project records to their evidence locations. |
| Lightweight baseline | `baseline_training_v1.json` | Records the ResNet-18–LSTM training configuration. |
| RQ1 | `rq1_final_evidence_v2.json` and the related RQ1 component configurations | Records the uncertainty, MC Dropout, calibration and selective-prediction settings. |
| RQ2 validation protocol | `rq2_egtp_validation_protocol_v1.json` | Records the EGTP implementation conventions and validation procedure. |
| RQ2 operating-point selection | `rq2_egtp_validation_selection_v1.json` | Records the validation-based selection of the main EGTP setting. |
| RQ2 test protocol | `rq2_egtp_test_protocol_v1.json` | Records the frozen test evaluation for Raw, Persistence-5 and EGTP. |
| Supporting RQ2 audit | `rq2_egtp_transition_type_audit_v1.json` | Records the conditional transition-type audit. This is a supporting analysis rather than part of the main RQ2 comparison. |
