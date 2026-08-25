# Public Release Scope

This document records which materials were copied from the original local project into the public repository. It also explains why other materials were excluded. The original project was not deleted or modified during this process.

## Included Code

| Area | Files | Purpose |
|---|---|---|
| Data preparation | `01_extract_frames_make_manifest.py`, `02_extract_resnet_features.py` | Prepare the 1 fps inputs and extract ResNet-18 features. |
| Lightweight baseline | `03_train_temporal_baseline.py` | Train the causal unidirectional LSTM baseline for seeds 0, 1 and 2. |
| RQ1 analysis | Numbered scripts `08`, `23`, `27`–`30` and `42`, together with their metric helpers | Run MC Dropout, calibration, selective-prediction analysis and final evidence preparation. |
| Main RQ2 analysis | Numbered scripts `39` and `40`, together with the EGTP, boundary and temporal-metric helpers | Select the EGTP operating point on validation data and run the frozen test comparison. |
| Supporting RQ2 analysis | Numbered scripts `38` and `41`, together with `tec_loss.py` | Retain the TEC and transition-type analyses as supporting records. They are not part of the main Raw, Persistence-5 and EGTP comparison. |
| Operating-point selection | `operating_point_selection.py` | Apply the Strict Set and Basic Set rules used for EGTP selection. This file was renamed from a legacy internal filename without changing its calculations. |

## Included Configurations, Evidence and Tests

- `configs/` contains portable copies of the baseline, RQ1 and EGTP protocols. Machine-specific path prefixes were replaced with repository-relative paths.
- `evidence/frozen_records/configs/` contains the original config files with their historical absolute paths and hashes.
- `evidence/results/` contains the final RQ1 evidence, MC Dropout convergence records, three-seed MC summaries, Temperature Scaling results, EGTP validation selection and frozen RQ2 test evidence. It also contains the supporting transition-type audit.
- `tests/` contains focused checks for EGTP, TEC, operating-point selection, boundary-event accounting and evidence integrity.

The English RQ2 conclusion record is included. A duplicate Chinese-language narrative summary created for internal working use is not part of the public package. The historical run manifest remains unchanged and therefore still records that file as an output of the original run.

## Public Script List

The following scripts and helper modules are included:

```text
01_extract_frames_make_manifest.py
02_extract_resnet_features.py
03_train_temporal_baseline.py
08_eval_mc_dropout.py
23_mc_dropout_convergence_validation.py
27_finalize_multiseed_mc_protocol.py
28_fit_temperature_scaling_validation.py
29_evaluate_temperature_scaling_test.py
30_finalize_rq1_evidence.py
38_train_rq2_tec_lstm.py
39_select_rq2_egtp_validation.py
40_evaluate_frozen_rq2_egtp_test.py
41_add_egtp_transition_type_audit.py
42_repair_rq1_evidence_manifest.py
boundary_metrics.py
calibration_metrics.py
egtp_transition_policy.py
operating_point_selection.py
reproducibility_utils.py
rq2_stability_metrics.py
selective_prediction_metrics.py
tec_loss.py
temperature_scaling_logits.py
```

## Excluded Code

| Category | Original files | Reason for exclusion |
|---|---|---|
| Early dataset checks and superseded baselines | `00_probe_dataset.py`, `03_train_bilstm_baseline.py`, `04_eval_boundary_baseline.py`, `05_eval_boundary_with_smoothing.py` | These files are not required for the final baseline pipeline. |
| Superseded RQ1 exploration | Numbered scripts `06`, `07`, `10`–`14`, `20`, `24` and `25` | These analyses were replaced by the frozen three-seed RQ1 pipeline. |
| OOD exploration | `09_eval_visual_shift_ood.py` | OOD evaluation was removed from the dissertation scope. |
| Earlier temporal-refinement work | Numbered scripts `15`–`19`, `21`, `22`, `26` and `31`–`37`, together with `cumulative_transition_gate_v1.py` and `reliability_weighted_transition_gate.py` | These exploratory gates were superseded by the final RQ2 comparison and do not support the main conclusions. |

The exact excluded script list is:

```text
00_probe_dataset.py
03_train_bilstm_baseline.py
04_eval_boundary_baseline.py
05_eval_boundary_with_smoothing.py
06_analyze_boundary_reliability.py
07_eval_risk_coverage.py
09_eval_visual_shift_ood.py
10_eval_boundary_aware_qc.py
11_eval_boundary_aware_conformal.py
12_make_baseline_diagnostics.py
13_make_per_second_output_table.py
14_analyze_uncertainty_error_risk.py
15_uncertainty_guided_temporal_refinement.py
16_analyze_refinement_tradeoffs.py
17_simplified_upr_prototype_refinement.py
18_multiseed_smoothing_refinement_comparison.py
19_per_video_refinement_analysis.py
20_aggregate_multiseed_rq1.py
21_audit_boundary_matching.py
22_rq2_validation_robustness.py
24_calibrate_multiseed_baseline.py
25_aggregate_multiseed_rq1_v2.py
26_evaluate_frozen_multiseed_rq2.py
31_finalize_rq2_failure_analysis.py
32_freeze_gate_v2_label_free_candidates.py
33_select_gate_v2_validation.py
34_evaluate_gate_v2_frozen_test.py
35_prepare_rq2_dissertation_bundle.py
36_prepare_gate_v1_canonical_bundle.py
37_add_gate_v1_transition_type_audit.py
cumulative_transition_gate_v1.py
reliability_weighted_transition_gate.py
```

The generic selection logic from `rq2_gate_v2_selection.py` was retained under the clearer public name `operating_point_selection.py`.

## Excluded Configurations

The public `configs/` directory excludes the earlier `frozen_protocol_v1/v2` family, the `rq2_gate_v1/v2` and `rq2_proposed_gate_v1` configurations, superseded RQ1 and MC Dropout configurations, and earlier RQ2 evidence configurations. These files describe exploratory methods or evidence versions that are no longer authoritative. They remain unchanged in the original project.

The related legacy output directories were also excluded. The original project still contains the full development history.

## Excluded Data and Binary Files

The public repository does not include:

- original surgical videos or annotations;
- extracted frames or ResNet feature arrays;
- trained checkpoints or raw prediction archives;
- intermediate outputs that are not referenced by the final evidence index;
- the duplicate Chinese-language RQ2 narrative summary created for internal working use;
- local caches, virtual environments, IDE files or archive files.

These materials were excluded because of dataset restrictions, file size, privacy or licensing concerns, or because they are not needed to support the final thesis claims. Where required for provenance, their paths and SHA-256 hashes remain in the immutable manifests.

## Historical Absolute Paths

Historical manifests and frozen configurations are evidence records. Their absolute paths were therefore left unchanged. Portable copies are stored separately in `configs/` and use repository-relative paths. Changes made to a portable copy are not presented as changes to the original frozen file, whose SHA-256 value remains available in `evidence/frozen_records/`.
