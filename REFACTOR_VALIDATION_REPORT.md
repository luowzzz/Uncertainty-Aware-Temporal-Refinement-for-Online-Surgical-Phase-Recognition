# Public Repository Refactoring and Validation Report

Date: 22 August 2026

## Scope

All packaging and validation work was carried out in a separate public-repository copy. No file in the original local project directory was deleted or edited.

## Changes Made for the Public Repository

1. The public repository retains the baseline and final RQ1 code, the main EGTP RQ2 code, and the supporting TEC and transition-type analysis code.
2. The generic selection helper was renamed from `rq2_gate_v2_selection.py` to `operating_point_selection.py`. Only its documentation and the importing module were changed.
3. Unused loaders for the earlier multi-threshold Gate configurations were removed from `reproducibility_utils.py`. The hash, seed, project-root and safe-output utilities required by the public scripts were kept unchanged.
4. Frozen-record lookup in the public RQ2 test runner was changed so that relative paths are resolved from the repository root.
5. One generated README string containing the original project path was replaced with “repository root”.
6. Portable configuration copies were created with repository-relative paths. The original files remain byte-preserved in `evidence/frozen_records/configs/`.
7. Evidence tests were adjusted so that historical output paths resolve to the public copies in `evidence/results/`. The evidence files themselves were not edited.
8. A duplicate Chinese-language RQ2 narrative summary was excluded from the public package, and the public RQ2 runner was limited to the equivalent English summary. No quantitative output was changed.

These changes did not alter any model equation, EGTP state update, TEC loss, Temperature Scaling calculation, MC Dropout calculation, boundary-matching rule or reported metric.

## Validation Results

| Check | Result |
|---|---|
| Public test suite | 21 tests passed |
| Original core test suite | 12 tests passed |
| Python compilation | All public scripts and tests passed `compileall` |
| Main command-line import and help checks | 7 entry points passed |
| Portable JSON parsing | 16 JSON configurations passed |
| Runtime path scan | No machine-specific absolute path was found in `scripts/`, `configs/` or `tests/` |
| Secret-pattern scan | No likely API key, password, bearer token or secret was found |
| Frozen evidence comparison | All 91 included files matched the original copies; one non-quantitative internal narrative summary was intentionally excluded |

## Basis for Algorithm Equivalence

The following calculation modules are byte-identical to those in the original project:

- `boundary_metrics.py`;
- `calibration_metrics.py`;
- `egtp_transition_policy.py`;
- `rq2_stability_metrics.py`;
- `selective_prediction_metrics.py`;
- `temperature_scaling_logits.py`;
- `tec_loss.py`;
- the final RQ1 calculation scripts `08`, `23`, `28`, `29` and `30`;
- TEC training script `38` and transition-type audit script `41`;
- data-preparation and baseline-training scripts `01`, `02` and `03`.

The public changes to scripts `27`, `39` and `40` concern documentation, module naming, path resolution or omission of a duplicate narrative summary. Their numerical helper modules were not changed. The same EGTP and TEC unit cases passed in both project copies, and all 91 distributed evidence files were preserved byte for byte. Together, these checks support the behavioural equivalence of the packaged implementation.

## Reproduction Limitation

The public repository does not contain the restricted videos, extracted features, model checkpoints or raw prediction archives. The included tests and evidence records can be checked directly. End-to-end training or inference requires users to obtain the dataset and regenerate or provide the omitted materials.
