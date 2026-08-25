# MSc Dissertation Implementation and Reproducibility Package

Project title: Uncertainty-Aware Temporal Refinement for Online Surgical Phase Recognition

This repository contains the implementation, portable configurations, tests and evidence records supporting Deliverables 1–4 of the MSc project. The MSc project report is submitted separately and is not stored in this repository. Restricted AutoLaparo data, extracted features, model checkpoints and raw prediction archives are not included.

## Project Deliverables

| Deliverable | Material provided in this repository |
|---|---|
| Lightweight online surgical phase recognition baseline | ResNet-18 feature extraction, unidirectional LSTM training code and baseline configurations |
| RQ1 uncertainty evaluation framework | Deterministic and MC Dropout uncertainty analysis, Temperature Scaling and selective-prediction evaluation |
| Independent EGTP reimplementation and backbone-transfer evaluation | EGTP policy code, validation-based operating-point selection and frozen RQ2 evaluation |
| Versioned reproducibility and evidence records | Frozen protocols, run manifests, result records, file-integrity checks and automated tests |

The main RQ2 comparison contains Raw, causal Persistence-5 and validation-selected uncalibrated EGTP with `k = 0.6`. Other retained analyses are supporting records and are not part of the main comparison. The test split was inspected during earlier iterative development. The test results are therefore described as three-seed iterative-development evidence rather than a fully independent confirmatory evaluation.

## Repository Guide

| Resource | Description | Location |
|---|---|---|
| Implementation code | Data preparation, baseline training, RQ1 analysis and RQ2 temporal-refinement scripts | [Open implementation scripts](scripts/) |
| Experiment configurations | Portable copies of the frozen study protocols and selected settings | [Open configurations](configs/) |
| Automated checks | Tests for the main algorithms and evidence records | [Open tests](tests/) |
| Result and run records | Result tables, figures and run manifests retained for the final and supporting analyses | [Open result records](evidence/results/) |
| Frozen provenance records | Byte-preserved protocol records retained for traceability | [Open frozen records](evidence/frozen_records/) |
| Public release scope | Materials included in and excluded from the public repository | [Read release scope](PUBLIC_RELEASE_SCOPE.md) |
| Packaging validation | Checks performed after preparing the public repository | [Read validation report](REFACTOR_VALIDATION_REPORT.md) |
| Data and method notices | Dataset restrictions, third-party attribution and method ownership | [Read notices](NOTICE.md) |
| Python requirements | Packages required by the public implementation | [View requirements](requirements.txt) |

## Environment

Python 3.10 or newer is recommended. Create an isolated environment, install the listed packages and run the automated tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For GPU use, PyTorch should be installed using the command that matches the local CUDA version. Where available, the frozen run manifests record the environment used for the original experiments.

## Reproduction Outline

The AutoLaparo videos, extracted frames, features, predictions and trained checkpoints are not redistributed. They must be obtained or regenerated before the complete workflow can be run. After access to the dataset has been obtained under the provider's terms, the main stages are:

```bash
python scripts/01_extract_frames_make_manifest.py --help
python scripts/02_extract_resnet_features.py --help
python scripts/03_train_temporal_baseline.py --seed 0 --out_dir outputs/v2_lstm_online_resnet18_seed00
python scripts/03_train_temporal_baseline.py --seed 1 --out_dir outputs/v2_lstm_online_resnet18_seed01
python scripts/03_train_temporal_baseline.py --seed 2 --out_dir outputs/v2_lstm_online_resnet18_seed02
python scripts/23_mc_dropout_convergence_validation.py --help
python scripts/28_fit_temperature_scaling_validation.py --help
python scripts/39_select_rq2_egtp_validation.py --help
python scripts/40_evaluate_frozen_rq2_egtp_test.py --help
```

The script numbers follow the original experimental workflow. Some numbers are absent because superseded exploratory stages were excluded from the main public workflow. The remaining supporting scripts are retained where they help explain or verify the evidence records.

## Frozen Records and Portable Configurations

Files in `evidence/frozen_records/` are unchanged historical records. Machine-specific absolute paths in these files are retained only to preserve provenance. They are not runnable paths for other users and should not be edited within the frozen records.

The files in `configs/` are portable copies. Their original absolute path prefixes were replaced with repository-relative paths, and the dependent configuration hashes were updated. The quantitative records in `evidence/results/` remain byte-identical to the original frozen outputs.

The inclusion and exclusion decisions are explained in [Public Release Scope](PUBLIC_RELEASE_SCOPE.md).

## Data and Licensing

This repository does not provide rights to the AutoLaparo dataset or other third-party materials. Users must obtain the dataset separately and follow the conditions set by its provider.

No open-source licence had been assigned to this code when the public package was prepared. Access to the source files should not be interpreted as permission to reuse or redistribute them. Dataset attribution, method citations and implementation notices are provided in [Data and Method Notices](NOTICE.md).
