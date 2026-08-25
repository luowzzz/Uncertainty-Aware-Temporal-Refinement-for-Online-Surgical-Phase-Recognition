# Data, Method and Third-Party Notices

This repository contains a project-specific implementation used for the MSc dissertation. The sources below are acknowledged because they provide the dataset, methods or technical basis used by the project. Their inclusion does not mean that the original authors endorse this repository.

## Dataset

AutoLaparo-T1 is the surgical workflow recognition task used in this project. Access to the dataset must be obtained from its provider and remains subject to the provider's release conditions. The dataset, surgical videos and annotations are not redistributed in this repository.

## Method Attribution

- Liu et al., *Stabilizing Temporal Inference Dynamics for Online Surgical Phase Recognition* (2026 preprint), introduced the EGTP and TEC method family. This project independently reimplements EGTP and evaluates it on the lightweight ResNet-18–LSTM baseline. TEC is retained only as a supporting analysis and is not part of the main RQ2 comparison.
- Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017), provides the basis for the Temperature Scaling procedure used in RQ1.
- Gal and Ghahramani, *Dropout as a Bayesian Approximation* (ICML 2016), provides the basis for the MC Dropout procedure used in RQ1.

The independent EGTP implementation includes numerical conventions that were not fully specified in the source paper. These conventions are recorded in `configs/rq2_egtp_validation_protocol_v1.json`. EGTP is not presented as an original method developed by this project, and the authors' official implementation was not used.

## Included and Excluded Third-Party Materials

No third-party source tree was copied into this repository. Dataset files, video frames, extracted features, model checkpoints and prediction archives are intentionally excluded. Users are responsible for obtaining any required external materials and following their applicable licences and terms of use.
