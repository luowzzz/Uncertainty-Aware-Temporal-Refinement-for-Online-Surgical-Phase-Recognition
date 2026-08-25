# Temperature Scaling Calibration v2

## Frozen data flow

- One positive scalar temperature was fitted separately for each training seed.
- Fitting read only validation videos 11-14 and minimised pooled per-second multiclass NLL.
- The test stage read videos 15-21 only after the temperature file and protocol were frozen.
- Inputs are original logits saved by each validation-selected best checkpoint; no probabilities were transformed back into logits.
- Calibration is an RQ1 reliability analysis and does not enter the frozen RQ2 gate.

## Metric definitions

- NLL: `-(1/N) * sum_i log(p_i,y_i)`; lower is better.
- Multiclass Brier: `(1/N) * sum_i sum_c (p_i,c - 1[y_i=c])^2`; lower is better.
- ECE-15: top-label ECE with 15 equal-width confidence bins on [0,1], weighted by bin frequency; confidence 1 is included in the final bin.
- Accuracy: pooled per-second argmax accuracy.
- Macro-F1: unweighted mean of seven class F1 values (`zero_division=0`).
- Each seed pools all seconds in its split. Three-seed summaries are the arithmetic mean and sample standard deviation (`ddof=1`) of seed-level metrics.
- Per-video metrics use the same definitions within each video; they are diagnostic and are not used to fit temperature.

## Validation-fitted temperatures

- Seed 00: T=1.574526; validation NLL 0.873468 -> 0.758987.
- Seed 01: T=1.362818; validation NLL 0.807915 -> 0.761612.
- Seed 02: T=1.621691; validation NLL 1.030345 -> 0.899504.

## Three-seed test summary

- NLL: 1.159871 ± 0.165159 -> 0.990308 ± 0.151245.
- Multiclass Brier: 0.530516 ± 0.048088 -> 0.489272 ± 0.058096.
- ECE-15: 0.172977 ± 0.027922 -> 0.081565 ± 0.029682.
- Accuracy: 0.656439 ± 0.023059 before and after calibration.
- Macro-F1: 0.561934 ± 0.032481 before and after calibration.

## Correctness and interpretation

- Temperature scaling improved NLL, multiclass Brier score and 15-bin ECE for every seed without changing phase recognition accuracy.
- Every second retained the same argmax class; accuracy and macro-F1 are exactly unchanged within each seed.
- Probability rows sum to one within numerical precision, and the saved video/time/label order was cross-checked against feature metadata and deterministic per-second tables.
- The validation fitting manifest records that no test file or label was read before temperatures were frozen.
- These results are three-seed iterative-development evidence, not a fully independent confirmatory test, because the test split had been inspected earlier in the project.

## Reproduction commands

```powershell
python scripts\28_fit_temperature_scaling_validation.py
python scripts\29_evaluate_temperature_scaling_test.py
```
