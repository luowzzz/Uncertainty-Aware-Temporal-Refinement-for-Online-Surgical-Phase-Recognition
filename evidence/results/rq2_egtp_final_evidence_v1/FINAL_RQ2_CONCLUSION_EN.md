# Final RQ2 evidence

The validation-selected method was the uncalibrated EGTP applied to the frozen LSTM baseline with k=0.6. No strict operating point existed; the method was selected from the basic feasible set and must therefore be interpreted as a stability-transition-sensitivity trade-off.

Across the three test seeds, EGTP achieved Macro-F1 0.5564 ± 0.0294, produced 203.3 ± 25.4 boundaries, and obtained boundary precision/recall/F1 of 0.0610/0.2242/0.0957 at ±10 s. Relative to raw argmax, Macro-F1 changed by -0.0055, boundary F1 by +0.0421, boundary recall by -0.2485, and mean-video TFI by -6.9979.

The paper-derived TEC training ablation did not satisfy the validation feasibility constraints and was rejected before test evaluation. Temperature scaling produced almost identical EGTP decisions because dynamic normalisation largely cancels global logit-scale changes.

The evidence supports a conditional conclusion: causal normalised evidence accumulation reduces temporal fragmentation, but genuine transitions can still be delayed or suppressed. Results are three-seed iterative-development evidence rather than a fully independent confirmatory test because the test split had been inspected earlier in the project.
