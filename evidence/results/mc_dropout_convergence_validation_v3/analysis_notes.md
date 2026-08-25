# Validation-Only MC Dropout Convergence

- Training seeds: 0, 1 and 2.
- Inference seed: 0.
- Nested pass counts: [10, 20, 30, 50].
- Reference for numerical differences: T=50.
- Test files read: no.

## Frozen practical stability rule

The smallest T must satisfy every criterion for every training seed relative
to the nested T=50 reference.  The project-specific tolerances are stored in
`selected_mc_config.json`; they cover prediction disagreement, entropy and MI
rank correlation, error-AUROC/AUPR differences, and Exact AURC difference.

- Selected minimum stable T: 30.
- Stable candidates across all three seeds: [30, 50].

## T=30 versus T=50

- MC entropy error-AUROC mean: 0.738846
  versus 0.739067.
- MC mutual-information error-AUROC mean:
  0.755398
  versus 0.758354.
- MC entropy AURC mean: 0.156644
  versus 0.156662.
- Mean prediction disagreement rate between T=30 and T=50:
  0.004898.
- Mean MC entropy rank correlation between T=30 and T=50:
  0.999002.
- Mean MC mutual-information rank correlation between T=30 and T=50:
  0.988160.

The thresholds are explicit project operating tolerances, not universal laws.
They were formalised during reproducibility hardening after the initial
exploratory analysis, so the evidence remains development-stage rather than
confirmatory.
