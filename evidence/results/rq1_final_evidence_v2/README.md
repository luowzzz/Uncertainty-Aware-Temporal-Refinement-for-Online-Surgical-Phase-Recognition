# RQ1 final thesis evidence v2

This is the authoritative RQ1 thesis evidence bundle. All quantitative CSV
files are byte-identical copies of `rq1_final_evidence_v1`; no training,
inference, metric recomputation, parameter selection, or conclusion change was
performed.

Version 2 exists because the v1 `README.md` changed after its manifest was
written. The stale README hash caused one reproducibility test to fail even
though the quantitative evidence files remained intact. This bundle preserves
v1 and records the repair instead of overwriting history.

Core thesis evidence:

- overall error detection: AUROC, AUPR and Exact AURC;
- near/far conditional AUROC and AUPR, with +/-10 s as the primary window;
- T=30 MC Dropout selected by validation convergence;
- raw-logit Temperature Scaling fitted independently on validation for each
  training seed, reported using NLL, Brier score and ECE-15.

The test split had been inspected during earlier iterative development. These
results are therefore traceable three-seed iterative-development evidence, not
a fully independent confirmatory test.
