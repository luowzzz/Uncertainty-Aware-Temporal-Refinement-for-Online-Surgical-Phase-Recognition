# Three-Seed MC Dropout Reproducibility Commands

Run from `E:\AutoLaparoProject` with the project Python environment.

## Validation-only nested convergence

```powershell
python scripts\23_mc_dropout_convergence_validation.py --out-dir outputs\mc_dropout_convergence_validation_v3
```

## Frozen test consistency runs

```powershell
python scripts\08_eval_mc_dropout.py --training_seed 0 --checkpoint outputs\v2_lstm_online_resnet18_seed00\checkpoints\best.pt --out_dir outputs\rq1_mc_dropout_t30_seed00_v4 --T 30 --inference_seed 0 --splits test --windows 5 10 20 --mc_config configs\mc_dropout_evaluation_v4.json
```

```powershell
python scripts\08_eval_mc_dropout.py --training_seed 1 --checkpoint outputs\v2_lstm_online_resnet18_seed01\checkpoints\best.pt --out_dir outputs\rq1_mc_dropout_t30_seed01_v4 --T 30 --inference_seed 0 --splits test --windows 5 10 20 --mc_config configs\mc_dropout_evaluation_v4.json
```

```powershell
python scripts\08_eval_mc_dropout.py --training_seed 2 --checkpoint outputs\v2_lstm_online_resnet18_seed02\checkpoints\best.pt --out_dir outputs\rq1_mc_dropout_t30_seed02_v4 --T 30 --inference_seed 0 --splits test --windows 5 10 20 --mc_config configs\mc_dropout_evaluation_v4.json
```

## Aggregation

```powershell
python scripts\27_finalize_multiseed_mc_protocol.py
```

## Interpretation

Incorrect MC-averaged predictions are positive errors. Deterministic and MC scores are not treated as strictly paired when their prediction error sets differ.

The three-seed protocol improves reproducibility and robustness, but the results remain iterative-development evidence because the test set had previously been inspected.
