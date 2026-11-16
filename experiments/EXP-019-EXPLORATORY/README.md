# EXP-019: 500-Epoch Extended Training Benchmark

## Overview
Extended 500-epoch benchmark evaluating Baseline Model A (`ConventionalCNNBiGRU`, 163,940 parameters) against Mutated Model A (`MultiScaleConventionalCNNBiGRU`, 192,692 parameters, +17.54%) on Fold 1 under identical training hyperparameters. Assesses asymptotic learning dynamics, peak generalization, stage-specific F1 gains, and overfitting resistance.

## Interface
- `run_exp019_500epochs.py`: Parallel dual-GPU orchestrator executing Baseline on `cuda:0` and MultiScale on `cuda:1`.
- `train_single_model` (`run_exp019_500epochs.py`): Full 500-epoch training loop tracking inner-val peak and terminal metrics.
- `evaluate_loader` (`run_exp019_500epochs.py`): Evaluates Macro-F1, Accuracy, Cohen's Kappa, Matthews Correlation Coefficient (MCC), and per-stage F1.
- `load_fold1_datasets` (`run_exp019_500epochs.py`): Loads deterministic Fold 1 subject partition.
- `synthesize_and_report` (`run_exp019_500epochs.py`): Compiles markdown report comparing peak and asymptotic metrics.

## Invariants
- 500 epochs trained continuously without premature termination.
- Hyperparameters identical between models: Adam ($lr=10^{-4}$), batch size 4, sequence length 600, seed 42.
- Parameter ceiling capped within $+20\%$ over baseline (+17.54% actual, maintaining $>96\%$ compression vs. paper benchmark).

## Side Effects
- Saves best-checkpoint and terminal-epoch models in `checkpoints/`.
- Exports summary metrics to `exp019_results/` and comprehensive synthesis to `EXP019_REPORT.md`.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_exp019_pipeline.py`
