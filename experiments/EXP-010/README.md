# EXP-010: Long-Training Convergence Benchmark

## Overview
Extended 500-epoch convergence benchmark analyzing asymptotic learning behavior, late-stage overfitting, and validation peak stability for `SleepAdaptiveNet`.

## Interface
- `run_exp010` (`run_exp010_convergence.py`): Orchestrates 500-epoch training and validation monitoring.

## Invariants
- Maximum training epochs strictly bounded to 500.
- Peak validation checkpoint selected by Macro-F1 independently of terminal epoch state.

## Side Effects
- Writes checkpoint models to `checkpoints_exp010/`.
- Exports training curves to `exp010_convergence_curves.png`.
- Serializes complete per-epoch loss and validation metrics to `exp010_training_history.pkl`.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
