# EXP-008: Boundary Weight Sweep & GRL Disentanglement

## Overview
Grid sweep evaluating boundary loss penalty weight ($\lambda_{\text{boundary}} \in [0.0, 0.05, 0.1, 0.2, 0.5]$) and domain-adversarial loss weight to disentangle the impact of transition-aware regularization from inter-subject alignment.

## Interface
- `run_exp008` (`run_exp008_boundary_sweep.py`): Entrypoint running grid search across boundary weight values.
- `train_sweep_model` (`run_exp008_boundary_sweep.py`): Configurable training loop for arbitrary loss coefficient combinations.
- `evaluate_validation_sweep` (`run_exp008_boundary_sweep.py`): Evaluates epoch-level Macro-F1, Cohen's Kappa, and stage-specific F1 on the validation set.

## Invariants
- Execution confined strictly to training and validation splits; frozen test cohort is never loaded.
- Model architecture fixed to `SleepAdaptiveNet` base specification.

## Side Effects
- Saves sweep evaluation records to `exp008_boundary_sweep_results.pkl`.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
