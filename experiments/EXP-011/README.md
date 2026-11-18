# EXP-011: Multi-Seed Long-Training Convergence Audit

## Overview
Replication of the 500-epoch training horizon across three random seeds (42, 123, 2024) to establish statistical variance bounds on convergence rates, terminal generalization, and optimal early-stopping windows.

## Interface
- `run_exp011` (`run_exp011_multiseed_convergence.py`): Executes the multi-seed convergence benchmark suite.
- `train_seed_convergence` (`run_exp011_multiseed_convergence.py`): Single-seed training pipeline logging metric curves.

## Invariants
- Random seeds locked to `[42, 123, 2024]`.
- Identical data partitioning and learning rate schedule across all seed iterations.

## Side Effects
- Saves intermediate and final model checkpoints in `checkpoints_exp011/`.
- Writes aggregated convergence trajectories to `exp011_multiseed_history.pkl`.
- Generates composite convergence visualization `exp011_convergence_curves.png`.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
