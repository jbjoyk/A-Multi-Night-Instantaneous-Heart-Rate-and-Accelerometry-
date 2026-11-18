# EXP-017: Sequence Horizon Evaluation ($T=600$ vs $T \le 1200$)

## Overview
Evaluates the impact of extended temporal context horizons on sleep staging performance by comparing standard truncated sequences ($T=600$ epochs / 5 hours) against full-night sequences ($T \le 1200$ epochs / up to 10 hours) for Model A across seeds 42, 123, and 2024.

## Interface
- `run_governed_exp017` (`run_exp017_execution.py`): Orchestrates execution of paired sequence length runs.
- `evaluate_model_on_epochs` (`run_exp017_execution.py`): Evaluates accuracy, Macro-F1, and stage F1 under variable sequence lengths.
- `compute_paired_cluster_bootstrap` (`run_exp017_execution.py`): Paired cluster bootstrap calculating delta distributions between $T=600$ and $T=1200$.

## Invariants
- Recurrent sequence padding preserves temporal alignment without backward-leakage.
- Random seeds fixed to `[42, 123, 2024]`.

## Side Effects
- Saves checkpoints to `checkpoints/`.
- Writes JSON summaries (`EXP017_BOOTSTRAP.json`, `ModelA_T*.json`) and reports (`EXP017_FINAL_REPORT.md`, `EXP017_SUMMARY.csv`).

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_recurrent_padding.py tests/test_dataset_gate.py`
