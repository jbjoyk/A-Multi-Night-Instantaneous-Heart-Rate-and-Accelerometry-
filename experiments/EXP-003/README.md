# EXP-003: Baseline Artifact QC Audit & Validation Search Mapping

> [!NOTE]
> **Governance & Naming Note:** In `EXPERIMENTS.md` (DEC-005), EXP-003 formally designates the QC lag correlation audit and baseline artifact verification on saved outputs from EXP-001 (inheriting the 33/7/7 split, hash `a6b569f63126`, Macro-F1 0.67791). The 12-candidate validation-only optimizer search described below was conducted and reported under EXP-004 (DEC-007, DEC-008, DEC-010).

## Overview
Hyperparameter and optimization search on the validation cohort, tuning AdamW learning rate, grouped weight decay, dropout rates, and learning rate scheduling for SleepChannelNet.

## Interface
- Historical execution cell 23 in `experiments/EXP-001/sleepchannelnet_protocol_patch_v2_UPDATED.ipynb`.
- Verification metrics: Accuracy: 0.68659, Macro-F1: 0.67791, Cohen's Kappa: 0.54722.

## Invariants
- Evaluated strictly on the 7-subject validation cohort; zero test-set exposure during hyperparameter selection.
- Split hash locked to `a6b569f63126` (Seed 0, 33 train / 7 val / 7 test).

## Side Effects
- Allocates CUDA device memory during validation sweeps.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
