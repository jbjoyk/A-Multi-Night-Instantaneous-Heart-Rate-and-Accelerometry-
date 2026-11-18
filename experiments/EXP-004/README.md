# EXP-004: Frozen Test Set Baseline Evaluation

> [!NOTE]
> **Governance & Naming Note:** In `EXPERIMENTS.md` (DEC-007, DEC-008), EXP-004 designates the 12-candidate validation-only optimizer search and subsequent frozen test evaluation on Cohort Alpha (Accuracy 0.6777, Macro-F1 0.664803, Cohen's $\kappa$ 0.5311). The metric 0.6779 cited below represents the earlier EXP-001 / EXP-003 baseline.

## Overview
Baseline evaluation of the validation-selected optimal SleepChannelNet model on the 7 frozen held-out test subjects, establishing the initial non-adapted benchmark.

## Interface
- Historical execution cell 24 in `experiments/EXP-001/sleepchannelnet_protocol_patch_v2_UPDATED.ipynb`.
- Verification metrics: Accuracy: 0.6866, Macro-F1: 0.6779, Cohen's Kappa: 0.5472.

## Invariants
- Test set evaluated once using best validation checkpoint; no training or threshold adjustments on test data.
- Split hash locked to `a6b569f63126` (7 held-out test subjects).

## Side Effects
- Loads held-out test subject records.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
