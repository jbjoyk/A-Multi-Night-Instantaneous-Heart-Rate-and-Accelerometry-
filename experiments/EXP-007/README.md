# EXP-007: Controlled Ablation Study (A0–A5)

## Overview
Controlled ablation study isolating individual architectural components of `SleepAdaptiveNet` across variants A0 through A5 on Seed 42, establishing baseline contributions of reliability gating, boundary loss, and domain adaptation.

## Interface
- `exp007_controlled_ablation_results.pkl`: Serialized performance records for ablation models A0, A1, A2, A3, A4, and A5.

## Invariants
- Random seed locked to 42.
- Split hash locked to `a6b569f63126`.
- Evaluated on validation split.

## Side Effects
- Reads serialized ablation metrics.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
