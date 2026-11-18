# EXP-009: Multi-Seed Stability Audit

## Overview
Multi-seed stability experiment comparing `B0_GRL` (SleepAdaptiveNet with subject domain adaptation) against `B2` (ablation without GRL) across independent random seeds (42, 123, 2024). Validates whether observed domain alignment performance represents robust optimization or stochastic variance.

## Interface
- `run_exp009` (`run_exp009_multiseed.py`): Top-level runner executing dual-model comparisons across random seed triplets.

## Invariants
- Random seeds fixed to `[42, 123, 2024]`.
- Batch size, learning rate ($10^{-4}$), optimizer (Adam), and split definitions (`a6b569f63126`) held constant.

## Side Effects
- Serializes metric summaries and per-seed results to `exp009_multiseed_results.pkl`.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
