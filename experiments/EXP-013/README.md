# EXP-013: Factorial Component Attribution Study

## Overview
Rigorous single-component factorial ablation study comparing five matched-capacity architectures (Models A–E) across 34,241 held-out test windows. Disentangles the individual contributions of conventional CNN-BiGRU (Model A), adaptive reliability gating (Model B), domain adversarial subject adaptation (Model C), full SleepAdaptiveNet (Model D), and boundary regularization (Model E).

## Interface
- `ConventionalCNNBiGRU` (`exp013_models.py`): Clean, capacity-matched baseline model without gating or GRL (~163k parameters).
- `build_exp013_model` (`exp013_models.py`): Factory function constructing variants A, B, C, D, and E with identical backbone dimensions.
- `run_exp013` (`run_exp013_attribution.py`): Top-level runner executing multi-seed training, validation triage, and outer-test evaluation.
- `compute_paired_bootstrap` (`run_exp013_attribution.py`): Computes subject-clustered paired bootstrap distributions (1,000 resamples) and 95% confidence intervals.
- `run_preflight_audit` (`preflight_audit_exp013.py`): Verifies parameter matching, gradient flow, and freeze protection.

## Invariants
- Parameter budget strictly matched within ~162k–164k parameters across all five variants.
- Statistical significance evaluated via paired subject-clustered bootstrapping.

## Side Effects
- Saves trained model checkpoints in `EXP013_OUTPUTS/checkpoints/`.
- Compiles findings into `EXP013_RESULTS.md`.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_evaluator.py tests/test_dataset_gate.py`
