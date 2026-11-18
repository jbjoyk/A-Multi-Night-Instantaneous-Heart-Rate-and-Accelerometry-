# EXP-018: Multi-Night Within-Person Adaptation Protocol (AUTORESEARCH)

## Overview
Implements the multi-night within-person adaptation research protocol (`AUTORESEARCH`). Evaluates token exposure schedule invariance, random-crop data sampling, sliding-window full-night inference, and parametric saturation curves to test whether personalization on sequential recording nights improves staging accuracy.

## Interface
- `RandomCropSleepDataset` (`exp018_dataset.py`): PyTorch Dataset performing on-the-fly random temporal cropping.
- `verify_compute_schedule_invariant` (`exp018_dataset.py`): Asserts that total token exposure ($N_{\text{steps}} \times B \times L$) remains invariant across training regimes.
- `create_infinite_batch_stream` (`exp018_dataset.py`): Yields infinite shuffled training mini-batches for exact step-budgeted schedules.
- `SlidingWindowPredictor` (`exp018_inference.py`): Aggregates overlapping window predictions with boundary tapering to score full nights.
- `run_phase1_orchestrator`, `run_phase1b_orchestrator`, `run_phase2_orchestrator` (`run_exp018_execution.py`): Multi-phase adaptation benchmarks.

## Invariants
- Total training token exposure strictly preserved ($N_{\text{steps}} \times B \times L = \text{const}$) across all sequence configurations.
- Temporal adaptation strictly causal: adaptation to night $k$ uses only nights $1 \dots k-1$; future night data is never accessed.
- Subject disjointness strictly verified before training or adaptation.

## Side Effects
- Creates directories `phase1_results/`, `phase1b_results/`, `phase2_results/`.
- Generates markdown analysis reports (`EXP018_PHASE1_REPORT.md`, `EXP018_PHASE1B_REPORT.md`, `EXP018_PHASE2_REPORT.md`).

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_exp018_pipeline.py`
