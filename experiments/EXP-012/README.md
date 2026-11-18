# EXP-012: Frozen Test Benchmark & Audit Dossier

## Overview
Comprehensive frozen test evaluation of `SleepAdaptiveNet` across the 7 held-out test subjects over three independent seeds (42, 123, 2024). Generates exhaustive per-subject metrics, paired comparisons, confusion matrices, and formal review dossiers.

## Interface
- `run_exp012` (`run_exp012_final_benchmark.py`): Orchestrates evaluation across seeds and subjects on held-out test partitions.
- `evaluate_test_set` (`run_exp012_final_benchmark.py`): Evaluates accuracy, Macro-F1, per-stage F1, Cohen's Kappa, and calibration error (ECE).
- `compile_pdf.py`: Compiles HTML and markdown assets into formal benchmark PDF dossier.
- `generate_report_charts.py`: Generates confusion matrices and subject-level performance distributions.

## Invariants
- Test set strictly evaluated without parameter updating or threshold tuning on test data.
- Dataset split integrity verified via split fingerprint `a6b569f63126`.

## Side Effects
- Serializes evaluation payloads to `exp012_test_results.json` and `exp012_test_results.pkl`.
- Exports CSV breakdown to `exp012_subject_metrics.csv`.
- Generates visual and publication artifacts (`exp012_confusion_matrices.png`, `EXP012_BENCHMARK_REPORT.pdf`).

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_evaluator.py tests/test_dataset_gate.py`
