# Wearable Sleep Staging Research Project Blueprint

## System Overview
This repository contains reproducible research code, experimental benchmarks, cross-validation studies, and scientific reports for wearable four-class sleep staging (Wake, Light, Deep, REM) using instantaneous heart rate and accelerometry against EEG-derived polysomnography labels on the *Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels* (47 subjects, 253 recorded nights, 246 usable nights, ~145k valid 30-second windows).

Key architectural milestones include:
- **SLAMSS-IFS Baseline Reproduction (EXP-005)**: Independent reproduction audit of the official IEEE TBME 2026 architecture (`BIDSLabUMass/SLAMSS-IFS`).
- **SleepAdaptiveNet (EXP-006 to EXP-012)**: Compact dual-branch architecture (~162k parameters, 33.7x compression) incorporating reliability gating, BiGRU temporal context, domain-adversarial subject alignment (GRL), and boundary-aware loss.
- **Factorial Component Attribution (EXP-013)**: Rigorous single-component ablation across 34,241 held-out test windows.
- **75-Run 5-Fold Cross-Validation & NAS (EXP-014)**: Definitive subject-level cross-validation across all 47 subjects and continuous Neural Architecture Search.
- **Within-Person Adaptation Protocol (EXP-018)**: Multi-night longitudinal personalization under compute schedule invariance.
- **Extended Convergence Benchmark (EXP-019)**: 500-epoch asymptotic training comparing baseline and multi-scale temporal fusion.

## Local Interface & Invariants
- `DatasetGate` (`dataset_gate.py`): Enforces dataset verification, split disjointness, cache checksums, and strict zero-test-set-leakage gates (`verify_dataset_structure`, `verify_split`, `assert_no_test_leakage`).
- `SleepEvaluator` (`evaluator.py`): Hierarchical metric evaluator computing primary Macro-F1, secondary Accuracy, Cohen's Kappa, stage-specific F1, and expected calibration error (`evaluate_predictions`, `evaluate_model`, `compute_ece`).
- `ClinicalHypnogramEvaluator` (`evaluator.py`): Computes clinical hypnogram statistics including total sleep time, sleep efficiency, sleep onset latency, and WASO (`compute_night_metrics`, `evaluate_cohort`).
- `ResearchGovernor` (`governor.py`): Autonomy controller enforcing multi-level safety barriers, token-based execution authorization, and epoch budgets (`generate_governance_token`, `verify_governance_token`, `check_experiment_safety`).
- `run_data_sanity_check` (`data_sanity_check.py`): Validates physical signal ranges, NaN absence, and sampling alignment.
- `run_all_ablations_seed42` (`run_controlled_experiment.py`): Controlled ablation driver executing model variants A0 through A5 on Seed 42.

## Subdirectories & System Boundaries
- [experiments](./experiments/README.md) - Complete suite of experimental investigations (EXP-001 through EXP-019), model definitions, and proxy evaluation suites.
- [research_engine](./research_engine/README.md) - Governed autonomous research engine including state-machine lifecycle enforcement, hypothesis generation, dual-GPU resource management, and peer-review gates.

## Global Invariants
- **Subject-Level Disjointness**: All train/val/test splits must be strictly partitioned by Subject ID; no subject's nights may appear across multiple partitions (`dataset_gate.py`).
- **Frozen Test Set Protection**: Test partitions are strictly evaluated using validation-selected checkpoints; test data is never exposed during training or hyperparameter search (`evaluator.py`).
- **Standardized Classification Space**: Fixed 4-class target domain across all models: Wake (0), Light (1), Deep (2), REM (3).
- **GPU Resource Isolation**: Multi-GPU execution on dual RTX 3060 GPUs (`cuda:0`, `cuda:1`) requires active non-expired lease tokens (`research_engine/resource_manager.py`).

## Global Side Effects
- Reads raw sensor recordings and preprocessed cache arrays from the PhysioNet dataset directory.
- Updates persistent research governance and experiment logs in `research_memory.json`.
- Manages lockfiles for CUDA hardware allocation in system temporary paths.

## System Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/`
