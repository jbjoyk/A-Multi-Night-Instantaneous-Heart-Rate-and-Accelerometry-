# Scope: Continuous Neural Architecture Search (NAS) Loop

## Architecture & Problem Overview
- **Baseline Model A**: `ConventionalCNNBiGRU` (163,940 params in `exp013_models.py`)
- **Iteration 1 Model**: `MultiScaleConventionalCNNBiGRU` (192,692 params in `experiments/EXP-014/nas_iteration1_model.py`)
- **Published Benchmark Target**: Table V in SLAMSS-IFS paper: **71.04% Accuracy, 0.7079 Weighted-F1** across 47 subjects (5-fold CV).
- **Core Hypotheses**:
  - **Hypothesis H-F**: Cross-modal interaction between instantaneous heart rate (IHR) and accelerometry (ACC) captures autonomic-somatomotor coupling essential for sleep staging.
  - **Hypothesis H-A**: Cross-modal / adaptive representations produce statistically significant and subject-invariant gains across all 47 subjects in BIDSleep.
- **Hardware Constraints**:
  - Fast Proxy: Single GPU `cuda:0`, 50 epochs, Fold 1, seed 42.
  - Full Cohort: 5-fold CV, 80 epochs, 3 seeds parallelized across `cuda:0` and `cuda:1`.
- **Parameter Budget**:
  - Target: ~162k–196k parameters (>90% compression vs 5.47M paper model). Soft constraint if performance exceeds target.

## Feature Inventory & Search Space
| # | Feature / Mutation Target | Description | Milestone | Source |
|---|---------------------------|-------------|-----------|--------|
| 1 | Baseline & Iteration 1 Benchmarks | Verification of Model A (163.9k) & Iteration 1 (192.7k) | Complete | EXP-013 / EXP-014 |
| 2 | Iteration 2: Cross-Modal Interaction (H-F) | Tensor-safe cross-modal attention / gating module between IHR & ACC streams | Iteration 2 | R1, H-F |
| 3 | Fast Proxy Filter (Fold 1, 50ep) | Fold 1 evaluation (Acc >= 70.0%, Weighted-F1 >= 0.69) | Iteration 2 | R1, R3 |
| 4 | Full 5-Fold Evaluation (47 subjects, 80ep) | 5 folds, 3 seeds on cuda:0 & cuda:1 (Acc > 71.04%, Weighted-F1 > 0.7079) | Iteration 2 | R1, R3 |
| 5 | Victory Synchronization | Final report, git commit/push, victory statement | Victory | R4 |

## Search Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Setup & NAS Survey | Survey H-F & H-A candidate mechanisms, existing codebases and evaluation scripts | none | DONE |
| M2 | Iteration 2–6 Mutations | Propose, implement, and proxy-evaluate Iterations 2–6 (all reverted per proxy filter) | M1 | DONE |
| M3 | Continuous NAS Search Loop | Propose, implement, proxy-evaluate Iteration 7+; trigger 5-fold CV if proxy passes | M2 | IN_PROGRESS |
| M4 | Victory Condition | Generate report, git sync, output victory confirmation | M3 | PLANNED |

