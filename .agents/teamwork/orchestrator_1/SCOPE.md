# Scope: NAS Iteration 1 - Model A Mutation & Evaluation

## Architecture
- Baseline: Model A (`ConventionalCNNBiGRU`, 163,940 parameters in `exp013_models.py`)
- Theoretical Basis: Hypothesis H-A from `RESEARCH_DIRECTION_v3_FULL.md` (multi-night within-person adaptation, capturing temporal and multi-scale dynamics in instantaneous heart rate and accelerometry)
- Target Mutation: Isolated structural mutation (e.g. multi-scale feature fusion or depthwise separable convolutions)
- Constraints:
  - Parameter count <= 196,728 (<= +20% over baseline 163,940)
  - Dimension compatibility with projection layers where channels change
  - Fixed training hyperparameters (batch size, optimizer, learning rate unchanged)
  - Single GPU execution on `cuda:0` using `./.venv/bin/python`
  - Fold 1 evaluation (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) reporting Macro-F1

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Baseline & Hypothesis Audit | Analyze Model A in `exp013_models.py` and H-A in `RESEARCH_DIRECTION_v3_FULL.md` | M1 | ORIGINAL_REQUEST |
| 2 | Structural Mutation Design | Propose isolated structural mutation conforming to parameter budget (<= 196,728) | M1 | ORIGINAL_REQUEST |
| 3 | Tensor-Safe PyTorch Module | Implement mutated architecture as `nn.Module` with dimension-preserving projection layers | M1 | ORIGINAL_REQUEST |
| 4 | Forward Pass Verification | Test forward pass locally with expected BIDSleep input tensor shapes on `cuda:0` | M1 | ORIGINAL_REQUEST |
| 5 | 1-Epoch Eval Loop on Fold 1 | Execute 1-epoch loop on Fold 1 using `./.venv/bin/python` without modifying batch size, lr, or optimizer | M1 | ORIGINAL_REQUEST |
| 6 | Metric Reporting | Calculate and log Macro-F1 metric | M1 | ORIGINAL_REQUEST |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: NAS Iteration 1 | Model mutation, implementation, forward test, 1-epoch Fold 1 eval, Macro-F1 | none | IN_PROGRESS |

## Acceptance Criteria
- [ ] Mutated architecture implements an isolated structural mutation grounded in hypothesis H-A from `RESEARCH_DIRECTION_v3_FULL.md`.
- [ ] Total parameter count does not exceed 196,728 parameters (<= 20% increase over the baseline 163,940 parameters).
- [ ] All tensor operations and dimension transitions are dimension-compatible, with projection layers implemented for any channel changes.
- [ ] Forward pass executes without shape or runtime errors on expected BIDSleep input tensor dimensions.
- [ ] 1-epoch evaluation loop completes successfully on Fold 1 (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) on single GPU `cuda:0` using `./.venv/bin/python`.
- [ ] Evaluation script logs the Macro-F1 score without altering batch size, optimizer, or learning rate.
