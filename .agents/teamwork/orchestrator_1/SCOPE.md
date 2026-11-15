# Scope: NAS Iteration 1 - Model A Mutation & Evaluation

## Architecture
- Baseline: Model A (`ConventionalCNNBiGRU`, 163,940 parameters in `exp013_models.py`)
- Mutated Model: `MultiScaleConventionalCNNBiGRU` (192,692 parameters in `experiments/EXP-014/nas_iteration1_model.py`)
- Theoretical Basis: Hypothesis H-A from `RESEARCH_DIRECTION_v3_FULL.md` (multi-night within-person adaptation via Pre-Recurrent Multi-Scale Temporal Fusion, MSTF-Pre, capturing temporal and multi-scale dynamics across 30s, 90s, 150s contexts)
- Structural Mutation:
  - Pointwise bottleneck projection: Conv1d(144, 96, 1)
  - 3 parallel depthwise convolutions: kernel sizes $k \in \{1, 3, 5\}$, groups=32
  - Channel concatenation: (B, 96, T)
  - Pointwise restoration projection: Conv1d(96, 144, 1)
  - Residual connection: GELU(LayerNorm(144)(X + out))
- Constraints:
  - Parameter count: 192,692 (<= 196,728 cap, +17.54% over baseline, +4,036 headroom)
  - Dimension compatibility: (B, T, 144) preserved, full downstream GRU/classifier compatibility
  - Hyperparameters: batch size = 4, Adam lr = 1e-4, weight_decay = 0.0, sequence length = 600, CE + 0.01*RCCLoss
  - Single GPU execution on `cuda:0` using `./.venv/bin/python`
  - Fold 1 evaluation on 54 test nights (32,098 windows): Macro-F1 = 0.3556 (+5.01 pp over baseline)

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | Baseline & Hypothesis Audit | Analyze Model A in `exp013_models.py` and H-A in `RESEARCH_DIRECTION_v3_FULL.md` | M1 | ORIGINAL_REQUEST | DONE |
| 2 | Structural Mutation Design | Propose isolated structural mutation conforming to parameter budget (<= 196,728) | M1 | ORIGINAL_REQUEST | DONE |
| 3 | Tensor-Safe PyTorch Module | Implement mutated architecture as `nn.Module` with dimension-preserving projection layers | M1 | ORIGINAL_REQUEST | DONE |
| 4 | Forward Pass Verification | Test forward pass locally with expected BIDSleep input tensor shapes on `cuda:0` | M1 | ORIGINAL_REQUEST | DONE |
| 5 | 1-Epoch Eval Loop on Fold 1 | Execute 1-epoch loop on Fold 1 using `./.venv/bin/python` without modifying batch size, lr, or optimizer | M1 | ORIGINAL_REQUEST | DONE |
| 6 | Metric Reporting | Calculate and log Macro-F1 metric | M1 | ORIGINAL_REQUEST | DONE |

## Milestones
| # | Name | Scope | Dependencies | Status | Key Outputs |
|---|------|-------|-------------|--------|-------------|
| 1 | M1: NAS Iteration 1 | Model mutation, implementation, forward test, 1-epoch Fold 1 eval, Macro-F1 | none | DONE | `experiments/EXP-014/nas_iteration1_model.py`, `experiments/EXP-014/evaluate_nas_iteration1.py`, `nas_iteration1_eval_results.json` |

## Acceptance Criteria
- [x] Mutated architecture implements an isolated structural mutation grounded in hypothesis H-A from `RESEARCH_DIRECTION_v3_FULL.md`.
- [x] Total parameter count does not exceed 196,728 parameters (actual: 192,692, +17.54% over baseline 163,940).
- [x] All tensor operations and dimension transitions are dimension-compatible, with projection layers implemented for any channel changes.
- [x] Forward pass executes without shape or runtime errors on expected BIDSleep input tensor dimensions.
- [x] 1-epoch evaluation loop completes successfully on Fold 1 (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) on single GPU `cuda:0` using `./.venv/bin/python`.
- [x] Evaluation script logs the Macro-F1 score without altering batch size, optimizer, or learning rate.
