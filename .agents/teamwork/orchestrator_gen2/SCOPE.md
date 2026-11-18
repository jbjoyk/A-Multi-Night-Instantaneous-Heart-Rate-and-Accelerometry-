# Scope: Continuous Neural Architecture Search (NAS) Loop

## Architecture
- Baseline: `ConventionalCNNBiGRU` (163,940 params in `exp013_models.py`)
- Candidate Theta (Iteration 10 record): Dual-Scale ACC DSLBS + CASCI (196,373 params)
- Candidate Kappa (Iteration 12): Bounded-Margin Cross-Modal Synergy (BM-CMS) with learned asymmetric thresholds ($\tau_{\text{acc}} > 0, \tau_{\text{ihr}} < 0$) + ArcMargin/Cosine head (196,400 params)
- Hardware: Single GPU `cuda:0` for proxy, dual GPUs `cuda:0` & `cuda:1` for 5-fold CV.
- Virtualenv: `./.venv/bin/python`

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| 1 | Baseline Feature Caching | Pre-computed IHR and ACC feature tensors | M1 | `ORIGINAL_REQUEST.md` |
| 2 | Fast Proxy Pipeline | Fold 1 50-epoch single-seed proxy filter | M2 | `ORIGINAL_REQUEST.md` |
| 3 | Candidate Kappa Deployment | Bounded-Margin Synergy PyTorch module | M3 | Explorer r12_1 |
| 4 | Candidate Kappa Proxy Eval | 50-epoch Fold 1 evaluation on cuda:0 | M3 | Worker r12_1 |
| 5 | Full Cohort 5-Fold Evaluation | 5 folds, 47 subjects, 80 epochs, 3 seeds | M4 | `ORIGINAL_REQUEST.md` |
| 6 | Victory Verification & Final Report | LaTeX/PDF-ready report & git sync | M5 | `ORIGINAL_REQUEST.md` |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Prior Search Grounding | Ground state from Gen 0 & Gen 1 (Iterations 1-11) | none | DONE |
| M2 | Iteration 12 Candidate Kappa Formulation | Verify BM-CMS formulation and unit tests | M1 | DONE |
| M3 | Iteration 12 Deployment & Proxy Eval | Deploy `nas_iteration12_model.py` and run 50-ep proxy | M2 | IN_PROGRESS |
| M4 | 5-Fold Cross-Validation | Full cohort evaluation across all 47 subjects | M3 (if proxy passes) | PLANNED |
| M5 | Victory Reporting & Sync | Final paper benchmark comparison and terminal announcement | M4 (if 5-fold passes) | PLANNED |

## Interface Contracts
### `nas_iteration12_model.py` ↔ `evaluate_nas_iteration12_proxy.py`
- Model Class: `BoundedMarginSynergyPyramidCNNBiGRU`
- Input: Dict with keys `ihr` (B, T, 30), `accel` (B, T, 30), `meta` (B, T, 5), `y` (B, T), `mask` (B, T)
- Output: Dict with keys `logits` (B, T, 4), `loss` (scalar tensor)
- Metric requirements: Accuracy >= 70.0%, Weighted-F1 >= 0.6900
