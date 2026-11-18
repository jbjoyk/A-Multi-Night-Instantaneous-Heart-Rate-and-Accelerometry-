# Project: Autonomous NAS Syndicate for BIDSleep Staging

## Architecture
- Baseline Model A: `ConventionalCNNBiGRU` (163,940 params in `exp013_models.py`).
- Iteration 1 Model: `MultiScaleConventionalCNNBiGRU` (192,692 params in `experiments/EXP-014/nas_iteration1_model.py`).
- Iteration 2 Mutation: `CrossModalFiLMCNNBiGRU` (193,380 params) targeting Hypothesis H-F (somatomotor-autonomic cross-modal FiLM conditioning) and Hypothesis H-A (LayerNorm and identity initialization for inter-subject invariance across all 47 subjects).
- Published Benchmark Target: Song et al., IEEE TBME 2026 (Table V): 71.04% Accuracy, 0.7079 Weighted-F1 across all 47 subjects.

## Feature Inventory
| # | Feature / Search Target | Description | Milestone | Source |
|---|-------------------------|-------------|-----------|--------|
| 1 | Baseline & Iteration 1 Model Auditing | Complete audit of Model A (163,940) and Iteration 1 (192,692) | Complete | EXP-013, EXP-014 |
| 2 | Fast Proxy Pipeline Design | Fold 1, 50 epochs, seed 42 on cuda:0 (Threshold: Acc >= 70.0%, WF1 >= 0.69) | Iteration 2 | R1, R3 |
| 3 | Dual-GPU 5-Fold CV Design | 5 folds x 3 seeds, 80 epochs across cuda:0 & cuda:1 (Threshold: Acc > 71.04%, WF1 > 0.7079) | Iteration 2 | R1, R3 |
| 4 | Iteration 2 Candidate B Implementation | Tensor-safe `CrossModalFiLMFusion` (193,380 params) | Iteration 2 | R1, R2, H-F |
| 5 | Iteration 2 Proxy Evaluation | Execute 50-epoch Fold 1 proxy eval on cuda:0 | Iteration 2 | R1, R3 |
| 6 | Iteration 2 5-Fold Evaluation | Execute full 5-fold CV if proxy passes | Iteration 2 | R1, R3 |
| 7 | Victory Synchronization | Final report, git commit/push, terminal victory statement | Victory | R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Survey & Mutation Selection | Map problem space, hardware, candidate mutations | none | DONE |
| M2 | Iteration 2 Implementation & Proxy Eval | Implement Candidate B and run 50-epoch Fold 1 proxy eval | M1 | IN_PROGRESS |
| M3 | Iteration 2 5-Fold CV & Gating | Full 5-fold CV across 47 subjects on dual GPUs | M2 | PLANNED |
| M4 | Victory Condition & Git Push | Generate PDF-ready report, sync git, declare victory | M3 | PLANNED |

## Interface Contracts
### `CrossModalFiLMFusion`
- Inputs: `h_hr` $(B, T, 64)$, `h_acc` $(B, T, 64)$
- Affine Modulation: ACC generates $\gamma_{hr}, \beta_{hr} \in \mathbb{R}^{B \times T \times 64}$; HR generates $\gamma_{acc}, \beta_{acc} \in \mathbb{R}^{B \times T \times 64}$
- Joint Gating: $[h_{hr}^{film} ; h_{acc}^{film}] \in \mathbb{R}^{B \times T \times 128} \to g_{hr}, g_{acc} \in \mathbb{R}^{B \times T \times 64}$
- Outputs: `h_hr_out` $(B, T, 64)$, `h_acc_out` $(B, T, 64)$
- Downstream Concatenation: $[h_{hr\_out} ; h_{acc\_out} ; h_{meta}] \in \mathbb{R}^{B \times T \times 144} \to \text{BiGRU}(144 \to 96)$.

## Code Layout
- Model: `experiments/EXP-014/nas_iteration2_model.py`
- Proxy Evaluator: `experiments/EXP-014/evaluate_nas_iteration2_proxy.py`
- 5-Fold Evaluator: `experiments/EXP-014/evaluate_nas_iteration2_5fold.py`
- Results JSON: `experiments/EXP-014/nas_iteration2_proxy_results.json`, `experiments/EXP-014/nas_iteration2_5fold_results.json`
