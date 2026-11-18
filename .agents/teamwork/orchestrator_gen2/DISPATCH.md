## 2026-11-15T09:56:52Z
Act as an autonomous Neural Architecture Search (NAS) syndicate to continuously iterate architectural mutations until definitively beating the published SLAMSS-IFS paper model on its primary metrics and resolving hypotheses H-F and H-A on the BIDSleep dataset.

Working directory: .agents/teamwork/orchestrator_gen2/
Identity: Project Orchestrator (Generation 2)
Parent Sentinel: fb288bfc-537f-41c9-8fa6-e473a32ff9a1

## Requirements

### R1. Continuous Structural Mutation and Architecture Search Loop
Execute a continuous neural architecture search loop starting at Iteration 12 targeting Hypothesis H-F (cross-modal IHR-accelerometry interaction module) and Hypothesis H-A (adaptation/architecture with statistically significant gain across all 47 subjects). For each iteration:
1. Propose one new structural mutation targeting H-F and H-A.
2. Implement the tensor-safe PyTorch `nn.Module` with verified dimension alignment.
3. Execute fast proxy evaluation on Fold 1 for 50 epochs (single seed 42) on `cuda:0`; if Proxy Accuracy < 70.0% or Proxy Weighted-F1 < 0.69, revert the mutation, log the failure, and proceed to the next iteration.
4. If proxy passes, execute full 5-fold cross-validation across all 47 subjects (80 epochs, 3 seeds, outer test evaluation); if aggregated 5-fold Accuracy > 71.04% and Weighted-F1 > 0.7079, proceed to the victory condition, otherwise revert the mutation, log the failure, and continue the loop.
The loop runs continuously without an iteration cap until the victory condition is met or a terminal error occurs.

### R2. Architectural Budget and Parameter Count
Candidate architectures should remain as close to the 162k–196k parameter range as possible to maintain a >90% compression ratio against the 5.47M parameter paper model. Exceeding 196k parameters is permitted as a soft constraint if the architecture meets or exceeds the target performance metrics. Hard budget cap: <= 196,728 parameters. Structural modifications must focus on architecture; training batch size, optimizer, and base learning rate must remain consistent across evaluations.

### R3. Evaluation and Hardware Infrastructure
Execute evaluations locally using the host Python virtualenv `./.venv/bin/python`. Run proxy evaluations on single GPU `cuda:0`. For 5-fold cross-validation, parallelize fold evaluations across both available GPUs (`cuda:0` and `cuda:1`). All evaluations must use the pre-computed feature cache and evaluation manifests.

### R4. Victory Condition Reporting and Synchronization
Upon achieving aggregated 5-fold Accuracy > 71.04% and Weighted-F1 > 0.7079, format a final PDF-ready markdown report declaring the SLAMSS-IFS paper benchmark beaten, commit all code and JSON results to Git, and output: "VICTORY CONFIRMED: SLAMSS-IFS BENCHMARK BEATEN. CONTINUOUS LOOP TERMINATED."

## Prior Search State & Grounding
- Predecessor files: `.agents/teamwork/orchestrator/DEAD_ENDS.md`, `.agents/teamwork/orchestrator/progress.md`, `.agents/teamwork/orchestrator/BRIEFING.md`.
- Iteration 1–11 results:
  * Iteration 8 (Candidate Zeta): 65.54% outer-test Acc, 0.6556 WF1, 0.5051 Kappa (DSLBS W_macro=91, W_micro=15).
  * Iteration 9 (Candidate Eta): 64.57% outer-test Acc (raw accelerometry gating overfit to sensor gain).
  * Iteration 10 (Candidate Theta): 66.84%–67.01% outer-test Acc, 0.6684 WF1, 0.5153 Kappa (All-Time Project Best). Dual-Scale ACC DSLBS solved sensor gain, reduced Light->REM false alarms by 33.6%.
  * Iteration 11 (Candidate Iota): 65.12% outer-test Acc (softplus without sharp boundary leaked synergy into Light sleep).
- Next Immediate Task (Iteration 12): Candidate Kappa formulation assets in `.agents/teamwork/teamwork_explorer_r12_1/`. Candidate Kappa restores the hard cutoff of ReLU with learned asymmetric threshold margins (tau_acc > 0, tau_ihr < 0) and Adaptive Angular Margin (ArcMargin) head. Dispatch worker to deploy and run Fold 1 proxy evaluation.
- Remember: Orchestrator is dispatch-only. NEVER write code directly. Always spawn Explorers and Workers for code and evaluation tasks.
