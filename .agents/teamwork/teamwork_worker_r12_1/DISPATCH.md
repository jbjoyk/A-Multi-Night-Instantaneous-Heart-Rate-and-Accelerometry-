# DISPATCH: NAS Iteration 12 (Candidate Kappa) Deployment and Proxy Evaluation

**To**: `teamwork_worker_r12_1`  
**From**: Project Orchestrator Gen 2 (`5a45d55d-cdec-4610-8dcf-b95f2285b08e`)  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r12_1`  
**Timestamp**: 2026-11-15T16:00:00Z  

## Objective
Deploy Candidate Kappa (`BoundedMarginSynergyPyramidCNNBiGRU`) into `experiments/EXP-014/nas_iteration12_model.py`, verify its parameter count and tensor alignment, implement `experiments/EXP-014/evaluate_nas_iteration12_proxy.py`, execute the 50-epoch Fold 1 fast proxy evaluation on `cuda:0` using `./.venv/bin/python`, evaluate the outer-test performance against the proxy gate thresholds (Accuracy >= 70.0% and Weighted-F1 >= 0.6900), and write a comprehensive handoff report.

## Mandatory Context Files (Read Before Starting Work)
1. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
2. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_gen2/SCOPE.md`
3. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_gen2/DEAD_ENDS.md`
4. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r12_1/verify_candidate_kappa.py`
5. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/evaluate_nas_iteration11_proxy.py`

## Exclusive File Ownership
- `experiments/EXP-014/nas_iteration12_model.py`
- `experiments/EXP-014/evaluate_nas_iteration12_proxy.py`
- `experiments/EXP-014/nas_iteration12_proxy_results.json`
- `experiments/EXP-014/nas_iteration12_best_proxy_ckpt.pt`
- `.agents/teamwork/teamwork_worker_r12_1/*`

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Protocol & Requirements
1. Parameter Budget: Must be <= 196,728 parameters (target ~196,400).
2. Optimization / Hyperparameters: Batch size 4, Adam(lr=1e-4, weight_decay=0.0), seq_len 600, RCCLoss(lambda=0.01). Do NOT change training hyperparameters or optimizer.
3. Verification: Run the unit verification script or self-contained tests first to confirm parameter count, loss gradient flow, and shape preservation.
4. Fast Proxy Evaluation:
   - Run 50 epochs on Fold 1 using `./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration12_proxy.py` on `cuda:0`.
   - Save full evaluation metrics to `experiments/EXP-014/nas_iteration12_proxy_results.json`.
5. Outer-Test Gating:
   - Calculate outer-test Accuracy, Weighted-F1, Macro-F1, Cohen's Kappa, and 4x4 Confusion Matrix.
   - Filter condition: Accuracy >= 70.0% AND Weighted-F1 >= 0.6900.
6. Deliverables:
   - Handoff report in `.agents/teamwork/teamwork_worker_r12_1/handoff.md` with:
     * Observation (parameter count, runtime, validation & outer-test metrics, confusion matrix)
     * Logic Chain (why Candidate Kappa performed as it did, REM recall vs Light false alarms)
     * Caveats & Gate Verdict (`PROXY_PASS` or `PROXY_FAIL (REVERT)`)
   - Send completion message to orchestrator via `send_message`.
