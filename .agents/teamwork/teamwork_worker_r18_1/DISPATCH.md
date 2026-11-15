# DISPATCH: NAS Iteration 18 — Candidate Pi Deployment & Proxy Evaluation

## Mission
You are dispatched as `teamwork_worker_r18_1` to deploy `evaluate_nas_iteration18_proxy.py`, execute the 50-epoch Fold 1 fast proxy evaluation on `cuda:0` for **Candidate Pi** (`UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU`), extract all validation and outer-test metrics, commit research artifacts to Git, and deliver your handoff report.

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Required Reading
1. `ORIGINAL_REQUEST.md` (project root)
2. `.agents/teamwork/orchestrator_gen2/SCOPE.md`
3. `.agents/teamwork/teamwork_explorer_r18_1/handoff.md` (detailed formulation, exact parameter count 196,374, and verified architecture properties)
4. `experiments/EXP-014/nas_iteration18_model.py` (the verified model implementation)
5. `experiments/EXP-014/evaluate_nas_iteration17_proxy.py` (reference evaluation script for proxy setup)

## Execution Tasks
1. **Deploy Evaluation Script**:
   - Create `experiments/EXP-014/evaluate_nas_iteration18_proxy.py` adapted from iteration 17/16/12 proxy scripts.
   - Import `build_nas_iteration18_model` from `nas_iteration18_model`.
   - Configure training for Fold 1, 50 epochs, batch size 4, lr 1e-4, Adam optimizer, seq_len 600, RCCLoss($\lambda=0.01$), seed 42 on `cuda:0`.
   - Save results JSON to `experiments/EXP-014/nas_iteration18_proxy_results.json`.
   - Save model checkpoint to `experiments/EXP-014/nas_iteration18_best.pt`.
2. **Execute Fast Proxy Evaluation**:
   - Run the script via `./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration18_proxy.py` on `cuda:0`.
   - Monitor training progress and record epoch-by-epoch metrics.
3. **Analyze Results**:
   - Extract peak inner-validation metrics (Epoch, Acc, WF1, MF1, Kappa).
   - Extract final epoch 50 metrics.
   - Extract outer-test evaluation metrics (Acc, WF1, MF1, Kappa, per-class recall and confusion matrix).
   - Compare outer-test Acc against the 70.0% proxy threshold and WF1 against the 0.6900 threshold.
   - Check value of learned temperature scale $s$ and margins ($\tau_{\text{acc}}, \tau_{\text{ihr}}$).
4. **Git Synchronization**:
   - Mandatory sync: commit and push all newly generated code, JSON results, and reports to GitHub (`origin main`).
5. **Handoff Report**:
   - Write your report to `.agents/teamwork/teamwork_worker_r18_1/handoff.md` containing Observation, Logic Chain, Caveats, Conclusion, and Verification Method.
   - Use `send_message` to report your completion and verdict.

## 2026-11-15T11:53:45Z
Sender: 5a45d55d-cdec-4610-8dcf-b95f2285b08e
Priority: MESSAGE_PRIORITY_HIGH
Content:
You are teamwork_worker_r18_1. Your working directory is `.agents/teamwork/teamwork_worker_r18_1/`.
Read your dispatch instructions in `.agents/teamwork/teamwork_worker_r18_1/DISPATCH.md` and read `ORIGINAL_REQUEST.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your mission:
1. Deploy `experiments/EXP-014/evaluate_nas_iteration18_proxy.py` adapting from iteration 17/16/12 proxy scripts and importing `build_nas_iteration18_model` from `nas_iteration18_model`.
2. Run the 50-epoch Fold 1 proxy evaluation on `cuda:0` using `./.venv/bin/python`.
3. Save results to `experiments/EXP-014/nas_iteration18_proxy_results.json` and checkpoint to `experiments/EXP-014/nas_iteration18_best.pt`.
4. Analyze all metrics: peak inner-val, final epoch 50, outer-test evaluation (Acc, WF1, MF1, Kappa, per-class recall, confusion matrix, learned temperature scale s and margins).
5. Commit and push all experimental code and results to GitHub `origin main`.
6. Write your comprehensive handoff report to `.agents/teamwork/teamwork_worker_r18_1/handoff.md`.
7. Send a message to the orchestrator upon completion.
