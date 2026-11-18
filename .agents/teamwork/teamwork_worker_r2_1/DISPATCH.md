# Task Assignment: Worker 1 (NAS Iteration 2 PyTorch Engineer & Evaluator)

- Path to authoritative user request: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
- Working Directory: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r2_1/`
- Owned Files:
  - `experiments/EXP-014/nas_iteration2_model.py`
  - `experiments/EXP-014/evaluate_nas_iteration2_proxy.py`
  - `experiments/EXP-014/evaluate_nas_iteration2_5fold.py`
  - `experiments/EXP-014/nas_iteration2_proxy_results.json`
  - `experiments/EXP-014/nas_iteration2_5fold_results.json`

## 2026-11-15T05:30:22Z
You are teamwork_worker_r2_1 (NAS PyTorch Engineer & Evaluator).
Your working directory is: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r2_1

MANDATORY FIRST STEP: Read the authoritative user request at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Write Ownership (Exclusive):
- `experiments/EXP-014/nas_iteration2_model.py`
- `experiments/EXP-014/evaluate_nas_iteration2_proxy.py`
- `experiments/EXP-014/evaluate_nas_iteration2_5fold.py`
- `experiments/EXP-014/nas_iteration2_proxy_results.json`
- `experiments/EXP-014/nas_iteration2_5fold_results.json`
- Test files inside `.agents/teamwork/teamwork_worker_r2_1/`

Reference Artifacts to Read:
- Architecture specifications: `.agents/teamwork/teamwork_explorer_r2_1/analysis.md` (Sections 4.2 & 7)
- Evaluation pipeline design: `.agents/teamwork/teamwork_explorer_r2_2/analysis.md`
- Multi-GPU & parallelization setup: `.agents/teamwork/teamwork_explorer_r2_3/analysis.md`

Tasks:
1. Implement Candidate B (`CrossModalFiLMFusion` and `CrossModalFiLMCNNBiGRU`) in `experiments/EXP-014/nas_iteration2_model.py`.
   - Modality encoders: `hr_encoder` (64 dims) and `acc_encoder` (64 dims).
   - FiLM conditioning: ACC features dynamically scale ($\gamma$) and shift ($\beta$) HR features; HR features dynamically scale and shift ACC features.
   - Dynamic Gating: Joint gating network with Sigmoid activation.
   - Normalization & Identity Init: LayerNorm on modulated streams, zero-initialization on generator projection layers so training begins near the identity of Baseline Model A.
   - Total parameter count: Exactly 193,380 parameters (or 185,156 with compact gate), strictly within the 162k–196k budget cap.
2. Create unit test script `.agents/teamwork/teamwork_worker_r2_1/test_nas_iteration2.py`:
   - Verify parameter count assertion.
   - Verify forward pass on shape `(B=4, T=600, 30)` for IHR & ACC, `(B=4, T=600, 5)` for metadata.
   - Verify backward pass and gradient flow through all FiLM and Gating parameters.
   - Run via `./.venv/bin/python .agents/teamwork/teamwork_worker_r2_1/test_nas_iteration2.py`.
3. Implement `experiments/EXP-014/evaluate_nas_iteration2_proxy.py`:
   - Fast proxy evaluation on Fold 1 for 50 epochs (single seed 42) on single GPU `cuda:0` using `./.venv/bin/python`.
   - Data loading: Fold 1 manifest (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) and cache `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b`.
   - Fixed hyperparameters: batch size 4, Adam(lr=1e-4, weight_decay=0.0), seq_len 600, composite RCCLoss ($\lambda=0.01$), grad clipping 1.0.
   - Evaluate on outer test set (10 subjects, 54 nights, 32,098 windows).
   - Compute and log: Accuracy, Weighted-F1, Macro-F1, Per-class F1, Cohen's Kappa, Confusion Matrix.
   - Save results to `experiments/EXP-014/nas_iteration2_proxy_results.json`.
   - Run proxy evaluation and capture output.
4. Check Proxy Gating Filter:
   - If Proxy Accuracy < 70.0% or Proxy Weighted-F1 < 0.69: candidate failed proxy gate. Record failure in handoff report.
   - If Proxy Accuracy >= 70.0% and Proxy Weighted-F1 >= 0.69: candidate passed proxy gate!
5. If proxy passed:
   - Implement `experiments/EXP-014/evaluate_nas_iteration2_5fold.py`:
     Full 5-fold cross-validation across all 47 subjects (80 epochs, 3 seeds: 42, 123, 2024), parallelized across `cuda:0` and `cuda:1`.
     Run evaluation, aggregate metrics, and save to `experiments/EXP-014/nas_iteration2_5fold_results.json`.
     Compare against published benchmark: Accuracy > 71.04% and Weighted-F1 > 0.7079.
6. Write full handoff report to `.agents/teamwork/teamwork_worker_r2_1/handoff.md` with:
   - Observation: Exact parameter count, proxy evaluation metrics, 5-fold evaluation metrics (if run), execution logs.
   - Logic Chain: Technical rationale.
   - Caveats: Observed limitations.
   - Conclusion: Pass/Fail gate recommendations.
   - Verification: Exact terminal commands and results.
7. Send a message to parent orchestrator upon completion.
