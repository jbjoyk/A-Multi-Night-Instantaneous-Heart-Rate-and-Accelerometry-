## 2026-11-15T06:16:21Z
You are teamwork_worker_r4_1 (NAS PyTorch Engineer & Evaluator for Iteration 4).
Your working directory is: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r4_1

MANDATORY FIRST STEP: Read the authoritative user request at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Write Ownership (Exclusive):
- `experiments/EXP-014/nas_iteration4_model.py`
- `experiments/EXP-014/evaluate_nas_iteration4_proxy.py`
- `experiments/EXP-014/evaluate_nas_iteration4_5fold.py`
- `experiments/EXP-014/nas_iteration4_proxy_results.json`
- `experiments/EXP-014/nas_iteration4_5fold_results.json`
- Test files inside `.agents/teamwork/teamwork_worker_r4_1/`

Reference Artifacts to Read:
- Verified architecture model: `.agents/teamwork/teamwork_explorer_r4_1/proposed_nas_iteration4_model.py`
- Verified evaluation script: `.agents/teamwork/teamwork_explorer_r4_1/proposed_evaluate_nas_iteration4_proxy.py`
- Unit test suite: `.agents/teamwork/teamwork_explorer_r4_1/test_candidate_beta.py`
- Explorer handoff report: `.agents/teamwork/teamwork_explorer_r4_1/handoff.md`

Tasks:
1. Implement Candidate Beta (`CircadianAnchoredCrossAttention` and `CircadianCrossAttentionCNNBiGRU`) in `experiments/EXP-014/nas_iteration4_model.py`.
   - Modality encoders: `hr_encoder` (64 dims) and `acc_encoder` (64 dims).
   - Masked Temporal Instance Normalization (`MaskedTemporalInstanceNorm`): standardizes representations across valid sequence length T per recording to zero nocturnal mean and unit variance.
   - Modality-specific multi-scale temporal convolutions: `ms_hr` (kernels 1, 3, 5, mid 48) and `ms_acc` (kernels 1, 3, mid 32) isolated before fusion.
   - Circadian context projection: `circadian_proj` projecting 16-dim meta to 32-dim circadian context $c_{meta}$.
   - Localized Multi-Head Cosine Cross-Attention: Window size $W=5$ ($t \pm 2$ epochs), $N_h=2$, $d_{attn}=16$, $L_2$-normalized cosine similarity logits in $[-1, 1]$, zero-initialized output projection ($W_o \equiv 0$).
   - Total model parameter count: Exactly 186,870 parameters, strictly within the 162k–196k budget cap.
2. Implement unit test script `.agents/teamwork/teamwork_worker_r4_1/test_nas_iteration4.py`:
   - Exact parameter count assertion (186,870).
   - Test step-0 zero identity initialization ($W_o \equiv 0$).
   - Test cosine similarity logits bounded in $[-1, 1]$.
   - Test Masked TIN on sequence dimension with boolean masking.
   - Test forward pass on expected BIDSleep shapes (B=4, T=600, 30 for IHR/ACC, 5 for meta).
   - Test backward pass and gradient flow through all CACA components (`attn_hr_from_acc`, `attn_acc_from_hr`, `tin_hr`, `ms_hr`, `ms_acc`, `circadian_proj`).
   - Run via `./.venv/bin/python .agents/teamwork/teamwork_worker_r4_1/test_nas_iteration4.py`.
3. Implement `experiments/EXP-014/evaluate_nas_iteration4_proxy.py`:
   - Fast proxy evaluation on Fold 1 for 50 epochs (single seed 42) on single GPU `cuda:0` using `./.venv/bin/python`.
   - Data loading: Fold 1 manifest (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) and cache `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b`.
   - Fixed hyperparameters: batch size 4, Adam(lr=1e-4, weight_decay=0.0), seq_len 600, RCCLoss ($\lambda=0.01$), grad clipping 1.0.
   - Checkpointed evaluation: Evaluate inner-validation set every epoch, track `best_val_macro_f1`, save the peak model checkpoint state dict, and restore this peak checkpoint for outer test evaluation (10 subjects, 54 nights, 32,098 windows).
   - Compute and log: Accuracy, Weighted-F1, Macro-F1, Per-class F1, Cohen's Kappa, Confusion Matrix.
   - Save results to `experiments/EXP-014/nas_iteration4_proxy_results.json`.
   - Run proxy evaluation and capture output.
4. Check Proxy Gating Filter:
   - If Proxy Accuracy < 70.0% or Proxy Weighted-F1 < 0.69: candidate failed proxy gate. Record failure in handoff report.
   - If Proxy Accuracy >= 70.0% and Proxy Weighted-F1 >= 0.69: candidate passed proxy gate!
5. If proxy passed:
   - Implement and execute full 5-fold cross-validation `experiments/EXP-014/evaluate_nas_iteration4_5fold.py` across all 47 subjects (80 epochs, 3 seeds: 42, 123, 2024), parallelized across `cuda:0` and `cuda:1`.
   - Save to `experiments/EXP-014/nas_iteration4_5fold_results.json`.
   - Check victory condition: Aggregated 5-fold Accuracy > 71.04% AND Weighted-F1 > 0.7079.
6. Write full handoff report to `.agents/teamwork/teamwork_worker_r4_1/handoff.md`.
7. Send a message to parent orchestrator upon completion.
