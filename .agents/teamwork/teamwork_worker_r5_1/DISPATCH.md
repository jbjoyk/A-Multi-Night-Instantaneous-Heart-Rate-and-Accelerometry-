# Dispatch: NAS Iteration 5 PyTorch Implementation & Fast Proxy Evaluation

**Target Agent**: `teamwork_worker_r5_1` (NAS PyTorch Engineer & Evaluator for Iteration 5)  
**Date**: 2026-11-15T06:40:00Z  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r5_1`  

---

## 1. Mandatory First Step & Constraints
1. **Authoritative Request**: Read `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
2. **MANDATORY INTEGRITY WARNING**:
   > DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

---

## 2. Write Ownership (Exclusive)
- `experiments/EXP-014/nas_iteration5_model.py`
- `experiments/EXP-014/evaluate_nas_iteration5_proxy.py`
- `experiments/EXP-014/evaluate_nas_iteration5_5fold.py`
- `experiments/EXP-014/nas_iteration5_proxy_results.json`
- `experiments/EXP-014/nas_iteration5_5fold_results.json`
- Test files inside `.agents/teamwork/teamwork_worker_r5_1/`

---

## 3. Reference Artifacts to Read
- Verified Architecture Model: `.agents/teamwork/teamwork_explorer_r5_1/proposed_nas_iteration5_model.py`
- Explorer Unit Tests: `.agents/teamwork/teamwork_explorer_r5_1/test_candidate_gamma.py`
- Explorer Analysis & Handoff: `.agents/teamwork/teamwork_explorer_r5_1/analysis.md` and `handoff.md`
- Previous Proxy Evaluation Harness: `experiments/EXP-014/evaluate_nas_iteration4_proxy.py`
- 5-Fold Dual-GPU Runner: `experiments/EXP-014/evaluate_nas_iteration4_5fold.py`

---

## 4. Tasks
1. **Implement Candidate Gamma** in `experiments/EXP-014/nas_iteration5_model.py`:
   - `hr_encoder` (CNN 64 dims) + `acc_encoder` (CNN 64 dims) + `meta_proj` (16 dims).
   - `MaskedTemporalInstanceNorm` for HR and ACC across valid sequence length $T$.
   - `temporal_gru` (BiGRU 2 layers, input 144, hidden 48 -> 96).
   - `InterEpochSequenceAttention`: 4-head scaled dot-product attention ($d_{\text{attn}}=48$), with $-10^9$ padding key-masking and zero-initialized output projection ($W_o \equiv 0$).
   - Direct 144-dim Feature Skip Connection bypassing the BiGRU into the classification head.
   - Classifier head: `Linear(96 [GRU] + 48 [Attn] + 144 [Skip] = 288 -> 64 -> 4)`.
   - Exact parameter count assertion: Exactly 193,156 parameters ($\le 196,728$ budget cap).
2. **Implement and Execute Unit Tests** in `.agents/teamwork/teamwork_worker_r5_1/test_nas_iteration5.py`:
   - Exact parameter count assertion (193,156).
   - Step-0 zero identity initialization.
   - Attention key-masking strictly zeroes out padded tokens.
   - Masked TIN nocturnal normalization.
   - Forward pass on BIDSleep shapes (B=4, T=600, 30 for IHR/ACC, 5 for meta).
   - Backward pass and gradient flow through all components including direct skip highway.
   - Run via `./.venv/bin/python .agents/teamwork/teamwork_worker_r5_1/test_nas_iteration5.py`.
3. **Implement and Execute Fast Proxy Evaluation** in `experiments/EXP-014/evaluate_nas_iteration5_proxy.py`:
   - Fast proxy evaluation on Fold 1 for 50 epochs (single seed 42) on single GPU `cuda:0` using `./.venv/bin/python`.
   - Fixed hyperparameters: batch size 4, Adam(lr=1e-4, weight_decay=0.0), seq_len 600, RCCLoss ($\lambda=0.01$), grad clipping 1.0.
   - Checkpointed evaluation: Evaluate inner-validation set every epoch, track `best_val_macro_f1`, save the peak model checkpoint state dict, and restore this peak checkpoint for outer test evaluation (10 subjects, 54 nights, 32,098 windows).
   - Compute and log: Accuracy, Weighted-F1, Macro-F1, Per-class F1, Cohen's Kappa, Confusion Matrix.
   - Save results to `experiments/EXP-014/nas_iteration5_proxy_results.json`.
   - Run proxy evaluation and capture output.
4. **Check Proxy Gating Filter**:
   - If Proxy Accuracy < 70.0% or Proxy Weighted-F1 < 0.69: candidate failed proxy gate. Record failure in handoff report.
   - If Proxy Accuracy >= 70.0% and Proxy Weighted-F1 >= 0.69: candidate passed proxy gate!
5. **If Proxy Passed**:
   - Implement and execute full 5-fold cross-validation `experiments/EXP-014/evaluate_nas_iteration5_5fold.py` across all 47 subjects (80 epochs, 3 seeds: 42, 123, 2024), parallelized across `cuda:0` and `cuda:1`.
   - Save to `experiments/EXP-014/nas_iteration5_5fold_results.json`.
   - Check victory condition: Aggregated 5-fold Accuracy > 71.04% AND Weighted-F1 > 0.7079.
6. **Write full handoff report** to `.agents/teamwork/teamwork_worker_r5_1/handoff.md`.
7. **Send a message to parent orchestrator** upon completion.


## 2026-11-15T06:40:12Z
You are teamwork_worker_r5_1 (NAS PyTorch Engineer & Evaluator for Iteration 5). Read your dispatch instructions at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r5_1/DISPATCH.md
and the authoritative request at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md.

Deploy Candidate Gamma into experiments/EXP-014/nas_iteration5_model.py, implement and pass unit tests, execute the fast proxy evaluation on Fold 1 on cuda:0, check the proxy gate filter, execute 5-fold CV if passed, commit and push artifacts, write handoff report, and send completion message via send_message.
