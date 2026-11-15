# NAS Iteration 4: Architectural Exploration & Mutation Handoff Report

**Agent**: `teamwork_explorer_r4_1` (Neural Architecture Explorer for NAS Iteration 4)  
**Date**: 2026-11-15T06:15:00Z  
**Target Architecture**: Candidate Beta — `CircadianCrossAttentionCNNBiGRU` (`CircadianAnchoredCrossAttention`, CACA)  
**Working Directory**: `.agents/teamwork/teamwork_explorer_r4_1/`  
**Target Hypotheses**: Hypothesis H-F (Autonomic–Somatomotor Non-linear Coupling) & Hypothesis H-A (Subject-Invariant Generalization across All 47 Subjects)  
**Status**: Exploration Complete — High-Confidence Architecture Formulated, Mathematically Verified, Unit-Tested (6/6 Passed), Ready for Implementation & Fast Proxy Evaluation  

---

## 1. Observation

### 1.1 Direct Observations from Prior Iterations
1. **Iteration 3 Proxy Evaluation Results (`nas_iteration3_proxy_results.json:11-124`)**:
   - Inner Validation (5 subjects, 24 nights, 14,089 windows):
     * Accuracy: **72.53%**
     * Weighted-F1: **0.7256**
     * Macro-F1: **0.7384**
     * Cohen's Kappa: **0.6119**
     * Per-class F1: Wake: 0.8125, Light: 0.7112, Deep: 0.7353, REM: 0.6947
   - Outer Test (10 locked subjects, 54 nights, 32,098 windows):
     * Accuracy: **63.09%** (Gating threshold $\ge 70.0\%$ $\to$ FAILED)
     * Weighted-F1: **0.6314** (Gating threshold $\ge 0.6900$ $\to$ FAILED)
     * Macro-F1: **0.6314**
     * Cohen's Kappa: **0.4681**
     * Per-class F1: Wake: 0.6391, Light: 0.6286, Deep: 0.6909, REM: 0.5668
   - Outer-Test Confusion Matrix (`nas_iteration3_proxy_results.json:75-99`):
     ```
     [[1945,  682,  272,  307],
      [ 691, 8633, 1806, 2734],
      [ 186, 1837, 5503,  393],
      [  59, 2450,  430, 4170]]
     ```
     * Verbatim observation: 2,734 Light sleep windows misclassified as REM, and 2,450 REM windows misclassified as Light sleep, generating **5,184 cross-modal confusion windows** on held-out test subjects.

2. **Methodological Defect in Evaluation Harness (`evaluate_nas_iteration3_proxy.py:206-250`)**:
   - Lines 206–237 show the 50-epoch training loop: inner validation was *never* evaluated inside the loop, and no model checkpoints were saved.
   - Lines 242–250 show that after epoch 50 completed, the script evaluated inner-val and outer-test on the final epoch 50 weights.
   - In `nas_iteration3_proxy_results.json:136-385`, training loss dropped continuously from $1.3018$ (epoch 1) down to $0.7319$ (epoch 50).
   - In contrast, `experiments/EXP-019/run_exp019_500epochs.py:276-305` evaluates `val_loader` at every epoch and maintains `best_val_macro_f1` and `best_ckpt_path`, evaluating outer test on the best validation checkpoint.

3. **Proposed Candidate Beta Implementation & Parameter Count**:
   - Code files written in explorer workspace:
     * `.agents/teamwork/teamwork_explorer_r4_1/proposed_nas_iteration4_model.py`
     * `.agents/teamwork/teamwork_explorer_r4_1/proposed_evaluate_nas_iteration4_proxy.py`
     * `.agents/teamwork/teamwork_explorer_r4_1/test_candidate_beta.py`
   - Analytical parameter count:
     * `hr_encoder`: 29,568 parameters
     * `acc_encoder`: 29,568 parameters
     * `meta_proj`: 96 parameters
     * `cross_attention` (`CircadianAnchoredCrossAttention`): **22,930 parameters**
       - `tin_hr` + `tin_acc` (`MaskedTemporalInstanceNorm`): $128 + 128 = 256$ parameters
       - `norm_hr_in` + `norm_acc_in` (`LayerNorm(64)`): $128 + 128 = 256$ parameters
       - `ms_hr` (`MultiScaleModalityBlock`, kernels 1, 3, 5, mid 48): 6,672 parameters
       - `ms_acc` (`MultiScaleModalityBlock`, kernels 1, 3, mid 32): 4,480 parameters
       - `circadian_proj` (`Linear(16, 32) + GELU`): 544 parameters
       - `attn_hr_from_acc` (`LocalizedCosineCrossAttention`, $W=5, d_{attn}=16, N_h=2$): 5,233 parameters
       - `attn_acc_from_hr` (`LocalizedCosineCrossAttention`, $W=5, d_{attn}=16, N_h=2$): 5,233 parameters
       - `norm_hr_out` + `norm_acc_out` (`LayerNorm(64)`): $128 + 128 = 256$ parameters
     * `temporal_gru` (BiGRU 2 layers, 144 -> 96): 97,920 parameters
     * `temp_norm`: 192 parameters
     * `classifier_head`: 6,596 parameters
     * **Total Model Parameters**: **Exactly 186,870 parameters** (+13.99% over Baseline Model A; 9,858 parameters under the 196,728 cap; 96.58% compression vs 5.47M published SLAMSS-IFS benchmark).

4. **Empirical Unit Test Results**:
   - Command: `./.venv/bin/python .agents/teamwork/teamwork_explorer_r4_1/test_candidate_beta.py`
   - Result: **6 of 6 tests PASSED in 1.186 seconds**:
     * `test_01_parameter_counts`: Exact 186,870 parameters confirmed.
     * `test_02_step0_identity_initialization`: Output projections zero-initialized ($W_o \equiv 0, b_o \equiv 0$), residual delta strictly $0.0$ ($< 10^{-6}$).
     * `test_03_cosine_similarity_bounds`: Cosine similarity logits strictly bounded in $[-1.0, 1.0]$.
     * `test_04_masked_tin_normalization`: Valid nocturnal sequence standardized to zero mean ($< 10^{-3}$) and unit variance ($| \sigma - 1.0 | < 10^{-2}$).
     * `test_05_forward_backward_gradient_flow`: Non-zero gradient backpropagation verified across all CACA submodules.
     * `test_06_variable_length_packing`: Verified packed padded sequences with variable length nights and boolean masking.

5. **Hardware Throughput on NVIDIA RTX 3060 (`cuda:0`)**:
   - 42 batches of CACA forward + backward completed in **0.573 seconds** (**13.64 ms per batch**).
   - Estimated 50-epoch training + validation proxy runtime: **~65 seconds**.

---

## 2. Logic Chain

1. **Step 1 — Deductions from Masked TIN Success**:
   - Observation 1.1 confirms that Masked TIN elevated inner-validation accuracy to 72.53% (from 67.5% in Baseline and 69.56% in Iteration 2).
   - Therefore, per-recording sequence standardization is fundamentally sound and must be retained in Candidate Beta to preserve inter-subject baseline invariance (Hypothesis H-A).

2. **Step 2 — Diagnosing the Generalization Gap**:
   - Observation 1.2 demonstrates that the evaluation harness trained blindly for 50 epochs, dropping train loss from 1.30 to 0.73 without validation checkpointing.
   - On 32 training subjects, unregularized late-epoch training causes overfitting to subject-specific idiosyncratic patterns.
   - Observation 1.1 shows 5,184 boundary misclassifications between Light sleep and REM. Point-wise static MLP gating cannot disambiguate motor stillness in REM from quiet non-REM sleep without inspecting temporal neighbors.

3. **Step 3 — Formulating Localized Multi-Head Cosine Cross-Attention (CACA)**:
   - To resolve boundary confusion, we expand the temporal horizon of cross-modal interaction to $W=5$ epochs ($t \pm 2$ epochs / 2.5 minutes) via 1D temporal unfold. This allows the model to determine whether motor stillness is accompanied by sustained autonomic stability (SWS/N2) or autonomic storminess (REM).
   - To prevent the amplitude-scaling explosion that failed in Iteration 2, queries and keys are $L_2$-normalized, strictly constraining interaction logits to cosine similarities $\cos(\theta) \in [-1, 1]$. This eliminates inter-subject amplitude drift.
   - Circadian phase context ($c_{meta}$, 32 dims) is injected into queries and keys, providing continuous time-of-night awareness (e.g. early-night SWS vs late-night REM).
   - Output projections are initialized to exact zero ($W_o \equiv 0$), guaranteeing step-0 baseline manifold preservation.

4. **Step 4 — Designing the Checkpointed Proxy Evaluation Harness**:
   - The harness in `proposed_evaluate_nas_iteration4_proxy.py` evaluates the 6-batch inner validation set at every epoch (~0.10s overhead).
   - The best model checkpoint (`best_val_macro_f1`) is saved and loaded before running outer-test evaluation.
   - This prevents late-epoch overfitting and ensures that the model evaluated on the outer test set represents peak cross-subject generalization.

5. **Step 5 — Analytical Parameter Compliance**:
   - Observation 1.3 and Observation 1.4 confirm that Candidate Beta has exactly 186,870 parameters.
   - This is within +0.4% of the ~186,054 target and provides 9,858 parameters of headroom under the 196,728 budget cap (+13.99% over baseline), strictly satisfying Requirement R2.

---

## 3. Caveats

1. **Fold 1 Distributional Severity**: Fold 1 has the most challenging subject demographics in the BIDSleep cohort (baseline 63.3%, vs 71.8% on Fold 3). Achieving $\ge 70.0\%$ Accuracy and $\ge 0.6900$ Weighted-F1 on Fold 1 outer test within 50 epochs is a rigorous gate.
2. **Hyperparameter Invariance**: In accordance with Requirement R2, optimizer (Adam), learning rate (1e-4), batch size (4), loss weights ($\lambda=0.01$ RCCLoss), and sequence length (600) remain strictly frozen without scheduling or tuning.
3. **Primary vs Final Checkpoint Reporting**: The harness evaluates and reports both the best-validation checkpoint and the epoch-50 final model. In accordance with the enhanced protocol, the primary gate decision is based on the best-validation checkpoint.

---

## 4. Conclusion

1. **Formulation Complete**: Candidate Beta (`CircadianCrossAttentionCNNBiGRU` with `CircadianAnchoredCrossAttention` CACA) is fully formulated, mathematically grounded, and verified in PyTorch.
2. **Parameter Compliance**: Total parameters = **186,870**, strictly compliant with the 162k–196k budget cap ($186,870 \le 196,728$, 9,858 parameter headroom).
3. **Harness Enhanced**: Evaluation harness incorporates epoch-level inner-validation tracking and outer-test evaluation on the peak Macro-F1 checkpoint.
4. **Action Directive for Worker Agent (`teamwork_worker_r4_1`)**:
   - Copy `.agents/teamwork/teamwork_explorer_r4_1/proposed_nas_iteration4_model.py` to `experiments/EXP-014/nas_iteration4_model.py`.
   - Copy `.agents/teamwork/teamwork_explorer_r4_1/proposed_evaluate_nas_iteration4_proxy.py` to `experiments/EXP-014/evaluate_nas_iteration4_proxy.py`.
   - Execute fast proxy evaluation via `./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration4_proxy.py` on `cuda:0`.
   - Evaluate gate filter: If Proxy Acc $\ge 70.0\%$ and Proxy WF1 $\ge 0.6900$, trigger dual-GPU 5-fold cross-validation (`cuda:0` and `cuda:1`); otherwise revert and log failure.

---

## 5. Verification Method

To independently verify the implementation, tensor safety, and analytical parameter counts:

1. **Run Complete Unit Test Suite**:
   ```bash
   ./.venv/bin/python .agents/teamwork/teamwork_explorer_r4_1/test_candidate_beta.py
   ```
   *Expected Output*: 6 of 6 tests pass with 0 failures, asserting exact 186,870 parameters, zero-initialization identity, and cosine bounded logits.

2. **Verify Proposed Model Instantiation and Dimensions**:
   ```bash
   ./.venv/bin/python -c "
   import sys
   sys.path.insert(0, '.agents/teamwork/teamwork_explorer_r4_1')
   from proposed_nas_iteration4_model import build_nas_iteration4_model
   model = build_nas_iteration4_model()
   p = sum(param.numel() for param in model.parameters())
   print(f'Candidate Beta Parameters: {p:,}')
   assert p == 186870, f'Expected 186,870, got {p}'
   "
   ```
   *Expected Output*: `Candidate Beta Parameters: 186,870`

3. **Verify Proposed Evaluation Script Compilation**:
   ```bash
   ./.venv/bin/python -m py_compile .agents/teamwork/teamwork_explorer_r4_1/proposed_evaluate_nas_iteration4_proxy.py
   ```
   *Expected Output*: Clean exit with return code 0.

4. **Invalidation Conditions**:
   - Parameter count $\ne 186,870$.
   - Any test failure in `test_candidate_beta.py`.
   - Cosine similarity logits outside $[-1.0, 1.0]$.
   - Output projection not initialized to exact zero.
