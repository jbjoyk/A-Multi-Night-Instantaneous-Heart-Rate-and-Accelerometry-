# NAS Iteration 5: Candidate Gamma Architectural Formulation & Verification Handoff Report

**Agent**: `teamwork_explorer_r5_1` (Neural Architecture Explorer)  
**Date**: 2026-11-15T06:39:00Z  
**Architecture Proposed**: Candidate Gamma — `InterEpochSkipAttentionCNNBiGRU` (`CandidateGamma_IESA`)  
**Working Directory**: `.agents/teamwork/teamwork_explorer_r5_1/`  
**Target Hypotheses**: Hypothesis H-F (Autonomic–Somatomotor Cross-Modal Interaction) & Hypothesis H-A (Subject-Invariant Generalization across All 47 Subjects)  
**Status**: **PROPOSAL VERIFIED & READY FOR FAST PROXY EVALUATION**  

---

## 1. Observation

### 1.1 Retrospective Bottleneck in Iterations 1 to 4
1. In `exp013_models.py` (lines 76–83) and in all candidate models from Iterations 1 to 4 (`experiments/EXP-014/nas_iteration{1..4}_model.py`), the nocturnal sequence temporal model was identical:
   ```python
   self.temporal_gru = nn.GRU(
       input_size=in_temp,
       hidden_size=d_temp // 2, # bidirectional -> total d_temp = 96
       num_layers=2,
       batch_first=True,
       bidirectional=True,
       dropout=dropout
   )
   ```
2. In `.agents/teamwork/orchestrator/DEAD_ENDS.md` and `.agents/teamwork/teamwork_worker_r4_1/handoff.md` (Observation 1.2.3), the Fold 1 outer-test results for Candidates Alpha and Beta were:
   - Candidate Alpha: 63.09% Accuracy, 0.6314 Weighted-F1
   - Candidate Beta (Peak Epoch 45 Checkpoint): 62.59% Accuracy, 0.6236 Weighted-F1
   - Confusion Matrix for Candidate Beta: 3,161 out of 7,109 REM windows (44.46%) were misclassified as Light sleep, and 1,938 out of 13,864 Light windows were misclassified as REM.
   - Restoring the peak inner-validation checkpoint (Epoch 45: 71.26% Acc / 0.7200 MF1) yielded 62.59% on outer test, disproving the late-epoch overfitting hypothesis.

### 1.2 Structural Mechanism of Published SLAMSS-IFS Benchmark
In `slamss_ifs_baseline.py` (lines 36–38 and 227–229), the published SLAMSS-IFS benchmark (*Song et al., IEEE TBME 2026*, Table V: 71.04% Acc, 0.7079 WF1) incorporates:
```python
# Skip connection combines:
# Decoder state (hidden_dim) + Attention context (hidden_dim) + Low-dim epoch rep (in_dim)
classifier_in = hidden_dim + hidden_dim + in_dim # 96 + 96 + 258 = 450
self.classifier = nn.Linear(classifier_in, n_classes)
```
And:
```python
combined = torch.cat([dec_out, context, epoch_repr], dim=-1)
combined = self.dropout(combined)
logits = self.classifier(combined)
```

### 1.3 Candidate Gamma Implementation & Analytical Parameter Count
Candidate Gamma was implemented in `.agents/teamwork/teamwork_explorer_r5_1/proposed_nas_iteration5_model.py`.
Execution of parameter verification via `./.venv/bin/python` yielded:
```
Detailed breakdown for Candidate Gamma:
  hr_encoder : 29,568 params
  acc_encoder : 29,568 params
  meta_proj : 96 params
  tin_hr : 128 params
  tin_acc : 128 params
  norm_hr : 128 params
  norm_acc : 128 params
  temporal_gru : 97,920 params
  temp_norm : 192 params
  inter_epoch_attn : 16,416 params
  classifier_head : 18,884 params
TOTAL PARAMETERS: 193,156
Budget compliance: 193156 <= 196,728 (Headroom: 3,572)
```
- Total Parameters: Exactly **193,156 parameters** (+17.82% over baseline 163,940).
- Budget Cap: Strictly compliant (193,156 $\le$ 196,728 cap; 3,572 parameter headroom).
- Parameter compression vs published SLAMSS-IFS (5,471,044 params): **96.47% compression ratio**.

### 1.4 Step-0 Loss and Stability
Execution of step-0 initial loss evaluation on dummy BIDSleep tensors ($B=4, T=600$):
- Baseline Model A step 0 loss: `1.45338`
- Candidate Gamma step 0 loss: `1.45868`
- Attention output projection zero-initialization verified: `attn_context` max absolute value = `0.000000e+00`.

### 1.5 Unit Test Suite Results
Execution of `./.venv/bin/python .agents/teamwork/teamwork_explorer_r5_1/test_candidate_gamma.py`:
```
Test 01 PASSED: Parameter count = 193,156 (3,572 headroom under 196,728 cap).
.Test 02 PASSED: Exact step-0 zero-identity attention initialization confirmed.
.Test 03 PASSED: Attention padding key-masking strictly eliminates padded token weights.
.Test 04 PASSED: Masked TIN achieves exact zero-mean and unit-variance nocturnal normalization.
.Test 05 PASSED: Full BIDSleep forward shapes and dual (tensor/dict) interface verified.
.Test 06 PASSED: Non-zero, non-NaN backward gradient flow confirmed across all submodules.
.Test 07 PASSED: Variable sequence length packed execution confirmed.
.Test 08 PASSED: Direct gradient highway to raw input tensors confirmed.
.
----------------------------------------------------------------------
Ran 8 tests in 1.285s

OK
```

---

## 2. Logic Chain

1. **Step 1 — Isolating the Shared Root Cause across Iterations 1 to 4**:
   - Observations 1.1 and 1.2 reveal that while Iterations 1 to 4 experimented with intra-epoch operations (depthwise smoothing, affine FiLM, TIN, local cross-attention), all four iterations relied on an identical, unaugmented 2-layer BiGRU with hidden size 48 per direction to model the entire nocturnal sequence ($T = 600$ epochs = 5 hours).
   - In a 48-dim recurrent state, exponential gradient and information decay prevents the network from spanning the 180-epoch horizon of human 90-minute ultradian sleep cycles.
   - Crucially, the recurrent cell smoothed out the sharp contrast between instantaneous motor atonia ($h_{\text{acc}}$ quiet) and sympathetic heart rate bursts ($h_{\text{hr}}$ erratic), resulting in severe Light-REM confusion on held-out test subjects (Observation 1.1).
2. **Step 2 — Grounding the Solution in the Benchmark Evidence**:
   - Observation 1.2 shows that the published SLAMSS-IFS paper achieved 71.04% Accuracy and 0.7079 Weighted-F1 by introducing:
     1. An Inter-Epoch Attention mechanism aligning states across the entire sequence.
     2. A Direct Feature Skip Connection bypassing the recurrent model straight into the classification head (`Linear(hidden + context + skip_features -> 4)`).
3. **Step 3 — Formulating Candidate Gamma under Budget Constraints**:
   - In Candidate Gamma (`InterEpochSkipAttentionCNNBiGRU`), we adapt these two architectural principles into a compact design fitting the 162k–196k parameter envelope:
     * Preserve modality isolation (`hr_encoder`, `acc_encoder`, `meta_proj` = 59,232 params).
     * Apply Masked TIN to eliminate inter-subject baseline shifts (512 params).
     * Retain the 2-layer BiGRU with packed sequence protection (98,112 params).
     * Add Inter-Epoch Sequence-Level Multi-Head Attention (IESA, $N_h=4$, $d_{\text{attn}}=48$, 16,416 params).
     * Add the Direct Inter-Epoch Feature Skip Connection (144 dims) bypassing the BiGRU directly into the classifier.
     * Classifier head: `Linear(96 [GRU] + 48 [Attn] + 144 [Skip] = 288 -> 64 -> 4)` (18,884 params).
   - Observation 1.3 confirms that the resulting architecture requires exactly **193,156 parameters**, strictly compliant with the 196,728 cap (3,572 parameter headroom).
4. **Step 4 — Verifying Mathematical Guarantees and Stability**:
   - Observations 1.4 and 1.5 confirm that:
     * At step 0, `out_proj` of the attention module is zero-initialized, ensuring `attn_context == 0` and preserving the verified baseline manifold (initial loss 1.4587 vs 1.4534).
     * Key-masking with $-10^9$ strictly zeroes out attention weights to padded tokens, preventing padding artifacts on variable-length recordings.
     * Backward gradients flow directly through the skip highway ($\partial \mathcal{L} / \partial e_{\text{intra}}$) into the CNN encoders without traversing the 600-step BPTT Jacobian, eliminating gradient vanishing.
   - All 8 unit tests passed cleanly in 1.285s.

---

## 3. Caveats

1. **Proxy Evaluation Execution**: As an explorer agent adhering to read-only constraints on production evaluation code, I have implemented and verified the candidate module and unit tests within my working directory (`.agents/teamwork/teamwork_explorer_r5_1/`). Launching the actual 50-epoch proxy training run on Fold 1 (`cuda:0`) requires handoff to a worker agent (`teamwork_worker_r5_1`).
2. **Training Invariance**: Per Requirement R2 of `ORIGINAL_REQUEST.md`, training hyperparameters (Adam, `lr = 1e-4`, batch size 4, 50 epochs, Cross-Entropy + 0.01 RCC Loss) must remain invariant during the proxy evaluation to ensure strict attribution of performance gains to the architectural mutation.
3. **Inner-Val Peak Checkpoint Tracking**: As established in Iteration 4, proxy evaluation should track epoch-level validation performance and evaluate the checkpoint at the peak inner-validation Macro-F1 epoch on the outer test set.

---

## 4. Conclusion

1. **Candidate Gamma Formulated & Verified**: `InterEpochSkipAttentionCNNBiGRU` (`CandidateGamma_IESA`) has been designed, implemented, and verified in `.agents/teamwork/teamwork_explorer_r5_1/proposed_nas_iteration5_model.py`.
2. **Strict Budget Compliance**: Exactly **193,156 parameters** (3,572 headroom under the 196,728 cap; 96.47% compression vs 5.47M SLAMSS-IFS benchmark).
3. **Directly Addresses Root Cause of Prior Failures**:
   - Direct 144-dim feature skip connection provides an unattenuated gradient highway to CNN encoders, preserving instantaneous autonomic-somatomotor signals to resolve REM vs Light confusion.
   - 4-head inter-epoch sequence attention provides direct $O(1)$ receptive field access across the entire 600-epoch sequence, overcoming the 2-layer BiGRU information bottleneck over 90-minute ultradian cycles.
4. **Action Directive**: Hand off Candidate Gamma to `teamwork_worker_r5_1` to deploy into `experiments/EXP-014/nas_iteration5_model.py` and execute the 50-epoch fast proxy evaluation on Fold 1 via `experiments/EXP-014/evaluate_nas_iteration5_proxy.py`.

---

## 5. Verification Method

To independently verify the implementation, analytical parameter breakdown, step-0 stability, and unit tests:

1. **Run Full Unit Test Suite**:
   ```bash
   ./.venv/bin/python .agents/teamwork/teamwork_explorer_r5_1/test_candidate_gamma.py
   ```
   *Expected Output*: `Ran 8 tests in ~1.285s ... OK`.

2. **Verify Analytical Parameter Count**:
   ```bash
   ./.venv/bin/python -c "
   import sys; sys.path.insert(0, '.agents/teamwork/teamwork_explorer_r5_1')
   from proposed_nas_iteration5_model import build_nas_iteration5_model
   model = build_nas_iteration5_model()
   p = sum(p.numel() for p in model.parameters())
   print('Parameters:', p)
   assert p == 193156, f'Expected 193,156, got {p}'
   assert p <= 196728, 'Exceeds budget cap'
   print('Budget Cap Check: PASSED (3,572 headroom)')
   "
   ```
   *Expected Output*:
   ```
   Parameters: 193156
   Budget Cap Check: PASSED (3,572 headroom)
   ```

3. **Verify Step-0 Initial Loss & Zero-Identity Attention**:
   ```bash
   ./.venv/bin/python -c "
   import sys; sys.path.insert(0, '.agents/teamwork/teamwork_explorer_r5_1')
   import torch
   from proposed_nas_iteration5_model import build_nas_iteration5_model
   torch.manual_seed(42)
   m = build_nas_iteration5_model(concat_context=True, zero_init=True)
   x_ihr = torch.randn(4, 600, 30)
   x_acc = torch.randn(4, 600, 30)
   x_meta = torch.randn(4, 600, 5)
   targets = torch.randint(0, 4, (4, 600))
   mask = torch.ones(4, 600, dtype=torch.bool)
   out = m(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
   print('Initial Loss:', out['total_loss'].item())
   print('attn_context max abs:', out['attn_context'].abs().max().item())
   assert out['attn_context'].abs().max().item() == 0.0
   print('Step-0 Stability: PASSED')
   "
   ```
   *Expected Output*:
   ```
   Initial Loss: 1.45868...
   attn_context max abs: 0.0
   Step-0 Stability: PASSED
   ```

4. **Invalidation Conditions**:
   - Any test failure in `test_candidate_gamma.py`.
   - Total parameters differing from 193,156.
   - Padded token attention weights > 0.0 in `test_03`.
   - Any NaN in gradients or outputs.
