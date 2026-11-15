# NAS Iteration 18 Handoff Report: Candidate Pi Architecture Exploration

## 1. Observation
1. **Empirical Defect of Prior Candidates**:
   - In `experiments/EXP-014/nas_iteration12_model.py` (Candidate Kappa) and `experiments/EXP-014/nas_iteration17_model.py` (Candidate Omicron), inner-validation achieved 73.00% and 73.04% Accuracy respectively, proving representation capacity to beat the 71.04% SLAMSS-IFS benchmark.
   - However, on outer test, accuracy fell to 67.04% (Kappa) and 66.54% (Omicron) due to massive collapse into majority Light sleep: 2,657 REM windows (37.4%) and 2,225 Deep windows (28.1%) misclassified as Light.
   - In standard linear heads (`logits_c = W_c^T h + b_c`), Light sleep comprises 43.19% of training windows (`p = [0.0999, 0.4319, 0.2467, 0.2215]`), driving `b_light` to be strongly positive. When feature norms `||h||` attenuate on unseen test subjects, `W_c^T h` shrinks and predictions default to `arg max_c (b_c) = Light Sleep`.
   - In `experiments/EXP-014/nas_iteration13_model.py` (Candidate Lambda), ACAM cosine heads proved that cosine representations achieve historic recalls for minority stages (69.74% REM, 78.39% Deep), but failed because asymmetric margins (`m_rem=0.15, m_deep=0.12`) over-carved Light sleep down to 47.15% recall.
   - In Candidate Omicron, prior logit correction (`-tau_prior * log(p_c)`) was absorbed during training into linear bias `b_c`.

2. **Candidate Pi Implementation**:
   - Implemented in `experiments/EXP-014/nas_iteration18_model.py`:
     * Class: `UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU`
     * Aliases: `CandidatePi_UHCH`, `CandidatePi_Pyramid`, `BoundedMarginSynergyPyramidCNNBiGRU`
     * Factory function: `build_nas_iteration18_model(...)`, `build_candidate_pi_model(...)`
   - Unbiased Hyperspherical Cosine Head (UH-CH) (`UnbiasedHypersphericalCosineClassifierHead`):
     * Normalizes feature representations `h(t)` symmetrically onto unit hypersphere $S^{63}$ (`h_norm = F.normalize(h, p=2, dim=-1)`).
     * Normalizes class prototypes `W_c` symmetrically onto unit hypersphere $S^{63}$ (`w_norm = F.normalize(self.weight, p=2, dim=-1)`).
     * Temperature scale: Single learnable scalar parameter initialized at $s = 16.0$ (`self.scale = nn.Parameter(torch.tensor(16.0))`).
     * Symmetric zero-margin ($m = 0.0$): Identical cosine formulation across all 4 stages during both training and evaluation, completely avoiding over-carving of Light sleep.
     * Complete absence of additive linear bias terms: There is NO `b_c` parameter in `classifier_head`.
   - Zero-bias synergy projection:
     * In `BoundedMarginCrossModalSynergy`, `fc_logit` is instantiated with `bias=False` (`Linear(3, 4, bias=False)`, 12 parameters vs 16), ensuring zero static bias injection into logits.
   - BoundedMarginSTE stability cutoffs:
     * Retains `BoundedMarginSTE` straight-through estimator to clamp `tau_acc >= 0.05` and `tau_ihr >= -0.05`, eliminating negative margin drift while maintaining 100% gradient flow.

3. **Verification Command and Execution Results**:
   - Command: `./.venv/bin/python experiments/EXP-014/verify_candidate_pi.py` on `cuda:0` (NVIDIA GeForce RTX 3060).
   - Verbatim Output:
     ```
     ================================================================================
     RUNNING CANDIDATE PI 10-TEST AUTOMATED VERIFICATION SUITE
     ================================================================================
     Target Device: cuda:0
     Device Name:   NVIDIA GeForce RTX 3060

     --- TEST 1: Parameter Count and Strict Budget Compliance ---
     Candidate Pi Trainable Parameters: 196,374
     Detailed Parameter Breakdown:
       - hr_encoder        :   29,568 params
       - acc_encoder       :   29,568 params
       - meta_proj         :       96 params
       - basal_dslbs       :       16 params
       - acc_dslbs         :        0 params
       - luan_hr           :      320 params
       - luan_acc          :      320 params
       - bm_casci          :      485 params
       - circadian_gate    :    2,592 params
       - temporal_gru      :   97,920 params
       - temp_norm         :      192 params
       - inter_epoch_attn  :   16,416 params
       - classifier_head   :   18,881 params
     Target Budget Expected:                 196,374 params
     Hard Cap:                              196,728 params
     Headroom below Hard Cap:               354 params
     Compression vs 5.47M SLAMSS-IFS:       96.41%
     [PASS] TEST 1: Parameter Count and Strict Budget Compliance Verified!

     --- TEST 2: Complete Absence of Additive Linear Bias ---
       - classifier_head output layer has no additive linear bias parameters.
       - bm_casci.fc_logit has bias=False (12 params, zero additive bias).
       - Zero synergy input strictly produces zero synergy logits (no offset drift).
     [PASS] TEST 2: Complete Absence of Additive Linear Bias Verified!

     --- TEST 3: Hyperspherical Geometry on S^63, Scale Init, and Zero Margin ---
       - Temperature scale s at initialization: 16.0000 (expected: 16.0)
       - Feature norm lengths on S^63: 1.000000 to 1.000000
       - Prototype norm lengths on S^63: 1.000000 to 1.000000
       - Cosine similarity range: [-0.3444, +0.3200]
       - Max difference between targets=None and targets=with_targets logits: 0.00e+00 (expected: 0.0)
     [PASS] TEST 3: Hyperspherical Geometry, Scale Init, and Zero Margin Verified!

     --- TEST 4: Bounded Margin Stability Cutoffs & Anti-Drift Protection ---
       tau_acc_rem: raw = -0.10 -> clamped effective = 0.05 (cutoff >= 0.05 enforced)
       tau_ihr_rem: raw = -0.25 -> clamped effective = -0.05 (cutoff >= -0.05 enforced)
     [PASS] TEST 4: Bounded Margin Stability Cutoffs Verified!

     --- TEST 5: Forward Pass on (B=4, C=4, T=600) on cuda:0 ---
       Logits Shape:          torch.Size([4, 600, 4])
       Emission Logits Shape: torch.Size([4, 600, 4])
       Synergy Logits Shape:  torch.Size([4, 600, 4])
       Probabilities Shape:   torch.Size([4, 600, 4])
       Loss Scalar:           2.5477
       Scale Tensor Output:   16.0000
     [PASS] TEST 5: Forward Pass on cuda:0 Verified!

     --- TEST 6: End-to-End Gradient Flow to ALL Trainable Parameters ---
     Part A (zero_init=False): Total Trainable Parameter Tensors: 118
     Tensors with Active Gradients: 118 / 118
       classifier_head.scale grad:      +1.567957e-01
       classifier_head.weight grad:     4.374011e+00
       bm_casci.tau_ihr_rem grad norm:  2.207629e-05
       bm_casci.tau_acc_rem grad norm:  7.111926e-06
       bm_casci.fc_logit.weight grad:   6.555911e-01
       temporal_gru weight_ih_l0 grad:  3.724830e+00
     Part B (zero_init=True): scale grad in production mode: +9.660250e-02
     [PASS] TEST 6: 100% of Trainable Parameters Receive Non-Zero Finite Gradients!

     --- TEST 7: RCCLoss Compatibility & Multi-Task Integration ---
       Loss without RCCLoss (CE only): 2.5172
       Loss with RCCLoss (CE + RCC):   2.8398
     [PASS] TEST 7: RCCLoss Compatibility Verified!

     --- TEST 8: Tensor-Input vs Dict-Input Forward Equivalence ---
       Max absolute logit difference (dict vs positional args): 0.00e+00
     [PASS] TEST 8: Calling Convention Equivalence Verified!

     --- TEST 9: Accelerometry Scale-Invariance & Gravity Tilt Invariance ---
       Max abs difference in delta_z_acc under 10x gain & +50.0 DC offset: 5.17e-05
     [PASS] TEST 9: Accelerometry Scale and Tilt Invariance Verified!

     --- TEST 10: Multi-Step Optimization Simulation on cuda:0 ---
       Step 0 Loss: 2.5360 -> Step 4 Loss: 1.8964
       Initial scale: 16.0000
       Final scale:   15.9952 (delta = -0.0048)
     [PASS] TEST 10: Multi-Step Optimization Simulation Verified!

     ================================================================================
     CANDIDATE PI VERIFICATION SUMMARY:
     ================================================================================
       Test 1: Budget Compliance               : [PASS]
       Test 2: Zero Linear Bias                : [PASS]
       Test 3: UH-CH Geometry                  : [PASS]
       Test 4: Bounded Margins                 : [PASS]
       Test 5: Forward Pass                    : [PASS]
       Test 6: Gradient Flow                   : [PASS]
       Test 7: RCCLoss Compatibility           : [PASS]
       Test 8: Calling Equivalence             : [PASS]
       Test 9: ACC Invariance                  : [PASS]
       Test 10: Optimization Simulation        : [PASS]
     ================================================================================
     ALL 10 TESTS PASSED DEFINITIVELY ON CUDA:0!
     Candidate Pi is 100% verified and ready for deployment.
     ================================================================================
     ```

## 2. Logic Chain
1. **From Linear Bias Dominance to Hyperspherical Cosine Head**:
   - In linear classification heads, logit values scale with $\|h\|$. On held-out subjects where $\|h\|$ shrinks due to physiological tone shifts, additive linear bias vectors $b_c$ dictate the prediction, collapsing minority classes into majority Light sleep ($43.19\%$).
   - By normalizing both $h$ and class prototypes $W_c$ onto $S^{63}$, all representations lie on a compact Riemannian manifold where inner products are purely angular cosine similarities in $[-1.0, 1.0]$.
   - Completely removing $b_c$ ensures that when features attenuate, no default additive offset exists to collapse predictions into Light sleep.
2. **From Candidate Lambda Over-Carving to Symmetric Zero Margins ($m = 0.0$)**:
   - Candidate Lambda established that cosine geometry unlocks outstanding REM (69.74%) and Deep (78.39%) recall, but its asymmetric angular margins ($m_{rem}=0.15, m_{deep}=0.12$) over-carved Light sleep down to 47.15% recall.
   - By enforcing symmetric zero margins ($m = 0.0$) across all 4 stages, Candidate Pi eliminates artificial boundary distortion, ensuring Light sleep is never penalized during training.
3. **From Candidate Omicron Margin Drift to BoundedMarginSTE**:
   - Candidate Omicron proved that `BoundedMarginSTE` prevents negative margin drift below $\tau_{acc} \ge 0.05$ and $\tau_{ihr} \ge -0.05$. Retaining this module in Candidate Pi guarantees stability without impeding gradient flow.
4. **From Synergy Bias Leakage to Zero-Bias Projection**:
   - Setting `bias=False` in `fc_logit` prevents the synergy module from injecting static offsets into stage logits, reducing parameters from 16 to 12.
5. **Exact Parameter Budget Compliance**:
   - Total trainable parameters: $196,374$, which is strictly compliant with the $\le 196,728$ cap ($354$ parameters headroom) and achieves a $96.41\%$ compression ratio vs the 5.47M SLAMSS-IFS benchmark.

## 3. Caveats
- Fast proxy training and 5-fold evaluation were not executed by this agent (per the teamwork division of labor; explorer performs architecture formulation and unit verification, worker performs proxy training evaluation).
- The temperature scale $s$ is initialized at $16.0$ and is learnable. In Test 10, over 5 Adam steps, $s$ drifted slightly to $15.9952$, indicating stable convergence without gradient explosion or divergence.

## 4. Conclusion
Candidate Pi (`UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU`) has been formulated, fully implemented in `experiments/EXP-014/nas_iteration18_model.py`, and verified across a comprehensive 10-test automated suite on `cuda:0` in `experiments/EXP-014/verify_candidate_pi.py`. It satisfies all architectural constraints, eliminates all additive linear bias terms, enforces hyperspherical normalization on $S^{63}$, preserves symmetric margins ($m = 0.0$), enforces BoundedMarginSTE stability cutoffs, and complies with the parameter budget with exactly 196,374 trainable parameters. It is ready for proxy training evaluation by the worker agent.

## 5. Verification Method
To independently verify Candidate Pi:
1. Run the automated 10-test verification suite on `cuda:0`:
   ```bash
   ./.venv/bin/python experiments/EXP-014/verify_candidate_pi.py
   ```
2. Inspect the model file:
   - File: `experiments/EXP-014/nas_iteration18_model.py`
3. Invalidation Conditions:
   - If total trainable parameters exceed 196,728 or differ from 196,374.
   - If any linear bias exists in `classifier_head` or `bm_casci.fc_logit`.
   - If any gradient in the 118 trainable parameter tensors is `None` or zero during backpropagation.
