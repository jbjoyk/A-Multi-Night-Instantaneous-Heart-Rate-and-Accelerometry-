# Handoff Report: NAS Iteration 19 — Candidate Rho Architecture Exploration

## 1. Observation
- **Root-Cause Analysis of Prior Candidates (Kappa Iter 12, Omicron Iter 17, Pi Iter 18)**:
  * In `experiments/EXP-014/nas_iteration12_model.py` (lines 430 and 703-704) and `experiments/EXP-014/nas_iteration17_model.py` (lines 471-478), the synergy term for Light sleep was defined as:
    $$s_{\text{light}} = \text{ReLU}(\tau_{\text{ihr\_light}} - u_{\text{ihr}}) \cdot \text{ReLU}(u_{\text{acc}} - \tau_{\text{acc\_light}})$$
  * Accelerometry deviation $u_{\text{acc}} = \Delta z_{\text{acc}}$ during quiet resting sleep satisfies $u_{\text{acc}} \le 0.0$.
  * With margin stability cutoff $\tau_{\text{acc\_light}} \ge 0.05$ (or initial $\tau_{\text{acc\_light}} = 0.0$), the term $u_{\text{acc}} - \tau_{\text{acc\_light}} \le 0.0$, making $\text{ReLU}(u_{\text{acc}} - \tau_{\text{acc\_light}}) \equiv 0.0$ identically throughout the entire night.
  * Verified directly in `experiments/EXP-014/verify_candidate_rho.py` Test 2:
    `Prior Kappa/Omicron s_light during quiet sleep: 0.000000 (MATHEMATICALLY ZERO!)`
  * Consequently, all prior candidates completely lacked an active discriminative synergy feature for Light sleep, forcing the linear classifier bias $b_{\text{light}}$ to inflate, which caused catastrophic collapse on held-out subjects (collapsing 2,657 REM windows [37.4%] and 2,225 Deep windows [28.1%] into Light sleep, per `DEAD_ENDS.md` line 20).
- **Candidate Rho Implementation (`experiments/EXP-014/nas_iteration19_model.py`)**:
  * Implemented Autonomic & Somatomotor Homeostasis Contrast (AH-CASCI):
    $$s_{\text{light}} = \text{ReLU}(\tau_{\text{ihr\_light}} - |u_{\text{ihr}}|) \cdot \text{ReLU}(\tau_{\text{acc\_light}} - |u_{\text{acc}}|)$$
    where $|u_{\text{ihr}}| = |\Delta z_{\text{ihr}}|$ and $|u_{\text{acc}}| = |\Delta z_{\text{acc}}|$ measure absolute deviations from the 91-epoch ultradian basal mean.
  * Initialized $\tau_{\text{ihr\_light}} = 0.50$ and $\tau_{\text{acc\_light}} = 0.25$.
  * Increased `hidden_dim` in `bm_casci` from 3 to 4 dimensions (`fc1 = Linear(8, 4)`, `norm = LayerNorm(4)`, `fc2 = Linear(4, 144)`, `fc_logit = Linear(4, 4)`).
  * Enforced `BoundedMarginSTE` stability cutoffs:
    - $\tau_{\text{acc\_rem}} \ge 0.05, \tau_{\text{acc\_deep}} \ge 0.05, \tau_{\text{acc\_wake}} \ge 0.0, \tau_{\text{acc\_light}} \ge 0.05$
    - $\tau_{\text{ihr\_rem}} \ge -0.05, \tau_{\text{ihr\_deep}} \ge 0.0, \tau_{\text{ihr\_wake}} \ge 0.0, \tau_{\text{ihr\_light}} \ge 0.05$
- **Verification Execution Output**:
  * Ran `./.venv/bin/python experiments/EXP-014/verify_candidate_rho.py` on `cuda:0` (NVIDIA GeForce RTX 3060).
  * Output verbatim:
    ```
    ================================================================================
    RUNNING CANDIDATE RHO 10-TEST AUTOMATED VERIFICATION SUITE
    ================================================================================
    Target Device: cuda:0
    Device Name:   NVIDIA GeForce RTX 3060

    --- TEST 1: Parameter Count and Strict Budget Compliance ---
    Candidate Rho Trainable Parameters: 196,540
    Detailed Parameter Breakdown:
      - hr_encoder        :   29,568 params
      - acc_encoder       :   29,568 params
      - meta_proj         :       96 params
      - basal_dslbs       :       16 params
      - acc_dslbs         :        0 params
      - luan_hr           :      320 params
      - luan_acc          :      320 params
      - bm_casci          :      648 params
      - circadian_gate    :    2,592 params
      - temporal_gru      :   97,920 params
      - temp_norm         :      192 params
      - inter_epoch_attn  :   16,416 params
      - classifier_head   :   18,884 params
    Target Budget Expected:                 196,540 params
    Hard Cap:                              196,728 params
    Headroom below Hard Cap:               188 params
    Compression vs 5.47M SLAMSS-IFS:       96.41%
    [PASS] TEST 1: Parameter Count and Strict Budget Compliance Verified!

    --- TEST 2: Autonomic & Somatomotor Homeostasis Contrast ---
      Prior Kappa/Omicron s_light during quiet sleep: 0.000000 (MATHEMATICALLY ZERO!)
      Candidate Rho s_light during quiet sleep:        0.080000 (ACTIVELY FIRING > 0!)
      Candidate Rho s_light during active Wake movement: 0.000000 (Cleanly 0.0)
      Candidate Rho s_light during Deep sleep bradycardia: 0.000000 (Cleanly 0.0)
      Candidate Rho s_light during REM autonomic surge:    0.000000 (Cleanly 0.0)
    [PASS] TEST 2: Autonomic & Somatomotor Homeostasis Contrast Verified!

    --- TEST 3: 4-Stage Context Bottleneck Rank-4 Subspace ---
      fc1 weight shape: torch.Size([4, 8]), rank: 4 (full row rank = 4)
      fc2 weight shape: torch.Size([144, 4])
      fc_logit weight shape: torch.Size([4, 4])
    [PASS] TEST 3: 4-Stage Context Bottleneck Rank-4 Subspace Verified!

    --- TEST 4: BoundedMarginSTE Stability Cutoffs & Anti-Drift Protection ---
      tau_acc_rem:   raw = -0.50 -> clamped = +0.05 (cutoff >= +0.05 enforced)
      tau_ihr_rem:   raw = -0.50 -> clamped = -0.05 (cutoff >= -0.05 enforced)
      tau_acc_deep:  raw = -0.50 -> clamped = +0.05 (cutoff >= +0.05 enforced)
      tau_ihr_deep:  raw = -0.50 -> clamped = +0.00 (cutoff >=  0.00 enforced)
      tau_acc_wake:  raw = -0.50 -> clamped = +0.00 (cutoff >=  0.00 enforced)
      tau_ihr_wake:  raw = -0.50 -> clamped = +0.00 (cutoff >=  0.00 enforced)
      tau_acc_light: raw = -0.50 -> clamped = +0.05 (cutoff >= +0.05 enforced)
      tau_ihr_light: raw = -0.50 -> clamped = +0.05 (cutoff >= +0.05 enforced)
      STE gradient pass-through on clamped parameter: -1.9000
    [PASS] TEST 4: BoundedMarginSTE Stability Cutoffs and STE Gradient Flow Verified!

    --- TEST 5: Forward Pass on (B=4, C=4, T=600) on cuda:0 ---
      Logits Shape:          torch.Size([4, 600, 4])
      Emission Logits Shape: torch.Size([4, 600, 4])
      Synergies Shape:       torch.Size([4, 600, 4])
      Probabilities Shape:   torch.Size([4, 600, 4])
      Loss Scalar:           1.4704
    [PASS] TEST 5: Forward Pass on cuda:0 Verified!

    --- TEST 6: End-to-End Gradient Flow to ALL Trainable Parameters ---
    Part A (zero_init=False): Total Trainable Parameter Tensors: 119
    Tensors with Active Gradients: 119 / 119
      bm_casci.tau_ihr_rem grad norm:   1.452932e-05
      bm_casci.tau_acc_rem grad norm:   7.210683e-07
      bm_casci.tau_ihr_light grad norm: 4.233172e-06
      bm_casci.tau_acc_light grad norm: 9.327196e-06
      bm_casci.fc1.weight grad norm:    5.655761e-03
      bm_casci.fc2.weight grad norm:    1.077569e-02
      bm_casci.fc_logit.weight grad:    1.296477e-01
      temporal_gru weight_ih_l0 grad:   1.282232e-01
      classifier_head.linear.weight:    5.149208e-01
    Part B (zero_init=True): Production mode gradients active on classifier head and BM-CASCI.
    [PASS] TEST 6: 100% of Trainable Parameters Receive Non-Zero Finite Gradients!

    --- TEST 7: RCCLoss Compatibility & Multi-Task Integration ---
      Loss without RCCLoss (CE only): 1.4309
      Loss with RCCLoss (CE + RCC):   1.5113
    [PASS] TEST 7: RCCLoss Compatibility Verified!

    --- TEST 8: Tensor-Input vs Dict-Input Forward Equivalence ---
      Max absolute logit difference (dict vs positional args): 0.00e+00
    [PASS] TEST 8: Calling Convention Equivalence Verified!

    --- TEST 9: Accelerometry Scale-Invariance & Gravity Tilt Invariance ---
      Max abs difference in delta_z_acc under 10x gain & +50.0 DC offset: 5.72e-05
    [PASS] TEST 9: Accelerometry Scale and Tilt Invariance Verified!

    --- TEST 10: Multi-Step Optimization Simulation on cuda:0 ---
      Step 0 Loss: 1.4651 -> Step 4 Loss: 1.4223
      Initial tau_ihr_light: 0.5000
      Final tau_ihr_light:   0.4975 (delta = -2.4917e-03)
    [PASS] TEST 10: Multi-Step Optimization Simulation Verified!

    ================================================================================
    CANDIDATE RHO VERIFICATION SUMMARY:
    ================================================================================
      Test 1: Budget Compliance               : [PASS]
      Test 2: AH-CASCI Active Firing          : [PASS]
      Test 3: Rank-4 Bottleneck               : [PASS]
      Test 4: Bounded Margins                 : [PASS]
      Test 5: Forward Pass                    : [PASS]
      Test 6: Gradient Flow                   : [PASS]
      Test 7: RCCLoss Compatibility           : [PASS]
      Test 8: Calling Equivalence             : [PASS]
      Test 9: ACC Invariance                  : [PASS]
      Test 10: Optimization Simulation        : [PASS]
    ================================================================================
    ALL 10 TESTS PASSED DEFINITIVELY ON CUDA:0!
    ```

## 2. Logic Chain
1. **From Observation 1**: Prior candidates suffered from a mathematical identity flaw: with $u_{\text{acc}} \le 0$ during quiet sleep and $\tau_{\text{acc\_light}} \ge 0$, $\text{ReLU}(u_{\text{acc}} - \tau_{\text{acc\_light}}) \equiv 0.0$. Thus $s_{\text{light}} \equiv 0.0$ across the entire night.
2. **From Physiological Principles**: Light sleep (N1/N2) represents autonomic and somatomotor basal homeostasis / quiescence: heart rate fluctuates within normal resting limits ($|u_{\text{ihr}}| < \tau_{\text{ihr\_light}}$) and gross body movement is absent ($|u_{\text{acc}}| < \tau_{\text{acc\_light}}$).
3. **From Formulation**: Defining $s_{\text{light}} = \text{ReLU}(\tau_{\text{ihr\_light}} - |u_{\text{ihr}}|) \cdot \text{ReLU}(\tau_{\text{acc\_light}} - |u_{\text{acc}}|)$ converts quiet sleep into a positive, active synergy feature ($s_{\text{light}} = 0.080000$ in Test 2), while automatically shutting off during movement ($s_{\text{light}}=0.0$), deep bradycardia ($s_{\text{light}}=0.0$), and REM tachycardia ($s_{\text{light}}=0.0$).
4. **From Bottleneck Dimension**: Expanding `hidden_dim` in `bm_casci` from 3 to 4 provides a 1-to-1 rank-4 subspace matching the 4 sleep stages [Wake, Light, Deep, REM], adding exactly +159 parameters.
5. **From Parameter Budget**: Baseline 196,381 + 159 = 196,540 parameters. This is strictly compliant with the $\le 196,728$ cap (188 parameters headroom, 96.41% compression ratio).
6. **From Stability Enforcement**: `BoundedMarginSTE` guarantees that even under adverse gradient forces, margin drift cannot penetrate below physiological limits ($\tau_{\text{acc\_rem}} \ge 0.05$, etc.), while the straight-through estimator maintains uninhibited backpropagation gradients (100% of all 119 trainable parameter tensors actively receive gradients).

## 3. Caveats
- Fast proxy evaluation on Fold 1 for 50 epochs (seed 42) has not yet been executed; this task is scoped to the worker agent per NAS syndicate protocol.
- The default initial margins $\tau_{\text{ihr\_light}}=0.50$ and $\tau_{\text{acc\_light}}=0.25$ assume standardized deviations ($z$-scores) where $\pm 0.5$ SD captures the central ~38% of HR variations and $\pm 0.25$ SD captures quiet sleep immobility; these are learnable and free to adapt.

## 4. Conclusion
- Candidate Rho (`AutonomicHomeostasisBoundedMarginPyramidCNNBiGRU`) successfully and definitively solves the fatal zero-collapse defect of prior candidates' Light sleep synergy signal.
- The model architecture is implemented in `experiments/EXP-014/nas_iteration19_model.py`.
- The architecture is fully verified across 10 automated unit and integration tests in `experiments/EXP-014/verify_candidate_rho.py` on `cuda:0`.
- Total trainable parameters: **Exactly 196,540 parameters** ($\le 196,728$ cap, 188 parameters headroom, 96.41% compression ratio).
- Ready for immediate proxy evaluation by the worker agent.

## 5. Verification Method
- **Command**:
  ```bash
  ./.venv/bin/python experiments/EXP-014/verify_candidate_rho.py
  ```
- **Files to Inspect**:
  - `experiments/EXP-014/nas_iteration19_model.py` (Candidate Rho implementation)
  - `experiments/EXP-014/verify_candidate_rho.py` (Verification suite)
  - `.agents/teamwork/teamwork_explorer_r19_1/BRIEFING.md` (Agent state)
  - `.agents/teamwork/teamwork_explorer_r19_1/progress.md` (Agent heartbeat)
- **Invalidation Conditions**:
  - Total trainable parameters $\ne 196,540$ or $> 196,728$.
  - $s_{\text{light}} == 0.0$ when $|u_{\text{ihr}}| < \tau_{\text{ihr\_light}}$ and $|u_{\text{acc}}| < \tau_{\text{acc\_light}}$.
  - Any parameter tensor has `grad is None` or `grad == 0.0` after backward pass with `zero_init=False`.
  - Any verification test fails on `cuda:0`.
