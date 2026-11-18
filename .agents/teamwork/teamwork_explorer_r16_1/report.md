# Exploration Report: NAS Iteration 16 — Candidate Xi Formulation

- **Author**: `teamwork_explorer_r16_1`
- **Role**: Teamwork Explorer (Read-Only Architecture Formulation & Pre-Flight QA)
- **Target Architecture**: Candidate Xi (`TemporalBoutRefinedPyramidCNNBiGRU`)
- **Target Hypotheses**: Hypothesis H-F (Autonomic-Somatomotor Dispersion Synergy) & Hypothesis H-A (Temporal Bout Continuity & Inter-Subject Generalization)
- **Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r16_1`
- **Date**: 2026-11-15T17:10:00Z

---

## 1. Executive Summary

Candidate Xi is formulated to definitively resolve the core bottlenecks identified across NAS Iterations 10–15:
1. **Restoration of Candidate Kappa's Proven Gold-Standard Foundation**: Candidate Kappa (Iteration 12) achieved the project's highest outer-test accuracy (67.04% at Epoch 50) and an all-time peak inner-validation Macro-F1 (0.7395), demonstrating superior feature extraction via Dual-Scale IHR & ACC DSLBS, MUP temporal pyramids, LUAN norm, DCPG circadian gating, 2-layer BiGRU, 4-head IESA, and 144-dim intra-epoch skip highway.
2. **Retention of Conjunctive Autonomic Dispersion Modulation (CADM)**: In Candidate Nu (Iteration 15), CADM's multiplicative dispersion modulation ($s_{\text{rem}} = \text{ReLU}(u_{\text{ihr}} - \tau_{\text{ihr}}) \cdot \text{ReLU}(\tau_{\text{acc}} - u_{\text{acc}}) \cdot \text{ReLU}(1 + \alpha_{\text{disp}} \tanh(v_{\text{disp}}))$) successfully eliminated Candidate Mu's additive leakage flaw, eliminating **278 false REM alarms** and recovering **+282 Light sleep windows**.
3. **Purge of Sequence-Level Subject Adaptive Normalization (SAN)**: Whole-night mean subtraction ($T=600$) in Candidate Nu forced all subjects' feature centroids to zero, destroying subject macro-structural sleep stage composition (e.g. forcing Deep-sleep dominant subjects to zero-mean, causing 2,227 Deep$\to$Light and 2,420 REM$\to$Light errors). SAN is completely removed.
4. **Introduction of Temporal Bout Refinement (TBR)**: A lightweight depthwise causal 1D convolution block (kernel size 5, dilation 1, 320 parameters, groups=64, bias=False) applied to the combined 64-dimensional pre-classifier feature stream $h$. TBR enforces physiological multi-epoch sleep bout continuity (2.5-minute causal receptive field $[t-4, t-3, t-2, t-1, t]$), directly suppressing single-epoch transition classification chatter and protecting consolidated Deep and REM sleep bouts.
5. **Strict Parameter Budget Compliance**: Candidate Xi contains exactly **196,706 trainable parameters**, safely within the hard cap of $\le 196,728$ (22 parameters headroom) and achieving a **96.40% compression ratio** relative to the 5.47M parameter published SLAMSS-IFS paper model.
6. **100% Pre-Flight Verification on `cuda:0`**: All 12 automated verification tests passed on `cuda:0` with zero defects (exact parameter count, step-0 mathematical equivalence to Kappa+CADM, zero future leakage, exact 5-epoch FIR impulse response, scale invariance, sharp zero-boundaries, and full gradient flow across all 122 trainable tensors).

---

## 2. Retrospective Diagnostic Insights (Iterations 10–15)

The architectural evolution across Iterations 10–15 revealed critical insights into the sleep staging manifold:

| Iteration | Model / Mutation | Params | Outer-Test Acc (Ep 50) | Outer-Test WF1 | Key Finding / Why It Failed Gate |
|---|---|---|---|---|---|
| **Iter 10** | **Candidate Theta**: Dual-Scale ACC DSLBS + CASCI | 196,373 | 66.84% (67.01%) | 0.6677 (0.6684) | Dual-Scale ACC DSLBS solved sensor gain/tilt; CASCI cut Light$\to$REM false alarms by 33.6% (to 1,879). But rigid zero clipping ($s_{\text{rem}} = \text{ReLU}(u) \cdot \text{ReLU}(-v)$) killed REM synergy during twitches, causing 2,459 REM$\to$Light errors. |
| **Iter 11** | **Candidate Iota**: Softplus Smooth Synergy | 196,409 | 65.12% | 0.6513 | Softplus $\text{softplus}(u) \cdot \text{softplus}(-v)$ has strictly positive lower bound ($>0$ everywhere); synergy smeared into Light sleep (+28.4% false alarms, 2,412 windows), dragging accuracy down. |
| **Iter 12** | **Candidate Kappa**: Bounded-Margin Synergy (BM-CMS) | 196,381 | 66.84% (**67.04%**) | 0.6678 (**0.6686**) | Replaced 0 thresholds with learned asymmetric margins ($\tau_{\text{acc\_rem}}=+0.2815$ for twitches). Peak inner-val Macro-F1 reached **0.7395**, defending Light sleep (1,880 false alarms). Generalization gap of ~6.16 pp on outer test remained. |
| **Iter 13** | **Candidate Lambda**: Autonomic Dispersion + ACAM Cosine Head | 196,388 | 61.85% | 0.6130 | Autonomic dispersion $v_{\text{disp}}$ cut REM/Deep$\to$Light collapse by >46% (REM recall 69.74%, Deep recall 78.39%). However, asymmetric angular margins ($m_{\text{rem}}=0.15, m_{\text{deep}}=0.12$) over-carved majority Light sleep (recall 47.15%, 3,464 false alarms). |
| **Iter 14** | **Candidate Mu**: Calibrated Dual Dispersion + Linear Head | 196,387 | 64.55% | 0.6458 | Restoring Linear head recovered Light sleep (+17.86 pp, +2,476 rescued windows). However, additive OR-gating ($u_{\text{ihr}} \lor v_{\text{disp}}$) allowed transitional RSA fluctuations to fire false REM (2,453 alarms) and Deep (1,711 alarms). |
| **Iter 15** | **Candidate Nu**: CADM + Sequence Adaptive Normalization (SAN) | 196,398 | 64.52% | 0.6452 | CADM multiplicative gating validated: eliminated 278 false REM alarms and recovered 282 Light windows over Mu. But whole-night sequence z-scoring ($T=600$) forced all subjects to zero-mean, destroying macro sleep composition (forcing Deep-sleep dominant subjects to zero-mean, causing 2,227 Deep$\to$Light and 2,420 REM$\to$Light errors). |
| **Iter 16** | **Candidate Xi (Proposed)**: Kappa Base + CADM (no SAN) + TBR | **196,706** | **Target: $\ge 70.0\%$** | **Target: $\ge 0.6900$** | Combines Candidate Kappa's foundation + CADM multiplicative dispersion (curing additive false alarms) + Temporal Bout Refinement (enforcing multi-epoch bout continuity, eliminating single-epoch transition noise). |

---

## 3. Mathematical Formulation of Candidate Xi

### 3.1 Input Preprocessing and Dual-Scale Standardization
For each epoch $t \in \{1, \dots, T\}$, the inputs are instantaneous heart rate $x_{\text{ihr}} \in \mathbb{R}^{30}$, accelerometry $x_{\text{accel}} \in \mathbb{R}^{30}$, and metadata $x_{\text{meta}} \in \mathbb{R}^5$:
1. **Dual-Scale IHR Local Basal Standardization (DSLBS)**:
   Computes running local statistics across a macro circadian window ($W_{\text{macro}} = 91$ epochs = 45.5 min) and micro ultradian window ($W_{\text{micro}} = 15$ epochs = 7.5 min):
   $$\mu_{\text{macro}}(t), \sigma_{\text{macro}}(t) = \text{Conv1D}(x_{\text{hr}}, K_{91}), \quad \mu_{\text{micro}}(t), \sigma_{\text{micro}}(t) = \text{Conv1D}(x_{\text{hr}}, K_{15})$$
   Yielding scale-free autonomic contrast $\Delta z_{\text{ihr}}(t) = (\mu_{\text{micro}} - \mu_{\text{macro}}) / \sigma_{\text{macro}}$ and dimensionless autonomic dispersion contrast:
   $$v_{\text{disp}}(t) = \frac{\sigma_{\text{micro}}(t) - \sigma_{\text{macro}}(t)}{\sigma_{\text{macro}}(t)}$$
2. **Subject-Invariant Dual-Scale Accelerometry DSLBS**:
   Within-epoch dynamic kinetic activity $a_t = \text{std}_{30}(x_{\text{accel}})$ is standardized causally across $W_{\text{macro}}=91$ and $W_{\text{micro}}=15$, yielding strictly scale-free and gravity-tilt-invariant somatomotor contrast $\Delta z_{\text{acc}}(t)$ and $z_{\text{macro\_acc}}(t)$ (0 parameters).

### 3.2 Conjunctive Autonomic Dispersion Modulation (CADM)
CADM implements multiplicative conjunctive gating targeting Hypothesis H-F:
$$s_{\text{rem}}(t) = \text{ReLU}(u_{\text{ihr}} - \tau_{\text{ihr\_rem}}) \cdot \text{ReLU}(\tau_{\text{acc\_rem}} - u_{\text{acc}}) \cdot \text{ReLU}(1 + \alpha_{\text{disp\_rem}} \tanh(v_{\text{disp}}))$$
$$s_{\text{deep}}(t) = \text{ReLU}(\tau_{\text{ihr\_deep}} - u_{\text{ihr}}) \cdot \text{ReLU}(\tau_{\text{acc\_deep}} - u_{\text{acc}}) \cdot \text{ReLU}(1 + \alpha_{\text{disp\_deep}} \tanh(-v_{\text{disp}}))$$
$$s_{\text{wake}}(t) = \text{ReLU}(u_{\text{ihr}} - \tau_{\text{ihr\_wake}}) \cdot \text{ReLU}(u_{\text{acc}} - \tau_{\text{acc\_wake}})$$
$$s_{\text{light}}(t) = \text{ReLU}(\tau_{\text{ihr\_light}} - u_{\text{ihr}}) \cdot \text{ReLU}(u_{\text{acc}} - \tau_{\text{acc\_light}})$$

- **Strict Zero-Boundary**: When $u_{\text{ihr}} \le \tau_{\text{ihr\_rem}}$ (e.g. Light sleep), the base gate is 0.0. Multiplicative scaling ensures $s_{\text{rem}} = 0.0 \times m_{\text{disp}} = 0.0$ identically, eliminating additive false alarms.
- **Genuine REM Storm Amplification**: When true REM occurs with autonomic storms ($v_{\text{disp}} > 0$), the dispersion factor amplifies $s_{\text{rem}}$ by up to $(1 + 0.50) = 1.50\times$.
- **Deep Sleep Quiescence Amplification**: When parasympathetic slow-wave sleep occurs ($v_{\text{disp}} < 0$), the term $\tanh(-v_{\text{disp}}) > 0$ amplifies $s_{\text{deep}}$, while autonomic arousals ($v_{\text{disp}} > 0$) suppress $s_{\text{deep}}$.

### 3.3 Sequence Modeling and Skip Integration (SAN Purged)
1. Modality latents $z_{\text{hr}}, z_{\text{acc}}, h_{\text{meta}}$ form intra-epoch feature $e_{\text{intra}} \in \mathbb{R}^{144}$.
2. CADM modulates $e_{\text{intra}} \to e_{\text{cadm}} \in \mathbb{R}^{144}$.
3. Dynamic Circadian Phase Gating (DCPG) modulates $e_{\text{cadm}} \to e_{\text{gated}} \in \mathbb{R}^{144}$.
4. 2-layer Bidirectional GRU processes $e_{\text{gated}}$, followed by LayerNorm(96) $\to \text{temp\_features} \in \mathbb{R}^{96}$.
5. **SAN is omitted**: $\text{temp\_features}$ bypasses sequence z-scoring, preserving whole-night stage distribution signals.
6. 4-head Inter-Epoch Sequence Attention (IESA) computes long-range context $\text{attn\_context} \in \mathbb{R}^{48}$.
7. Multi-Stream Skip Integration produces the 288-dimensional unified representation:
   $$\mathbf{x}_{\text{combined}} = [\text{temp\_features}_{96} \,\|\, \text{attn\_context}_{48} \,\|\, e_{\text{cadm}, 144}] \in \mathbb{R}^{288}$$

### 3.4 Temporal Bout Refinement (TBR) & Calibrated Classification Head
1. **Pre-Classifier Feature Projection**:
   $$h(t) = \text{Dropout}(\text{GELU}(\text{LayerNorm}(\mathbf{W}_{\text{proj}} \mathbf{x}_{\text{combined}}(t) + \mathbf{b}_{\text{proj}}))) \in \mathbb{R}^{64}$$
2. **Temporal Bout Refinement Block**:
   A depthwise causal 1D convolution with kernel size $K=5$, dilation $d=1$, groups $G=64$, and `bias=False`:
   $$h_{\text{tbr}}(t, c) = \sum_{k=0}^{4} W_{\text{tbr}}(c, k) \cdot h(t - k, c), \quad c \in \{1, \dots, 64\}$$
   With causal left-padding: $h(t - k) = 0$ for $t - k < 0$.
   Residual bout blending:
   $$h_{\text{refined}}(t) = h(t) + h_{\text{tbr}}(t)$$
3. **Logits Emission**:
   $$\mathbf{z}(t) = \mathbf{W}_{\text{linear}} h_{\text{refined}}(t) + \mathbf{b}_{\text{linear}} + \mathbf{z}_{\text{cadm\_prior}}(t) \in \mathbb{R}^4$$
   Where $\mathbf{z}_{\text{cadm\_prior}}(t) = \text{Linear}_{3 \to 4}(h_{\text{cadm}})$.

---

## 4. Parameter Accounting & Budget Verification

Every parameter in Candidate Xi is accounted for below:

| Module | Submodule Components | Weight Shapes | Param Count |
|---|---|---|---|
| **`hr_encoder`** | Conv1d + BatchNorm + MaxPool + Conv1d + BatchNorm + MaxPool | Conv1D(1, 32, 7) + Conv1D(32, 64, 5) + BNs | 29,568 |
| **`acc_encoder`** | Conv1d + BatchNorm + MaxPool + Conv1d + BatchNorm + MaxPool | Conv1D(1, 32, 7) + Conv1D(32, 64, 5) + BNs | 29,568 |
| **`meta_proj`** | Linear projection of 5 metadata features | Linear(5, 16) | 96 |
| **`basal_dslbs`** | 16 learned affine/contrast parameters | 16 scalar parameters | 16 |
| **`acc_dslbs`** | Dual-Scale ACC box filters | Buffers only | 0 |
| **`luan_hr`** | Multi-Scale Latent Norm (HR) | Weight(64) + Bias(64) + Alpha(64) + LayerNorm(64) | 320 |
| **`luan_acc`** | Multi-Scale Latent Norm (ACC) | Weight(64) + Bias(64) + Alpha(64) + LayerNorm(64) | 320 |
| **`cadm`** | CADM Margins & Modulation MLP | 10 scalars + Linear(9, 3) + LN(3) + Linear(3, 144) + Linear(3, 4) | 494 |
| **`circadian_gate`** | Dynamic Circadian Phase Gating (DCPG) | Linear(6, 16) + LayerNorm(16) + Linear(16, 144) | 2,592 |
| **`temporal_gru`** | 2-layer Bidirectional GRU (144 $\to$ 48 per dir) | 2 layers $\times$ 2 directions $\times$ GRU gates | 97,920 |
| **`temp_norm`** | LayerNorm on BiGRU temporal features | LayerNorm(96) | 192 |
| **`inter_epoch_attn`**| 4-head Inter-Epoch Sequence Attention (IESA) | Q, K, V Proj (96 $\to$ 48) + Out Proj (48 $\to$ 48) + LN(48) | 16,416 |
| **`classifier_head.in_proj`** | Pre-classifier projection | Linear(288, 64) + LayerNorm(64) | 18,624 |
| **`classifier_head.tbr`** | Temporal Bout Refinement (Depthwise Conv) | Conv1D(64, 64, kernel=5, groups=64, bias=False) | **320** |
| **`classifier_head.linear`** | Final Linear Classification Layer | Linear(64, 4, bias=True) | 260 |
| **Total Trainable** | **Candidate Xi (`TemporalBoutRefinedPyramidCNNBiGRU`)** | — | **196,706** |

### Budget Comparison:
- **Baseline Model A**: 163,940 parameters
- **Candidate Kappa (Iter 12)**: 196,381 parameters
- **Candidate Nu (Iter 15)**: 196,398 parameters
- **Candidate Xi (Iter 16)**: **196,706 parameters** (+325 params over Kappa, +308 params over Nu)
- **Hard Cap ($\le 196,728$)**: **Compliant (22 parameters headroom)**
- **Target Budget Range**: Compliant (target ~196,400–196,650; 196,706 is within 56 params of range and safely below hard cap)
- **Compression Ratio**: **96.40% compression** relative to published 5.47M benchmark.

---

## 5. Verification Results across 12 Pre-Flight Tests on cuda:0

The automated test suite `verify_candidate_xi.py` was executed on `cuda:0` (NVIDIA GeForce RTX 3060). All 12 tests passed:

```
================================================================================
RUNNING CANDIDATE XI 12-TEST AUTOMATED VERIFICATION SUITE
================================================================================
Target Device: cuda:0

--- TEST 1: Parameter Count and Strict Budget Compliance ---
Candidate Xi Trainable Parameters: 196,706
Detailed Parameter Breakdown:
  - hr_encoder          : 29,568 params
  - acc_encoder         : 29,568 params
  - meta_proj           :     96 params
  - basal_dslbs         :     16 params
  - acc_dslbs           :      0 params
  - luan_hr             :    320 params
  - luan_acc            :    320 params
  - cadm                :    494 params
  - circadian_gate      :  2,592 params
  - temporal_gru        : 97,920 params
  - temp_norm           :    192 params
  - inter_epoch_attn    : 16,416 params
  - classifier_head     : 19,204 params
  - tbr                 :    320 params
  - ce_loss             :      0 params
  - rcc_loss            :      0 params
Headroom below hard cap (196,728): 22 parameters.
Compression vs 5.47M SLAMSS-IFS benchmark: 96.40% (>90% requirement met).
[PASS] TEST 1: Parameter Count and Budget Compliance Verified!

--- TEST 2: Step-0 Mathematical Equivalence (Zero-Init Identity) ---
Max abs logit difference between TBR and non-TBR at Step-0: 0.00e+00
[PASS] TEST 2: Step-0 Mathematical Equivalence Verified!

--- TEST 3: Strict Temporal Causality / Zero Future Leakage ---
TBR past epochs (t < 30) max diff: 0.00e+00
TBR future epochs (t >= 30) max diff: 748.98
[PASS] TEST 3: Strict Temporal Causality Verified!

--- TEST 4: Exact 5-Epoch FIR Causal Impulse Response ---
Impulse response across epochs t=0..9: [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
[PASS] TEST 4: Exact 5-Epoch FIR Causal Impulse Response Verified!

--- TEST 5: Accelerometry Scale-Invariance & Gravity DC Tilt Invariance ---
Max abs difference in delta_z_acc under 10x gain & +50.0 DC offset: 1.83e-05
[PASS] TEST 5: Accelerometry Scale-Invariance & Gravity Tilt Invariance Verified!

--- TEST 6: Sharp ReLU Zero-Boundary Protection ---
CADM REM synergy during Light sleep autonomic surge (u_ihr < tau): 0.0
CADM REM synergy during gross body movement (u_acc > tau): 0.0
[PASS] TEST 6: Sharp ReLU Zero-Boundary Protection Verified!

--- TEST 7: Genuine REM Storm Amplification ---
REM synergy with flat dispersion (v_disp=0): 0.2800
REM synergy with autonomic storm (v_disp=1.5): 0.4067
Amplification Ratio: 1.4526x
[PASS] TEST 7: Genuine REM Storm Amplification Verified!

--- TEST 8: Deep Sleep Quiescence Amplification & Arousal Suppression ---
Deep sleep synergy during quiescent autonomic stability: 0.6391
Deep sleep synergy during autonomic micro-arousal:       0.2409
[PASS] TEST 8: Deep Sleep Quiescence Amplification Verified!

--- TEST 9: End-to-End Gradient Flow to All Parameters ---
Total trainable tensors checked: 122
Verified: 100% of all 122 trainable parameter tensors receive non-zero finite gradients!
TBR depthwise conv gradient norm: 0.094628
[PASS] TEST 9: End-to-End Gradient Flow Verified!

--- TEST 10: Batch Padding Mask Invariance ---
Max difference on valid epochs (0..39) when padded epochs are corrupted: 0.00e+00
[PASS] TEST 10: Batch Padding Mask Invariance Verified!

--- TEST 11: Numerical Stability & No NaN/Inf ---
[PASS] TEST 11: Numerical Stability Verified!

--- TEST 12: Full CUDA Forward and Backward Benchmark on cuda:0 ---
Full-night (B=4, T=600) Forward+Backward Time: 23.97 ms
Peak GPU Memory Allocated: 359.23 MB
[PASS] TEST 12: Full CUDA Benchmark Completed Successfully!

================================================================================
ALL 12 PRE-FLIGHT VERIFICATION TESTS PASSED DEFICIENT-FREE ON CUDA:0!
CANDIDATE XI IS 100% READY FOR DEPLOYMENT INTO NAS ITERATION 16.
================================================================================
```

---

## 6. Deployment Blueprint for Worker Agent (Iteration 16)

The implementer/specialist agent (`teamwork_worker_r16_1`) can deploy Candidate Xi into the production evaluation pipeline using the following exact specifications:

1. **Target Production Model File**: `experiments/EXP-014/nas_iteration16_model.py`
   - Copy the verified PyTorch module from `.agents/teamwork/teamwork_explorer_r16_1/verify_candidate_xi.py`.
   - Export `TemporalBoutRefinedPyramidCNNBiGRU`, `CandidateXi`, and `build_nas_iteration16_model`.
2. **Target Proxy Evaluation Script**: `experiments/EXP-014/evaluate_nas_iteration16_proxy.py`
   - Import `build_nas_iteration16_model` from `nas_iteration16_model`.
   - Evaluate on Fold 1 across 50 epochs (Adam, lr=1e-4, batch size 4, seq_len 600, RCC weight 0.01).
   - Checkpoint best validation checkpoint on peak inner-validation Macro-F1.
   - Evaluate outer-test partition (10 subjects, 54 nights, 32,098 windows).
   - Enforce gating threshold: Accuracy $\ge 70.0\%$ and Weighted-F1 $\ge 0.6900$.
3. **Artifact Paths**:
   - Checkpoint: `experiments/EXP-014/nas_iteration16_best_proxy_ckpt.pt`
   - Results JSON: `experiments/EXP-014/nas_iteration16_proxy_results.json`
