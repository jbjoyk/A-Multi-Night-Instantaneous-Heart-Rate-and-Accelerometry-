# Handoff Report: NAS Iteration 8 Architecture Exploration (Candidate Zeta)

**Agent**: `teamwork_explorer_r8_1`  
**Milestone**: NAS Iteration 8 — Candidate Zeta Architecture Specification  
**Recipient**: Parent Orchestrator (`5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91`) and Implementer Worker (`teamwork_worker_r8_1`)  
**Timestamp**: 2026-11-15T07:44:00Z  

---

## 1. Observation

1. **Candidate Epsilon Outer-Test Evaluation & Confusion Matrix**:
   In `experiments/EXP-014/nas_iteration7_proxy_results.json` (lines 84–116):
   - Checkpoint Epoch 43 reached:
     - Outer Test Accuracy: **64.27%**
     - Outer Test Weighted-F1: **0.6435**
     - Outer Test Macro-F1: **0.6435**
     - REM Recall: **59.64%** (up from 49.37% in Candidate Delta, a +10.27 pp gain).
     - Light->REM false positive count dropped from 3,740 in Delta to 2,447 in Epsilon (a 34.6% reduction).
   - Verbatim Outer Test Confusion Matrix (lines 91–116):
     ```
     [[2024,  751,  151,  280],  # Wake (support: 3,206)
      [ 710, 9239, 1468, 2447],  # Light (support: 13,864)
      [  90, 2290, 5127,  412],  # Deep (support: 7,919)
      [ 118, 2396,  355, 4240]]  # REM (support: 7,109)
     ```
   - **Remaining Failure Mode (Deep-Light Confusion)**:
     - 2,290 Deep sleep windows were misclassified as Light sleep (28.92% of all Deep sleep).
     - 1,468 Light sleep windows were misclassified as Deep sleep (10.59% of Light sleep).
     - Deep-Light confusion accounts for $2,290 + 1,468 = 3,758$ windows out of 32,098 total windows (**11.71% of all outer-test data**).

2. **Physiological Root Cause Diagnosis**:
   In `experiments/EXP-014/nas_iteration7_model.py` (lines 62–161), Candidate Epsilon used a single ultradian window size ($W = 91$ epochs = 45.5 minutes) for Local Ultradian Basal Standardization (LUBS):
   - A single 45.5-minute window captures the broad 90-minute ultradian macro-drift, but acts as a low-pass filter that washes out transient autonomic dynamics.
   - Physiological sleep staging requires resolving two distinct autonomic timescales:
     * **Macro timescale** ($W_{\text{macro}} \approx 91$ epochs / 45.5 min): Captures the 90-minute REM-NREM ultradian cycle and circadian baseline drift.
     * **Micro timescale** ($W_{\text{micro}} \approx 15$ epochs / 7.5 min): Resolves transient autonomic micro-activations, sympathetic bursts, arousals, and sleep spindles/K-complexes.
   - **Slow Wave Sleep (Deep / N3)** requires sustained, quiescent parasympathetic suppression without transient sympathetic spikes.
   - **Light Sleep (N1 / N2)** is punctuated by frequent transient autonomic perturbations and micro-arousals against a fluctuating baseline.
   - When only $W=91$ is used, a quiet period of Light sleep without arousals looks identical to Deep sleep, causing 1,468 Light windows to be misclassified as Deep; conversely, a single arousal during Light sleep averages into the 45.5-minute window, diluting the signal, while Deep sleep with slight baseline variability gets labeled Light (2,290 windows).

3. **Autonomic Contrast Differential Formulation**:
   By defining dual complementary running local statistics:
   $$\mu_{\text{macro}}(t), \sigma_{\text{macro}}(t) \quad (W_{\text{macro}} = 91 \text{ epochs} = 45.5 \text{ min})$$
   $$\mu_{\text{micro}}(t), \sigma_{\text{micro}}(t) \quad (W_{\text{micro}} = 15 \text{ epochs} = 7.5 \text{ min})$$
   The **Autonomic Contrast Differential** $\Delta z(t)$ is defined as:
   $$\Delta z(t) = \frac{\mu_{\text{micro}}(t) - \mu_{\text{macro}}(t)}{\sigma_{\text{macro}}(t)}$$
   - Sustained Deep sleep: $\Delta z(t) \approx 0.0$ and $z_{\text{macro}}(t) < -0.5$ (parasympathetic tone stably suppressed below nightly mean).
   - Transient arousal in Light sleep: $\Delta z(t) > 0.0$ (micro mean sharply exceeds local macro baseline).
   - REM bursts: $\Delta z(t) \gg 0.0$ and $z_{\text{macro}}(t) > 0.0$ (sympathetic storm on elevated circadian baseline).

4. **Parameter Budget Constraints**:
   Per `ORIGINAL_REQUEST.md` (Requirement R2) and `PROJECT.md`:
   - Target parameter range: 162k–196k parameters (>90% compression vs 5.47M published SLAMSS-IFS benchmark).
   - Parameter budget cap: strictly $\le 196,728$ parameters.
   - Baseline Model A: 163,940 parameters.
   - Candidate Epsilon (Iter 7): 195,708 parameters (1,020 headroom).
   - Candidate Zeta (Iter 8): **195,892 parameters** (836 headroom under cap; 96.42% parameter reduction vs 5.47M benchmark).

5. **Implementation & Unit Test Suite Execution**:
   - Proposed architecture implementation written to:
     `.agents/teamwork/teamwork_explorer_r8_1/proposed_nas_iteration8_model.py`
   - Complete unit test suite written to:
     `.agents/teamwork/teamwork_explorer_r8_1/test_nas_iteration8.py`
   - Executed `./.venv/bin/python .agents/teamwork/teamwork_explorer_r8_1/test_nas_iteration8.py`:
     ```
     [Setup] Running TestNASIteration8CandidateZeta on device: cuda:0
     [Test 01 PASSED] Params: 195,892 | Cap: 196,728 | Headroom: 836 | Compression: 96.42%
     .[Test 02 PASSED] Dual-Scale Local Basal Standardization verified across 52 bpm vs 80 bpm baselines.
     .[Test 03 PASSED] Multi-Scale Latent Normalization dimensions and zero-init verified.
     .[Test 04 PASSED] Dynamic Circadian Phase Gating exact step-0 identity confirmed.
     .[Test 05 PASSED] Forward pass across all 4 interface formats verified on BIDSleep shapes (B=4, T=600).
     .[Test 06 PASSED] Variable sequence length packed execution confirmed.
     .[Test 07 PASSED] Step-0 numerical stability verified (Loss: 1.5080).
     .[Test 08 PASSED] Backward gradient coverage confirmed across all 104 named parameters.
     .[Test 09 PASSED] Direct intra-epoch skip gradient highway to raw input tensors confirmed.
     .[Test 10 PASSED] Autonomic contrast verified: Deep Delta z = 0.0000 (~0) vs Light Arousal Delta z = 1.8141 (>0.5).
     ----------------------------------------------------------------------
     Ran 10 tests in 1.249s
     OK
     ```

---

## 2. Logic Chain

1. **Identification of the Dominant Bottleneck (Observation 1 $\to$ Step 1)**:
   Candidate Epsilon proved that local running standardization (LUBS) successfully resolves the macro non-stationarity of whole-night sleep, establishing a new project best (64.27% Accuracy, 0.6435 Weighted-F1) and dropping Light->REM false positives by 34.6%. However, 3,758 windows (11.71% of all outer-test data) remain confused across the Deep-Light boundary, forming the single largest error cluster in the current architecture.

2. **Dual-Scale Temporal Dynamics (Observation 2 & 3 $\to$ Step 2)**:
   Slow Wave Sleep (Deep / N3) and Light Sleep (N1 / N2) operate at the same macro circadian basal heart rate level, but differ in their high-frequency temporal stability:
   - Deep sleep is characterized by sustained parasympathetic tone with near-zero derivative and near-zero micro-macro divergence:
     $\Delta z(t) = \frac{\mu_{\text{micro}}(t) - \mu_{\text{macro}}(t)}{\sigma_{\text{macro}}(t)} \approx 0.0$ and $z_{\text{macro}}(t) < -0.5$.
   - Light sleep is interrupted by micro-arousals (timescale $\approx 7.5$ minutes, $W_{\text{micro}}=15$), where sympathetic tone spikes sharply above the 45.5-minute local baseline:
     $\Delta z(t) > 0.0$.
   Introducing dual complementary box kernels ($W_{\text{micro}}=15$ and $W_{\text{macro}}=91$) enables the model to explicitly quantify transient autonomic activations relative to the basal sleep stage background.

3. **Architectural Formulation of Candidate Zeta (Observation 3 & 4 $\to$ Step 3)**:
   Candidate Zeta (`LocalUltradianPyramidCNNBiGRU`) introduces:
   - **Dual-Scale Local Basal Standardization (DSLBS)**:
     Vectorized parameter-free dual 1D depthwise convolutions computing micro and macro statistics with exact padding mask protection.
     Standardizes raw IHR: $z_{\text{ihr}} = (z_{\text{ihr\_macro}} + \alpha_{\text{ihr\_diff}} \cdot \Delta z_{\text{ihr}}) \cdot \gamma_{\text{ihr}} + \beta_{\text{ihr}}$.
     With $\alpha_{\text{ihr\_diff}} = 0.0$ at initialization, $z_{\text{ihr}}$ identically reproduces Candidate Epsilon at step 0 while unlocking gradient-directed adaptation.
   - **Multi-Scale Local Ultradian Latent Normalization (Latent MUP LUAN)**:
     Applied independently to latent HR (64) and ACC (64) representations.
     Computes depthwise 1D micro and macro statistics across all channels (`groups=64`), synthesizing:
     $x_{\text{pyramid}} = (x_{\text{norm\_macro}} + \alpha \odot \Delta x_{\text{norm}}) \cdot w + b$.
     With $\alpha \equiv \mathbf{0}$, this achieves exact step-0 identity to Candidate Epsilon's Latent LUAN.
   - **Dynamic Circadian Phase Gating (DCPG)**:
     Conditions on a 6-dimensional Multi-Scale Ultradian Temporal Pyramid state:
     $c_{\text{mup}}(t) = [t/T, \text{circ\_trend}, \text{circ\_var\_ratio}, \Delta z(t), z_{\text{macro}}(t), \sigma_{\text{micro}}/\sigma_{\text{macro}}]$.
     With `zero_init=True`, $\text{fc2} \equiv 0 \implies 2 \cdot g(t) \equiv 1.0$, preserving exact step-0 identity.
   - **Inter-Epoch Sequence Attention (IESA)**:
     Retains the 4-head sequence attention ($d_{\text{attn}}=48$, 16,416 params) providing direct $O(1)$ receptive field access across all 600 epochs.
   - **Direct 144-dimensional Intra-Epoch Skip Highway**:
     Directly bridges intra-epoch features ($64 \text{ HR} + 64 \text{ ACC} + 16 \text{ Meta} = 144$) into the classifier head (`Linear(288, 64) -> LayerNorm -> GELU -> Dropout -> Linear(64, 4)`, 18,884 params), ensuring direct 1-step backpropagation into the encoders.

4. **Budget and Capacity Verification (Observation 4 & 5 $\to$ Conclusion)**:
   Component parameter breakdown:
   - `hr_encoder`: 29,568 params
   - `acc_encoder`: 29,568 params
   - `meta_proj`: 96 params
   - `basal_dslbs` (DSLBS): 16 params (+8 params over Epsilon)
   - `luan_hr` (Latent MUP LUAN): 320 params (+64 params over Epsilon)
   - `luan_acc` (Latent MUP LUAN): 320 params (+64 params over Epsilon)
   - `circadian_gate` (DCPG with 6-dim context): 2,592 params (+48 params over Epsilon)
   - `temporal_gru` (BiGRU 2-layer): 97,920 params
   - `temp_norm`: 192 params
   - `inter_epoch_attn` (IESA): 16,416 params
   - `classifier_head` (Multi-Stream): 18,884 params
   - **Total parameters**: **195,892 parameters** (exactly 836 parameter headroom under the 196,728 cap; 96.42% parameter compression vs 5.47M benchmark).
   All 10 unit tests passed on CUDA in 1.25s with 100% gradient coverage across all 104 named parameters.

---

## 3. Caveats

1. **Dual Timescale Selection**:
   The default window sizes are set to $W_{\text{macro}} = 91$ epochs (45.5 min, half-ultradian cycle) and $W_{\text{micro}} = 15$ epochs (7.5 min, typical micro-arousal cluster duration). While both are configurable, these values were selected based on classical sleep physiology literature.
2. **Proxy Training Execution**:
   This report encompasses architectural specification, mathematical derivation, parameter accounting, and unit test verification. Fast proxy training on Fold 1 (50 epochs on `cuda:0`) is delegated to the implementer worker agent (`teamwork_worker_r8_1`).

---

## 4. Conclusion

**Candidate Zeta (`LocalUltradianPyramidCNNBiGRU`)** is fully specified, verified, and ready for deployment in NAS Iteration 8:
- **Architecture Name**: `LocalUltradianPyramidCNNBiGRU` (aliases: `CandidateZeta_MUP`, `CandidateZeta_DSLBS`).
- **Target Hypotheses**:
  - **Hypothesis H-F**: Autonomic-somatomotor cross-modal coupling with 144-dimensional direct skip highway into multi-stream classification head.
  - **Hypothesis H-A**: Multi-Scale Ultradian Temporal Pyramids (MUP) and Dual-Scale Local Basal Standardization (DSLBS), resolving Deep-Light sleep confusion via the Autonomic Contrast Differential $\Delta z(t)$.
- **Parameter Capacity**:
  - Exactly **195,892 parameters** (Cap: 196,728; Headroom: 836; Compression: 96.42% vs 5.47M SLAMSS-IFS benchmark).
- **Step-0 Identity**:
  - Guaranteed exact mathematical identity to Candidate Epsilon at step 0 via zero-initialization of $\alpha$ differentials and DCPG gate projection.
- **Unit Test Status**:
  - 10/10 tests passed on CUDA (`cuda:0`), verifying all 4 interface shapes, sequence packing safety, numerical stability, and 100% gradient coverage across all 104 named parameters.

---

## 5. Verification Method

To independently verify Candidate Zeta:

1. **Execute Complete Unit Test Suite (10/10 pass)**:
   ```bash
   ./.venv/bin/python .agents/teamwork/teamwork_explorer_r8_1/test_nas_iteration8.py
   ```
   *Expected output*:
   ```
   [Setup] Running TestNASIteration8CandidateZeta on device: cuda:0
   [Test 01 PASSED] Params: 195,892 | Cap: 196,728 | Headroom: 836 | Compression: 96.42%
   .[Test 02 PASSED] Dual-Scale Local Basal Standardization verified across 52 bpm vs 80 bpm baselines.
   .[Test 03 PASSED] Multi-Scale Latent Normalization dimensions and zero-init verified.
   .[Test 04 PASSED] Dynamic Circadian Phase Gating exact step-0 identity confirmed.
   .[Test 05 PASSED] Forward pass across all 4 interface formats verified on BIDSleep shapes (B=4, T=600).
   .[Test 06 PASSED] Variable sequence length packed execution confirmed.
   .[Test 07 PASSED] Step-0 numerical stability verified (Loss: 1.5080).
   .[Test 08 PASSED] Backward gradient coverage confirmed across all 104 named parameters.
   .[Test 09 PASSED] Direct intra-epoch skip gradient highway to raw input tensors confirmed.
   .[Test 10 PASSED] Autonomic contrast verified: Deep Delta z = 0.0000 (~0) vs Light Arousal Delta z = 1.8141 (>0.5).
   ----------------------------------------------------------------------
   Ran 10 tests in 1.249s
   OK
   ```

2. **Inspect Parameter Count & Headroom**:
   ```bash
   ./.venv/bin/python -c "
   import sys
   sys.path.insert(0, '.agents/teamwork/teamwork_explorer_r8_1')
   from proposed_nas_iteration8_model import build_nas_iteration8_model
   m = build_nas_iteration8_model()
   total = sum(p.numel() for p in m.parameters())
   print('Candidate Zeta Parameters: {:,}'.format(total))
   print('Budget Cap:                196,728')
   print('Headroom:                  {:,}'.format(196728 - total))
   print('Cap Compliant:             {}'.format(total <= 196728))
   "
   ```
   *Expected output*:
   ```
   Candidate Zeta Parameters: 195,892
   Budget Cap:                196,728
   Headroom:                  836
   Cap Compliant:             True
   ```

3. **Deploy Code to Experiments Directory (Implementer Worker Action)**:
   ```bash
   cp .agents/teamwork/teamwork_explorer_r8_1/proposed_nas_iteration8_model.py experiments/EXP-014/nas_iteration8_model.py
   ```
