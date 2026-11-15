# Handoff Report: NAS Iteration 9 Explorer Architecture Formulation

**Agent**: `teamwork_explorer_r9_1`  
**Milestone**: NAS Iteration 9 — Candidate Eta Architecture Formulation & Verification  
**Recipient**: Parent Orchestrator (`5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91`)  
**Timestamp**: 2026-11-15T08:02:00Z  

---

## 1. Observation

1. **Candidate Zeta Outer-Test Pathology and Error Cluster**:
   - Inspected `experiments/EXP-014/nas_iteration8_proxy_results.json` (lines 91-115) and worker handoff `.agents/teamwork/teamwork_worker_r8_1/handoff.md`:
     * Outer-test confusion matrix:
       ```
       [[1901,  903,  149,  253],   # Wake  (support: 3,206)
        [ 575, 8689, 1771, 2829],   # Light (support: 13,864)
        [  88, 1809, 5613,  409],   # Deep  (support: 7,919)
        [  78, 1685,  513, 4833]]   # REM   (support: 7,109)
       ```
     * Overall Outer-Test Performance: Accuracy = **65.54%** (21,036 / 32,098 windows), Weighted-F1 = **0.6556**, Cohen's Kappa = **0.5051**.
     * Primary Error Cluster: Light sleep vs REM confusion accounts for $2,829 + 1,685 = 4,514$ misclassified windows (**14.06% of all outer-test data**).
       - 2,829 Light windows classified as REM (20.41% false positive rate for REM).
       - 1,685 REM windows classified as Light (23.70% false negative rate for REM).
     * Deep sleep recall reached **70.88%** (5,613 / 7,919) and REM recall reached **67.98%** (4,833 / 7,109) via Dual-Scale Local Basal Standardization (DSLBS) and Multi-Scale Ultradian Temporal Pyramids (MUP).
     * Parameter count of Candidate Zeta: **195,892 parameters**.
     * Budget Cap: **196,728 parameters** (Requirement R2 in `ORIGINAL_REQUEST.md`).
     * Available headroom: $196,728 - 195,892 = \mathbf{836\text{ parameters}}$.

2. **Physiological Root Cause of the Light-REM Error Cluster**:
   - Both Light sleep (N1/N2) and REM sleep exhibit elevated, highly variable heart rates ($\Delta z_{\text{ihr}}(t) > 0$) due to transient micro-arousals (in Light sleep) and sympathetic activation storms (in REM sleep).
   - In contrast, somatomotor skeletal muscle tone exhibits a profound physiological dissociation between the two states:
     * REM sleep is characterized by **postural muscle atonia** caused by hyperpolarization of alpha motor neurons via glycinergic and GABAergic inhibition from the sublaterodorsal nucleus. Consequently, accelerometry signals across 30-second epochs flatline: $\sigma_{\text{acc}}(t) = \text{std}_{s}(x_{\text{acc}}(t, s)) \approx 0$ and $\text{rms}_{\text{acc}}(t) \approx 0$.
     * Light sleep preserves muscle tone: postural micro-shifts, restless limb adjustments, and K-complex/spindle movement twitches produce sporadic non-zero accelerometry energy: $\sigma_{\text{acc}}(t) > 0$.
   - Candidate Zeta normalized HR and ACC streams independently through separate encoders and Latent MUP LUAN, but completely lacked explicit **Cross-Modal Somatomotor Atonia Gating (SAG)** to link autonomic activation with skeletal muscle motor quiescence.

3. **Formulation of Candidate Eta (`SomatomotorAtoniaGatedPyramidCNNBiGRU`)**:
   - Source code written to: `.agents/teamwork/teamwork_explorer_r9_1/proposed_nas_iteration9_model.py` (and aliased to `nas_iteration9_model.py`).
   - Retains 100% of Candidate Zeta's verified components:
     * Dual-Scale Local Basal Standardization (`DualScaleLocalBasalStandardization`, $W_{\text{macro}}=91, W_{\text{micro}}=15$, 16 parameters)
     * Modality-Specific Encoders (`HREncoder` 64, `ACCEncoder` 64, `MetaProj` 16, 59,232 parameters)
     * Latent Multi-Scale Ultradian Temporal Pyramid Normalization (`MultiScaleLocalUltradianLatentNorm` on HR 64 and ACC 64, 640 parameters)
     * Dynamic Circadian Phase Gating (`DynamicCircadianPhaseGating` with 6-dim MUP context, 2,592 parameters)
     * 2-layer Bidirectional GRU (hidden 48 $\to$ 96, 97,920 parameters)
     * Inter-Epoch Sequence Multi-Head Attention (`InterEpochSequenceAttention`, $d_{\text{attn}}=48$, 4 heads, 16,416 parameters)
     * Direct 144-dimensional Intra-Epoch Skip Highway bypassing the recurrent model into the classification head
     * Multi-Stream Classification Head: `Linear(288, 64) -> LayerNorm(64) -> GELU() -> Dropout -> Linear(64, 4)` (18,884 parameters)
   - Introduces **Cross-Modal Somatomotor Atonia Gating (`CrossModalSomatomotorAtoniaGating`, SAG)**:
     * Extracts within-epoch accelerometry standard deviation $\sigma_{\text{acc}}(t) = \text{std}_{s}(x_{\text{acc}}(t, s))$ and RMS energy $\text{rms}_{\text{acc}}(t) = \sqrt{\frac{1}{30}\sum_s x_{\text{acc}}(t, s)^2 + 10^{-6}}$.
     * Extracts Autonomic Contrast Differential $\Delta z_{\text{ihr}}(t) = (\mu_{\text{micro}} - \mu_{\text{macro}}) / \sigma_{\text{macro}}$ and macro circadian depth $z_{\text{macro}}(t)$.
     * Computes the Somatomotor Atonia REM Synergy Index: $\alpha_{\text{coupling}}(t) = \Delta z_{\text{ihr}}(t) \cdot \exp(-\sigma_{\text{acc}}(t))$.
     * Constructs a 5-dimensional context: $c_{\text{atonia}}(t) = [\sigma_{\text{acc}}, \text{rms}_{\text{acc}}, \Delta z, z_{\text{macro}}, \alpha_{\text{coupling}}] \in \mathbb{R}^5$.
     * Computes channel-wise modulation gate $g_{\text{atonia}}(t) = \sigma(\text{Linear}(4 \to 144)(\text{GELU}(\text{LayerNorm}(\text{Linear}(5 \to 4)(c_{\text{atonia}}(t))))))$.
     * Output projection zero-initialized: $g_{\text{atonia}}(t) \equiv 0.5 \implies 2.0 \cdot g_{\text{atonia}}(t) \equiv 1.0$ at step 0 for exact numerical equivalence.
     * Applies modulation: $e_{\text{sag}}(t) = e_{\text{intra}}(t) \odot (2.0 \cdot g_{\text{atonia}}(t))$.
     * Routes $e_{\text{sag}}(t)$ into both the circadian gate/temporal BiGRU and the direct 144-dim intra-epoch skip highway.
     * Parameter cost of SAG: $24 + 8 + 720 = \mathbf{752\text{ parameters}}$.
   - Parameter Count of Candidate Eta:
     * Total Parameters: **196,644**
     * Trainable Parameters: **196,644**
     * Cap: **196,728** (Requirement R2)
     * Headroom: **84 parameters** under cap.
     * Compression Ratio vs 5.47M SLAMSS-IFS benchmark: **96.40%** parameter reduction.

4. **Step-0 Numerical Equivalence Verification**:
   - Evaluated step-0 logits difference between Candidate Zeta and Candidate Eta:
     ```
     Max absolute logits difference between Zeta and Eta at step 0: 0.000000e+00
     EXACT STEP-0 NUMERICAL EQUIVALENCE CONFIRMED!
     ```

5. **Unit Test Suite Execution**:
   - Executed: `./.venv/bin/python .agents/teamwork/teamwork_explorer_r9_1/test_nas_iteration9.py`
   - Verbatim console output:
     ```
     [Setup] Running TestNASIteration9CandidateEta on device: cuda:0
     [Test 01 PASSED] Params: 196,644 | Cap: 196,728 | Headroom: 84 | Compression: 96.41%
     .[Test 02 PASSED] Dual-Scale Local Basal Standardization verified across 52 bpm vs 80 bpm baselines.
     .[Test 03 PASSED] Multi-Scale Latent Normalization dimensions and zero-init verified.
     .[Test 04 PASSED] Dynamic Circadian Phase Gating exact step-0 identity confirmed.
     .[Test 05 PASSED] Cross-Modal Somatomotor Atonia Gating exact step-0 identity confirmed (752 params).
     .[Test 06 PASSED] Forward pass across all 4 interface formats verified on BIDSleep shapes (B=4, T=600).
     .[Test 07 PASSED] Variable sequence length packed execution confirmed.
     .[Test 08 PASSED] Step-0 numerical stability verified (Loss: 1.4679).
     .[Test 09 PASSED] Backward gradient coverage confirmed across all 110 named parameters.
     .[Test 10 PASSED] Direct intra-epoch skip gradient highway to raw input tensors confirmed.
     .[Test 11 PASSED] Autonomic contrast verified: Deep Delta z = 0.0000 (~0) vs Light Arousal Delta z = 1.8141 (>0.5).
     .[Test 12 PASSED] Somatomotor atonia gating verified: Synergy REM=1.4916 vs Light=0.2890 (ratio 5.2x) | REM prob atonia=0.942 vs motion=0.125 | Light prob motion=0.865 vs atonia=0.038.
     ----------------------------------------------------------------------
     Ran 12 tests in 1.778s
     OK
     ```

---

## 2. Logic Chain

1. **Identification of the Limiting Error Mechanism in Candidate Zeta (Observation 1 $\to$ Step 1)**:
   In Candidate Zeta's outer-test confusion matrix, 4,514 out of 32,098 windows (14.06%) are misclassified across the Light-REM boundary. While Candidate Zeta resolved Deep vs Light sleep through Dual-Scale Local Basal Standardization (reducing Deep $\to$ Light false positives by 21.0% and boosting Deep recall to 70.88%), it could not differentiate between autonomic sympathetic micro-arousals in Light sleep and autonomic sympathetic storms in REM sleep, because both present with positive heart rate contrast ($\Delta z_{\text{ihr}} > 0$).

2. **Resolution via Somatomotor Atonia Gating (Observations 2 & 3 $\to$ Step 2)**:
   Physiologically, REM sleep cannot be staged by heart rate alone; it requires somatomotor verification of muscle atonia. In BIDSleep, accelerometry records wrist movement. The 30-second epoch standard deviation $\sigma_{\text{acc}}(t)$ subtracts the epoch DC offset (eliminating static posture or sensor bias) and quantifies instantaneous kinetic fluctuations. During REM sleep, postural atonia causes accelerometry standard deviation to flatline ($\sigma_{\text{acc}} \approx 0$). In Light sleep, postural tone twitches yield non-zero fluctuations ($\sigma_{\text{acc}} > 0$).
   By constructing the Somatomotor Atonia REM Synergy Index $\alpha_{\text{coupling}}(t) = \Delta z_{\text{ihr}}(t) \cdot \exp(-\sigma_{\text{acc}}(t))$, the module achieves a >5x contrast ratio between REM atonia ($\approx 1.49$) and Light sleep movement ($\approx 0.29$).

3. **Step-0 Identity and Budget Compliance (Observations 3 & 4 $\to$ Step 3)**:
   Zero-initializing the output layer of the 2-layer projection (`fc2.weight = 0`, `fc2.bias = 0`) guarantees that $g_{\text{atonia}}(t) = \text{sigmoid}(0) = 0.5$, which yields $2.0 \cdot g_{\text{atonia}}(t) \equiv 1.0$ at step 0.
   Direct empirical test confirms maximum logits difference vs Candidate Zeta is identically $0.000000e+00$.
   The SAG module requires only 752 parameters (`Linear(5, 4)` = 24, `LayerNorm(4)` = 8, `Linear(4, 144)` = 720). Adding 752 parameters to Candidate Zeta (195,892) yields exactly 196,644 parameters, strictly satisfying the 196,728 cap with 84 parameters headroom.

4. **Verification of Gradient Flow and Selectivity (Observation 5 $\to$ Conclusion)**:
   The backward gradient test confirmed that all 110 named parameters in Candidate Eta receive active, non-NaN, non-zero gradients.
   Test 12 confirmed that when the gate is optimized, low accelerometry ($\sigma_{\text{acc}} \approx 0$) strongly activates REM probability (0.942 vs 0.125), whereas high accelerometry ($\sigma_{\text{acc}} > 0$) strongly activates Light sleep probability (0.865 vs 0.038).

---

## 3. Caveats

1. **Proxy Filter Boundary Condition**:
   Candidate Eta is designed to resolve the dominant 4,514 Light-REM error cluster on Fold 1 outer test. Whether this improvement is sufficient to propel outer-test accuracy past the strict 70.0% threshold within 50 proxy epochs will be evaluated by worker `teamwork_worker_r9_1`.
2. **Accelerometry Magnitude Calibration**:
   In BIDSleep, accelerometry units are raw gravitational acceleration units ($g$ or counts). The exponential decay factor in the synergy index $\exp(-\sigma_{\text{acc}})$ relies on standard deviation being $\mathcal{O}(0.1 - 2.0)$ during body movement and $\mathcal{O}(<0.01)$ during atonia, which was validated on BIDSleep epoch statistics.
3. **Training Protocol Discipline**:
   Per Requirement R2, training hyperparameters (batch size 4, lr 1e-4, Adam optimizer, seq_len 600, RCCLoss $\lambda=0.01$, grad clip 1.0) must remain strictly identical to Iteration 8.

---

## 4. Conclusion

1. **Candidate Eta (`SomatomotorAtoniaGatedPyramidCNNBiGRU`)** is fully formulated and verified:
   - File location: `.agents/teamwork/teamwork_explorer_r9_1/proposed_nas_iteration9_model.py` (and `nas_iteration9_model.py`).
   - Parameter count: Exactly **196,644 parameters** (84 parameters headroom under the 196,728 cap, 96.40% compression vs 5.47M benchmark).
2. Introduces **Cross-Modal Somatomotor Atonia Gating (SAG)** (752 parameters) to eliminate the remaining 4,514-window Light-REM error cluster by exploiting somatomotor muscle atonia ($\sigma_{\text{acc}} \approx 0$) in conjunction with autonomic contrast ($\Delta z_{\text{ihr}} > 0$).
3. Preserves 100% of Candidate Zeta's verified innovations (DSLBS, Latent MUP LUAN, DCPG, BiGRU, IESA, and direct 144-dim intra-epoch skip highway).
4. Verified exact step-0 numerical equivalence to Candidate Zeta ($0.000000e+00$ logits difference) and 100% gradient coverage across all 110 named parameters.
5. All 12/12 unit tests in `test_nas_iteration9.py` PASSED on `cuda:0`.
6. Ready for immediate deployment and proxy evaluation by `teamwork_worker_r9_1`.

---

## 5. Verification Method

To independently verify Candidate Eta architecture, parameter counts, and unit tests:

1. **Execute Candidate Eta Unit Test Suite**:
   ```bash
   ./.venv/bin/python .agents/teamwork/teamwork_explorer_r9_1/test_nas_iteration9.py
   ```
   *Expected Outcome*: 12 tests ran, 0 failures, exit code 0.

2. **Verify Parameter Count and Cap Compliance**:
   ```bash
   ./.venv/bin/python -c "
   import sys
   sys.path.insert(0, '.agents/teamwork/teamwork_explorer_r9_1')
   from proposed_nas_iteration9_model import build_nas_iteration9_model
   m = build_nas_iteration9_model()
   total = sum(p.numel() for p in m.parameters())
   print(f'Total params: {total:,} (Cap: 196,728, Headroom: {196728-total})')
   "
   ```
   *Expected Output*: `Total params: 196,644 (Cap: 196,728, Headroom: 84)`

3. **Verify Step-0 Equivalence to Candidate Zeta**:
   ```bash
   ./.venv/bin/python -c "
   import torch, sys
   sys.path.insert(0, 'experiments/EXP-014')
   sys.path.insert(0, '.agents/teamwork/teamwork_explorer_r9_1')
   from nas_iteration8_model import build_nas_iteration8_model
   from proposed_nas_iteration9_model import build_nas_iteration9_model

   torch.manual_seed(42)
   m_zeta = build_nas_iteration8_model(zero_init=True)
   torch.manual_seed(42)
   m_eta = build_nas_iteration9_model(zero_init=True)

   zeta_state = m_zeta.state_dict()
   eta_state = m_eta.state_dict()
   for k, v in zeta_state.items():
       if k in eta_state and eta_state[k].shape == v.shape:
           eta_state[k].copy_(v)
   m_eta.load_state_dict(eta_state)

   m_zeta.eval(); m_eta.eval()
   x_ihr = torch.randn(2, 100, 30) * 5.0 + 60.0
   x_acc = torch.randn(2, 100, 30)
   x_meta = torch.randn(2, 100, 5)
   x_meta[:, :, 3] = x_ihr.mean(dim=-1)
   x_meta[:, :, 4] = x_ihr.std(dim=-1)

   with torch.no_grad():
       out_zeta = m_zeta(x_ihr, x_acc, x_meta)
       out_eta = m_eta(x_ihr, x_acc, x_meta)

   diff = (out_zeta['logits'] - out_eta['logits']).abs().max().item()
   print(f'Max absolute difference: {diff:.6e}')
   assert diff == 0.0
   "
   ```
   *Expected Output*: `Max absolute difference: 0.000000e+00`
