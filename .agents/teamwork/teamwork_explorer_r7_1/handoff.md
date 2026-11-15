# Handoff Report: NAS Iteration 7 Architecture Exploration (Candidate Epsilon)

**Agent**: `teamwork_explorer_r7_1`  
**Milestone**: NAS Iteration 7 — Candidate Epsilon Architecture Specification  
**Recipient**: Parent Orchestrator (`5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91`) and Implementer Worker (`teamwork_worker_r7_1`)  
**Timestamp**: 2026-11-15T07:21:00Z  

---

## 1. Observation

1. **Dead Ends Analysis of Candidate Delta (Iteration 6)**:
   In `experiments/EXP-014/nas_iteration6_proxy_results.json` (lines 14–19, 87–91, 98–115):
   - At best checkpoint (epoch 40), inner-validation reached Accuracy 70.60% (weighted F1 0.7056, macro F1 0.7215), but dropped to 62.11% Accuracy (weighted F1 0.6231) on the locked outer-test partition (10 subjects, 54 nights, 32,098 windows).
   - In the outer-test confusion matrix (lines 98–115):
     - Light Sleep (ground truth = 13,864 windows): 3,740 windows were misclassified as REM (26.98% false positive rate for REM).
     - REM Sleep (ground truth = 7,109 windows): 1,854 windows were misclassified as Light sleep (26.08% false negative rate for REM).
     - Together, Light-REM misclassifications accounted for $3,740 + 1,854 = 5,594$ windows out of 32,098 total windows (**17.43% of all outer-test windows** were lost solely to Light-REM confusion).

2. **Root Cause Diagnosis**:
   In `experiments/EXP-014/nas_iteration6_model.py` (lines 108–135), Candidate Delta applied whole-night global basal z-scoring:
   ```python
   mean_hr = (x_ihr * m_samples).sum(dim=(1, 2), keepdim=True) / valid_sample_counts
   var_hr = (((x_ihr - mean_hr) ** 2) * m_samples).sum(dim=(1, 2), keepdim=True) / valid_sample_counts
   std_hr = torch.sqrt(var_hr + self.eps)
   z_ihr = (x_ihr - mean_hr) / std_hr
   ```
   Because human nocturnal autonomic regulation is inherently non-stationary across the night (early sleep hours 0–3 are dominated by Slow Wave Sleep with low baseline HR ~50–55 bpm; late sleep hours 5–8 are dominated by REM and circadian cortisol rise with baseline HR ~65–70 bpm), global nightly standardization maps early-night REM surges (HR ~65 bpm, $z > +0.8$) and late-night quiet Light sleep (HR ~68 bpm, $z > +0.8$) into the exact same positive relative z-score band. Because accelerometry is near zero in both Light sleep and REM atonia, the classifier cannot disambiguate them.

3. **Comparison with Candidate Gamma (Iteration 5)**:
   In `experiments/EXP-014/nas_iteration5_model.py` (lines 172–257) and `DEAD_ENDS.md` (line 8):
   - Candidate Gamma achieved the highest outer-test accuracy so far (63.16% Acc, 0.6312 WF1) and boosted REM recall to 54.76% via 4-head Inter-Epoch Sequence Attention (`IESA`, 16,416 params) and the direct intra-epoch skip highway (144 dims).
   - However, Gamma passed unstandardized raw HR and used whole-night global `MaskedTemporalInstanceNorm` (TIN), remaining vulnerable to inter-subject baseline shifts and non-stationarity.

4. **Parameter Budget Constraints**:
   Per `ORIGINAL_REQUEST.md` (Requirement R2) and `PROJECT.md`:
   - Target parameter range: 162k–196k parameters (>90% compression vs 5.47M published SLAMSS-IFS benchmark).
   - Parameter budget cap: strictly $\le 196,728$ parameters.

5. **Implementation & Unit Test Verification**:
   Executed `./.venv/bin/python .agents/teamwork/teamwork_explorer_r7_1/test_nas_iteration7.py`:
   ```
   [Test 01 PASSED] Params: 195,708 | Cap: 196,728 | Headroom: 1,020 | Compression: 96.42%
   [Test 02 PASSED] Local Ultradian Basal Standardization verified across 50 bpm vs 85 bpm baselines.
   [Test 03 PASSED] Latent Local Ultradian Adaptive Normalization dimensions verified.
   [Test 04 PASSED] Dynamic Circadian Phase Gating exact step-0 identity confirmed.
   [Test 05 PASSED] Forward pass across all 4 interface formats verified on BIDSleep shapes (B=4, T=600).
   [Test 06 PASSED] Variable sequence length packed execution confirmed.
   [Test 07 PASSED] Step-0 numerical stability verified (Loss: 1.4652).
   [Test 08 PASSED] Backward gradient coverage confirmed across all 94 named parameters.
   [Test 09 PASSED] Direct intra-epoch skip gradient highway to raw input tensors confirmed.
   Ran 9 tests in 0.850s - OK
   ```
   CUDA execution verified on `cuda:0` with forward pass loss 1.5070 and backward grad norm $8.54 \times 10^{-5}$.

---

## 2. Logic Chain

1. **Failure Diagnosis (Observation 1 & 2 $\to$ Step 1)**:
   Global whole-night z-scoring fails because sleep is composed of 90-minute ultradian cycles with slow circadian baseline drift. Normalizing against the 8-hour mean $\mu_{\text{night}}$ shifts early-night Light sleep to negative z-scores and late-night Light sleep to positive z-scores, directly colliding with REM surges and producing 5,594 misclassifications.
2. **Local Ultradian Window Solution (Step 1 $\to$ Step 2)**:
   Instead of a static whole-night scalar, we compute running local mean $\mu_{\text{local}}(t)$ and running local standard deviation $\sigma_{\text{local}}(t)$ over an **ultradian window** of $W = 91$ epochs (45.5 minutes, covering half of an ultradian cycle).
   Depthwise 1D conv with non-trainable box kernel $\mathbf{1}_{1, 1, W}$ and exact boolean padding mask:
   $$\mu_{\text{local}}(t) = \frac{\sum_{\tau=-P}^P x(t+\tau) \cdot m(t+\tau)}{\max(1.0, \sum_{\tau=-P}^P m(t+\tau))}$$
   $$\sigma_{\text{local}}^2(t) = \frac{\sum_{\tau=-P}^P \left(\frac{1}{30} \sum_{s=1}^{30} (x_{\text{ihr}}(t+\tau, s) - \mu_{\text{local}}(t))^2\right) \cdot m(t+\tau)}{\max(1.0, \sum_{\tau=-P}^P m(t+\tau))}$$
   $$\sigma_{\text{local}}(t) = \sqrt{\sigma_{\text{local}}^2(t) + \epsilon}$$
   Under this running local standardizer:
   - Early-night Light sleep: HR matches $\mu_{\text{local}} \approx 54 \implies z_{\text{local}} \approx 0.0$.
   - Late-night Light sleep: HR matches $\mu_{\text{local}} \approx 66 \implies z_{\text{local}} \approx 0.0$.
   - Early-night REM: HR surges above local baseline $\implies z_{\text{local}} > +1.5$.
   - Late-night REM: HR surges above local baseline $\implies z_{\text{local}} > +1.5$.
   Both early and late Light sleep are centered at zero, and REM surges are distinct positive peaks, eliminating the Light-REM collision.
3. **Dynamic Circadian Phase Gating (Observation 2 $\to$ Step 3)**:
   The macro circadian drift $\Delta \mu(t) = \frac{\mu_{\text{local}}(t) - \mu_{\text{night}}}{\sigma_{\text{night}} + \epsilon}$ and local variability ratio $\Delta \sigma(t) = \frac{\sigma_{\text{local}}(t)}{\sigma_{\text{night}} + \epsilon}$ along with $t/T$ define the 3-dimensional Circadian State Vector $c(t)$.
   A lightweight 2-layer MLP (2,544 params) produces adaptive gate $g(t) \in (0, 1)^{144}$. With `zero_init=True`, $\text{fc2} \equiv 0 \implies \sigma(0) = 0.5 \implies 2 \cdot g(t) \equiv 1.0$, guaranteeing exact step-0 identity preservation while allowing the network to adaptively modulate autonomic sensitivity along the circadian trajectory.
4. **Integration with Proven Gamma Components (Observation 3 & 4 $\to$ Step 4)**:
   Retaining Inter-Epoch Sequence Attention (`IESA`, 4 heads, $d_{\text{attn}}=48$, 16,416 params) provides every epoch with direct $O(1)$ receptive field access to the 90-minute sleep cycle. Retaining the direct 144-dim intra-epoch skip highway into the classification head (`Linear(288, 64) -> LN -> GELU -> Dropout -> Linear(64, 4)`, 18,884 params) guarantees a direct 1-step backpropagation gradient path into the encoders, bypassing the recurrent BPTT chain.
5. **Exact Budget Verification (Observation 4 & 5 $\to$ Conclusion)**:
   Total parameter count is analytically and empirically verified at **195,708 parameters**, providing 1,020 parameter headroom below the 196,728 cap and delivering 96.42% parameter compression vs the 5.47M published benchmark.

---

## 3. Caveats

1. **Window Size Hyperparameter**: Default window size is set to $W = 91$ epochs (45.5 minutes), matching the canonical half-ultradian NREM-REM alternation. While $W=61$ (30.5 min) is also supported, $W=91$ was selected to prevent individual 10–15 min REM surges from artificially elevating the local baseline.
2. **Proxy Training Execution**: This exploration report encompasses architecture design, tensor math, parameter auditing, and unit test verification. Actual training on Fold 1 (50 epochs on `cuda:0`) will be conducted by the implementer worker agent.

---

## 4. Conclusion

**Candidate Epsilon (`LocalUltradianAdaptiveNormCNNBiGRU`)** is ready for deployment in NAS Iteration 7:
- **Architecture Name**: `LocalUltradianAdaptiveNormCNNBiGRU` (aliases: `CandidateEpsilon_LUAN`, `CandidateEpsilon_LUAP`).
- **Core Structural Mutations**:
  1. `LocalUltradianBasalStandardization` (LUBS, $W=91$, 8 params) for running local standard deviation and mean tracking.
  2. `LocalUltradianLatentNorm` (Latent LUAN on HR 64 & ACC 64, 512 params) for local ultradian contrast in latent space.
  3. `DynamicCircadianPhaseGating` (DCPG, 2,544 params) with exact step-0 identity preservation ($2 \cdot g \equiv 1.0$).
  4. `InterEpochSequenceAttention` (IESA, 4 heads, $d_{\text{attn}}=48$, 16,416 params) for sequence-level ultradian cycle context.
  5. Direct 144-dimensional intra-epoch skip highway bypassing BiGRU directly into the classifier head.
- **Parameters**: **195,708 parameters** (Cap: 196,728; Headroom: 1,020; Compression: 96.42% vs 5.47M paper benchmark).
- **Artifacts Prepared**:
  - Implementation: `.agents/teamwork/teamwork_explorer_r7_1/proposed_nas_iteration7_model.py`
  - Unit Test Suite: `.agents/teamwork/teamwork_explorer_r7_1/test_nas_iteration7.py` (all 9 unit tests passed).
- **Target File for Implementer Worker**:
  `experiments/EXP-014/nas_iteration7_model.py` and `experiments/EXP-014/evaluate_nas_iteration7_proxy.py`.

---

## 5. Verification Method

To independently verify Candidate Epsilon, execute the following commands using the workspace Python virtual environment:

1. **Run Full Candidate Epsilon Unit Test Suite**:
   ```bash
   ./.venv/bin/python "/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r7_1/test_nas_iteration7.py"
   ```
   *Expected outcome*: 9 tests run, 0 failures, 0 errors, status OK. Output confirms:
   - Total parameters = 195,708 $\le$ 196,728.
   - Dual interface and positional 3-argument `model(x, x_meta, mask)` support.
   - Non-zero gradient flow across 100% of named parameters (94 parameter tensors).

2. **Run GPU Sanity Forward & Backward Pass**:
   ```bash
   ./.venv/bin/python -c "
   import sys
   sys.path.insert(0, '/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r7_1')
   import torch
   from proposed_nas_iteration7_model import build_nas_iteration7_model

   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   model = build_nas_iteration7_model().to(device)
   B, T = 4, 600
   x_ihr = (torch.randn(B, T, 30, device=device) * 5.0 + 60.0).requires_grad_(True)
   x_acc = torch.randn(B, T, 30, device=device).requires_grad_(True)
   x_meta = torch.randn(B, T, 5, device=device).requires_grad_(True)
   targets = torch.randint(0, 4, (B, T), device=device)
   mask = torch.ones(B, T, dtype=torch.bool, device=device)
   out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
   out['total_loss'].backward()
   print('CUDA Forward & Backward verified. Params:', sum(p.numel() for p in model.parameters()))
   "
   ```
   *Expected outcome*: Outputs `CUDA Forward & Backward verified. Params: 195708`.

3. **Invalidation Conditions**:
   - Total parameter count exceeds 196,728.
   - Step-0 loss is NaN, Inf, or negative.
   - Variable-length sequences leak non-zero representations into padded positions.
