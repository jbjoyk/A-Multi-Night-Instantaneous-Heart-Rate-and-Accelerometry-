# Handoff Report: NAS Iteration 12 (Candidate Kappa) Deployment and Proxy Evaluation

**Agent**: `teamwork_worker_r12_1`  
**Role**: Implementer, QA, Specialist  
**To**: Project Orchestrator Gen 2 (`5a45d55d-cdec-4610-8dcf-b95f2285b08e`)  
**Date**: 2026-11-15T16:07:00Z  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r12_1`  

---

## 1. Observation

### 1.1 Architecture & Parameter Verification
- Candidate Kappa (`BoundedMarginSynergyPyramidCNNBiGRU`) was deployed into `experiments/EXP-014/nas_iteration12_model.py`.
- Parameter count was directly verified via PyTorch on Python 3:
  - Linear classifier head (default): **196,381 parameters** (strictly compliant with target budget $\le 196,500$ and hard cap $\le 196,728$, headroom = 119 parameters).
  - Cosine classifier head: **196,378 parameters** (headroom = 122 parameters).
  - Module breakdown:
    * Modality Encoders (`HREncoder`, `ACCEncoder`, `meta_proj`): 59,232 parameters
    * IHR Dual-Scale Local Basal Standardization (`DualScaleLocalBasalStandardization`): 16 parameters
    * Accelerometry Dual-Scale Standardization (`DualScaleAccelerometryDSLBS`): 0 parameters
    * Latent Multi-Scale Ultradian Latent Normalization (`MultiScaleLocalUltradianLatentNorm`): 640 parameters
    * Bounded-Margin Cross-Modal Synergy (`BoundedMarginCrossModalSynergy`): 489 parameters (8 learnable margin parameters $\tau$, Linear(8, 3): 27 params, LayerNorm(3): 6 params, Linear(3, 144): 432 params, Linear(3, 4): 16 params)
    * Dynamic Circadian Phase Gating (`DynamicCircadianPhaseGating`): 2,592 parameters
    * Temporal BiGRU & Norm (`temporal_gru`, `temp_norm`): 97,920 + 192 = 98,112 parameters
    * Inter-Epoch Sequence Attention (`InterEpochSequenceAttention`): 16,416 parameters
    * Multi-Stream Classification Head (`AdaptiveCosineClassifierHead`): 18,884 parameters
- Ran 12-test automated verification suite (`.agents/teamwork/teamwork_explorer_r12_1/verify_candidate_kappa.py`):
  * All 12 tests passed on `cuda:0` (Parameter budget, Step-0 mathematical equivalence to Theta, Zero future leakage / temporal causality, Gain scale-invariance, DC gravity tilt invariance, REM twitch tolerance, Sharp ReLU zero-boundary, Deep sleep quiescence vs arousal discrimination, Gradient flow to all 8 margins, Cosine angular margin head, Mask invariance, and Full CUDA backward pass).

### 1.2 Proxy Training Runtime & Inner-Validation Dynamics
- Executed `experiments/EXP-014/evaluate_nas_iteration12_proxy.py` on `cuda:0` using `./.venv/bin/python`.
- Training schedule: 50 epochs on Fold 1 inner-train partition (32 subjects, 168 nights, 42 batches/epoch, batch size 4, seq_len 600, lr 1e-4, Adam, RCC loss weight 0.01).
- Total training duration: **102.69s (~1.71m)** on NVIDIA GeForce RTX 3060 (`cuda:0`).
- Inner-Validation Progression across 50 epochs (evaluating 5 subjects, 24 nights, 14,089 windows):
  * Epoch 05: Acc 54.56%, Macro-F1 0.5285, Loss 1.0424
  * Epoch 10: Acc 58.91%, Macro-F1 0.5687, Loss 0.9699
  * Epoch 15: Acc 60.81%, Macro-F1 0.6072, Loss 0.9105
  * Epoch 20: Acc 67.03%, Macro-F1 0.6861, Loss 0.8345
  * Epoch 25: Acc 68.42%, Macro-F1 0.6944, Loss 0.8027
  * Epoch 30: Acc 68.42%, Macro-F1 0.6956, Loss 0.8213
  * Epoch 35: Acc 72.96%, Macro-F1 0.7391, Loss 0.7493
  * Epoch 40: Acc 70.67%, Macro-F1 0.7155, Loss 0.7839
  * **Epoch 45 (Best Checkpoint)**: **Acc 73.00%**, **Weighted-F1 0.7300**, **Macro-F1 0.7395**, **Cohen's Kappa 0.6145**, Loss 0.7433
  * Epoch 50 (Final): Acc 72.99%, Weighted-F1 0.7294, Macro-F1 0.7337, Cohen's Kappa 0.6067, Loss 0.7510
- **Peak Inner-Validation Record**: Inner-validation Macro-F1 reached **0.7395** (an all-time high across all 12 NAS iterations, surpassing Candidate Theta's 0.7258 and Candidate Zeta's 0.7188).
- Learned Margins at Peak Epoch (Epoch 45):
  * `tau_ihr_rem`: -0.0839 (adapted slightly negative to capture subtle autonomic surges)
  * `tau_acc_rem`: +0.2815 (expanded from +0.2500 to accommodate physiological micro-twitches)
  * `tau_ihr_deep`: +0.0271 (adapted upward to accommodate marginal slow-wave autonomic shifts)
  * `tau_acc_deep`: +0.0569 (stable near +0.0500 motor stillness)
  * `tau_ihr_wake`: +0.0201
  * `tau_acc_wake`: -0.0036
  * `tau_ihr_light`: +0.1212
  * `tau_acc_light`: -0.0338

### 1.3 Outer-Test Evaluation Metrics (Locked Fold 1 Test Partition: 10 subjects, 54 nights, 32,098 windows)
- **Best Validation Checkpoint (Epoch 45)**:
  * **Accuracy**: **66.84%** (0.66842) [Proxy Gate Threshold: $\ge 70.0\%$ $\rightarrow$ **FAILED**]
  * **Weighted-F1**: **0.6678** (0.66779) [Proxy Gate Threshold: $\ge 0.6900$ $\rightarrow$ **FAILED**]
  * **Macro-F1**: **0.6633**
  * **Cohen's Kappa**: **0.5154**
  * Per-Class Breakdown:
    - Wake:  F1 = 0.6562 | Recall = 63.23% (2,027 / 3,206) | Precision = 68.20%
    - Light: F1 = 0.6740 | Recall = 70.61% (9,789 / 13,864)| Precision = 64.48%
    - Deep:  F1 = 0.7175 | Recall = 70.21% (5,560 / 7,919) | Precision = 73.36%
    - REM:   F1 = 0.6055 | Recall = 57.38% (4,079 / 7,109) | Precision = 64.08%
  * 4x4 Confusion Matrix (Wake, Light, Deep, REM):
    ```
    [[ 2027,   880,   110,   189],
     [  723,  9789,  1472,  1880],
     [   90,  2052,  5560,   217],
     [  132,  2461,   437,  4079]]
    ```
- **Final Epoch 50 Checkpoint (Ablation Comparison)**:
  * Accuracy: **67.04%** (0.67038)
  * Weighted-F1: **0.6686** (0.66862)
  * Macro-F1: **0.6626**
  * Cohen's Kappa: **0.5084**
  * Confusion Matrix:
    ```
    [[ 1835,  1077,   112,   182],
     [  425, 10592,  1256,  1591],
     [   34,  2446,  5215,   224],
     [   37,  2880,   316,  3876]]
    ```

---

## 2. Logic Chain

1. **Deficiency of Candidate Theta & Candidate Iota**:
   - Candidate Theta (Iteration 10) introduced CASCI with strict zero-threshold ReLU gates ($s_{\text{rem}} = \text{ReLU}(u_{\text{ihr}}) \cdot \text{ReLU}(-u_{\text{acc}})$). This successfully reduced Light$\rightarrow$REM false alarms by 33.6% (down to 1,879 windows), but rigid zero clipping destroyed REM synergy during micro-twitches ($u_{\text{acc}} > 0$), causing REM$\rightarrow$Light false negatives to jump to 2,459 windows.
   - Candidate Iota (Iteration 11) attempted to smooth the interaction via strictly positive $\text{softplus}$ activations. Because $\text{softplus}(u) > 0$ unconditionally, non-zero synergy smeared forward in time, causing Light$\rightarrow$REM false alarms to surge by +28.4% (back up to 2,412 windows), pulling outer test accuracy down to 65.12%.
2. **Impact of Bounded-Margin Synergy (Candidate Kappa)**:
   - Candidate Kappa retained sharp ReLU clipping (guaranteeing a strict zero lower-bound) while replacing zero thresholds with learnable asymmetric margins ($\tau_{\text{acc\_rem}}$ initialized to +0.25, $\tau_{\text{ihr\_rem}}$ initialized to 0.0).
   - In training, $\tau_{\text{acc\_rem}}$ learned to expand to $+0.2815$, and $\tau_{\text{ihr\_rem}}$ adapted to $-0.0839$.
   - **Protection of Light Sleep Verified**: Light$\rightarrow$REM false alarms remained strictly bounded at **1,880 windows** (virtually identical to Theta's 1,879 and dramatically superior to Zeta's 2,829 and Iota's 2,412). The sharp zero-boundary successfully prevented the catastrophic leakage suffered by Iota.
   - **REM Recall Behavior**: Candidate Kappa achieved 4,079 true positive REM windows (57.38% recall), slightly ahead of Theta (4,065 windows, 57.17% recall). At Epoch 50, outer test reached **67.04% Accuracy** and **0.6686 Weighted-F1**, establishing the highest final-epoch performance observed in the project to date (surpassing Theta's 67.01% / 0.6684).
3. **Generalization Gap Analysis**:
   - On inner-validation subjects, Candidate Kappa excelled, setting a project record of **73.00% Accuracy**, **0.7300 Weighted-F1**, and **0.7395 Macro-F1** (Kappa 0.6145).
   - However, when evaluated on the locked outer-test partition (10 unseen subjects), performance dropped by ~6.16 percentage points to 66.84% (best checkpoint) / 67.04% (Epoch 50).
   - The primary bottleneck remains the persistent Light-REM / Light-Deep overlap: 2,461 REM windows were classified as Light sleep, and 2,052 Deep windows were classified as Light sleep.
4. **Gate Threshold Assessment**:
   - The fast proxy gating criteria require Accuracy $\ge 70.0\%$ and Weighted-F1 $\ge 0.6900$.
   - Candidate Kappa achieved 66.84% Accuracy (< 70.0%) and 0.6678 Weighted-F1 (< 0.6900).
   - Therefore, Candidate Kappa fails the fast proxy filter and must be reverted per Requirement R1.3 of `ORIGINAL_REQUEST.md`.

---

## 3. Caveats

1. **Partition Specificity**: All proxy evaluations are conducted strictly on Fold 1 (32 train subjects, 5 val subjects, 10 test subjects). The held-out test subjects in Fold 1 exhibit significant physiological variance in resting baseline heart rate and accelerometry somatosensory coupling.
2. **Linear Head vs Cosine Head**: Candidate Kappa was evaluated with its default Linear Classifier Head (196,381 parameters) to ensure direct step-0 mathematical comparability with Candidate Theta. The Cosine Classification Head with Angular Margin was verified in unit tests (Test 10), but was not enabled in proxy training to preserve the uncompromised ablation of the BM-CMS synergy module.
3. **Hyperparameter Invariance**: Consistent with Requirement R2, optimizer (Adam, lr=1e-4), batch size (4), and loss formulation (RCCLoss 0.01) were held strictly constant.

---

## 4. Conclusion & Gate Verdict

- **Gate Verdict**: **`PROXY_FAIL (REVERT)`**
- **Justification**: While Candidate Kappa successfully cured Candidate Iota's leakage flaw (restoring strict suppression of Light$\rightarrow$REM false alarms to 1,880 windows), achieved the highest inner-validation Macro-F1 in project history (0.7395), and edged out Candidate Theta's final epoch accuracy (67.04% vs 67.01%), its outer-test metrics (66.84% Accuracy, 0.6678 Weighted-F1) fall short of the required proxy thresholds (Accuracy $\ge 70.0\%$, Weighted-F1 $\ge 0.6900$).
- **Action**: In accordance with the continuous NAS loop protocol, Candidate Kappa is formally logged as a dead end, the mutation is flagged for reversion, and search moves to Iteration 13.

---

## 5. Verification Method

To independently reproduce and verify this work:
1. **Parameter Count Verification**:
   ```bash
   ./.venv/bin/python -c "
   import sys; sys.path.insert(0, 'experiments/EXP-014')
   import nas_iteration12_model as m
   model = m.build_nas_iteration12_model()
   p = sum(x.numel() for x in model.parameters() if x.requires_grad)
   print('Params:', p)
   assert p == 196381
   "
   ```
2. **12-Test Automated Verification Suite**:
   ```bash
   ./.venv/bin/python .agents/teamwork/teamwork_explorer_r12_1/verify_candidate_kappa.py
   ```
3. **Inspect Output Files**:
   - Results JSON: `experiments/EXP-014/nas_iteration12_proxy_results.json`
   - Checkpoint: `experiments/EXP-014/nas_iteration12_best_proxy_ckpt.pt`
4. **Re-run Proxy Evaluation (Deterministic Seed 42)**:
   ```bash
   ./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration12_proxy.py
   ```
   Confirm that the script outputs `Proxy Accuracy: 66.84%` and `Proxy Weighted-F1: 0.6678` with verdict `PROXY GATE ACTION: REVERT`.
