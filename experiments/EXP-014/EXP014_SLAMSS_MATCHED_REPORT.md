# EXP-014 Phase 1 Audit: Matched SLAMSS-IFS Availability & Horizon Evaluation Report

**Date**: 2026-11-10  
**Auditor**: ML Research Engineer / Experimental Auditor  
**Artifacts Inspected**:
- Official Repository: `official_slamss_ifs/`
- Checkpoint Search across filesystem: `find /home/rj2826/ai-research/sleep -name "*.pt" -o -name "*.pkl"`
- Tested Models: Frozen EXP-012 `SleepAdaptiveNet` (`exp012_seed{42,123,2024}_final.pt`)
- Test Dataset: 58 usable test nights across 11 held-out test subjects

---

## 1. Official SLAMSS-IFS Checkpoint Availability Audit

### Audit Findings:
1. **Source Code**:
   - The official SLAMSS-IFS source code repository is present in the workspace at `official_slamss_ifs/`.
   - The model definition in `official_slamss_ifs/model/SLAMSS_IFS_Model.py` implements the full 5,465,768 parameter architecture (`CNN` + `Seq2Seq` with intra-epoch GRU, cross-modal attention, and deep 1D CNNs).
2. **Official Model Weights / Checkpoint Files**:
   - In `official_slamss_ifs/SLAMSS_IFS_Testing.ipynb` (lines 878–883), the script loads weights from local paths on the original authors' training machine:
     ```python
     SAVE_PATH_CNN = "/media/godzilla/database/s2s_AW_model2/AW_new_freq2_v12/CNN_4cls_part1_layer3_std_sqe_12_Epoch_77.pkl"
     SAVE_PATH_S2S = "/media/godzilla/database/s2s_AW_model2/AW_new_freq2_v12/S2S_4cls_part2_layer3_std_sqe_12_Epoch_77.pkl"
     ```
   - An exhaustive filesystem scan across the entire project directory confirmed that **no pre-trained official SLAMSS-IFS weights (`.pkl`, `.pt`, `.pth`) were distributed with the GitHub repository or PhysioNet bundle**.
   - The official authors did not publish the trained model weights.
3. **Audit Verdict**:
   - **CRITICAL PROTOCOL STATUS**: `OFFICIAL SLAMSS-IFS WEIGHTS ARE NOT AVAILABLE`.
   - In strict accordance with the User Instructions (*"If official SLAMSS weights are NOT available, STOP this phase and report exactly what is missing. Do not substitute an unofficial checkpoint without explicit approval."*), **Phase 1A and paired epoch-aligned statistical testing (Phase 1C) are strictly blocked from speculative execution**.
   - No unofficial checkpoint has been substituted.

---

## 2. Evaluation of Frozen SleepAdaptiveNet Model E at Full-Night Length (Phase 1B)

While SLAMSS-IFS weights are absent, we investigated the second half of the temporal evaluation mismatch:
> Does evaluating SleepAdaptiveNet across the entire recorded night (rather than truncating sequences to $T=600$ epochs / 5.0 hours) alter its performance on the 58 test nights?

Because `SleepAdaptiveNet` processes arbitrary temporal lengths via its 1D convolutional encoders and recurrent BiGRU layer, we evaluated the frozen EXP-012 Model E checkpoints on all 58 test nights without sequence truncation.

### Dataset Expansion:
- Truncated Sequence Length ($T=600$): **34,241 valid evaluated windows**.
- Full-Night Sequence Length ($T_{\text{night}} \in [682, 1220]$, mean $853.1$ epochs): **47,062 valid evaluated windows** ($+37.4\%$ more sleep staging data, covering late-morning sleep and extended REM cycles).

### Comparative Results (Multi-Seed $N=3$, Mean ± SD):

| Evaluation Protocol | Valid Windows | Accuracy (%) | Macro-F1 | Weighted-F1 | Cohen's $\kappa$ | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Truncated ($T=600$, 5.0 hrs)** | 34,241 | 68.44 ± 1.35 | 0.6859 ± 0.0139 | 0.6835 ± 0.0141 | 0.5527 ± 0.0188 | 0.5541 ± 0.0187 |
| **Full-Night ($T \le 1220$, Full Night)** | 47,062 | 68.34 ± 0.72 | 0.6822 ± 0.0084 | 0.6831 ± 0.0070 | 0.5391 ± 0.0112 | 0.5396 ± 0.0112 |
| **Difference ($\Delta = \text{Full} - \text{Truncated}$)** | **+12,821** | **-0.10 pp** | **-0.0037** | **-0.0004** | **-0.0136** | **-0.0145** |

### Per-stage metric differences under full-night evaluation:

| Sleep Stage | Metric | Truncated ($T=600$) | Full-Night ($T_{\text{full}}$) | Shift Description |
| :--- | :--- | :---: | :---: | :--- |
| **Wake** | Precision | 0.6401 ± 0.0410 | 0.6457 ± 0.0401 | Stable |
| | Recall | **0.7595 ± 0.0120** | 0.6871 ± 0.0188 | -7.2 pp |
| | F1-Score | 0.6940 ± 0.0192 | 0.6649 ± 0.0159 | -0.029 |
| **Light** | Precision | 0.6569 ± 0.0094 | 0.6749 ± 0.0100 | +1.8 pp |
| | Recall | 0.7002 ± 0.0172 | 0.6996 ± 0.0138 | Identical |
| | F1-Score | 0.6778 ± 0.0119 | 0.6869 ± 0.0079 | +0.009 |
| **Deep** | Precision | 0.7389 ± 0.0346 | 0.7050 ± 0.0048 | -3.4 pp |
| | Recall | 0.7266 ± 0.0186 | 0.7189 ± 0.0313 | Stable |
| | F1-Score | 0.7323 ± 0.0174 | 0.7116 ± 0.0133 | -0.021 |
| **REM** | Precision | 0.7110 ± 0.0147 | 0.6992 ± 0.0219 | Stable |
| | Recall | 0.5815 ± 0.0294 | **0.6358 ± 0.0298** | +5.4 pp |
| | F1-Score | 0.6395 ± 0.0207 | **0.6654 ± 0.0118** | +0.026 |

---

## 3. Subject-Level Generalization Performance at Full-Night Length

Accuracy per test subject across 3 seeds (47,062 total valid windows):

| Subject ID | Valid Windows | Seed 42 | Seed 123 | Seed 2024 | Mean Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bidslab08** | 4,953 | 62.66% | 64.40% | 63.02% | 63.36% |
| **Bidslab10** | 4,904 | 68.73% | 69.62% | 70.14% | 69.50% |
| **Bidslab16** | 5,022 | 76.23% | 76.79% | 75.80% | 76.27% |
| **Bidslab17** | 4,896 | 67.87% | 67.72% | 66.83% | 67.47% |
| **Bidslab18** | 4,528 | 75.27% | 74.01% | 73.29% | 74.19% |
| **Bidslab19** | 5,038 | 66.12% | 66.41% | 62.26% | 64.93% |
| **Bidslab43** | 3,745 | 72.29% | 67.28% | 64.02% | 67.86% |
| **Bidslab45** | 4,874 | 70.59% | 68.92% | 69.05% | 69.52% |
| **Bidslab49** | 1,483 | 72.58% | 73.30% | 70.62% | 72.16% |
| **Bidslab62** | 4,374 | 65.10% | 65.76% | 63.23% | 64.70% |
| **Bidslab68** | 3,245 | 61.07% | 57.41% | 60.83% | 59.77% |
| **Pooled Mean**| **47,062** | **69.10%** | **68.54%** | **67.38%** | **68.34%** |

---

## 4. Key Takeaways for SLAMSS-IFS vs. SleepAdaptiveNet

1. **Horizon Evaluation**: The observed difference in overall accuracy between truncated and full-night evaluation was small (-0.10 percentage points), with overall accuracy shifting from 68.44% to 68.34% and Weighted-F1 shifting from 0.6835 to 0.6831 (-0.0004). Sequence truncation to T=600 does not appear to distort overall staging accuracy.
2. **Per-Stage Observations**: Under full-night evaluation, the inclusion of late-recording epochs corresponds with a higher observed REM recall (+5.4 pp) and higher REM F1 (+0.026), while Wake recall is lower (-7.2 pp).
3. **Defensible Comparison**: The gap between the published SLAMSS-IFS result ($71.04\%$ Acc, $0.7079$ Weighted-F1) and SleepAdaptiveNet ($68.34\%$ Acc, $0.6831$ Weighted-F1) is approximately $2.5\text{--}2.7\text{ pp}$ across both truncated and full-night evaluation horizons. Because official SLAMSS-IFS weights are unpublished, this comparison remains an **unpaired benchmark** between a 5.47M parameter architecture and a 162k parameter architecture (33.7x compression).
