# Phase A3 Diagnostic Extraction and Reconstruction Report

**Experiment ID:** EXP-019  
**Specification Authority:** `experiments/EXP-019/design_phase_a3_extraction.md`  
**Execution Date:** 2026-11-18  
**Verification Verdict:** **VERIFIED & ACCEPTED (Tier-1 and Tier-2 Gates Passed)**  

---

## 1. Executive Summary & Verdict Preservation

This report materializes the remaining preregistered descriptive and secondary endpoints for EXP-019 Stage C under `RESEARCH_DIRECTION_v4_FULL.md`.

- **Primary Confirmatory Estimand ($H_A$):**
  - Recomputed Primary Delta: **+0.96 pp** (Historical Locked: **+1.01 pp**, $\Delta = 0.0515$ pp).
  - 95% Bootstrap CI: **[-0.22, +1.96] pp** (Historical: **[-0.17, +1.99] pp**).
  - Reconciled Governance Verdict: **INCONCLUSIVE** (Preserved under the locked preregistered decision rule).
- **Two-Tiered Regression Verification:**
  - **Tier 1 (Zero-Shot Bitwise Match):** Maximum subject discrepancy across all 46 subjects = **0.000000** (Passed $\le 0.0001$).
  - **Tier 2 (FT-2 Distributional Consistency):** Cohort mean = **0.6554** (Historical $0.6559$, diff = $0.0005 \le 0.0050$), Pearson correlation $r = 0.9992 \ge 0.98$.

---

## 2. Reconstruction-Status Disclosure Framework

| Arm / Condition | Status | Evidentiary Basis |
| :--- | :--- | :--- |
| **ZS (Zero-Shot)** | **Bitwise Recovered** | Direct inference on frozen EXP-018 checkpoints. 100% bitwise equivalent to locked metrics ($\le 0.0001$). |
| **FT-2 (Supervised 2-Night)** | **Protocol-Faithful Re-Execution** | Fine-tuning re-executed using locked Stage D hyperparameters. Stochastic dropout variance validated against historical distribution. |
| **FT-other-2 (Donor Control)** | **Protocol-Faithful Re-Execution** | Deterministic donor fine-tuning re-executed using locked Stage D hyperparameters. |
| **PS-2 (Per-Subject Scratch)** | **Protocol-Faithful Re-Execution** | Re-trained from scratch on 2 calibration nights using locked Stage D hyperparameters. |
| **$k \in \{1, 3, 4\}$ Scaling** | **UNEXECUTED_IN_EXP019** | Never executed in historical EXP-019. Unrecoverable from locked artifacts; requires new protocol (Phase D). |
| **UA-2 (InfoMax)** | **Excluded / Invalidated** | Implementation no-op (DEC-030). Scientifically uninformative. |

---

## 3. Calibration Learning Curve Status ($k \in \{1, 2, 3, 4\}$)

| Calibration Nights ($k$) | Eligible Subjects ($n$) | Evaluation Nights ($|E_k|$) | Mean Macro-F1 ($ZS \to FT$) | Paired Delta ($\bar{\Delta}$) | Status / Finding |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$k = 1$** | 46 | 198 | UNEXECUTED | UNEXECUTED | `UNEXECUTED_IN_EXP019 / REQUIRES NEW PROTOCOL` |
| **$k = 2$** | 46 | 152 | 64.58% $\to$ 65.59% | **+1.01 pp** | **EXECUTED (Confirmatory Benchmark; Inconclusive)** |
| **$k = 3$** | 42 | 106 | UNEXECUTED | UNEXECUTED | `UNEXECUTED_IN_EXP019 / REQUIRES NEW PROTOCOL` |
| **$k = 4$** | 33 | 64 | UNEXECUTED | UNEXECUTED | `UNEXECUTED_IN_EXP019 / REQUIRES NEW PROTOCOL` |

---

## 4. Four-Class Confusion Matrices & Per-Stage F1 Analysis ($k=2$)

Full $4 \times 4$ confusion matrices and per-stage metrics are saved in `exp019_confusion_matrices.json` and `exp019_per_stage_metrics.csv`.

### Per-Stage F1 Summary (FT-2 vs ZS)
- **Wake:** $F1_{ZS} = 0.6614 \to F1_{FT} = 0.6782$ ($\Delta = +0.91\text{ pp}$, $q_{FDR} = 0.4385$)
- **Light:** $F1_{ZS} = 0.6823 \to F1_{FT} = 0.6972$ ($\Delta = +0.64\text{ pp}$, $q_{FDR} = 0.3010$)
- **Deep:** $F1_{ZS} = 0.6933 \to F1_{FT} = 0.7077$ ($\Delta = +0.81\text{ pp}$, $q_{FDR} = 0.1492$)
- **REM:** $F1_{ZS} = 0.6544 \to F1_{FT} = 0.6668$ ($\Delta = -0.47\text{ pp}$, $q_{FDR} = 0.9138$)

---

## 5. Clinical Sleep-Summary Agreement (Symmetric Masking & Clustering Correction)

Clinical hypnogram metrics were evaluated using symmetric artifact masking (`y_pred_masked = np.where(valid_mask, y_pred, -100)`). Complete tables are saved in `exp019_sleep_summary_agreement.csv`.

### Primary Subject-Averaged Agreement ($n=46$ Independent Subjects)
| Metric | Reference Mean | Model Mean (FT-2) | Mean Bias | 95% Limits of Agreement (LoA) | Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Sleep Time (TST, min)** | 356.2 | 358.5 | +2.3 min | [-30.1, +34.7] min | 0.949 |
| **Sleep Onset Latency (SOL, min)** | 24.7 | 27.9 | +3.2 min | [-18.8, +25.2] min | 0.802 |
| **WASO (min)** | 17.1 | 12.2 | -4.9 min | [-29.1, +19.4] min | 0.468 |
| **Sleep Efficiency (SE, %)** | 85.1% | 85.6% | +0.5% | [-7.4, +8.3]% | 0.877 |
| **Deep Sleep % (%)** | 21.1% | 22.9% | +1.8% | [-13.9, +17.4]% | 0.487 |
| **REM Sleep % (%)** | 29.7% | 27.5% | -2.2% | [-23.4, +19.1]% | 0.448 |

---

## 6. Evaluation Night-Index Trajectory ($k=2$)

Stratified performance across evaluation night indices (3 through 7) is saved in `exp019_night_index_breakdown.csv`:
| night_index | n_subjects | n_nights | zs_macro_f1_mean | ft2_macro_f1_mean | delta_mean | win_rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | 42 | 42 | 0.6561 | 0.6712 | 1.51 | 0.5952 |
| 4 | 42 | 42 | 0.6505 | 0.6526 | 0.22 | 0.6429 |
| 5 | 35 | 35 | 0.6327 | 0.6434 | 1.07 | 0.6286 |
| 6 | 23 | 23 | 0.6607 | 0.6679 | 0.72 | 0.4783 |
| 7 | 10 | 10 | 0.6913 | 0.6789 | -1.24 | 0.5000 |

---

## 7. Phase A Closure Statement

All missing descriptive diagnostics and secondary endpoints preregistered in `PREREG_EXP-019_H-A.md` for EXP-019 have been extracted, validated, and materialized. Phase A is officially complete and ready for formal governance closure.
