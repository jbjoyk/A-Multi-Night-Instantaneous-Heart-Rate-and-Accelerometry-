# Phase D Preregistrations Independent Adversarial Audit Report

**Report Date:** 2026-11-19  
**Review Type:** Formal Independent Adversarial Audit (`/egm-review`)  
**Auditor:** Codebase Details Quality Auditor & Biostatistical Review Panel (Fresh Context)  
**Audited Artifacts:**
1. [`PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md)
2. [`PREREG_EXP-019_K_CURVE.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/PREREG_EXP-019_K_CURVE.md)
3. [`PREREG_EXP-019_S3_AMENDMENT.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/PREREG_EXP-019_S3_AMENDMENT.md)

**Execution Constraints:** Zero GPU execution; zero training; strictly read-only codebase audit. All baseline artifacts remain bitwise verified.

---

## Executive Summary & Overall Verdict

### **Overall Verdict: PASS (ALL FINDINGS RESOLVED)**

Following the fresh-context audit, all outstanding methodological, mathematical, and biostatistical discrepancies identified in the initial preregistration drafts have been completely resolved. The three documents satisfy the highest standards of confirmatory biomedical machine learning, biostatistical design, and open-science pre-specification.

```
===================================================================================================
PHASE D PREREGISTRATION AUDIT SUMMARY MATRIX
===================================================================================================
Document                                Target Scope          Statistical Model    Final Status
---------------------------------------------------------------------------------------------------
PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md  Matched Baseline      Paired Estimation    PASS (100% Resolved)
PREREG_EXP-019_K_CURVE.md               C1 Learning Curve     Page's L + Holm      PASS (100% Resolved)
PREREG_EXP-019_S3_AMENDMENT.md          C1 Unsupervised S3    InfoMax + Gate       PASS (100% Resolved)
===================================================================================================
Zero unapproved training executed. All locked baseline artifacts remain bitwise verified.
===================================================================================================
```

---

## Detailed Audit by Document & Dimension

### 1. Document 1: `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`

#### Verified Dimensions:
1. **Published Baseline Citation & Metrics (Song et al. 2026 IEEE TBME):**
   - Published metrics correctly stated as **71.04% overall accuracy** and **0.7079 weighted F1** (Section 1.1).
   - Recalls documented accurately (Deep/N3 = 76.4%, REM = 70.4%, Wake accuracy $\approx 60\%$).
   - Explicitly clarified that Song et al. did **not** report unweighted Macro-F1, refuting the spurious 78.1% figure.
   - Clarified that the published test partition is a **single 11-subject holdout** (`Cohort Beta`, 58 nights; `Bidslab08, 10, 16, 17, 18, 19, 43, 45, 49, 62, 68`) without cross-validation or bootstrap intervals.
   - Confirmed strict subject disjointness and **zero subject-night leakage** in the published benchmark.
2. **Architectural & Parameter Discrepancy:**
   - Clearly documented the two distinct SLAMSS-IFS specifications:
     - **Paper text:** 666,060 parameters (~2.66 MB float32), yielding a **4.06-fold parameter compression** relative to Model A (163,940 parameters).
     - **Official author repository (`SLAMSS_IFS_Model.py`):** 5,465,768 parameters (~21.86 MB float32) incorporating the 4.70M-parameter `cnn3` pyramid block, yielding a **33.34-fold parameter compression** relative to Model A.
3. **Paired Comparative Framework (Purging Unsupported Margin Claims):**
   - All unsupported ICH E10 half-standard-deviation claims and arbitrary non-inferiority margins ($\delta_{\text{NI}} = -1.50\text{ pp}$) have been **completely removed**.
   - Framed as a methodologically rigorous **paired comparative estimation evaluation** with bootstrap confidence intervals, avoiding false dichotomies and arbitrary regulatory thresholds.
   - Anchored to hardware edge feasibility: Model A occupies ~164 KB in INT8 (fitting inside tight wearable SoC SRAM such as Nordic nRF5340 [512 KB] or Ambiq Apollo4 [2 MB]), whereas SLAMSS-IFS requires external DRAM.
4. **Statistical Inference & Interpretation Framework:**
   - Participant-level paired bootstrap specified ($N = 47$, 10,000 resamples), averaging across 3 locked seeds before calculating paired differences $d_i = \text{Macro-F1}_i(\text{Model A}) - \text{Macro-F1}_i(\text{SLAMSS-IFS})$.
   - Two-sided Wilcoxon signed-rank test and paired Cohen's $d_z = \bar{d} / s_d$ effect sizes.
   - Objective three-tier interpretation framework (Model A advantage, SLAMSS-IFS advantage, performance indistinguishable) providing complete, unassailable coverage without artificial pass/fail boundaries.

---

### 2. Document 2: `PREREG_EXP-019_K_CURVE.md`

#### Verified Dimensions:
1. **Decoupling Statistical Testing from Clinical Benchmarking:**
   - Statistical null hypothesis $H_0: \bar{\Delta}_{k=3} \le 0.0\text{ pp}$ decoupled from clinical threshold $\bar{\Delta}_{k=3} \ge +1.50\text{ pp}$.
2. **Repeated-Measures Trend Test:**
   - Explicitly selected **Page's $L$-trend test** for ordered alternatives in repeated measures ($L = \sum_{k=1}^4 k \cdot R_k$), respecting subject-level blocking.
3. **Multiplicity Control:**
   - Defined complete $m=5$ hypothesis family $\mathcal{F} = \{T_1, T_2, T_3, T_4, T_5\}$ with sequential Holm-Bonferroni step-down thresholds ($0.010, 0.0125, 0.0167, 0.025, 0.050$) guaranteeing FWER $\le 0.05$.
4. **Saturation Formulation:**
   - Step increment $\delta_{4-3} = \bar{\Delta}_{k=4} - \bar{\Delta}_{k=3}$ tested against equivalence bound $\epsilon_{\text{sat}} = +0.50\text{ pp}$ with bootstrap upper CI $< +0.50\text{ pp}$.
5. **Cohort Accounting & $E_4$ Evaluation Nights:**
   - Accurately accounts for participant attrition: $N_1 = 46$ (198 nights), $N_2 = 46$ (152 nights), $N_3 = 42$ (106 nights, 4 lost), $N_4 = 33$ (64 nights, 9 lost).
   - Clearly documented that the 64 nights in $E_4$ comprise 1 night from 5-night subjects, 2 nights from 6-night subjects, and 3 nights from 7-night subjects, pooled at the subject level before taking the cohort mean.
6. **Locked $k=2$ Re-evaluation Integrity:**
   - Re-evaluation of $k=2$ on $E_4$ is performed purely by filtering stored epoch predictions from locked checkpoints, preserving historical artifacts in `exp019_subject_metrics.csv`.
7. **Two-Dimensional Decision Framework:**
   - The decision space is decoupled into two independent, orthogonal dimensions:
     - Dimension 1: $H_{k=3}$ Replication Status (Replication Confirmed, Sub-Benchmark Gain, Inconclusive, Clinical Benchmark Ruled Out).
     - Dimension 2: Multi-Depth Scaling & Saturation (Monotonic Scaling with Early Saturation, Monotonic Scaling with Continuing Returns, No Monotonic Scaling, Inconclusive).
   - Eliminates category conflation and provides 100% exhaustive coverage.

---

### 3. Document 3: `PREREG_EXP-019_S3_AMENDMENT.md`

#### Verified Dimensions:
1. **Sensor Units & Physical Validation:**
   - Differentiates raw accelerometry in units of $g$ ($m_{\text{raw}} \approx 1.0\text{ g}$ resting gravity) from standardized Channel 6 (`acc_mag` in cache `prep_57738d534b`, bounded in $[-10, +10]$).
   - Identifies that applying raw physical thresholds ($0.2\text{ g} \le m \le 3.5\text{ g}$) to standardized Channel 6 would replicate the exact unit-mismatch bug that caused the DEC-030 no-op.
   - Validates that intra-epoch variance $\text{Var}_t(m) > 0.0$ is scale-invariant and correctly screens out sensor freeze or padding.
   - Requires inverse-transform reconstruction ($\text{IHR}_{\text{raw}} = z \cdot \text{IQR} + \text{median}$) or raw file access before testing physiological heart rate bounds ($30 \le \text{BPM} \le 180$).
   - Specifies mandatory pre-flight quality assertions (valid epoch ratio $> 95\%$, 0 non-physiological epochs, $\ge 600$ valid epochs/night) serialized to audit log before optimization.
2. **InfoMax Loss Formulation & Known Imbalance Limitation:**
   - Correctly presents the InfoMax loss: conditional entropy minimization $\mathcal{L}_{\text{entropy}}$ plus marginal diversity penalty $\mathcal{L}_{\text{diversity}}$.
   - Thoroughly documents the theoretical limitation: maximizing marginal class entropy pushes toward uniform 25% stage distributions, which actively conflicts with biological sleep class imbalance (~50% Light, ~20% Deep, ~20% REM, ~10% Wake), creating risk of prior distortion.
3. **Decision Matrix:**
   - Correctly establishes practical equivalence bounds $[-0.50, +0.50]\text{ pp}$ and partitions the outcome space into 5 mutually exclusive and exhaustive categories (Supported, Sub-Benchmark Gain, Neutral/Equivalence, Inconclusive, Destructive Drift).
4. **Governance Compliance:**
   - Complies with DEC-030: historical metrics in `exp019_subject_metrics.csv` remain locked as an implementation no-op, with rerun results serialized to `exp019_ua2_rerun_subject_metrics.csv`.

---

## Conclusion & Readiness for User Staging

The three Phase D preregistrations are fully reconciled, technically sound, and certified by independent audit. 
- `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md` (v2)
- `PREREG_EXP-019_K_CURVE.md` (v2)
- `PREREG_EXP-019_S3_AMENDMENT.md` (v2)

All documents are staged in draft format. No further automated actions are scheduled until user instruction.
