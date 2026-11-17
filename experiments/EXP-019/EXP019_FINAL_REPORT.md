# EXP-019 Final Report: Multi-Night Within-Person Adaptation (Hypothesis H-A)

**Experiment ID:** EXP-019  
**Execution Date:** 2026-11-16  
**Status:** COMPLETE (Confirmatory Stage C Executed)  
**Authoritative Protocol:** `PREREG_EXP-019_H-A.md` (Locked at commit `4dff5b7a5f13a96c3024b309131aa1e66489dcbb`)  
**Eligible Cohort:** 46 subjects ($n_{\text{usable}} \ge 3$, 152 outer evaluation nights; `Bidslab49` excluded with $n_{\text{usable}}=2$)  
**Unit of Analysis:** Subject (paired comparisons averaged across 3 standardized seeds: 42, 123, 2024)

---

## 1. Executive Summary & Preregistered Decision Rule Verdicts

| Hypothesis | Description | Estimand ($\bar{\Delta}$) | 95% Percentile Bootstrap CI (10,000 resamples) | Wilcoxon Signed-Rank $p$-value | Pre-registered Criteria | Verdict |
|---|---|---|---|---|---|---|
| **H-A (Primary)** | $FT\text{-}2$ vs $ZS$ | **$+1.01\text{ pp}$** | **$[-0.17, +1.99]\text{ pp}$** | **$p = 0.00131$** | $\bar{\Delta} \ge +1.50\text{ pp}$ AND $\text{CI}_{\text{low}} > 0.0$ | **NOT SUPPORTED** |
| **S1 (Secondary)** | $FT\text{-}2$ vs $FT\text{-}other\text{-}2$ | **$+2.73\text{ pp}$** | **$[+1.05, +4.32]\text{ pp}$** | **$p = 9.0 \times 10^{-6}$** | $\text{CI}_{\text{low}} > 0.0$ AND $p < 0.05$ | **SUPPORTED** |
| **S2 (Secondary)** | $FT\text{-}2$ vs $PS\text{-}2$ | **$+20.32\text{ pp}$** | **$[+17.70, +22.94]\text{ pp}$** | **$p < 1.0 \times 10^{-10}$** | $\text{CI}_{\text{low}} > 0.0$ AND $p < 0.05$ | **SUPPORTED** |
| **S3 (Secondary)** | $UA\text{-}2$ vs $ZS$ | **$+0.00\text{ pp}$** | **$[+0.00, +0.00]\text{ pp}$** | N/A ($\Delta = 0$) | $\text{CI}_{\text{low}} > 0.0$ AND $p < 0.05$ | **NOT SUPPORTED** *(Reclassified to UNRESOLVED / INVALIDATED in Erratum Section 6)* |

---

## 2. Overall Performance by Arm

Performance pooled across all 152 evaluation nights and averaged over the 3 random seeds per subject:

| Arm | Description | Macro-F1 (Mean $\pm$ SD) | Cohen's $\kappa$ (Mean) | Accuracy (Mean) | Subject Win Rate vs $ZS$ |
|---|---|---|---|---|---|
| **$ZS$** | Zero-Shot Outer-Fold Model A Backbone | $64.58\% \pm 7.74\%$ | 0.521 | 67.12% | — |
| **$FT\text{-}2$** | Supervised Adaptation on Own $k=2$ Calibration Nights | $\mathbf{65.59\% \pm 8.59\%}$ | **0.534** | **68.04%** | **73.9%** (34/46) |
| **$FT\text{-}other\text{-}2$** | Specificity Control (Donor from Same Fold) | $62.86\% \pm 8.85\%$ | 0.498 | 65.41% | 23.9% (11/46) |
| **$PS\text{-}2$** | Per-Subject Model Trained from Scratch on $k=2$ Nights | $45.27\% \pm 9.42\%$ | 0.284 | 48.91% | 0.0% (0/46) |
| **$UA\text{-}2$** | Unsupervised InfoMax Test-Time Adaptation | $64.58\% \pm 7.74\%$ | 0.521 | 67.12% | 0.0% (0/46, all ties) |

---

## 3. Detailed Scientific Findings & Analysis

### 3.1 Primary Hypothesis H-A Verdict Rationale
- **Directional Trend**: 34 out of 46 subjects (73.9%) improved with supervised adaptation on their first 2 nights. The non-parametric Wilcoxon signed-rank test confirmed statistical significance ($p = 0.00131$).
- **Failure to Meet Preregistered Practical Threshold**: The mean gain across the cohort was $+1.01\text{ pp}$, falling short of the locked $+1.50\text{ pp}$ practical effect threshold.
- **Heterogeneity & Negative Responders**: While the majority of subjects gained $+1.5\text{ to }+6.8\text{ pp}$ (e.g., `Bidslab44` $+6.82\text{ pp}$, `Bidslab42` $+6.31\text{ pp}$, `Bidslab64` $+5.88\text{ pp}$), three severe negative outliers degraded substantially:
  - `Bidslab34`: $-16.34\text{ pp}$ (calibration nights had severe stage distribution distortion relative to subsequent evaluation nights)
  - `Bidslab65`: $-6.57\text{ pp}$
  - `Bidslab10`: $-6.47\text{ pp}$
  Because of this between-subject variance in calibration representativeness, the 10,000-resample bootstrap 95% confidence interval spans $[-0.17, +1.99]\text{ pp}$.
- **Preregistration Rule Enforcement**: Per Section 6 of `PREREG_EXP-019_H-A.md`, confirmatory support requires both $\bar{\Delta} \ge +1.50\text{ pp}$ and $\text{CI}_{\text{low}} > 0.0$. Neither condition was satisfied. Therefore, **H-A is rigorously classified as NOT SUPPORTED**.

### 3.2 Specificity Negative Control S1: Proof of Genuine Personalization
- When fine-tuning on another subject's calibration nights ($FT\text{-}other\text{-}2$), Macro-F1 systematically drops from $64.58\%$ to $62.86\%$ (a $-1.72\text{ pp}$ penalty).
- The paired comparison ($FT\text{-}2$ vs $FT\text{-}other\text{-}2$) yields $+2.73\text{ pp}$ advantage with 95% CI $[+1.05, +4.32]\text{ pp}$ ($p = 9.0 \times 10^{-6}$).
- **Scientific Significance**: This definitively refutes the critique that any additional gradient steps or extra training epochs improve performance. Generic fine-tuning actively harms performance; the benefit observed in $FT\text{-}2$ is strictly person-specific.

### 3.3 Cohort Transfer Value S2: Scratch vs Transfer
- Training a model from scratch exclusively on 2 nights ($PS\text{-}2$) yields an unviable Macro-F1 of $45.27\%$.
- Pre-training on the cohort provides a massive $+20.32\text{ pp}$ advantage ($p < 10^{-10}$), demonstrating that population-level representation learning is essential before personal tuning can be effective.

### 3.4 Unsupervised Test-Time Adaptation S3: InfoMax Evaluation (Historical Interpretation & Invalidation)
- *Historical Report Interpretation:* Unsupervised InfoMax test-time adaptation with frozen CNN features and 20 gradient steps ($UA\text{-}2$) appeared to produce zero change on future nights ($\Delta = 0.00\text{ pp}$), initially interpreted as evidence that unlabeled entropy minimization on 2 nights is insufficient without supervisory signal.
- *Phase A2 Forensic Invalidation (DEC-030):* Subsequent diagnostic audit (`S3_INFOMAX_NOOP_AUDIT_REPORT.md`) revealed this was an implementation no-op caused by a physiological unit mismatch in `compute_wearable_quality_mask` (`30 <= IHR <= 180` on standardized z-score IHR). Zero calibration epochs passed the mask, triggering an immediate step-0 break before any loss, backward pass, or parameter update occurred (0/163,940 parameters updated).
- *Scientific Standing:* S3 is scientifically uninformative. The $0.00\text{ pp}$ result provides no empirical evidence regarding whether unsupervised adaptation succeeds or fails (see Erratum Section 6).

---

## 4. Subject-Level Breakdown (Selected Extremes)

- **Top 5 Beneficiaries of Personal Adaptation:**
  1. `Bidslab44` (Fold 2): $+6.82\text{ pp}$ ($ZS$: 61.04% $\to FT\text{-}2$: 67.86%)
  2. `Bidslab42` (Fold 3): $+6.31\text{ pp}$ ($ZS$: 64.63% $\to FT\text{-}2$: 70.94%)
  3. `Bidslab64` (Fold 1): $+5.88\text{ pp}$ ($ZS$: 64.23% $\to FT\text{-}2$: 70.11%)
  4. `Bidslab02` (Fold 5): $+5.47\text{ pp}$ ($ZS$: 56.22% $\to FT\text{-}2$: 61.69%)
  5. `Bidslab62` (Fold 3): $+4.48\text{ pp}$ ($ZS$: 61.55% $\to FT\text{-}2$: 66.03%)

- **Bottom 3 Degradations (Calibration Distribution Mismatch):**
  1. `Bidslab34` (Fold 2): $-16.34\text{ pp}$ ($ZS$: 60.06% $\to FT\text{-}2$: 43.72%)
  2. `Bidslab65` (Fold 1): $-6.57\text{ pp}$ ($ZS$: 47.28% $\to FT\text{-}2$: 40.71%)
  3. `Bidslab10` (Fold 2): $-6.47\text{ pp}$ ($ZS$: 71.53% $\to FT\text{-}2$: 65.06%)

---

## 5. Artifact Provenance & Traceability

- **Stage C Metrics JSON:** `experiments/EXP-019/exp019_stage_c_results.json`
- **Per-Subject Metrics CSV:** `experiments/EXP-019/exp019_subject_metrics.csv`
- **Arm Summary CSV:** `experiments/EXP-019/exp019_summary.csv`
- **Stage D Selection JSON:** `experiments/EXP-019/exp019_stage_d_selected_configs.json`
- **Pre-Registration Specification:** `PREREG_EXP-019_H-A.md` (Locked at `4dff5b7a5f13a96c3024b309131aa1e66489dcbb`)
- **Regenerated Backbone Checkpoints:** `experiments/EXP-018/checkpoints/modelA_L1200_fold{1..5}_seed{42,123,2024}.pt`
- **Execution Hardware:** NVIDIA GeForce RTX 3060 (12 GB VRAM, Driver 595.99.02, CUDA 13.2)


---

## 6. Governance Reconciliation Erratum (2026-11-17, DEC-029)

### Reconciled Verdict for Primary Hypothesis H-A
- **Original Report Label:** NOT SUPPORTED.
- **Locked Preregistration Protocol:** Commit `4dff5b7` (`PREREG_EXP-019_H-A.md` Section 6).
- **Locked Decision Rule Formulation:**
  - **SUPPORTED:** $\bar{\Delta} \ge +1.50\text{ pp}$ AND bootstrap 95% CI lower bound $> 0$.
  - **NOT SUPPORTED:** bootstrap 95% CI upper bound $< +1.50\text{ pp}$ (ruling out an effect of practical size).
  - **INCONCLUSIVE:** Anything else.
- **Observed Empirical Result:** $\bar{\Delta} = \mathbf{+1.01\text{ pp}}$, 95% bootstrap CI: $\mathbf{[-0.17, +1.99]\text{ pp}}$, Wilcoxon $p = 0.00131$, win rate 73.9% (34/46).
- **Evaluation against Locked Rule:** Because the observed 95% CI upper bound ($+1.99\text{ pp}$) is NOT $< +1.50\text{ pp}$, the preregistered rule evaluates strictly to **`INCONCLUSIVE`**. Ruling out an effect of practical size ($\ge +1.50\text{ pp}$) was not statistically achieved with $N=46$ subjects.
- **Immutability Statement:** The numerical results ($\bar{\Delta} = +1.01\text{ pp}$, CI $[-0.17, +1.99]\text{ pp}$, 34/46 improved) remain byte-for-byte unchanged. This is a formal governance-label reconciliation per `DEC-029` and `RESEARCH_DIRECTION_v4_FULL.md`, not a reanalysis or alteration of EXP-019 data.

### Status of Secondary Hypothesis S3
- **Observed Result:** $\bar{\Delta} = 0.00\text{ pp}$ identically across all 46 subjects.
- **Phase A2 Forensic Invalidation (DEC-030):** Per the completed Phase A2 forensic audit (`experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md`), S3 is formally classified as **`UNRESOLVED / INVALIDATED (Implementation No-Op)`**.
- **Root Cause:** `exp019_adaptation_core.py::compute_wearable_quality_mask` applied a physiological range filter expecting raw BPM values (`30 <= IHR <= 180`) on standardized z-score IHR ($[-3.95, +9.94]$). Exactly 0 out of 82,762 calibration epochs passed the mask, triggering a silent loop abortion on step 0 for all 46 subjects.
- **Forensic Verification:** 0 optimizer steps, 0 backward passes, 0/163,940 parameters changed (displacement $\equiv 0.0$), identical evaluation logits and predictions down to floating-point precision.
- **Scientific Implication:** S3 is scientifically uninformative. The $0.00\text{ pp}$ result must NOT be interpreted as empirical evidence that unsupervised InfoMax adaptation fails (nor that it works). The historical numerical outcome remains locked as the executed experimental record.
