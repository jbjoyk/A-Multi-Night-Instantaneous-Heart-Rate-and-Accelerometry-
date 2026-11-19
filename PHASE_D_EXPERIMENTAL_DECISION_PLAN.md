# Phase D Experimental Decision Plan: BIDSleep Personalization Project
## Decision Framework, Discrepancy Resolution, Task Classification, and Protocol Roadmap (v2, Revised)

**Document Date:** 2026-11-19  
**Revision:** v2 (Incorporating Mandatory Corrections from `PHASE_D_PLAN_INDEPENDENT_REVIEW.md`)  
**Status:** DRAFT FOR USER REVIEW (No experiments authorized; zero code executed)  
**Governing Authority:** [`RESEARCH_DIRECTION_v4_FULL.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_DIRECTION_v4_FULL.md) (DEC-029), Section 9 (Phase D); [`DECISIONS.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/DECISIONS.md) (DEC-001 through DEC-032); [`RESEARCH_GOAL_ATTAINMENT_AUDIT.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_GOAL_ATTAINMENT_AUDIT.md)  
**Permanent Invariants:** All locked EXP-019 confirmatory results, checkpoints, split manifests, and preregistration records remain strictly immutable.

---

## 1. Resolution of the $k=1, 2, 3, 4$ Learning Curve Discrepancy

### 1.1 The Apparent Contradiction
Earlier project discussions and draft overviews occasionally referred to the "$k=1, 2, 3, 4$ learning curve" as an intended output of Experiment EXP-019, creating the impression that personal models had already been evaluated across 1, 2, 3, and 4 calibration nights. Conversely, recent adversarial audits (`RESEARCH_GOAL_ATTAINMENT_AUDIT.md`, `PHASE_A_INDEPENDENT_AUDIT_REPORT.md`) asserted that calibration depths $k \in \{1, 3, 4\}$ were **unexecuted**.

### 1.2 Forensic Origin and Definitive Resolution
A forensic inspection of the repository history, preregistration texts, and governance audit reports (`PREREG_EXP-019_H-A.md`, `experiments/EXP-019/A3_EXTRACTION_GOVERNANCE_REVIEW_REPORT.md`, `experiments/EXP-019/design_phase_a3_extraction.md`, and `experiments/EXP-019/exp019_k_learning_curve.csv`) reveals the exact historical reality:

1. **Preregistration Intent vs. Historical Execution:**
   - In `PREREG_EXP-019_H-A.md` (locked at commit `4dff5b7`) and `RESEARCH_DIRECTION_v4_FULL.md` (line 78), Question K listed the learning curve over $k = 1, 2, 3, 4$ nights as a pre-registered *descriptive objective*.
   - However, during the original historical execution of EXP-019, **only the $k=2$ calibration condition was ever trained, tuned, and evaluated**. 
   - Hyperparameter selection (Stage D) was conducted exclusively on inner-validation subjects for 2-night calibration ($k=2$). No model checkpoints, optimizer states, or predictions were ever computed or serialized for $k=1$, $k=3$, or $k=4$.
2. **The Phase A3 "Extraction" Attempt:**
   - During Phase A3 diagnostic recovery, an initial design draft (`design_phase_a3_extraction.md`) proposed generating results for $k \in \{1, 3, 4\}$ by launching 414 new fine-tuning runs using the locked $k=2$ hyperparameters, framing this computation as a retrospective "extraction."
   - The independent governance review (`A3_EXTRACTION_GOVERNANCE_REVIEW_REPORT.md`) intervened and rejected this plan, ruling that running gradient descent on unexecuted conditions constituted *un-preregistered new training* disguised as artifact extraction.
   - The governance review mandated that:
     * Numerical performance metrics must be reported exclusively for the historically executed $k=2$ condition ($N=46$ subjects, $152$ nights, $\bar{\Delta} = +1.01\text{ pp}$, $95\%\text{ CI: } [-0.17, +1.99]\text{ pp}$).
     * For $k \in \{1, 3, 4\}$, the artifact must record the theoretical subject census ($n=46, 42, 33$) and evaluation night totals ($|E_k| = 198, 106, 64$), but populate all performance metrics explicitly as **`UNEXECUTED_IN_EXP019`**.
3. **Definitive Reconciliation:**
   - **What exists in the repository:** The cohort manifests, night inventories, and evaluable subject counts for $k \in \{1, 2, 3, 4\}$ are fully established.
   - **What has executed experimentally:** **Only $k=2$ has executed.** 
   - **Conclusion:** There is no numerical data for $k \in \{1, 3, 4\}$. Completing the learning curve requires executing 121 new fine-tuning runs under a formally documented Phase D protocol.

---

## 2. Master Task Classification

In accordance with strict research governance and the audit corrections from `PHASE_D_PLAN_INDEPENDENT_REVIEW.md`, all proposed Phase D experimental tasks are classified into four mutually exclusive operational categories:

```
Category 1: Already Executed — No Rerun Needed (Preserved as locked baselines)
Category 2: Missing Implementation Fix under Prior Mandate (Requires protocol amendment)
Category 3: Requires Dedicated Preregistration Addendum or Formal Approval (New training/benchmarks)
Category 4: Deferred or Outside Approved Research Direction (Formally out of scope)
```

| Task ID | Proposed Task Description | Target Domain | Governance Classification | Justification & Standing |
|---|---|---|:---:|---|
| **TSK-01** | Zero-shot population baseline evaluation ($ZS$) | Baseline Staging | **Category 1: Already Executed** | 5-fold CV out-of-fold predictions locked across all 47 subjects in EXP-014/018/019. Zero rerun needed. |
| **TSK-02** | 2-Night Supervised Personal Fine-Tuning ($FT\text{-}2$) | C1 (Primary) | **Category 1: Already Executed** | Evaluated on $N=46$ test subjects (152 nights) in EXP-019. Result: $+1.01\text{ pp}$, $95\%\text{ CI: } [-0.17, +1.99]\text{ pp}$. Bitwise locked. |
| **TSK-03** | Donor-night specificity control ($FT\text{-}other\text{-}2$) | C1 Specificity | **Category 1: Already Executed** | Evaluated on $N=46$ test subjects in EXP-019 S1. Result: $+2.73\text{ pp}$ advantage over donor tuning ($p = 0.00249$). Bitwise locked. |
| **TSK-04** | From-scratch personal baseline ($PS\text{-}2$) | C1 Baseline | **Category 1: Already Executed** | Evaluated on $N=46$ test subjects in EXP-019 S2. Result: $+20.32\text{ pp}$ advantage for pre-training. Bitwise locked. |
| **TSK-05** | Subject-averaged sleep-summary agreement | C3 Agreement | **Category 1: Already Executed** | Bland–Altman agreement independently audited and verified in Phase B and Phase 3 Table 1. Bitwise locked. |
| **TSK-06** | Execution of $k=1, 3, 4$ calibration-depth adaptation | C1 Learning Curve | **Category 3: Requires Dedicated Preregistration** | **Corrected from Category 2.** Requires 121 new model fine-tuning runs. Preregistration `PREREG_EXP-019_H-A.md` omitted hyperparameter rules for $k \ne 2$. Requires formal addendum `PREREG_EXP-019_K_CURVE.md`. |
| **TSK-07** | Re-run of Unsupervised Adaptation ($UA\text{-}2$) with fixed mask | C1 Unsupervised | **Category 2: Implementation Amendment** | Mandated by `RESEARCH_DIRECTION_v4_FULL.md` line 131 and DEC-030 ("re-run S3 only after implementation fault"). Requires raw-unit quality threshold validation. |
| **TSK-08** | Matched 5-Fold Cross-Validation Re-Run of SLAMSS-IFS | Baseline Benchmark | **Category 3: Requires Dedicated Preregistration** | Mandated by `RESEARCH_DIRECTION_v4_FULL.md` Section 8 and Section 11. Requires formal non-inferiority preregistration `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`. |
| **TSK-09** | EXP-019B: Subject-level heterogeneity hypothesis testing | C2 Diagnosis | **Category 3: Requires Formal Approval** | Statistical testing of candidate predictors of negative transfer. Protocol drafted (`EXP-019B_HETEROGENEITY_ANALYSIS_PLAN.md`), but unapproved under DEC-029. |
| **TSK-10** | EXP-020: Regularized & Gated Adaptation (FT-2-KL, $\text{GATE}_\tau$) | C2 Mitigation | **Category 3: Requires Formal Approval** | Post-hoc exploratory adaptation algorithms. Protocol drafted (`EXP020_PREREGISTRATION_DRAFT.md`), but strictly gated under DEC-029. |
| **TSK-11** | EXP-027: External Zero-Shot Transfer to SleepAccel | C3 External Valid | **Category 4: Deferred** | Formally deferred under DEC-029 and excluded from active manuscript scope. Requires multi-site harmonization. |
| **TSK-12** | EXP-028: Multi-Night Agreement on Second Dataset | C3 External Valid | **Category 4: Deferred / Impossible** | Systematic search (C1) confirmed no second public multi-night wearable dataset exists in the scientific domain. |
| **TSK-13** | Neural Architecture Search (NAS Iteration 2) | Algorithmic Search | **Category 4: Deferred / Outside Scope** | Permanently terminated under DEC-029 and DEC-031. Strictly prohibited. |

---

## 3. Scientific Importance Ranking of Remaining Work

In alignment with `RESEARCH_DIRECTION_v4_FULL.md` (which explicitly designates Contribution C1 as the primary research contribution), the unexecuted tasks are ranked strictly by their scientific necessity to the paper's core thesis:

```
Rank 1: TSK-06 — Calibration-Depth Learning Curve Execution (k = 1, 3, 4)
   Target: Contribution C1 (Primary Research Thesis)
   Impact: ESSENTIAL. Directly tests the core question of the manuscript title ("Does a few labeled
           nights... help?") and determines whether personalization reaches significance at k >= 3
           as claimed by van der Aar et al. (2025).

Rank 2: TSK-08 — Matched 5-Fold Cross-Validation Re-Run of SLAMSS-IFS
   Target: Matched Baseline Evaluation & Cross-Study Comparability (Gap G2)
   Impact: CRITICAL. Eliminates immediate desk-rejection risk at IEEE TBME/JBHI by replacing
           defensive "different protocol" caveats with a rigorous non-inferiority benchmark.

Rank 3: TSK-07 — Re-Run of Unsupervised Adaptation (UA-2) with Corrected Mask
   Target: Contribution C1 (Completing the Personalization Control Suite)
   Impact: HIGH. Resolves DEC-030 implementation no-op and tests whether label-free adaptation
           is viable on low-rate cardiac vitals.

Rank 4: TSK-11 — External Zero-Shot Transfer to SleepAccel (EXP-027)
   Target: Contribution C3 (External Generalizability)
   Impact: HIGH. Breaks the single-cohort limitation and proves cross-sensor validity.

Rank 5: TSK-09 — Subject-Level Heterogeneity Hypothesis Testing (EXP-019B)
   Target: Contribution C2 (Failure Mechanism Diagnosis)
   Impact: MODERATE. Evaluates statistical predictors of negative transfer on existing data.

Rank 6: TSK-10 — Regularized & Gated Adaptation (EXP-020: FT-2-KL & GATE_tau)
   Target: Algorithmic Contribution & C2 (Failure Mitigation)
   Impact: MODERATE-LOW. Post-hoc exploratory algorithm with high risk of non-generalizability.
```

---

## 4. Detailed Specifications for Separately Governed Experiments

### 4.1 Experiment 1 (TSK-06): Supervised Personalization Learning Curve ($k = 1, 3, 4$)
*(Governed independently under proposed `PREREG_EXP-019_K_CURVE.md`)*

#### A. Scientific Question
Does the benefit of supervised wearable personalization scale monotonically with calibration depth ($k = 1, 2, 3, 4$ nights), and does adaptation reach statistical significance ($\bar{\Delta} \ge +1.50\text{ pp}$, lower CI $> 0.0$) when three or more calibration nights are available ($k \ge 3$), replicating the clinical finding of van der Aar et al. (2025) in healthy adults?

#### B. Resolving Changing Participant Composition
Because participants possess varying numbers of usable nights ($n_{\text{usable}} \in [2, 7]$), reserving $k$ calibration nights leaves varying numbers of eligible participants with at least one evaluation night ($n_{\text{usable}} \ge k + 1$):
- $k=1$: $N=46$ evaluable subjects (198 evaluation nights).
- $k=2$: $N=46$ evaluable subjects (152 evaluation nights).
- $k=3$: $N=42$ evaluable subjects (106 evaluation nights; 4 subjects have $n_{\text{usable}} = 3$, leaving 0 evaluation nights).
- $k=4$: $N=33$ evaluable subjects (64 evaluation nights; 9 subjects have $n_{\text{usable}} = 4$, leaving 0 evaluation nights).

To prevent participant attrition from confounding depth comparisons, Experiment 1 enforces a two-tier evaluation framework:

1. **Common-Cohort Primary Analysis ($N = 33$ Fixed Participants):**
   - Evaluated strictly on the identical cohort of **$N = 33$ participants** who possess at least 5 usable nights ($n_{\text{usable}} \ge 5$).
   - Participant composition is 100% invariant across $k \in \{1, 2, 3, 4\}$.
   - Evaluates within-subject paired trajectories across calibration depths on identical participants.
2. **Available-Cohort Sensitivity Analysis ($N_k \in \{46, 46, 42, 33\}$):**
   - Evaluates all available eligible participants at each depth $k$ on all their subsequent nights ($E_k$).
   - Maximizes statistical power at smaller $k$ ($N=46$ at $k=1, 2$; $N=42$ at $k=3$) and serves as an empirical sensitivity check against selection bias in the common cohort.

#### C. Exact Primary Hypotheses, Multiplicity Correction & Decision Rules
- **Primary Hypotheses:**
  1. *Trend Hypothesis ($H_{\text{trend}}$):* Mean adaptation gain $\bar{\Delta}_k$ increases monotonically with $k \in \{1, 2, 3, 4\}$ in the common cohort ($N=33$), tested via Page's trend test for ordered alternatives.
  2. *van der Aar Replication ($H_{k=3}$):* In the available cohort ($N=42$) and common cohort ($N=33$), 3-night calibration achieves:
     $$\bar{\Delta}_{k=3} \ge +1.50\text{ pp} \quad \text{AND} \quad 95\%\text{ bootstrap CI lower bound} > 0.0\text{ pp}$$
  3. *Saturation Hypothesis ($H_{\text{sat}}$):* The incremental step gain from $k=3$ to $k=4$ ($\Delta_{4-3} = \bar{\Delta}_{k=4} - \bar{\Delta}_{k=3}$) is negligible ($95\%\text{ CI upper bound} < +0.50\text{ pp}$).
- **Pairwise Step Comparisons:**
  - $\Delta_{2-1} = \bar{\Delta}_{k=2} - \bar{\Delta}_{k=1}$ (Incremental value of 2nd night)
  - $\Delta_{3-2} = \bar{\Delta}_{k=3} - \bar{\Delta}_{k=2}$ (Incremental value of 3rd night)
  - $\Delta_{4-3} = \bar{\Delta}_{k=4} - \bar{\Delta}_{k=3}$ (Incremental value of 4th night)
- **Multiplicity Control:**
  - The family of secondary pairwise tests ($\Delta_{2-1}, \Delta_{3-2}, \Delta_{4-3}$) and depth vs. ZS comparisons is controlled using the **Holm-Bonferroni step-down procedure** at family-wise $\alpha = 0.05$.
- **Formal Decision Rules:**
  - `SUPPORTED (van der Aar Replication)`: $\bar{\Delta}_{k=3} \ge +1.50\text{ pp}$ AND Holm-adjusted lower CI $> 0.0\text{ pp}$.
  - `PLATEAU / SATURATION`: Significant gain at $k=2$ or $k=3$, but incremental gain $\Delta_{4-3}$ upper CI $< +0.50\text{ pp}$.
  - `INCONCLUSIVE`: Confidence interval for $k=3$ spans $+1.50\text{ pp}$ while lower bound spans 0.0 pp.
  - `NOT SUPPORTED`: 95% bootstrap upper bound for $k=3$ and $k=4$ is strictly below $+1.50\text{ pp}$.

#### D. Training Protocol & Leakage Safeguards
- **Backbone & Hyperparameter Inheritance:** Frozen population Model A (163,940 parameters). To prevent hyperparameter overfitting on small cohorts, fine-tuning inherits the locked $k=2$ configuration: AdamW, 15 epochs, learning rate $\eta = 10^{-4}$, weight decay $10^{-2}$, batch size 4.
- **Leakage Prevention:** Chronological ordering enforced strictly within subject ($k < \min(\text{eval\_indices})$). Shuffling or random selection of calibration nights is strictly prohibited.
- **Computational Cost & Hardware Allocation:** 363 fine-tuning runs (121 subject-depth evaluations $\times$ 3 random seeds). Profiled memory per Model A instance is **460 MiB allocated / 486 MiB reserved VRAM** at batch size $B=4, L=1200$. On our actual workstation (2 $\times$ NVIDIA RTX 3060 12 GB, AMD Ryzen 7 7700, 32 GB RAM), execution requires ~30 seconds per run $\implies$ **~3.0 hours on a single RTX 3060**, or **~1.5 to 1.75 wall-clock hours** parallelized across both independent RTX 3060 GPUs (e.g., Folds 1–3 on GPU 1, Folds 4–5 on GPU 0). Memory consumption is $< 0.5\text{ GB}$ VRAM per card.

---

### 4.2 Experiment 2 (TSK-07): Unsupervised Test-Time Adaptation Re-Run ($UA\text{-}2$)
*(Governed independently under proposed `PREREG_EXP-019_S3_AMENDMENT.md`)*

#### A. Scientific Question
Does unsupervised test-time adaptation via Shannon entropy minimization ($UA\text{-}2$) adapt sequence models to individual autonomic baselines without ground-truth labels, or does label-free adaptation fail or diverge on low-rate cardiac vitals?

#### B. Scientific Justification & Validation of Raw-Unit Quality Thresholds
In DEC-030, the historical S3 execution was invalidated because `compute_wearable_quality_mask` applied raw physiological cutoffs ($[30, 180]$) to z-score normalized features ($[-3.95, +9.94]$), filtering 100% of calibration epochs.

To correct this without improperly bypassing quality control:
1. **Physiological Threshold Justification:**
   - *Heart Rate ($30 \le \text{IHR}_{\text{raw}} \le 180\text{ BPM}$):* Normal nocturnal resting heart rate in healthy adults ranges from 40 to 90 BPM; during deep slow-wave sleep, sinus bradycardia may reach 35–40 BPM, while transient micro-arousals reach 120–140 BPM. Commercial PPG specifications and clinical guidelines establish readings $< 30\text{ BPM}$ or $> 180\text{ BPM}$ as unambiguous hardware artifacts (loss of optical coupling or motion-induced clipping).
   - *Accelerometry Magnitude ($0.2\text{ g} \le m_{\text{raw}} \le 3.5\text{ g}$):* Stationary wrist gravity contributes $1.0\text{ g}$. Normal sleep movements produce up to $2.0–2.5\text{ g}$. Accelerations outside $[0.2\text{ g}, 3.5\text{ g}]$ represent sensor dropouts or non-wear handling.
2. **Validation Requirements Prior to Adaptation Execution:**
   - Quality screening **must be evaluated on raw physical units prior to feature normalization**.
   - The binary boolean mask must be serialized alongside normalized features.
   - **Validation Sanity Gate:** Before launching adaptation, verify and log that:
     * $> 95\%$ of calibration epochs across the cohort pass the raw validity filter.
     * Exactly 0 non-physiological epochs enter the unsupervised loss.
     * At least $80\%$ of epochs per calibration night are retained for entropy optimization.

#### C. Formal Decision Criteria for UA-2
- `SUPPORTED (Label-Free Gain)`: $\bar{\Delta}_{\text{UA-2}} \ge +1.00\text{ pp}$ Macro-F1 vs. $ZS$ with 95% bootstrap lower CI $> 0.0\text{ pp}$.
- `NEUTRAL / INEFFECTIVE`: $\bar{\Delta}_{\text{UA-2}} \in [-0.50, +0.50]\text{ pp}$ (CI spans 0).
- `DESTRUCTIVE DRIFT`: $\bar{\Delta}_{\text{UA-2}} < -0.50\text{ pp}$ with upper CI $< 0.0\text{ pp}$ (entropy minimization collapses onto majority Light-sleep class).
- **Computational Cost & Hardware Allocation:** 138 adaptation runs (46 subjects $\times$ 3 seeds, 20 gradient steps). Profiled memory is **460 MiB VRAM** at $B=4, L=1200$. On an RTX 3060, each run takes ~12 seconds $\implies$ **~28 minutes total GPU time**. Runs concurrently on GPU 0 while GPU 1 runs TSK-06, requiring $< 0.5\text{ GB}$ VRAM and $< 3\text{ GB}$ system RAM, with zero cross-process contention.

---

### 4.3 Experiment 3 (TSK-08): Matched 5-Fold Cross-Validation Paired Comparative Benchmark of SLAMSS-IFS
*(Governed independently under proposed `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`)*

#### A. Scientific Question
How does the lightweight Model A (163,940 parameters) compare to the official SLAMSS-IFS baseline (666k params paper text; 5.47M params repository) when evaluated under strictly matched, leakage-free 5-fold cross-validation and full-night context ($L = 1200$)?

#### B. Methodological Framework: Estimation with Bootstrap Confidence Intervals
In the absence of an established clinical consensus defining an artificial non-inferiority margin for wearable sleep staging, Experiment 3 adopts a rigorous paired estimation framework:
1. **Focus on Estimation over Threshold Verdicts:** Rather than imposing an arbitrary pass/fail margin, the evaluation precisely quantifies the paired participant difference $\bar{d} = \text{Macro-F1}(\text{Model A}) - \text{Macro-F1}(\text{SLAMSS-IFS})$ with 10,000-resample participant-level bootstrap 95% confidence intervals and Wilcoxon signed-rank testing.
2. **Contextualizing Parameter Efficiency:** The observed performance difference is reported alongside the 4.06-fold (vs. paper text) to 33.34-fold (vs. repository) parameter compression, directly addressing wearable on-chip SRAM feasibility (~164 KB in INT8).

#### C. Participant-Level Paired Analysis & Interpretation Framework
Statistical inference is conducted strictly at the independent participant level ($N = 47$):

1. For each participant $i \in \{1, \dots, 47\}$, compute out-of-fold paired difference:
   $$d_i = \text{Macro-F1}(\text{Model A})_i - \text{Macro-F1}(\text{SLAMSS-IFS})_i$$
2. Compute the 10,000-resample participant-level percentile bootstrap distribution of the mean difference $\bar{d}$ to construct the two-sided 95% bootstrap confidence interval $[\text{CI}_{\text{lower}}, \text{CI}_{\text{upper}}]$.
3. **Formal Interpretation Rules:**
   - **STATISTICALLY SIGNIFICANT MODEL A ADVANTAGE:** $\text{CI}_{\text{lower}} > 0.0\text{ pp}$ and $p < 0.05$.
   - **STATISTICALLY SIGNIFICANT SLAMSS-IFS ADVANTAGE:** $\text{CI}_{\text{upper}} < 0.0\text{ pp}$ and $p < 0.05$.
   - **STATISTICALLY INDISTINGUISHABLE (PARITY ON MATCHED PROTOCOL):** 95% bootstrap CI spans $0.0\text{ pp}$ ($p \ge 0.05$). The confidence interval width directly bounds the maximum plausible performance difference under matched evaluation.

#### D. Execution Protocol & Hardware Allocation
- **Protocol:** Exact 5-fold manifests (`EXP-014`/`EXP-018`/`EXP-019`), 3 random seeds (42, 123, 2024), full-night context ($L = 1200$), identical 10-channel inputs.
- **Hardware Allocation & Execution Budget:** 15 training runs (5 folds $\times$ 3 seeds, 50–100 epochs, $L=1200$). Empirically profiled on the workstation's RTX 3060 (12 GB) using the exact 5,465,768-parameter official repository model (DEC-011): under PyTorch AMP (`torch.cuda.amp.autocast`) with Adam optimizer, measured peak VRAM across batch sizes is:
  * $B = 1$: 1.38 GiB allocated / 1.64 GiB reserved (0.31 nights/s).
  * $B = 2$: 2.45 GiB allocated / 2.95 GiB reserved (0.60 nights/s).
  * $B = 4$: **4.63 GiB allocated / 5.12 GiB reserved** (1.23 nights/s), leaves $> 6.8\text{ GiB}$ free VRAM on GPU 1 and $> 6.0\text{ GiB}$ on GPU 0.
  * Optimizer State: 41.7 MiB (Adam 1st and 2nd momentum states).
  * Numerical Stability: Verified fully stable across all batch sizes (zero NaNs/Infs).
  The official model **fits comfortably on a single RTX 3060 (12 GB)**. The prior $\ge 24\text{ GB}$ requirement was an unbatched theoretical overestimate. Runtime on RTX 3060 with AMP is ~2.0–2.5 hours per fold-seed $\implies$ ~33.75 total GPU hours. Parallelized across both RTX 3060 cards (7 runs on GPU 0, 8 runs on GPU 1), expected wall-clock runtime is **~16 to 18 hours**.

---

## 5. Summary of Experimental Roadmaps & Approvals

```
================================================================================
REVISED PHASE D EXPERIMENTAL EXECUTION ROADMAP (Updated under DEC-034)
================================================================================

PHASE D, STAGE 1: IMMEDIATE EXECUTION (Contributions C1 & C2 Closure)
  Experiment 1 (TSK-06): Calibration-Depth Learning Curve (k = 1, 3, 4)
  - Status: ACTIVE / PRIORITIZED FOR EXECUTION
  - Governance: PREREG_EXP-019_K_CURVE.md
  - Design: Common-cohort (N=33) on E4 + Available-cohort (N=46, 42, 33)
  - Hardware: RTX 3060 12GB (Peak VRAM: 486 MiB, B=4)
  - Compute: ~3.0 GPU hours serial (or ~1.5-1.75 hours parallel across 2 GPUs)
  - Scientific Output: Resolves van der Aar replication and depth scaling.

  Experiment 2 (TSK-07): Unsupervised Adaptation Re-Run (UA-2)
  - Status: ACTIVE / PRIORITIZED FOR EXECUTION
  - Governance: PREREG_EXP-019_S3_AMENDMENT.md
  - Design: Raw-unit physiological mask (30-180 BPM) + entropy minimization
  - Hardware: RTX 3060 12GB (Peak VRAM: 460 MiB, B=4)
  - Compute: ~28 minutes GPU time (runs concurrently on GPU 0 while GPU 1 executes TSK-06)
  - Scientific Output: Resolves DEC-030 no-op; provides empirical label-free baseline.

PHASE D, STAGE 2: FINAL MODEL SYNTHESIS & FREEZING
  - Integrate insights from TSK-06 (depth curve), TSK-07 (unsupervised baseline), EXP-018 (context scaling), and EXP-019 (sleep-summary agreement).
  - Develop, validate on internal folds, and freeze the final project model architecture and checkpoints.

PHASE D, STAGE 3: DEFINITIVE MATCHED BASELINE BENCHMARK (Run Once)
  Experiment 3 (TSK-08): Matched 5-Fold Cross-Validation of SLAMSS-IFS
  - Status: DEFERRED PENDING FINAL MODEL FREEZE (DEC-034)
  - Governance: PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md
  - Design: Paired comparative estimation with 10,000 bootstrap CIs on identical 5-fold CV
  - Hardware: RTX 3060 12GB (Peak VRAM: 5.12 GiB reserved with AMP at B=4 for official 5.47M model; fits on 12GB card)
  - Compute: ~33.75 GPU hours total (~16-18 wall-clock hours distributed across 2 GPUs: 7 runs on GPU 0, 8 runs on GPU 1)
  - Execution Rule: Run exactly once as the final matched benchmark against the frozen final model.
================================================================================
```

---

## 6. Project Governance & Immediate Next Steps

Under user decision **DEC-034**, the sequencing has been formally ratified:
1. **TSK-08 Deferred:** SLAMSS-IFS benchmark is held in reserve until the final model is developed and frozen.
2. **Immediate Execution Focus:** Author execution scripts and run TSK-06 (EXP-019-K) and TSK-07 (EXP-019-S3) concurrently on the dual-RTX 3060 workstation (~1.5–1.75 wall-clock hours total).
3. **Synthesis & Freeze:** Consolidate empirical results and freeze final model architecture before running Stage 3.

---

*(Plan revision complete. Zero code executed; zero models trained; locked files preserved.)*
