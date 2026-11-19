# PRE-REGISTRATION ADDENDUM: EXP-019 Calibration-Depth Learning Curve (k = 1, 3, 4)

**Document ID:** `PREREG_EXP-019_K_CURVE`  
**Parent Preregistration:** [`PREREG_EXP-019_H-A.md`](./PREREG_EXP-019_H-A.md) (Locked at commit `4dff5b7`)  
**Experiment ID:** EXP-019 (Phase D, Experiment 1 / Task TSK-06)  
**Status:** DRAFT FOR USER REVIEW (No models trained; zero GPU execution; locked baselines preserved)  
**Governing Authority:** [`RESEARCH_DIRECTION_v4_FULL.md`](./RESEARCH_DIRECTION_v4_FULL.md), Section 9; [`PHASE_D_EXPERIMENTAL_DECISION_PLAN.md`](./PHASE_D_EXPERIMENTAL_DECISION_PLAN.md)  
**Date:** 2026-11-19 (Revised)  

---

## 1. Scientific Motivation and Core Questions

In `PREREG_EXP-019_H-A.md`, the primary confirmatory hypothesis $H\text{-}A$ evaluated within-person supervised fine-tuning using exactly $k = 2$ calibration nights ($FT\text{-}2$), yielding an empirical gain over population zero-shot ($ZS$) on evaluation nights $E_2$ (Nights 3+) of:
$$\bar{\Delta}_{k=2} = +1.01\text{ percentage points (pp)}, \quad 95\%\text{ bootstrap CI: } [-0.17, +1.99]\text{ pp} \quad (N = 46\text{ subjects, } 152\text{ nights})$$

While $FT\text{-}2$ demonstrated a positive point estimate, the confidence interval spanned zero, falling short of the pre-specified confirmatory threshold ($\ge +1.50\text{ pp}$ with lower bound $> 0.0\text{ pp}$). In contemporary clinical literature, van der Aar et al. (2025) reported that within-person wearable sleep adaptation requires at least 3 calibration nights ($k \ge 3$) to consistently surmount nocturnal biological variability and achieve statistically significant personalization gains.

Although the original preregistration listed the $k \in \{1, 2, 3, 4\}$ learning curve as a descriptive objective, only $k = 2$ was historically executed. This preregistration addendum formally establishes the confirmatory protocol, cohort stratification, statistical hypotheses, multiplicity control, and decision rules for executing the unexecuted calibration depths $k \in \{1, 3, 4\}$.

### Core Scientific Questions
1. **Monotonic Scaling ($H_{\text{trend}}$):** Does personalization gain scale monotonically as a function of calibration depth $k \in \{1, 2, 3, 4\}$?
2. **van der Aar Replication ($H_{k=3}$):** Does expanding calibration from 2 to 3 nights achieve a clinically meaningful and statistically significant gain ($\bar{\Delta}_{k=3} \ge +1.50\text{ pp}$ with lower bound $> 0.0\text{ pp}$)?
3. **Diminishing Returns / Saturation ($H_{\text{sat}}$):** Does personalization saturate between 3 and 4 calibration nights ($\delta_{4-3} < +0.50\text{ pp}$), indicating that 3 nights represent the optimal clinical trade-off between user burden and model accuracy?

---

## 2. Participant Cohort Stratification & Attrition Control

### 2.1 The Participant Attrition Confound
In the BIDSleep dataset (47 total participants, 46 with $n_{\text{usable}} \ge 3$), participants possess varying numbers of recorded nights ($n_{\text{usable}} \in [2, 7]$). Because reserving $k$ calibration nights leaves only participants with $n_{\text{usable}} \ge k + 1$ having at least one evaluation night, the pool of evaluable participants naturally shrinks as $k$ increases:

| Calibration Depth | Usability Requirement | Available Test Participants ($N_k$) | Evaluated Participants Lost | Total Outer Evaluation Nights ($|E_k|$) |
|:---:|:---:|:---:|:---:|:---:|
| **$k = 1$** | $n_{\text{usable}} \ge 2$ | $N_1 = 46$ | 0 | 198 nights |
| **$k = 2$** (Locked) | $n_{\text{usable}} \ge 3$ | $N_2 = 46$ | 0 | 152 nights |
| **$k = 3$** | $n_{\text{usable}} \ge 4$ | $N_3 = 42$ | 4 (Bidslab02, 17, 28, 47) | 106 nights |
| **$k = 4$** | $n_{\text{usable}} \ge 5$ | $N_4 = 33$ | 9 (Bidslab04, 05, 08, 09, 13, 20, 24, 25, 41) | 64 nights |

*Note on Bidslab49:* Excluded across all arms ($n_{\text{usable}} = 2$, but omitted from historical manifests due to sensor dropout, leaving 46 eligible subjects).

Comparing available participants across $k$ without correction introduces survivor bias: participants with $\ge 5$ nights may possess systematically different sleep regularities or autonomic profiles than participants with only 3 or 4 nights. Furthermore, evaluation nights would shift between depths.

### 2.2 Dual-Cohort Evaluation Design
To eliminate compositional confounding while maintaining maximal statistical power, this experiment preregisters a mandatory two-tier design:

```
+-----------------------------------------------------------------------------------------+
|                                    COHORT DESIGN                                        |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  PRIMARY ANALYSIS: Common-Cohort (N = 33 Fixed Participants)                            |
|  - Inclusion: Strictly participants with n_usable >= 5.                                 |
|  - Composition: 100% identical participants across k = 1, 2, 3, 4.                      |
|  - Primary Evaluation Nights: Fixed late-night evaluation set E_4 (Nights 5+).          |
|    * This holds BOTH the participant cohort AND the evaluation nights strictly constant.|
|    * Isolates the pure effect of adding calibration data:                               |
|      Model_k1(Night 1) vs Model_k2(Nights 1-2) vs Model_k3(Nights 1-3) vs               |
|      Model_k4(Nights 1-4) evaluated on identical Night 5+ epochs.                       |
|  - Evaluation-Night Distribution in E_4 (64 nights total):                              |
|    * Participants with 5 usable nights contribute exactly 1 evaluation night (Night 5). |
|    * Participants with 6 usable nights contribute 2 evaluation nights (Nights 5, 6).    |
|    * Participants with 7 usable nights contribute 3 evaluation nights (Nights 5, 6, 7). |
|  - Within-Subject Pooling: All valid epochs across a participant's E_4 nights are       |
|    pooled into a single confusion matrix to compute that participant's Macro-F1.        |
|    Participants are averaged with equal weight (1/33) across the cohort.                |
|                                                                                         |
|  SENSITIVITY ANALYSIS: Available-Cohort (N_k in {46, 46, 42, 33})                       |
|  - Inclusion: All usable participants at each respective depth (n_usable >= k + 1).     |
|  - Evaluation Nights: All chronological evaluation nights E_k for that depth.           |
|  - Purpose: Maximize sample size at k=1, 2, 3 and verify that findings in N=33          |
|    generalize to participants with shorter total recording durations.                   |
+-----------------------------------------------------------------------------------------+
```

### 2.3 Evaluating Locked $k=2$ Predictions on $E_4$ Without Modifying Historical Results
- The authoritative historical results for $k = 2$ on $E_2$ (152 nights across 46 subjects, $\bar{\Delta} = +1.01\text{ pp}$) are permanently locked in `exp019_subject_metrics.csv` and will not be altered.
- For the primary common-cohort analysis on $E_4$, the stored epoch-level out-of-fold predictions from the locked $k = 2$ model checkpoints are filtered strictly to the subset of epochs belonging to $E_4$ for the $N = 33$ common-cohort participants.
- This derives the $k = 2$ data point on identical evaluation nights without retraining, tuning, or modifying historical artifacts.

---

## 3. Formal Statistical Hypotheses & Multiplicity Control

All primary hypotheses are evaluated at the participant level ($N = 33$ for the primary common-cohort analysis). Let $\Delta_i(k) = \text{Macro-F1}_i(FT\text{-}k) - \text{Macro-F1}_i(ZS)$ denote the paired personalization delta for participant $i$ at calibration depth $k$ evaluated on $E_4$, averaged across the 3 seeds.

### 3.1 Primary Repeated-Measures Trend Test ($H_{\text{trend}}$)
To test whether personalization performance monotonically improves with calibration depth, we specify **Page's $L$-trend test for ordered alternatives in repeated measures**:
- In this design, participants $i \in \{1, \dots, 33\}$ act as blocks, and calibration depths $k \in \{1, 2, 3, 4\}$ act as ordered treatments.
- For each participant, ranks $r_{ik} \in \{1, 2, 3, 4\}$ are computed across depths.
- Page's $L$ statistic is calculated as:
  $$L = \sum_{k=1}^4 k \cdot R_k, \quad \text{where } R_k = \sum_{i=1}^{33} r_{ik}$$
- Hypotheses:
  $$H_0: \text{Median}(\Delta_1) = \text{Median}(\Delta_2) = \text{Median}(\Delta_3) = \text{Median}(\Delta_4)$$
  $$H_1: \text{Median}(\Delta_1) \le \text{Median}(\Delta_2) \le \text{Median}(\Delta_3) \le \text{Median}(\Delta_4) \quad \text{(with at least one strict inequality)}$$
- Page's test is the exact non-parametric test for ordered alternatives in repeated measures (unlike Jonckheere-Terpstra, which assumes independent groups).

### 3.2 van der Aar 3-Night Replication Hypothesis ($H_{k=3}$)
We rigorously decouple the statistical hypothesis of superiority from the clinical relevance benchmark:
- **Statistical Superiority Hypothesis ($H_{k=3}^{\text{stat}}$):**
  $$H_0: \bar{\Delta}_{k=3} \le 0.0\text{ pp} \quad \text{vs.} \quad H_1: \bar{\Delta}_{k=3} > 0.0\text{ pp}$$
  Tested via two-sided Wilcoxon signed-rank test and 10,000-resample participant-level percentile bootstrap.
- **Clinical Relevance Benchmark Threshold:**
  $$\bar{\Delta}_{k=3} \ge +1.50\text{ pp Macro-F1}$$
- **Replication Confirmed Criterion:** The statistical superiority test is significant (Holm-adjusted bootstrap 95% CI lower bound $> 0.0\text{ pp}$) **AND** the point estimate achieves clinical relevance ($\bar{\Delta}_{k=3} \ge +1.50\text{ pp}$).

### 3.3 Saturation Hypothesis ($H_{\text{sat}}$)
- Evaluates whether the step increment from 3 to 4 calibration nights is practically negligible:
  $$\delta_{4-3} = \bar{\Delta}_{k=4} - \bar{\Delta}_{k=3}$$
- **Negligible Increment Threshold:** $\epsilon_{\text{sat}} = +0.50\text{ pp}$.
- **Saturation Criterion:** The upper bound of the 95% bootstrap confidence interval on $\delta_{4-3}$ is strictly below $+0.50\text{ pp}$ ($\text{CI}_{\text{upper}} < +0.50\text{ pp}$), demonstrating that adding a 4th calibration night provides no meaningful benefit.

### 3.4 Multiplicity Family & Holm-Bonferroni Step-Down Procedure
To guarantee a Family-Wise Error Rate (FWER) $\le 0.05$, we define the complete primary hypothesis family $\mathcal{F}$ comprising exactly $m = 5$ tests:
1. $T_1$: Monotonic trend across $k \in \{1, 2, 3, 4\}$ via Page's $L$-test ($p_1$).
2. $T_2$: Statistical superiority of 3-night calibration ($k = 3$) vs. $ZS$ via Wilcoxon signed-rank ($p_2$).
3. $T_3$: Incremental step gain $\delta_{2-1} = \bar{\Delta}_{k=2} - \bar{\Delta}_{k=1} > 0$ via Wilcoxon signed-rank ($p_3$).
4. $T_4$: Incremental step gain $\delta_{3-2} = \bar{\Delta}_{k=3} - \bar{\Delta}_{k=2} > 0$ via Wilcoxon signed-rank ($p_4$).
5. $T_5$: Incremental step gain $\delta_{4-3} = \bar{\Delta}_{k=4} - \bar{\Delta}_{k=3} > 0$ via Wilcoxon signed-rank ($p_5$).

**Inference Procedure:**
- Sort the $m = 5$ obtained $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le p_{(3)} \le p_{(4)} \le p_{(5)}$.
- Compare each $p_{(j)}$ sequentially to the adjusted threshold $\alpha / (5 - j + 1)$ with $\alpha = 0.05$:
  - Step 1: Check $p_{(1)} \le 0.05 / 5 = 0.0100$. If not, stop; no tests are significant.
  - Step 2: Check $p_{(2)} \le 0.05 / 4 = 0.0125$.
  - Step 3: Check $p_{(3)} \le 0.05 / 3 = 0.0167$.
  - Step 4: Check $p_{(4)} \le 0.05 / 2 = 0.0250$.
  - Step 5: Check $p_{(5)} \le 0.05 / 1 = 0.0500$.

---

## 4. Experimental Constants & Hyperparameter Inheritance

To preserve comparability with locked EXP-019 results and eliminate post-hoc tuning degrees of freedom:
- **Architecture:** Model A (`ConventionalCNNBiGRU`, 163,940 parameters).
- **Population Backbones:** Exactly the 15 locked EXP-018 Phase 2 checkpoints ($5\text{ folds} \times 3\text{ seeds}$ [42, 123, 2024]). Circadian features neutralized (`time_feat = 0`).
- **Inference Context:** Full-night context ($L = 1200$) with sliding-window probability stitching for $N > L$.
- **Class Mapping:** 4-class (Wake = 0, Light = 1, Deep = 2, REM = 3; unknown/unscored = -100).
- **Hyperparameter Inheritance:** Fine-tuning hyperparameters for $k \in \{1, 3, 4\}$ are inherited directly from the locked Stage D winning configurations for $k = 2$:
  - Fold 1: `scope = all`, $\eta = 1 \times 10^{-4}$, 5 epochs.
  - Fold 2: `scope = all`, $\eta = 3 \times 10^{-5}$, 15 epochs.
  - Fold 3: `scope = all`, $\eta = 3 \times 10^{-5}$, 15 epochs.
  - Fold 4: `scope = head_gru`, $\eta = 3 \times 10^{-5}$, 5 epochs.
  - Fold 5: `scope = all`, $\eta = 3 \times 10^{-5}$, 5 epochs.
- **Chronological Night Partitioning:** Calibration nights $C_k = \{t_1, \dots, t_k\}$ precede evaluation nights $E_k = \{t_{k+1}, \dots, t_M\}$ strictly by `recStart` timestamp. Zero post-calibration epochs enter training or model selection.

---

## 5. Formal Two-Dimensional Decision Framework

To prevent category overlap and ensure mathematically exhaustive coverage across both the single-depth van der Aar replication hypothesis ($H_{k=3}$) and the multi-depth scaling dynamics ($H_{\text{trend}}$ and $H_{\text{sat}}$), experimental outcomes are evaluated across two independent, orthogonal decision dimensions:

```
+-----------------------------------------------------------------------------------------+
|                    DIMENSION 1: PRIMARY HYPOTHESIS H_{k=3} (van der Aar Status)          |
|                    (Mutually Exclusive & Exhaustive Partition over \bar{\Delta}_{k=3})   |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  1A. REPLICATION CONFIRMED (van der Aar Validated & Clinically Relevant):               |
|      - Criterion: Holm-adjusted bootstrap 95% CI lower bound > 0.0 pp                   |
|        AND point estimate \bar{\Delta}_{k=3} >= +1.50 pp.                               |
|      - Scientific Meaning: Confirms van der Aar et al. (2025); 3 calibration nights      |
|        surmount autonomic variability and achieve clinically meaningful personalization.|
|                                                                                         |
|  1B. STATISTICALLY SIGNIFICANT BUT SUB-BENCHMARK GAIN:                                  |
|      - Criterion: Holm-adjusted bootstrap 95% CI lower bound > 0.0 pp                   |
|        AND point estimate \bar{\Delta}_{k=3} < +1.50 pp.                                |
|      - Scientific Meaning: Personalization yields a genuine, statistically reliable     |
|        gain, but the effect magnitude falls short of the +1.50 pp clinical benchmark.   |
|                                                                                         |
|  1C. INCONCLUSIVE:                                                                      |
|      - Criterion: Bootstrap 95% CI spans 0.0 pp AND upper CI >= +1.50 pp.               |
|      - Scientific Meaning: Sample size / variance is insufficient to confirm or rule    |
|        out a meaningful adaptation gain at k=3.                                         |
|                                                                                         |
|  1D. CLINICAL BENCHMARK RULED OUT:                                                      |
|      - Criterion: Bootstrap 95% CI upper bound < +1.50 pp (regardless of lower bound).  |
|      - Scientific Meaning: The data definitively rule out an adaptation effect of       |
|        clinical practical size (+1.50 pp) at 3 calibration nights.                      |
+-----------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------+
|             DIMENSION 2: MULTI-DEPTH SCALING & SATURATION (H_{trend} & H_{sat})         |
|             (Mutually Exclusive & Exhaustive Partition over Curve Dynamics)             |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  2A. MONOTONIC SCALING WITH EARLY SATURATION AT 3 NIGHTS:                               |
|      - Criterion: Page's L-test p_adj < 0.05 AND \delta_{4-3} 95% CI upper < +0.50 pp.   |
|      - Scientific Meaning: Performance increases monotonically across initial depths,    |
|        but plateaus between k=3 and k=4; 3 nights represent the optimal clinical         |
|        trade-off between user burden and accuracy.                                      |
|                                                                                         |
|  2B. MONOTONIC SCALING WITH CONTINUING RETURNS BEYOND 3 NIGHTS:                         |
|      - Criterion: Page's L-test p_adj < 0.05 AND \delta_{4-3} 95% CI lower > 0.0 pp.    |
|      - Scientific Meaning: Personalization gains continue to grow linearly through      |
|        4 nights; additional calibration beyond 4 nights may yield further benefits.     |
|                                                                                         |
|  2C. NO MONOTONIC SCALING (FLAT / UNRESPONSIVE PROFILE):                                |
|      - Criterion: Page's L-test p_adj >= 0.05 AND \delta_{4-3} 95% CI spans 0.0 pp.     |
|      - Scientific Meaning: Increasing calibration depth does not reliably improve       |
|        model performance on this sensor modality.                                       |
|                                                                                         |
|  2D. SCALING DYNAMICS INCONCLUSIVE:                                                     |
|      - Criterion: All other parameter configurations not meeting 2A, 2B, or 2C.         |
|      - Scientific Meaning: The trajectory of the learning curve cannot be definitively  |
|        categorized due to statistical uncertainty.                                      |
+-----------------------------------------------------------------------------------------+
```

---

## 6. Computational Execution Requirements

- **Total Fine-Tuning Runs:**
  - $k = 1$: 46 subjects $\times$ 3 seeds = 138 runs
  - $k = 3$: 42 subjects $\times$ 3 seeds = 126 runs
  - $k = 4$: 33 subjects $\times$ 3 seeds = 99 runs
  - *Total:* 363 model runs (5–15 epochs on 1–4 nights of data).
- **Target Hardware & Memory Profile:**
  - Workstation: 2 $\times$ NVIDIA GeForce RTX 3060 (12 GB VRAM each), AMD Ryzen 7 7700 (8C/16T), 32 GB RAM.
  - Model A memory footprint: **460 MiB allocated / 486 MiB reserved VRAM** per instance at batch size $B=4, L=1200$ sequence length.
  - VRAM usage per GPU is $< 0.5\text{ GB}$, leaving $> 11.5\text{ GB}$ VRAM free.
- **Execution Strategy & Runtime:**
  - Runtime per subject-seed run: ~30 seconds on RTX 3060.
  - Serial execution: ~3.0 hours on a single RTX 3060.
  - Dual-GPU parallel execution: Distributing folds/subjects across GPU 0 and GPU 1 completes all 363 runs in **~1.5 to 1.75 wall-clock hours**.
  - Concurrency with TSK-07: Alternatively, TSK-06 can execute on GPU 1 while TSK-07 runs concurrently on GPU 0 with zero resource contention.
- **Artifact Serialization:** Output results written to `experiments/EXP-019/exp019_k_learning_curve_executed.csv` and `experiments/EXP-019/exp019_k_curve_subject_metrics.csv`. The locked historical artifact `exp019_k_learning_curve.csv` remains strictly untouched.
