# Controlled Evaluation of Sequence Truncation in Wearable Sleep Staging

*   **Author:** jbjoyk
*   **Status:** Proposed Protocol (Under Review — Revisions Applied)
*   **Date:** 2026-11-14

--------------------------------------------------------------------------------

## Overview

Wearable sleep staging models trained on consumer photoplethysmography (PPG) and accelerometry signals are typically evaluated on fixed-length temporal sequences. In prior project phases (EXP-010 through EXP-014), sequence length was constrained to the first 600 epoch positions ($T = 600$, representing the first 5.0 hours from recording onset). 

EXP-017 is designed as a narrowly scoped, controlled sequence-horizon experiment to answer one specific question:

> Does extending the sequence training horizon from $T=600$ to $T \le 1200$ materially affect sleep-staging performance when both models are evaluated on the identical first 600 chronological epoch positions, with invalid positions masked?

To eliminate the primary confounding factor between sequence length and evaluation coverage, EXP-017 establishes:
1.  **Primary Causal Analysis**: Both the $T=600$ model and the $T \le 1200$ model are evaluated strictly on the **identical first 600 chronological epoch positions, with invalid positions masked** ($t \in [0, 600)$ from `recStart`) across the validation nights. This strictly isolates the representational and contextual benefits of extended-sequence training from the trivial confounder of scoring additional late-night epochs.
2.  **Secondary Descriptive Analysis (Non-Causal)**: The extended-sequence model ($T \le 1200$) is evaluated across its available extended recording coverage ($T \le 1200$) to provide a purely descriptive characterization of operational performance on longer recordings. This is strictly non-causal and descriptive, with no zero-padding of $T=600$ predictions and no comparative attribution against the truncated model.

The study holds all architectural, optimization, cohort, and loss parameters constant, isolating sequence horizon as the single experimental variable.

## The Problem and Context

In the BIDSLab Apple Watch + Dreem 2 dataset (246 usable nights across 47 subjects), recording durations range from 610 to 1,282 epochs (mean: 861.1 epochs / 7.18 hours; median: 857.5 epochs / 7.15 hours; verified in `AUDIT_ANSWERS.md`). The fixed 5-hour cutoff (`[:600]`) discards an average of 28.55% of labeled epochs per night (58,055 discarded out of 203,338 valid labeled epochs across all 246 usable nights; verified in `AUDIT_ANSWERS.md` Q2).

Because human sleep follows an ultradian progression where slow-wave sleep (Deep, N3) concentrates in early sleep cycles (hours 1–3) and REM sleep episodes lengthen and concentrate in late sleep cycles (hours 5–8), 5-hour truncation introduces an empirically documented shift in stage prevalence (`AUDIT_ANSWERS.md` Q3):
*   **Deep Sleep (N3)**: Enriched from 18.91% in whole nights (38,451 epochs) to 25.26% in $T=600$ windows (36,701 epochs), an absolute shift of +6.35 percentage points (+33.6% relative increase).
*   **REM Sleep**: Depleted from 26.06% in whole nights (52,987 epochs) to 21.33% in $T=600$ windows (30,992 epochs), an absolute shift of -4.73 percentage points (-18.1% relative decrease; 21,995 REM epochs dropped, accounting for 41.51% of all REM sleep in the dataset).
*   **Wake**: Shifted from 10.83% (22,019 epochs) to 12.72% (18,487 epochs), an absolute shift of +1.90 percentage points (+17.5% relative).
*   **Light Sleep (N1/N2)**: Shifted from 44.20% (89,881 epochs) to 40.68% (59,103 epochs), an absolute shift of -3.52 percentage points (-8.0% relative).

Prior model evaluations under $T=600$ exhibited lower REM recall ($\sim 58\text{–}64\%$) compared to Wake and Deep sleep. It is currently unknown whether this lower recall reflects wrist-sensor limitations or an artifact of truncating late-night training context.

## Model Scope: Model A Solely

*   **Primary Comparison**: Model A (Conventional matched-capacity CNN-BiGRU, 163,940 parameters; identical layer dimensions, kernels, activations).
*   **Exclusion of Model E**: Model E (SleepAdaptiveNet) is explicitly removed from EXP-017 execution to prevent conflating architectural mechanisms (dynamic gating, transition loss, GRL) with sequence horizon effects. Model E is designated strictly as a potential future replication study (e.g., EXP-018) should EXP-017 demonstrate a meaningful effect.

--------------------------------------------------------------------------------

## Formal Protocol Specification (A–P)

### A. Research Question
Does extending the sequence training horizon from $T=600$ to the 10-hour-capped extended sequence condition ($T \le 1200$) materially affect sleep-staging performance when both models are evaluated on the identical first 600 chronological epoch positions, with invalid positions masked?

### B. Primary Hypothesis ($H_1$)
Training with the extended sequence condition ($T \le 1200$) improves validation REM recall relative to $T=600$ when both models are evaluated on the identical first 600 chronological epoch positions, with invalid positions masked.

### C. Null Hypothesis ($H_0$)
Training with the extended sequence condition ($T \le 1200$) produces no improvement or a reduction in validation REM recall relative to $T=600$ on the identical first 600 chronological epoch positions, with invalid positions masked ($\Delta \text{REM Recall} \le 0$).

### D. Primary Estimand
The paired subject-level difference in validation REM recall between the extended sequence model and the truncated sequence model, evaluated strictly on the identical first 600 chronological epoch positions, with invalid positions masked:
$$\Delta \text{REM Recall}_{\text{matched-600}} = \text{REM Recall}_{(T \le 1200)}[\text{eval } 0:600] - \text{REM Recall}_{(T = 600)}[\text{eval } 0:600]$$

### E. Primary Metric
Validation REM Recall (sensitivity for Stage 3/REM) on the first 600 chronological epoch positions, with invalid positions masked:
$$\text{Recall}_{\text{REM}} = \frac{\text{TP}_{\text{REM}}}{\text{TP}_{\text{REM}} + \text{FN}_{\text{REM}}}$$
computed over all valid, unmasked epochs where `target == 3` and `valid == True` within positions $t \in [0, 600)$.

### F. Secondary Metrics
1.  **Classification Performance (Matched 600 Epoch Positions)**: Accuracy, Macro-F1, Cohen's $\kappa$, Matthews Correlation Coefficient (MCC).
2.  **Per-Stage Performance (Matched 600 Epoch Positions)**: Per-class F1 and Recall for Wake (0), Light (1), and Deep (2).
3.  **Boundary vs. Stable Staging (Matched 600 Epoch Positions)**: Staging accuracy on transition epochs ($\text{target}_t \ne \text{target}_{t-1}$) vs. stable epochs ($\text{target}_t = \text{target}_{t-1}$).
4.  **Secondary Descriptive Analysis (Non-Causal Extended Coverage)**: Descriptive reporting of Model A trained on $T \le 1200$ across its extended sequence coverage ($T \le 1200$). This analysis reports per-stage recall and Macro-F1 purely to document longer-recording performance without comparative causal claims against $T=600$ or artificial extrapolation of $T=600$ predictions.
5.  **Exploratory Signal Quality Analysis**: Missingness rate, valid epoch fraction, and sensor coverage metrics (`cov_hr`, `cov_motion`) computed hourly (hours 1–5 vs. hours 6–10) across recording nights to audit late-night data quality without introducing unvalidated SNR constructs.

### G. Exact Changed Variable
**Sequence Training Horizon ($T$)**:
*   **Baseline Condition ($T=600$)**: Exactly the first 600 chronological 30-second epoch positions from `recStart` (18,000 seconds / 5.0 hours), retaining the established project indexing convention.
*   **Extended Sequence Condition ($T \le 1200$)**: Up to the first 1200 chronological 30-second epoch positions from `recStart` (36,000 seconds / 10.0 hours). Nights longer than 1200 epoch positions are capped at 1200.

### H. Exact Controlled Variables
All non-sequence factors are held strictly invariant:
*   **Model Architecture**: Model A (Conventional CNN-BiGRU, 163,940 parameters; identical layer dimensions, kernels, activations).
*   **Cohort & Partitions**: Official inner training split (31 subjects, 160 usable nights) and validation split (5 subjects, 28 usable nights). Split hash: `bf33573623f47f3f`.
*   **Normalization**: Strictly per-night robust IQR scaling computed independently per night (`(v - median) / IQR`).
*   **Optimizer & Learning Rate**: Adam optimizer, $\text{lr} = 10^{-4}$, weight decay $0.0$, no learning rate scheduler.
*   **Batch Policy**: Batch size = 4 nights per step; 1 step per batch. Gradient accumulation, dynamic sequence bucketing, and batch size alterations are strictly prohibited.
*   **Training Budget**: Exactly 80 optimization epochs per run (matching the EXP-010/011 convergence window; 500-epoch runs remain blocked under DEC-019/020 and `governor.py`).
*   **Loss Formulation**: Weighted CrossEntropy / RCCLoss with identical class weights.
*   **Random Seeds**: Seed 42 for primary run; seeds 123 and 2024 for multi-seed validation confirmation.
*   **Hardware Allocation**: Dedicated lease on `cuda:0` (NVIDIA RTX 3060 12GB).
*   **VRAM Limit**: Governed by `governance.md` Section 3 ceiling ($\le 6.0$ GB for tier `STANDARD`).

### I. Sequence Construction and Recurrent Padding Invariant
1.  **Chronological Epoch Selection**: Both conditions select consecutive 30-second epoch positions starting from `recStart` index 0 (`rec["ihr"][:T]`, `rec["y"][:T]`). Invalid epochs (due to recording termination, artifacts, or sensor coverage $< 0.5$) retain their native chronological positions with `mask = False` and `target = -100`. No chronological epoch position is dropped prior to indexing, exactly adhering to the verified codebase implementation in `run_controlled_experiment.py`.
2.  **Capped Length**: $T_{\max} = 1200$ epoch positions ($10.0\text{ hours}$). Nights $> 1200$ epoch positions (maximum observed in cohort is 1,282 epochs) are capped at epoch position 1200 (`[:1200]`).
3.  **Recurrent Padding Invariant (MANDATORY)**: Shorter nights ($L < 1200$) are padded to 1200. To prevent padded timesteps from leaking into recurrent representations:
    *   The recurrent stage **MUST** utilize `torch.nn.utils.rnn.pack_padded_sequence` (with `enforce_sorted=False`) prior to the BiGRU and `torch.nn.utils.rnn.pad_packed_sequence` immediately after, OR
    *   An explicitly mask-gated recurrent update must zero out recurrent hidden state updates at padded timesteps.
    *   **Invariance Requirement**: Padded positions ($t \ge L$) must produce mathematically zero effect on hidden states and predictions at valid positions ($t < L$).
4.  **Loss and Evaluation Masking**: All loss computations and metrics mask out invalid positions via `valid = (targets >= 0) & (targets < 4) & mask`.

### J. Statistical Analysis Plan and N=5 Sample Honest Assessment
1.  **Sample Size Constraint**:
    *   The validation cohort contains **strictly $N=5$ independent human subjects** (28 usable nights total).
    *   A paired cluster bootstrap with $B=1000$ iterations provides a non-parametric estimation of sampling variability across these 5 subjects, but does **NOT** generate 1,000 independent observations or overcome the fundamental limitation of $N=5$.
    *   Evaluating across 3 random seeds (42, 123, 2024) captures algorithmic optimization variance, but seeds are **NOT** additional biological subjects.
2.  **Paired Metrics Reporting**:
    *   Per-subject paired differences $\Delta_s = \text{Metric}_{(T \le 1200)}(s) - \text{Metric}_{(T = 600)}(s)$ will be reported individually for all 5 subjects ($s \in \{1, 2, 3, 4, 5\}$).
    *   Subject-level direction consistency (e.g., $5/5$, $4/5$, or $3/5$ subjects favoring extended sequences) must be explicitly presented alongside aggregate bootstrap metrics.
3.  **Paired Cluster Bootstrap**:
    *   Resampling unit: Subject ($N=5$). Iterations: $B=1000$.
    *   95% confidence interval: empirical percentiles $[\Delta_{2.5\%}, \Delta_{97.5\%}]$.
4.  **Non-Inferiority Margins for Non-Target Stages**:
    *   Wake Recall lower bound: $> -3.0$ percentage points.
    *   Light Recall lower bound: $> -3.0$ percentage points.
    *   Deep Recall lower bound: $> -3.0$ percentage points.
    *   Macro-F1 lower bound: $> -0.010$.
    *   *Provenance of Margins*: These margins are **researcher-defined practical tolerances** established to ensure that potential gains in REM recall do not come at the expense of clinically unacceptable degradation in non-target sleep stages. They are not asserted as objective statistical constants.

### K. Decision Rule
The empirical result will be mapped into one of four mutually exclusive, exhaustive outcomes:

*   **SUPPORTED**:
    1. Positive REM recall effect estimate ($\Delta \text{REM Recall} > 0$), AND
    2. Empirical 95% bootstrap confidence interval strictly excludes zero ($\Delta_{2.5\%} > 0.0$ pp), AND
    3. Practical effect threshold is met (point estimate $\Delta \text{REM Recall} \ge +3.0$ pp), AND
    4. Non-inferiority margins on Wake, Light, Deep, and Macro-F1 are all satisfied.
*   **WEAK / UNCERTAIN POSITIVE**:
    1. 95% bootstrap confidence interval strictly excludes zero ($\Delta_{2.5\%} > 0.0$ pp), BUT
    2. Practical effect threshold is not met ($\Delta \text{REM Recall} < +3.0$ pp), OR non-target stage non-inferiority bounds are breached.
*   **INCONCLUSIVE**:
    1. Empirical 95% bootstrap confidence interval spans zero ($\Delta_{2.5\%} \le 0 \le \Delta_{97.5\%}$), indicating that sampling variance among the 5 validation subjects is too large to discern an effect.
*   **REJECTED**:
    1. Point estimate is non-positive ($\Delta \text{REM Recall} \le 0.0$ pp), AND
    2. Empirical 95% bootstrap confidence interval upper bound is at or below zero ($\Delta_{97.5\%} \le 0.0$ pp).

### L. Interpretation Protocol
*   If the result falls into **INCONCLUSIVE** or **REJECTED**, the scientific conclusion will state:
    > "Hypothesis $H_1$ is not supported under the tested protocol."
*   A non-significant or inconclusive outcome will not be claimed as proof that sequence length has no effect in general; it will be reported as reflecting either true negligible effect size or insufficient statistical power/precision within the $N=5$ validation partition.

### M. Threats to Validity
1.  **Validation Sample Size ($N=5$)**: With only 5 validation subjects, the study has limited power to detect small effect sizes ($< 3$ pp) and confidence intervals will be wide.
2.  **Late-Night Recording Quality**: If sensor contact loosens or motion increases toward morning awakening, missing or corrupted epochs in hours 6–10 could inject noise into the extended-sequence training representations.
3.  **Recurrent Temporal Dispersion**: In a BiGRU of length 1200, gradients propagated across 10 hours could attenuate, potentially shifting representation focus away from early-night transitions.

### N. Required Artifacts
1.  `research/experiments/EXP-017/config.json`: Complete execution and model parameters.
2.  `research/experiments/EXP-017/metrics_matched600.json`: Metrics evaluated on the identical first 600 chronological epoch positions, with invalid positions masked.
3.  `research/experiments/EXP-017/metrics_extended_descriptive.json`: Secondary descriptive metrics across extended coverage for Model A ($T \le 1200$).
4.  `research/experiments/EXP-017/subject_level_paired.csv`: Per-subject metric deltas across all 5 validation subjects.
5.  `research/experiments/EXP-017/training_history.csv`: Per-epoch train/val loss and Macro-F1.
6.  `research/experiments/EXP-017/hourly_coverage_analysis.csv`: Hourly valid epoch fractions and missingness rates.
7.  `EXP017_REPORT.md`: Comprehensive final report documenting matched-600 and extended analyses.

### O. Test-Set Protection Mechanism
1.  `DatasetGate.assert_no_test_leakage`: Mandatory programmatic pre-flight assertion confirming that zero subjects from the 11 held-out test subjects appear in the train or validation partitions.
2.  `SleepEvaluator` Shield: Evaluation script executes strictly with `split_role='val'`. Any attempt to call `split_role='test'` is blocked by the evaluator gate.
3.  Test metrics from EXP-012/013/014 remain in locked directories and are not accessed.

### P. Expected Interpretation for Each Possible Outcome
*   **Outcome 1 (Supported)**: Extended sequence training genuinely improves early-night REM feature representations, confirming that 5-hour sequence truncation was an active experimental limitation. Protocol DEC-022 will be proposed to adopt extended sequence modeling.
*   **Outcome 2 (Inconclusive / Rejected)**: Extended sequence training does not provide a detectable benefit on the first 600 epoch positions. The $T=600$ condition is retained as a computationally efficient representation, redirecting research focus to feature engineering and loss formulations.
*   **Outcome 3 (Weak Positive / Trade-Off)**: Extended sequence training yields marginal REM improvements but degrades Wake or Deep sleep staging, indicating that longer context requires architectural weighting or chunked temporal attention.
