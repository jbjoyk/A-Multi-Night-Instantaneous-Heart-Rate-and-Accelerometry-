# Protocol & Preregistration Draft: EXP-019B (Exploratory Heterogeneity Analysis)

**Document Status:** Formal Preregistration Draft (Pending Independent Review & User Authorization)  
**Governance Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029), Section 9 (Step D2), Section 13, Section 14  
**Classification:** Exploratory Hypothesis Testing (Post-Stage C Diagnostic Protocol)  
**Protocol Version:** Draft v0.1  
**Author:** Antigravity / EGM Protocol Architect  
**Draft Date:** 2026-11-18  

---

## 1. Critical Disclosure & Provenance

> [!WARNING]
> **MANDATORY PRIOR-ACCESS DISCLOSURE:**
> This protocol is drafted **after** the execution and inspection of EXP-019 Stage C results. The primary dataset (BIDSleep) and subject-level outcomes ($\Delta = \text{Macro-F1}_{\text{FT}} - \text{Macro-F1}_{\text{ZS}}$, responder rate 34/46, worst-case outlier `Bidslab34` at $-16.34\text{ pp}$) were already inspected during the Phase A diagnostic audit (`HETEROGENEITY_AUDIT_REPORT.md`).
> 
> Therefore, EXP-019B is **strictly an exploratory, hypothesis-generating protocol**. It must NEVER be described as a prospective confirmatory preregistration drafted prior to data observation, nor can its statistical tests provide independent confirmation on the BIDSleep cohort. True confirmation of any mechanism identified here is deferred to external dataset validation (EXP-027 / Phase C).

---

## 2. Research Questions & Theoretical Mechanisms

### Primary Research Question
What patient-level, physiological, or data-structural characteristics explain the substantial inter-subject heterogeneity in response to 2-night supervised fine-tuning ($\Delta \in [-16.34\text{ pp}, +6.82\text{ pp}]$), and can pre-adaptation features reliably forecast adaptation failure?

### Candidate Theoretical Mechanisms (M1–M6)
- **M1 (Stage Prior Shift & Scarcity):** Atypical sleep architecture during the 2 calibration nights (e.g., absence or severe depletion of Deep sleep or REM) causes fine-tuning to suppress corresponding output logits, inducing severe misclassification when normal sleep stages return during subsequent evaluation nights.
- **M2 (Autonomic Stability & HRV Volatility):** High night-to-night variability in baseline cardiac tone or optical PPG noise impairs representation stability.
- **M3 (Mathematical Coupling & Headroom):** Subjects with low zero-shot performance have higher mathematical headroom to improve, while subjects with high baseline performance suffer from ceiling effects or regression to the mean (RTM).
- **M4 (Sleep Architecture & Transition Friction):** Severe sleep fragmentation (high epoch-to-epoch stage transition frequency) degrades sequence-level feature alignment.
- **M5 (Parameter Displacement & Overfitting):** Excessive gradient updates on small calibration batches overfit network weights away from generalizable representations.
- **M6 (Valid Token Exposure):** Lower valid calibration epoch counts (due to sensor dislodgement or motion artifact masking) increase gradient noise.

---

## 3. Cohort Eligibility & Denominators

- **Dataset:** PhysioNet BIDSleep dataset (Song et al., 2026; 47 subjects, 253 nights).
- **Cohort Eligibility:** Exactly **46 subjects** having at least 3 usable nights passing signal quality screening ($n_{\text{usable}} \ge 3$).
- **Exclusions:** Subject `Bidslab49` is formally excluded because it possesses only 2 usable nights in total; using 2 nights for calibration leaves 0 evaluation nights ($n_{\text{eval}} = 0$).
- **Unit of Analysis:** The individual subject ($N = 46$), evaluated across 152 outer evaluation nights pooled across 5 nested cross-validation folds.

---

## 4. Predictor Definitions & Extraction Specifications

All candidate predictors are extracted **blind** to evaluation-night predictions, computed strictly from the first 2 labeled nights (calibration nights) or pre-adaptation zero-shot runs:

| Predictor ID | Variable Name | Theoretical Mechanism | Mathematical Definition / Extraction Formula | Operational Status |
|---|---|---|---|:---:|
| **X1** | `D_JS_pop` | M1: Stage Prior Shift | Base-2 Jensen-Shannon divergence $D_{\text{JS}}(P_C \parallel P_{\text{pop}})$ between 4-class calibration distribution $P_C$ and training cohort empirical prior $P_{\text{pop}}$. | Verified |
| **X2** | `deep_frac_cal` | M1: Deep Scarcity | Fraction of labeled calibration epochs belonging to Deep sleep (N3): $N_{\text{Deep}} / N_{\text{cal}}$. | Verified |
| **X3** | `min_stage_frac_cal` | M1: Minimum Stage Scarcity | Minimum empirical class share during calibration: $\min_{c \in \{0,1,2,3\}} P_C(y=c)$. | Verified |
| **X4** | `hr_volatility_cal` | M2: Autonomic Volatility | Absolute difference in median nightly heart rate between Night 2 and Night 1: $\| \text{median}(\text{HR}_{\text{N2}}) - \text{median}(\text{HR}_{\text{N1}}) \|$. | Verified |
| **X5** | `oldham_mean` | M3: Headroom / RTM | Oldham's adjusted baseline metric: $(\text{Macro-F1}_{\text{ZS}} + \text{Macro-F1}_{\text{FT}}) / 2$, correcting for regression to the mean. | Verified |
| **X6** | `loss_zs_cal` | M3: Calibration Loss Headroom | Mean Cross-Entropy loss of the unadapted Zero-Shot model evaluated on the calibration nights. | Verified |
| **X7** | `transition_rate_cal` | M4: Sleep Fragmentation | Number of stage transitions divided by total labeled calibration epochs: $N_{\text{transitions}} / (N_{\text{cal}} - 1)$. | Verified |
| **X8** | `param_displacement_norm` | M5: Parameter Drift | Relative Euclidean parameter displacement $\| \theta_{\text{adapted}} - \theta_{\text{ZS}} \|_2 / \| \theta_{\text{ZS}} \|_2$, stratified by optimization scope. | Verified |
| **X9** | `n_valid_tokens_cal` | M6: Token Exposure | Total count of valid, non-padded, non-masked 30-second epochs across the 2 calibration nights. | Verified |
| **X10** | `rem_frac_cal` | M1: REM Scarcity | Fraction of labeled calibration epochs belonging to REM sleep: $N_{\text{REM}} / N_{\text{cal}}$. | Verified |

---

## 5. Statistical Hypotheses & Multiplicity Control

### Analysis Model
For each predictor $X_j \in \{X_1, \dots, X_{10}\}$, compute:
1. Two-sided Spearman rank correlation $\rho(X_j, \Delta)$ with adaptation gain $\Delta = \text{Macro-F1}_{\text{FT-2}} - \text{Macro-F1}_{\text{ZS}}$.
2. 95% confidence intervals derived from 10,000 subject-bootstrap resamples.

### Multiplicity Control
Because 10 primary candidate predictors are tested simultaneously against the same response vector $\Delta$:
- **Primary Multiplicity Control:** Benjamini-Hochberg False Discovery Rate (FDR) control at nominal $q < 0.05$ across the 10 hypotheses.
- **Reporting Invariant:** Both raw two-tailed $p$-values and FDR-adjusted $q$-values must be reported side by side.
- **Decision Rule:** Any association with $q \ge 0.05$ must be reported as null / unsupported. An association with $q < 0.05$ is designated as an **exploratory correlation candidate**, not a confirmed causal finding.

---

## 6. Gating & Failure Conditions

- **Gating Check G0 (Inconclusive Baseline Invariant):** The confirmatory hypothesis H-A remains strictly INCONCLUSIVE. EXP-019B cannot be used to retroactively overturn H-A.
- **Gating Check G1 (Zero-Shot Predictor Independence):** Predictors must correlate with the adaptation delta $\Delta$, not merely mirror raw zero-shot baseline capability. Oldham's correlation check is mandatory.
- **Gating Check G2 (Scope Stratification):** Parameter displacement (X8) must be analyzed stratified by optimization scope (`scope=all` vs `scope=head_gru`) to prevent artificial bimodal artifacts from pooling different layer update policies.
- **Inconclusive Threshold:** If all 10 candidate predictors fail to achieve FDR $q < 0.05$, the protocol formally concludes that **observable pre-adaptation calibration metrics do not explain personal adaptation variance**, and the finding must be published as an informative negative result.

---

## 7. Computational Environment & Frozen Checkpoints

- **Backbone Model:** Model A (`ConventionalCNNBiGRU`, 163,940 parameters).
- **Frozen Checkpoints:** `experiments/EXP-018/checkpoints/modelA_L1200_fold{1..5}_seed{42,123,2024}.pt` (15 checkpoints).
- **Predictions Source:** `experiments/EXP-019/exp019_subject_metrics.csv` (SHA-256 verified).
- **Dataset Fingerprint:** `8ae7a916a4cd5709` / `bidslab_47subs_253nights`.
- **Environment:** Python 3.14.7, PyTorch 2.14.1+cu130, CUDA 13.0, NVIDIA RTX 3060.
- **Execution Guard:** This document is a **draft protocol**. Execution is prohibited until explicit user authorization is granted.
