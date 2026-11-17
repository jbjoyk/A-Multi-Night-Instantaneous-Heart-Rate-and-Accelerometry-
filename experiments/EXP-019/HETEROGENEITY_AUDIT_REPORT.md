# Exploratory Diagnostic Audit: Deconstructing Adaptation Heterogeneity (EXP-019)

**Status:** COMPLETE (Exploratory Diagnostic Audit — Zero Modification to Confirmatory H-A Results)
**Cohort Size:** 46 subjects (152 outer evaluation nights across 5 folds)
**Responders:** 34/46 (73.9%) | **Non-Responders:** 12/46

> [!NOTE]
> **Experimental Context & Scope:**
> - **Task & Model:** EXP-019 within-person supervised adaptation of Model A (`ConventionalCNNBiGRU`, $L=1200$) using calibration nights 2–3, evaluated on nights 4+ across 46 subjects (5-fold CV).
> - **Target Metric:** 4-class sleep staging Macro-F1 (Wake, Light, Deep, REM); $\Delta = \text{Macro-F1}_{\text{FT}} - \text{Macro-F1}_{\text{ZS}}$ in percentage points (`pp`).
> - **Responder Criterion:** $\Delta > 0.0\text{ pp}$ (34/46 responders, 12/46 non-responders). Locked confirmatory H-A mean gain was $+1.01\text{ pp}$ (non-significant vs $+1.50\text{ pp}$ practical threshold, $p=0.1706$).
> - **Divergence Metric:** $D_{JS}$ denotes base-2 Jensen-Shannon divergence (bounded in $[0, 1]$) between 4-stage empirical probability vectors ($P_C$ calibration, $P_E$ evaluation, $P_{\text{pop}}$ training cohort prior).

## 1. Primary Exploratory Hypothesis Results (Pre-Adaptation Predictors)

| Mechanism | Hypothesis / Tested Predictor | Predictor Variable | Spearman $\rho$ | $p$-value | FDR $q$-value | Supported ($q < 0.05$)? |
|---|---|---|---|---|---|---|
| **M1: Stage Prior Shift** | **H-D1 (Deployable Label Shift: D_JS_pop vs Delta)** | `D_JS_pop` | **-0.0840** | 0.5787 | 0.9367 | NO |
| **M1: Stage Prior Shift** | **H-D2 (Deep Sleep Scarcity: deep_frac_cal vs Delta)** | `deep_frac_cal` | **+0.0126** | 0.9339 | 0.9367 | NO |
| **M1: Stage Prior Shift** | **H-D3 (Minimum Stage Scarcity: min_stage_frac_cal vs Delta)** | `min_stage_frac_cal` | **+0.1063** | 0.4819 | 0.9367 | NO |
| **M2: Autonomic Volatility** | **H-P1 (HR Volatility: hr_volatility_cal vs Delta)** | `hr_volatility_cal` | **-0.2094** | 0.1625 | 0.9367 | NO |
| **M3: Headroom / RTM** | **H-C1 (Oldham-Corrected Headroom: oldham_mean vs Delta)** | `oldham_mean` | **+0.0376** | 0.8043 | 0.9367 | NO |
| **M3: Headroom / RTM** | **H-C2 (Calibration Loss Headroom: loss_zs_cal vs Delta)** | `loss_zs_cal` | **-0.0373** | 0.8055 | 0.9367 | NO |
| **M4: Transition Friction** | **H-L1 (Transition Friction: transition_rate_cal vs Delta)** | `transition_rate_cal` | **-0.0227** | 0.8810 | 0.9367 | NO |
| **M5: Parameter Drift** | **H-R1 (Parameter Drift Stratified [scope=all, n=37])** | `param_displacement_norm` | **-0.0135** | 0.9367 | 0.9367 | NO |
| **M6: Token Exposure** | **H-S1 (Token Exposure: n_valid_tokens_cal vs Delta)** | `n_valid_tokens_cal` | **-0.0628** | 0.6782 | 0.9367 | NO |

## 2. Key Methodological & Forensic Findings

### A. Prior Probability Shift & Stage Absence (Mechanism 1)
- **Deployable Divergence from Population Prior ($D_{JS}(P_C \parallel P_{\text{pop}})$)**: Spearman $\rho = -0.0840$ ($p = 0.5787$, FDR $q = 0.9367$).
- **Deep Sleep Calibration Scarcity**: Spearman $\rho = +0.0126$ ($p = 0.9339$, FDR $q = 0.9367$).
- **Retrospective Diagnostic Invariance ($D_{JS}(P_C \parallel P_E)$)**: Spearman $\rho = -0.2933$ ($p = 0.0479$). When calibration nights diverge from evaluation nights, adaptation delta exhibits a moderate negative association with downstream gain.
- **Forensic Examination of Worst Outlier (`Bidslab34`, $-16.34\text{ pp}$)**:
  - Calibration Deep Sleep: **0.7%** | Wake: 26.8% | REM: 46.7%
  - $D_{JS}(P_C \parallel P_E)$: **0.3319** (highest divergence in entire cohort)
  - **Diagnostic Case Interpretation (Bidslab34):** Fine-tuning heavily penalized deep sleep output logits because deep sleep was virtually absent during calibration ($0.0\%$ on Night 2, $1.3\%$ on Night 3). When deep sleep returned to ~18% during subsequent evaluation nights, the model systematically underpredicted the stage.

### B. Regression to the Mean vs Headroom (Mechanism 3)
- **Naive Baseline Correlation (Confounded by Mathematical Coupling)**: Spearman $\rho = -0.1710$ ($p = 0.2559$), Pearson $r = -0.0132$ ($p = 0.9304$). Naive correlation is negatively biased by mathematical coupling $\text{Cov}(\text{ZS}, \text{FT} - \text{ZS}) = \text{Cov}(\text{ZS}, \text{FT}) - \text{Var}(\text{ZS})$, but neither metric is statistically distinguishable from zero ($p > 0.25$).
- **Oldham-Corrected Correlation**: Spearman $\rho = +0.0376$ ($p = 0.8043$), Pearson $r = +0.2269$ ($p = 0.1293$). Adjusting for regression to the mean via Oldham's method ($(\text{ZS} + \text{FT})/2$) attenuates the negative association, confirming that baseline Zero-Shot performance does not govern adaptation headroom in this cohort.
- **Pre-Adaptation Calibration Loss Headroom**: Spearman $\rho = -0.0373$ ($p = 0.8055$).

### C. Scope-Stratified Parameter Displacement (Mechanism 5)
- **Confounded Pooled Correlation**: Spearman $\rho = +0.0367$ ($p = 0.8087$).
- **Stratified within `scope=all` ($n=37$, Folds 1, 2, 3, 5)**: Spearman $\rho = -0.0135$ ($p = 0.9367$).
- **Methodological Implication:** Because Fold 4 adapted only the head/GRU (`scope=head_gru`, CNN displacement identically 0.0), pooling creates an artificial bimodal distribution. Within `scope=all`, parameter displacement is further modulated by fold-level optimization budgets (e.g. Fold 5 5-epoch low-lr vs Folds 2–3 15-epoch schedules); after stratification, displacement shows no significant association with adaptation delta.

### D. Uniformly Null Autonomic, Structural, and Exposure Predictors (Mechanisms 2, 4, 6)
- **Autonomic HR Volatility (M2, `H-P1`)**: Spearman $\rho = -0.2094$ ($p = 0.1625$, FDR $q = 0.9367$). Median nightly HR difference between calibration nights provides an exploratory proxy of autonomic baseline stability, but showed no predictive power. (Note: True beat-to-beat HRV metrics like RMSSD cannot be mathematically derived from 1 Hz sampled wearable IHR without interpolation artifacts).
- **Boundary Transition Friction (M4, `H-L1`)**: Spearman $\rho = -0.0227$ ($p = 0.8810$, FDR $q = 0.9367$). Sleep fragmentation and stage switching frequency during calibration nights did not forecast adaptation response.
- **Token Exposure & Coverage (M6, `H-S1`)**: Spearman $\rho = -0.0628$ ($p = 0.6782$, FDR $q = 0.9367$). Total valid calibration epoch count showed no correlation with adaptation success across the cohort.

## 3. Stress-Testing Static Pre-Adaptation Gating (Counterfactual Analysis)

To evaluate whether a static pre-adaptation screening rule could prevent catastrophic degradation, we evaluated a counterfactual decision rule based strictly on calibration data:
- **Evaluated Gating Condition**: $\min_c P_C(y=c) \ge 3.0\%$ AND $D_{JS}(P_C \parallel P_{\text{pop}}) < 0.35$.
- **Cohort Retention**: 44/46 (95.7%) subjects passed the gate; 2/46 subjects were excluded.
- **Mean Adaptation Gain (Passed Subjects)**: **+1.39 pp** (vs un-gated +1.01 pp across all 46 subjects).
- **Mean Delta (Excluded Subjects)**: **-7.54 pp** ($N=2$).
- **Counterfactual Audit & Methodological Limitations:**
  1. *Outlier Exclusion:* The rule excludes extreme outlier `Bidslab34` ($-16.34\text{ pp}$) due to calibration deep sleep frequency $< 3.0\%$.
  2. *False Positive Exclusion:* The rule also excludes `Bidslab15` ($+1.27\text{ pp}$), which is a confirmed positive responder (deep sleep $1.7\%$ on calibration nights), demonstrating that the gate discards genuine adaptation opportunities.
  3. *Low Sensitivity to Non-Responders:* The gate fails to detect 11 of the 12 non-responders in the cohort (including severe degradation cases such as `Bidslab65` [$-6.57\text{ pp}$] and `Bidslab10` [$-6.47\text{ pp}$]). True sensitivity to non-responders is only $1/12 = 8.3\%$.
  4. *Brittle Threshold Boundary:* High responders `Bidslab42` ($+6.31\text{ pp}$, min stage $3.07\%$) and `Bidslab62` ($+4.48\text{ pp}$, min stage $3.17\%$) barely cleared the $3.0\%$ threshold. A slight threshold shift to $3.5\%$ would discard two of the cohort's strongest responders.

> [!IMPORTANT]
> **Diagnostic Takeaway:** This counterfactual stress-test demonstrates that static pre-adaptation screening based on short calibration stage frequencies is an ineffective, blunt filter. Hypotheses for future protocol design (such as prior-preserving loss penalties or test-time adaptation safeguards) remain unverified prospective concepts and are not established by this retrospective audit.
