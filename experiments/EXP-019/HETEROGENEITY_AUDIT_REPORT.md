# Exploratory Diagnostic Audit: Deconstructing Adaptation Heterogeneity (EXP-019)

**Status:** COMPLETE (Exploratory Diagnostic Audit — Zero Modification to Confirmatory H-A Results)
**Cohort Size:** 46 subjects (152 outer evaluation nights across 5 folds)
**Responders:** 34/46 (73.9%) | **Non-Responders:** 12/46

## 1. Primary Exploratory Hypothesis Results (Pre-Adaptation Predictors)

| Hypothesis / Tested Mechanism | Predictor Variable | Spearman $\rho$ | $p$-value | FDR $q$-value | Supported ($q < 0.05$)? |
|---|---|---|---|---|---|
| **H-D1 (Deployable Label Shift: D_JS_pop vs Delta)** | `D_JS_pop` | **-0.0840** | 0.5787 | 0.9711 | NO |
| **H-D2 (Deep Sleep Scarcity: deep_frac_cal vs Delta)** | `deep_frac_cal` | **+0.0126** | 0.9339 | 0.9711 | NO |
| **H-D3 (Minimum Stage Scarcity: min_stage_frac_cal vs Delta)** | `min_stage_frac_cal` | **+0.1063** | 0.4819 | 0.9711 | NO |
| **H-P1 (HR Volatility: hr_volatility_cal vs Delta)** | `hr_volatility_cal` | **-0.2094** | 0.1625 | 0.9711 | NO |
| **H-C1 (Oldham-Corrected Headroom: oldham_mean vs Delta)** | `oldham_mean` | **+0.0376** | 0.8043 | 0.9711 | NO |
| **H-C2 (Calibration Loss Headroom: loss_zs_cal vs Delta)** | `loss_zs_cal` | **-0.0373** | 0.8055 | 0.9711 | NO |
| **H-L1 (Transition Friction: transition_rate_cal vs Delta)** | `transition_rate_cal` | **-0.0227** | 0.8810 | 0.9711 | NO |
| **H-R1 (Parameter Drift Stratified [scope=all, n=37])** | `param_displacement_norm` | **-0.0062** | 0.9711 | 0.9711 | NO |
| **H-S1 (Token Exposure: n_valid_tokens_cal vs Delta)** | `n_valid_tokens_cal` | **-0.0628** | 0.6782 | 0.9711 | NO |

## 2. Key Methodological & Forensic Findings

### A. Prior Probability Shift & Stage Absence (Mechanism 1)
- **Deployable Divergence from Population Prior ($D_{JS}(P_C \parallel P_{\text{pop}})$)**: Spearman $\rho = -0.0840$ ($p = 0.5787$, FDR $q = 0.9711$).
- **Deep Sleep Calibration Scarcity**: Spearman $\rho = +0.0126$ ($p = 0.9339$, FDR $q = 0.9711$).
- **Retrospective Diagnostic Invariance ($D_{JS}(P_C \parallel P_E)$)**: Spearman $\rho = -0.2933$ ($p = 0.0479$). When calibration nights diverge from evaluation nights, adaptation drops sharply.
- **Forensic Examination of Worst Outlier (`Bidslab34`, $-16.34\text{ pp}$)**:
  - Calibration Deep Sleep: **0.7%** | Wake: 26.8% | REM: 46.7%
  - $D_{JS}(P_C \parallel P_E)$: **0.3319** (highest divergence in entire cohort)
  - *Causal Diagnosis*: Fine-tuning drove Deep sleep emission bias to negative infinity because Deep sleep was virtually absent in calibration ($0.0\%$ on Night 2, $1.3\%$ on Night 3). When Deep sleep returned to ~18% on subsequent nights, the model systematically failed to classify it.

### B. Regression to the Mean vs Headroom (Mechanism 3)
- **Naive Baseline Correlation (RTM Confounded)**: $\rho = -0.1710$ ($p = 0.2559$). Naive correlation is negative purely due to measurement error variance $\text{Cov}(ZS, FT - ZS) = -\text{Var}(\epsilon) < 0$.
- **Oldham-Corrected Correlation**: $\rho = +0.0376$ ($p = 0.8043$). After adjusting for RTM via Oldham's method, the negative correlation disappears/attenuates, proving that the apparent 'ceiling effect' is partially an RTM artifact!
- **Pre-Adaptation Calibration Loss Headroom**: $\rho = -0.0373$ ($p = 0.8055$).

### C. Scope-Stratified Parameter Displacement (Mechanism 5)
- **Confounded Pooled Correlation**: $\rho = +0.0441$ ($p = 0.7712$).
- **Stratified within `scope=all` ($n=37$, Folds 1, 2, 3, 5)**: $\rho = -0.0062$ ($p = 0.9711$).
- *Methodological Implication*: Because Fold 4 adapted only the head/GRU (`scope=head_gru`, CNN displacement identically 0.0), pooling creates an artificial bimodal distribution. Stratification reveals true parameter movement dynamics.

## 3. Deployment Gating Simulation (Exploratory)

Simulating a zero-leakage pre-adaptation safety gate based strictly on calibration data:
- **Safety Condition**: $\min_c P_C(y=c) \ge 3.0\%$ AND $D_{{JS}}(P_C \parallel P_{{\text{{pop}}}}) < 0.35$.
- **Subjects Passed Gate**: 44/46 (95.7%)
- **Mean Adaptation Gain Among Passed Subjects**: **+1.39 pp** (vs un-gated +1.01 pp)
- **Mean Delta Among Rejected Subjects**: **-7.54 pp**
- **Severe Negative Outliers Blocked**: `Bidslab34` ($-16.34\text{ pp}$) was **REJECTED** by the gate because its Deep sleep was $< 3.0\%$!

*(Note: This gating simulation is strictly exploratory proof-of-concept; no confirmatory standing).*
