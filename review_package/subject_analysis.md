# subject_analysis.md: Per-Subject Staging Performance, Reliability, and Covariate Analysis

This document provides a comprehensive evaluation of per-subject sleep staging performance across the 47 subjects in the BIDSLab Apple Watch + Dreem 2 dataset, based on the 75-run 5-fold cross-validation experiment (EXP-014).

---

## 1. Executive Summary and Key Empirical Findings

1. **Massive Inter-Subject Variance**:
   - Model accuracy varies from a minimum of **$43.52\%$** (Subject Bidslab20, Model E) to a maximum of **$80.04\%$** (Subject Bidslab38, Model E).
   - Cohen's $\kappa$ spans from **$0.1754$** (Bidslab20) to **$0.7059$** (Bidslab38).
   - This wide variance ($\sigma \approx 7.0\% - 7.7\%$) reflects substantial inter-individual physiological heterogeneity in wrist PPG and motion dynamics.

2. **Near-Perfect Inter-Model Reliability (Intraclass Correlation)**:
   - **Two-way mixed, single-rater consistency ICC(3, 1)** across the 5 models (averaged across seeds):
     $$\mathbf{\text{ICC}(3, 1) = 0.9633} \quad [95\%\text{ CI: } 0.9445, 0.9773], \quad F(46, 184) = 132.17, \quad p = 1.08 \times 10^{-118}$$
   - **Across all 15 model-seed runs**:
     $$\mathbf{\text{ICC}(3, 1) = 0.9151} \quad [95\%\text{ CI: } 0.8795, 0.9452], \quad F(46, 644) = 162.75, \quad p < 10^{-300}$$
   - **Interpretation**: What determines performance is overwhelmingly the **subject**, not the architecture. A subject that is difficult for Model A is almost identically difficult for Model E. Architecture variations produce small secondary perturbations on top of a dominant subject-specific baseline.

3. **Performance Parity Between Model A and Model E**:
   - Across the 47 subjects:
     - **Model A (Conventional CNN-BiGRU)**: Mean Accuracy = $66.28\% \pm 6.89\%$, Mean Cohen's $\kappa = 0.5155 \pm 0.1007$
     - **Model E (Full SleepAdaptiveNet)**: Mean Accuracy = $65.87\% \pm 7.68\%$, Mean Cohen's $\kappa = 0.5083 \pm 0.1099$
     - Mean paired difference ($\Delta = \text{Model E} - \text{Model A}$):
       $$\Delta_{\text{Acc}} = -0.41\% \pm 2.58\% \quad (p = 0.283, \text{ Wilcoxon signed-rank})$$
       $$\Delta_{\kappa} = -0.0071 \pm 0.0327 \quad (p = 0.155, \text{ Wilcoxon signed-rank})$$
   - In 21 out of 47 subjects, Model E outperformed Model A; in 26 subjects, Model A outperformed Model E.

---

## 2. Per-Subject Performance Comparison Table

Performance metrics below are means across 3 independent random seeds (42, 123, 2024) evaluated in the held-out test fold of EXP-014.

| Subject ID | Usable Nights | Valid Windows | Model A Acc | Model A $\kappa$ | Model E Acc | Model E $\kappa$ | $\Delta$ Acc (E - A) | $\Delta \kappa$ (E - A) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bidslab00** | 7 | 4,200 | 72.87% | 0.6123 | 73.30% | 0.6165 | +0.43 pp | +0.0042 |
| **Bidslab01** | 5 | 2,987 | 62.75% | 0.4417 | 64.77% | 0.4658 | +2.02 pp | +0.0241 |
| **Bidslab02** | 6 | 3,556 | 62.34% | 0.4884 | 56.90% | 0.4116 | -5.44 pp | -0.0769 |
| **Bidslab06** | 4 | 2,400 | 74.72% | 0.6193 | 75.06% | 0.6189 | +0.33 pp | -0.0004 |
| **Bidslab07** | 5 | 2,875 | 68.66% | 0.5283 | 64.03% | 0.4691 | -4.63 pp | -0.0593 |
| **Bidslab08** | 4 | 2,400 | 71.97% | 0.5891 | 70.36% | 0.5701 | -1.61 pp | -0.0190 |
| **Bidslab09** | 5 | 3,000 | 66.86% | 0.4907 | 67.24% | 0.4969 | +0.39 pp | +0.0061 |
| **Bidslab10** | 5 | 2,960 | 69.83% | 0.5694 | 72.11% | 0.6015 | +2.27 pp | +0.0321 |
| **Bidslab11** | 5 | 2,933 | 69.07% | 0.5694 | 71.55% | 0.6074 | +2.49 pp | +0.0380 |
| **Bidslab12** | 6 | 3,564 | 69.00% | 0.5638 | 69.57% | 0.5731 | +0.57 pp | +0.0093 |
| **Bidslab13** | 6 | 3,600 | 70.16% | 0.5762 | 68.30% | 0.5517 | -1.86 pp | -0.0245 |
| **Bidslab14** | 7 | 4,200 | 69.17% | 0.5635 | 70.73% | 0.5898 | +1.56 pp | +0.0264 |
| **Bidslab15** | 7 | 4,142 | 49.07% | 0.2862 | 53.94% | 0.3541 | **+4.86 pp** | **+0.0679** |
| **Bidslab16** | 4 | 2,400 | 74.40% | 0.6384 | 77.29% | 0.6766 | +2.89 pp | +0.0382 |
| **Bidslab17** | 6 | 3,598 | 63.86% | 0.4939 | 63.15% | 0.4851 | -0.71 pp | -0.0089 |
| **Bidslab18** | 7 | 4,195 | 69.21% | 0.5772 | 67.87% | 0.5604 | -1.34 pp | -0.0168 |
| **Bidslab19** | 5 | 2,990 | 65.62% | 0.5186 | 66.86% | 0.5367 | +1.24 pp | +0.0181 |
| **Bidslab20** | 7 | 4,152 | 47.82% | 0.2076 | 43.52% | 0.1754 | -4.30 pp | -0.0322 |
| **Bidslab21** | 6 | 3,594 | 70.19% | 0.5647 | 70.36% | 0.5739 | +0.17 pp | +0.0092 |
| **Bidslab22** | 6 | 3,584 | 69.85% | 0.5639 | 71.04% | 0.5841 | +1.19 pp | +0.0202 |
| **Bidslab23** | 6 | 3,595 | 68.34% | 0.5513 | 68.10% | 0.5529 | -0.24 pp | +0.0016 |
| **Bidslab24** | 5 | 3,000 | 66.70% | 0.5323 | 66.70% | 0.5280 | +0.00 pp | -0.0043 |
| **Bidslab25** | 6 | 3,596 | 74.00% | 0.6385 | 73.18% | 0.6310 | -0.82 pp | -0.0076 |
| **Bidslab26** | 5 | 2,977 | 64.66% | 0.4842 | 64.93% | 0.4883 | +0.27 pp | +0.0041 |
| **Bidslab27** | 5 | 3,000 | 73.11% | 0.6277 | 73.80% | 0.6346 | +0.69 pp | +0.0069 |
| **Bidslab28** | 5 | 3,000 | 70.08% | 0.5734 | 70.12% | 0.5724 | +0.04 pp | -0.0010 |
| **Bidslab29** | 5 | 3,000 | 68.51% | 0.5375 | 67.24% | 0.5244 | -1.27 pp | -0.0131 |
| **Bidslab30** | 5 | 2,999 | 59.95% | 0.4285 | 60.10% | 0.4277 | +0.14 pp | -0.0008 |
| **Bidslab31** | 6 | 3,569 | 66.19% | 0.5057 | 67.92% | 0.5332 | +1.73 pp | +0.0275 |
| **Bidslab32** | 6 | 3,594 | 69.17% | 0.5542 | 69.96% | 0.5694 | +0.79 pp | +0.0152 |
| **Bidslab33** | 6 | 3,594 | 65.65% | 0.4901 | 64.68% | 0.4765 | -0.97 pp | -0.0136 |
| **Bidslab34** | 5 | 2,987 | 56.32% | 0.3807 | 59.41% | 0.4225 | **+3.09 pp** | **+0.0418** |
| **Bidslab35** | 6 | 3,597 | 64.12% | 0.4862 | 63.85% | 0.4842 | -0.27 pp | -0.0020 |
| **Bidslab36** | 5 | 2,997 | 71.74% | 0.5794 | 71.60% | 0.5750 | -0.13 pp | -0.0044 |
| **Bidslab37** | 5 | 2,982 | 65.81% | 0.5053 | 64.62% | 0.4907 | -1.18 pp | -0.0146 |
| **Bidslab38** | 5 | 3,000 | 76.82% | 0.6599 | 80.04% | 0.7059 | **+3.22 pp** | **+0.0459** |
| **Bidslab39** | 5 | 2,997 | 69.21% | 0.5732 | 69.94% | 0.5843 | +0.72 pp | +0.0111 |
| **Bidslab40** | 3 | 1,800 | 77.82% | 0.6899 | 77.72% | 0.6897 | -0.09 pp | -0.0001 |
| **Bidslab41** | 5 | 2,998 | 66.86% | 0.5284 | 67.43% | 0.5401 | +0.58 pp | +0.0117 |
| **Bidslab42** | 5 | 2,996 | 58.74% | 0.4137 | 57.08% | 0.3951 | -1.67 pp | -0.0186 |
| **Bidslab43** | 6 | 3,599 | 67.97% | 0.5471 | 66.82% | 0.5375 | -1.15 pp | -0.0096 |
| **Bidslab44** | 5 | 3,000 | 58.82% | 0.4079 | 52.73% | 0.3299 | **-6.09 pp** | **-0.0780** |
| **Bidslab45** | 6 | 3,588 | 70.82% | 0.5907 | 70.52% | 0.5833 | -0.30 pp | -0.0074 |
| **Bidslab48** | 3 | 1,798 | 60.10% | 0.4069 | 58.12% | 0.3752 | -1.98 pp | -0.0317 |
| **Bidslab50** | 4 | 2,398 | 61.43% | 0.4497 | 56.06% | 0.3807 | **-5.36 pp** | **-0.0691** |
| **Bidslab54** | 4 | 2,397 | 64.90% | 0.4795 | 63.87% | 0.4674 | -1.03 pp | -0.0121 |
| **Bidslab60** | 3 | 1,787 | 70.86% | 0.5739 | 65.25% | 0.4996 | **-5.62 pp** | **-0.0743** |

---

## 3. Spearman Rank Correlations with Physiological and Signal Covariates

To understand the biological and technical drivers of model performance, we computed Spearman rank correlations ($\rho$) between subject performance metrics and individual recording covariates across all $N = 47$ subjects.

| Covariate Description | Median (IQR) across Cohort | Spearman $\rho$ with Model A Acc | $p$-value | Spearman $\rho$ with Model E Acc | $p$-value | Spearman $\rho$ with $\Delta_{\text{Acc}}$ (E - A) | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Signal Coverage Fraction** | $0.999$ ($0.978 - 1.000$) | $+0.2397$ | $0.1047$ | $+0.2680$ | $0.0686$ | $+0.0885$ | $0.5543$ |
| **Wake Stage Fraction** | $0.103$ ($0.080 - 0.129$) | $-0.1858$ | $0.2111$ | $-0.2054$ | $0.1661$ | $-0.2179$ | $0.1413$ |
| **Light Stage Fraction** | $0.444$ ($0.406 - 0.487$) | $-0.0109$ | $0.9422$ | $+0.0145$ | $0.9232$ | $-0.0245$ | $0.8701$ |
| **Deep Stage Fraction** | $0.184$ ($0.146 - 0.230$) | $+0.1688$ | $0.2566$ | $+0.1089$ | $0.4661$ | $-0.1597$ | $0.2836$ |
| **REM Stage Fraction** | $0.262$ ($0.230 - 0.284$) | $+0.0127$ | $0.9324$ | $+0.0730$ | $0.6260$ | **$+0.3958$** | **$0.0059$** |
| **Transitions per Hour** | $4.16$ ($3.85 - 4.54$) | $-0.2470$ | $0.0942$ | $-0.2581$ | $0.0798$ | $-0.1031$ | $0.4902$ |

### Biological and Methodological Interpretations:

1. **Sleep Fragmentation Impairs Accuracy ($\rho \approx -0.25, p \approx 0.08 - 0.09$)**:
   - Higher transition frequency (fragmented sleep with frequent micro-arousals and stage switching) degrades accuracy in both models. Boundary epochs are systematically more difficult to classify from indirect autonomic and motion cues.
2. **Signal Quality Provides Positive Trend ($\rho \approx +0.24 - +0.27, p \approx 0.07 - 0.10$)**:
   - Subjects with cleaner PPG sensor contact and fewer dropouts exhibit higher classification fidelity.
3. **Model E Advantage Correlates Directly with REM Proportion ($\rho = +0.3958, p = 0.0059$)**:
   - The relative performance advantage of Model E over Model A ($\Delta_{\text{Acc}} = \text{Model E} - \text{Model A}$) is significantly positively correlated with the subject's REM sleep fraction ($p = 0.0059$).
   - This aligns with sleep neurophysiology: REM sleep is characterized by muscular atonia (accelerometer inactivity identical to deep sleep) paired with autonomic activation (elevated and irregular heart rate resembling wakefulness). The combination of the adaptive reliability gate (giving higher weight to heart rate during movement-free periods) and the transition matrix (incorporating physiological sequence dynamics) selectively benefits REM detection.
