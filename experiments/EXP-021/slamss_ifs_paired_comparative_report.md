# EXP-021 (TSK-08): SLAMSS-IFS Matched 5-Fold Cross-Validation Paired Comparative Report

- **Preregistration:** `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md` (DEC-011, DEC-033, DEC-034)
- **Generated (UTC):** `2026-11-22T02:39:04Z`
- **Independent Participants:** `N = 47` (3-seed participant means across seeds `42, 123, 2024`)
- **Bootstrap Inference:** `10,000` participant-level resamples (seed `42`)
- **Primary Classification:** **`SLAMSS-IFS ADVANTAGE (STATISTICALLY SIGNIFICANT)`**

---

## 1. Primary Estimand & Hardware Efficiency Trade-Off

- **Model A (`ConventionalCNNBiGRU`, 163,940 params, ~160.1 KiB INT8 weights):** **64.79%** (SD 7.45 pp)
- **Official SLAMSS-IFS (5,465,768 repo params / 666,060 paper text):** **67.37%** (SD 6.73 pp)
- **Paired Difference ($\bar{d} = \text{Model A} - \text{SLAMSS-IFS}$):** **-2.58 pp** [95% CI: **-3.53, -1.68 pp**]
- **Wilcoxon Signed-Rank Test (two-sided):** $W = 144.0$, $p = 1.9818e-06$
- **Paired Effect Size (Cohen's $d_z$):** `-0.780`
- **Participant Win / Loss / Tie:** Model A wins `13/47` (27.7%), SLAMSS-IFS wins `34/47`, ties `0`
- **Parameter Compression:** **4.06x** vs. paper text | **33.34x** vs. official repository

> **Interpretation:** SLAMSS-IFS statistically outperforms Model A on matched 5-fold CV. The empirical performance gap and its 95% confidence interval quantify the exact accuracy-vs-efficiency trade-off against 4.06x-33.34x parameter compression.

---

## 2. Complete Primary & Secondary Paired Metrics (N = 47)

| Metric | Model A (Mean ± SD) | SLAMSS-IFS (Mean ± SD) | Paired Diff ($\bar{d}$) | 95% Bootstrap CI | Cohen's $d_z$ | Wilcoxon $p$ | Model A Wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-F1** | 64.79% ± 7.45 | 67.37% ± 6.73 | -2.58 pp | [-3.53, -1.68] pp | -0.780 | `1.9818e-06` | 13/47 |
| **Overall Accuracy** | 66.18% ± 6.79 | 68.75% ± 6.08 | -2.57 pp | [-3.53, -1.65] pp | -0.784 | `3.3783e-06` | 10/47 |
| **Weighted-F1** | 65.96% ± 6.73 | 68.72% ± 5.90 | -2.76 pp | [-3.77, -1.79] pp | -0.800 | `7.4148e-07` | 12/47 |
| **Cohen's Kappa** | 0.5078 ± 0.1011 | 0.5378 ± 0.0941 | -0.0300 | [-0.0434, -0.0169] | -0.643 | `1.3095e-04` | 15/47 |
| **Wake F1** | 62.70% ± 11.05 | 64.25% ± 9.36 | -1.55 pp | [-2.87, -0.19] pp | -0.325 | `2.8726e-02` | 17/47 |
| **Light F1** | 66.61% ± 5.25 | 69.44% ± 5.05 | -2.83 pp | [-3.83, -1.86] pp | -0.829 | `5.9535e-07` | 8/47 |
| **Deep F1** | 66.76% ± 12.31 | 70.27% ± 12.14 | -3.51 pp | [-5.07, -2.12] pp | -0.678 | `6.8193e-06` | 11/47 |
| **REM F1** | 63.09% ± 13.11 | 65.51% ± 11.54 | -2.41 pp | [-4.68, -0.32] pp | -0.317 | `1.0471e-01` | 19/47 |
