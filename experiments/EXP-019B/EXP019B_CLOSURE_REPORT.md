# EXP-019B Subject-Level Heterogeneity Closure Report (Exploratory)

> **Prior-Access Notice:** Design-informed exploratory analysis on BIDSleep; no confirmatory claims.

- **Family 1 (`EXP-019` FT-2 - ZS):** N=46 (Responders Δ>0: 34, Non-responders: 12)
- **Family 2 (`EXP-021` Model A - SLAMSS-IFS):** N=47
- **Oldham Headroom Check (Family 1):** rho=0.0376, p=0.8043

## Primary Correlation Summary

| Outcome | Feature | N | Spearman rho | 95% CI | p-value | FDR q (family) | FDR q (pooled) | LOO Bidslab34 rho |
|---|---|---:|---:|---|---:|---:|---:|---:|
| EXP-019_FT2_minus_ZS | `min_stage_frac` | 46 | +0.1073 | [-0.2164, +0.4154] | 0.4778 | 0.8454 | 0.8454 | +0.0464 |
| EXP-019_FT2_minus_ZS | `stage_entropy` | 46 | +0.0073 | [-0.2885, +0.2985] | 0.9614 | 0.9614 | 0.9614 | -0.0603 |
| EXP-019_FT2_minus_ZS | `eff_ihr_hz` | 46 | +0.0968 | [-0.2167, +0.3990] | 0.5221 | 0.8454 | 0.8454 | +0.1233 |
| EXP-019_FT2_minus_ZS | `missingness_rate` | 46 | -0.0874 | [-0.3831, +0.2206] | 0.5636 | 0.8454 | 0.8454 | -0.0385 |
| EXP-019_FT2_minus_ZS | `baseline_hr_mean` | 46 | -0.2466 | [-0.5192, +0.0624] | 0.0985 | 0.5913 | 0.5913 | -0.2182 |
| EXP-019_FT2_minus_ZS | `baseline_hr_iqr` | 46 | -0.0402 | [-0.2918, +0.2216] | 0.7906 | 0.9487 | 0.9614 | -0.0295 |
| EXP-021_ModelA_minus_SLAMSS | `min_stage_frac` | 47 | +0.0169 | [-0.2446, +0.2763] | 0.9103 | 0.9103 | 0.9614 | +0.0178 |
| EXP-021_ModelA_minus_SLAMSS | `stage_entropy` | 47 | +0.2446 | [-0.0587, +0.5204] | 0.0976 | 0.5673 | 0.5913 | +0.2339 |
| EXP-021_ModelA_minus_SLAMSS | `eff_ihr_hz` | 47 | -0.1081 | [-0.3934, +0.1928] | 0.4694 | 0.7042 | 0.8454 | -0.1183 |
| EXP-021_ModelA_minus_SLAMSS | `missingness_rate` | 47 | +0.1478 | [-0.1551, +0.4297] | 0.3215 | 0.6430 | 0.8454 | +0.1408 |
| EXP-021_ModelA_minus_SLAMSS | `baseline_hr_mean` | 47 | +0.0206 | [-0.3083, +0.3605] | 0.8908 | 0.9103 | 0.9614 | +0.0182 |
| EXP-021_ModelA_minus_SLAMSS | `baseline_hr_iqr` | 47 | +0.1950 | [-0.0965, +0.4580] | 0.1891 | 0.5673 | 0.7564 | +0.1949 |
