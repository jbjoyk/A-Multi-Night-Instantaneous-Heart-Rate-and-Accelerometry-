# Master Results Summary (`DEC-035` – `DEC-044`)

Self-contained reference tables for manual manuscript writing. All values are extracted directly from ratified repository decision records (`DEC-035`–`DEC-044`) and locked experiment artifacts.

---

## Bundle Contents

- **Figures (`300 DPI` PNG + Vector PDF):**
  - `fig_architectures_baseline_vs_proposed.{pdf,png}`
  - `fig_dataset_splits.{pdf,png}`
  - `fig_confusion_matrices.{pdf,png}`
  - `fig_latency_memory_comparison.{pdf,png}`
  - `fig_exp030_exp031_pareto.{pdf,png}`
- **Machine-Readable Data (`data/`):**
  - `confusion_matrices.json` & `confusion_matrices_long.csv`
  - `dec035_dec039_k_learning_curve_clean.csv`
  - `dec038_slamss_paired_subject_deltas.csv`
  - `dec040_exp019b_correlation_summary.csv`
  - `dec041_exp028_sleepaccel_bland_altman.csv`
  - `dec044_exp031_width_table.csv`

---

## 1. Model Architectures, Memory Footprint & CPU Latency (`DEC-037`, `DEC-038`, `DEC-039`, `DEC-042`, `DEC-043`, `DEC-044`)

| Model | Experiment / Role | Total Params | Compression vs. SLAMSS-IFS | FP32 Payload (KiB) | Dynamic INT8 In-Memory (KiB) | INT8 Serialized (KiB) | MMACs ($L=1200$) | 1-Thread CPU INT8 (ms/night) | 4-Thread CPU INT8 (ms/night) | FP32 vs. INT8 OOF $\bar{\Delta}$ (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`p1_100k`** | `EXP-030` Pareto Selection | 98,940 | 55.24× | 386.48 | **182.00** | 207.97 | 456.1 | 56.58 | 46.35 | $-0.0134$ ($p=0.4656$) |
| **`Model A`** | `DEC-037` Frozen Baseline 1 | 163,940 | 33.34× | 640.39 | **309.54** | 339.99 | 794.1 | 69.39 | 47.73 | $+0.0011$ ($p=0.7892$) |
| **`full_v5`** | `EXP-029` FiLM + Pyramid | 190,788 | 28.65× | 745.27 | **360.50** | 388.62 | — | 67.31 | 44.02 | $-0.0113$ ($p=0.9041$) |
| **`P190`** | `EXP-031` Section 9 Selection | 190,940 | 28.63× | 745.86 | **375.16** | 402.91 | 984.2 | 73.30 | 53.07 | $+0.0244$ ($p=0.0836$) |
| **`Official SLAMSS-IFS`** | `EXP-021` Baseline 2 | 5,465,768 | 1.00× | 21,350.66 | **19,960.51** | 20,005.65 | — | — | 3,373.83 | — |

---

## 2. `DEC-035` & `DEC-039`: Calibration-Depth Learning Curve (`TSK-06 / EXP-019-K`, Clean-ZS Reconciled)

### 2A. Primary Common Cohort ($N = 33$, Fixed $E_4 = 64$ Nights, Clean Zero-Shot $= 63.54\%$)

| Depth $k$ | Clean ZS (%) | FT Macro-F1 (%) | $\bar{\Delta}$ (pp) | 95% Bootstrap CI (pp) | Wilcoxon $W$ | $p_{\text{exact, 2s}}$ | $p_{\text{Holm}}$ (5-test $E_4$) | Wins / Losses |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| $k=1$ | 63.54 | 65.28 | **+1.75** | $[+0.63, +2.93]$ | 136.0 | 0.00879 | 0.02637 | 24 / 9 |
| $k=2$ | 63.54 | 65.41 | **+1.88** | $[+0.59, +3.25]$ | 123.0 | 0.00402 | 0.01609 | 24 / 9 |
| $k=3$ | 63.54 | 65.82 | **+2.29** | $[+0.69, +3.97]$ | 144.0 | 0.01369 | 0.02637 | 24 / 9 |
| $k=4$ | 63.54 | 66.25 | **+2.71** | $[+1.17, +4.33]$ | 110.0 | 0.00170 | 0.00848 | 24 / 9 |

- **Page's $L$-Trend Test ($k=1\dots4$):** $L = 863.0$ ($\mu_L = 825.0, \sigma_L = 16.58$), $z = +2.261$, $p = 0.01187$ ($p_{\text{Holm}} = 0.02637$ in $E_4$ all-$k$ family; $p_{\text{Holm}} = 0.05935$ two-sided / $0.04748$ one-sided in strict §3.4 family).
- **Incremental Steps ($E_4$):** $\delta_{2-1} = +0.13\text{ pp}$ $[-0.98, +1.26]$ ($p=0.4688$); $\delta_{3-2} = +0.41\text{ pp}$ $[-0.37, +1.20]$ ($p=0.3303$); $\delta_{4-3} = +0.42\text{ pp}$ $[+0.08, +0.78]$ ($p_{\text{2s}}=0.02660$, $p_{\text{sat}(<0.50)}=0.26249$).

### 2B. Sensitivity Available Cohort ($E_k$, $N \in \{46, 46, 42, 33\}$)

| Depth $k$ | $N$ | Clean ZS (%) | FT Macro-F1 (%) | $\bar{\Delta}$ (pp) | 95% Bootstrap CI (pp) | Wilcoxon $W$ | $p_{\text{exact, 2s}}$ | Wins / Losses |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| $k=1$ | 46 | 63.95 | 64.18 | +0.23 | $[-0.59, +1.05]$ | 491.0 | 0.59568 | 25 / 21 |
| $k=2$ | 46 | 64.58 | 65.58 | +0.99 | $[-0.18, +1.99]$ | 271.0 | 0.00271 | 33 / 13 |
| $k=3$ | 42 | 64.25 | 65.67 | +1.42 | $[+0.15, +2.58]$ | 213.0 | 0.00232 | 29 / 13 |
| $k=4$ | 33 | 63.54 | 66.25 | +2.71 | $[+1.17, +4.33]$ | 110.0 | 0.00170 | 24 / 9 |

---

## 3. `DEC-036`: Unsupervised InfoMax Adaptation Re-Run (`TSK-07 / EXP-019-S3`, $N = 46$)

| Metric | Value |
|---|---|
| Pre-flight Quality Gate | 81,233 cal epochs, **97.26%** valid ($>95\%$), min **606** epochs/night ($\ge 600$), **0** leakage |
| Anti-Collapse Invariance | 0/138 runs collapsed; mean final marginal entropy $= 1.382\text{ nats}$ ($\ln 4 = 1.3863$) |
| Zero-Shot ($ZS$) Macro-F1 | $64.58\%$ |
| Unsupervised ($UA\text{-}2$) Macro-F1 | $64.79\%$ |
| Paired Delta ($UA\text{-}2 - ZS$) | **+0.21 pp** [95% CI: $+0.04, +0.39\text{ pp}$] |
| Wilcoxon Signed-Rank | $p_{\text{two-sided}} = 0.0645$, $p_{\text{superiority}} = 0.0322$ |
| Subject Win Rate | 27 / 46 (58.7%) |
| Official Verdict | **SUB-BENCHMARK GAIN** ($p_{\text{sup}} < 0.05$, CI low $> 0$, below $+1.00\text{ pp}$ benchmark) |

---

## 4. `DEC-037` & `DEC-038`: Stage 2 Freeze & Stage 3 Matched Benchmark (`EXP-021`, BIDSleep $N = 47$)

| Endpoint | Frozen Model A (163,940) | Official SLAMSS-IFS (5,465,768) | Paired $\bar{d}$ (A − SLAMSS) | 95% Bootstrap CI | Cohen's $d_z$ | Wilcoxon $p$ | Model A Wins |
|---|---:|---:|---:|---|---:|---:|---:|
| **Macro-F1 (%)** | $64.79 \pm 7.45$ | $67.37 \pm 6.73$ | **$-2.58\text{ pp}$** | $[-3.53, -1.68]$ | $-0.780$ | $1.98 \times 10^{-6}$ | 13 / 47 |
| **Accuracy (%)** | $66.18 \pm 6.79$ | $68.75 \pm 6.08$ | $-2.57\text{ pp}$ | $[-3.53, -1.65]$ | $-0.784$ | $3.38 \times 10^{-6}$ | 10 / 47 |
| **Weighted-F1 (%)** | $65.96 \pm 6.73$ | $68.72 \pm 5.90$ | $-2.76\text{ pp}$ | $[-3.77, -1.79]$ | $-0.800$ | $7.41 \times 10^{-7}$ | 12 / 47 |
| **Cohen's $\kappa$** | $0.5078 \pm 0.1011$ | $0.5378 \pm 0.0941$ | $-0.0300$ | $[-0.0434, -0.0169]$ | $-0.643$ | $1.31 \times 10^{-4}$ | 15 / 47 |
| **Wake F1 (%)** | $62.70 \pm 11.05$ | $64.25 \pm 9.36$ | $-1.55\text{ pp}$ | $[-2.87, -0.19]$ | $-0.325$ | $0.0287$ | 17 / 47 |
| **Light F1 (%)** | $66.61 \pm 5.25$ | $69.44 \pm 5.05$ | $-2.83\text{ pp}$ | $[-3.83, -1.86]$ | $-0.829$ | $5.95 \times 10^{-7}$ | 8 / 47 |
| **Deep F1 (%)** | $66.76 \pm 12.31$ | $70.27 \pm 12.14$ | $-3.51\text{ pp}$ | $[-5.07, -2.12]$ | $-0.678$ | $6.82 \times 10^{-6}$ | 11 / 47 |
| **REM F1 (%)** | $63.09 \pm 13.11$ | $65.51 \pm 11.54$ | $-2.41\text{ pp}$ | $[-4.68, -0.32]$ | $-0.317$ | **0.1047 (NS)** | 19 / 47 |

---

## 5. `DEC-040`: Exploratory Heterogeneity (`EXP-019B`) & Regularized/Gated Adaptation (`EXP-020`)

### 5A. `EXP-019B` Heterogeneity Predictors
- **Result:** **0 / 12** primary correlation tests reached BH-FDR $q < 0.05$ across 6 predictors (`min_stage_frac`, `stage_entropy`, `eff_ihr_hz`, `missingness_rate`, `baseline_hr_mean`, `baseline_hr_iqr`) against `EXP-019` ($N=46$) and `EXP-021` ($N=47$); all $q_{\text{family}} \ge 0.5673$, $q_{\text{pooled}} \ge 0.5913$; Oldham headroom check $\rho = +0.0376$ ($p = 0.8043$).

### 5B. `EXP-020` Regularized & Gated Adaptation (Model A, $N = 46$, Exploratory)

| Arm | Macro-F1 | $\bar{\Delta}$ vs. ZS (pp) | 95% CI (pp) | $p_{\text{Holm}}$ | $\bar{\Delta}$ vs. FT-2 (pp) | Harmed ($N_{\text{harm}}$) | Worst Outlier (pp) | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---|
| `FT-2-KL` | 0.6562 | +1.03 | $[-0.11, +2.01]$ | 0.0030 | +0.02 | 3 | $-15.14$ | KL regularization |
| **`GATE_tau`** | **0.6591** | **+1.33** | **$[+0.49, +2.12]$** | **0.0028** | **+0.32** | **2** | **$-6.40$** | `Bidslab34` abstained ($\text{min\_stage}=0.0068 < \tau^*=0.01$) |
| `FT-2-KL_GATE_tau` | 0.6559 | +1.00 | — | 0.0030 | — | — | — | Inner-val gate skipped Folds 2, 3, 5 |

---

## 6. `DEC-041`: Zero-Shot External Validation on PhysioNet SleepAccel (`EXP-027` & `EXP-028`)

### 6A. `EXP-027` 4-Class Staging (`Model A` vs. `Official SLAMSS-IFS`, 15+15 Checkpoints)

| Cohort | Eval Mode | Model A Macro-F1 (%) | SLAMSS-IFS Macro-F1 (%) | $\bar{\Delta}$ (pp) | 95% Bootstrap CI | $d_z$ | Wilcoxon $p$ | Wins (A/S) |
|---|---|---:|---:|---:|---|---:|---:|---|
| `qc_clean` ($n=25$) | `mean_single_model` (Primary) | 60.51 | 61.08 | $-0.57$ | $[-2.13, +0.94]$ | $-0.14$ | 0.5782 | 10 / 15 |
| `qc_clean` ($n=25$) | `prob_ensemble` (Secondary) | **62.94** | **62.79** | **+0.15** | $[-1.51, +1.86]$ | $+0.03$ | 0.9789 | 10 / 15 |
| `all` ($N=31$) | `mean_single_model` | 60.65 | 62.27 | $-1.62$ | $[-3.84, +0.28]$ | $-0.27$ | 0.3468 | 13 / 18 |
| `all` ($N=31$) | `prob_ensemble` | 62.94 | 64.14 | $-1.20$ | $[-3.96, +1.04]$ | $-0.17$ | 0.5551 | 13 / 18 |

### 6B. `EXP-028` External Single-Night PSG Bland-Altman Agreement (`qc_clean` $n = 25$, `mean_single_model`)

| Sleep Metric | Model A Bias | Model A 95% LoA | Model A MAE | Model A $r$ | SLAMSS-IFS Bias | SLAMSS-IFS 95% LoA | SLAMSS-IFS MAE | SLAMSS-IFS $r$ |
|---|---:|---|---:|---:|---:|---|---:|---:|
| `tst_min` | $+11.81$ min | $[-23.79, +47.40]$ | 15.87 min | 0.9091 | $+14.59$ min | $[-21.68, +50.87]$ | 19.11 min | 0.9042 |
| `sol_min` | $+2.71$ min | $[-10.91, +16.33]$ | 5.73 min | 0.6607 | $+0.53$ min (med $+1.73$) | $[-18.73, +10.29]$ (NP) | 5.53 min | — |
| `waso_min` | $-13.57$ min | $[-43.05, +15.90]$ | 15.09 min | 0.3968 | $-14.16$ min | $[-44.39, +16.07]$ | 16.34 min | 0.2824 |
| `sleep_efficiency_pct` | $+2.62\%$ | $[-5.60, +10.83]\%$ | 3.58% | 0.3781 | $+3.24\%$ | $[-5.02, +11.49]\%$ | 4.26% | 0.2601 |
| `deep_pct_tst` | $+10.65\%$ | $[-2.39, +23.69]\%$ | 10.72% | 0.4909 | $+8.67\%$ | — | 8.81% | 0.5023 |
| `rem_pct_tst` | $+10.13\%$ | $[-5.92, +26.18]\%$ | 11.12% | 0.4767 | $+7.21\%$ | — | 9.64% | 0.1971 |

---

## 7. `DEC-042`: `EXP-029` `Ultradian-Pyramidal-Coupler` (`v5`, 75 Runs)

| Arm | Params | BIDSleep Macro-F1 (%) | Acc (%) | $\kappa$ | $\Delta$ vs. Model A ($p_{\text{Holm}}$) | $\Delta$ vs. SLAMSS-IFS ($p_{\text{Holm}}$) | SleepAccel `qc_clean` (`single` / `ens`, `em_off`) | `EXP-020` (`ZS` -> `FT-2` -> `GATE_tau`) |
|---|---:|---:|---:|---:|---|---|---:|---|
| `cnn_bigru_200k` (`B`) | 223,268 | $66.61 \pm 7.06$ | — | — | — | — | — | — |
| `film_only` | 172,260 | $66.67 \pm 7.02$ | — | — | — | — | — | — |
| `pyramid_only` | 182,468 | $66.92 \pm 7.05$ | — | — | — | — | — | — |
| **`full_v5`** (`RCCLoss`) | **190,788** | **$66.74 \pm 6.81$** | **68.17** | **0.5295** | **+1.95 pp** $[+1.31, +2.61]$ ($1.43 \times 10^{-6}$, 38/47 wins) | **$-0.63$ pp** $[-1.36, +0.16]$ ($0.1212$, 28.65× comp.) | **61.78% / 63.51%** (Deep% bias $+7.59\%$) | `66.67%` -> `67.59%` -> **`67.86%`** ($+1.19$ pp) |
| `full_v5_ce` (`CE`) | 190,788 | $66.95 \pm 6.84$ | 68.35 | 0.5324 | $+0.21\text{ pp}$ vs. `full_v5` ($p_{\text{raw}}=0.0272, p_{\text{Holm}}=0.1212$) | — | — | — |

---

## 8. `DEC-043`: `EXP-030` Single-Factor Capacity Pareto Sweep (`p1_100k` Selection)

### 8A. 6-Point Capacity Pareto Grid (`ScaledConventionalCNNBiGRU`, BIDSleep $N = 47$)

| Grid Tag | Params | BIDSleep Macro-F1 (%) |
|---|---:|---:|
| **`p1_100k`** *(Elbow Selected)* | **98,940** | **$66.67 \pm 7.57$** |
| `p2_164k` (`Model A` anchor) | 163,940 | $64.79 \pm 7.45$ |
| `p3_268k` | 268,260 | $66.43 \pm 7.08$ |
| `p4_404k` | 403,524 | $66.50 \pm 7.16$ |
| `p5_666k` | 665,556 | $66.17 \pm 7.02$ |
| `p6_1051k` | 1,051,492 | $66.25 \pm 6.84$ |

### 8B. `p1_100k` Downstream Summary
- **BIDSleep ($N=47$):** $+1.88\text{ pp}$ vs. `Model A` $[+1.25, +2.49]$, $p_{\text{Holm}}=1.19 \times 10^{-6}$, $d_z=+0.875$, `40/47` wins (REM F1 $+3.56\text{ pp}$, $p=2.40 \times 10^{-10}$); $-0.69\text{ pp}$ vs. `Official SLAMSS-IFS` $[-1.57, +0.15]$, $p_{\text{Holm}}=0.1498$ (**55.24×** compression).
- **W0-B Dynamic INT8:** `182.00 KiB` in-memory (`69,796 / 98,940` quantized), `456.1 MMACs`, `56.58 ms` 1T / `46.35 ms` 4T, OOF $\bar{\Delta}=-0.0134\text{ pp}$ ($p=0.4656$).
- **SleepAccel (`em_off`):** `qc_clean` ($n=25$): `60.88%` (`mean_single_model`, $+0.36\text{ pp}$ vs. `Model A` $p=0.1645$, $-0.21\text{ pp}$ vs. `SLAMSS-IFS` $p=1.0000$) / `62.51%` (`prob_ensemble`); `all` ($N=31$): `61.10%` / `62.51%`.
- **`EXP-020` (`N=46`):** Nested `GATE_tau` ($\tau^*=0.05$ Folds 1–2, $0.01$ Folds 3–5; `3/46` abstained): `ZS = 66.84%`, `FT-2 = 66.27%` ($-0.57\text{ pp}$, $p=0.6571$), `GATE_tau = 66.74%` ($-0.10\text{ pp}$, $p=0.9519$).

---

## 9. `DEC-044`: `EXP-031` Locked Single-Factor Width Sweep & `P190` Confirmatory Verdicts

### 9A. Confirmatory Hypotheses ($N = 47$, Holm-Bonferroni across `{H-ARCH, H-SCALE}`)

| Hypothesis | Comparison | Estimand / Statistic | 95% CI | Raw $p$ | Holm $p$ | Verdict |
|---|---|---|---|---:|---:|---|
| **`H-ARCH`** | `full_v5` (`66.74%`) vs. `P190` (`66.60%`) | $\bar{\Delta} = +0.14\text{ pp}$ ($d_z=+0.086$, `26W/21L`) | $[-0.34, +0.61]$ | 0.4037 | 0.4037 | **NOT SUPPORTED (practically)** |
| **`H-SCALE`** | Page's $L$ across 9 plain widths (`W050`..`W250`) | $L = 10,938.0$ ($\mu=10,575.0, \sigma=145.43$), $z=+2.493$ | — | 0.01268 (2s) | **0.0254** | **SUPPORTED** |

### 9B. 9-Width Plain CNN-BiGRU Sweep Table (`BIDSleep N = 47`, `< 400 KiB` Eligibility)

| Arm | $w$ | Params | INT8 (KiB) | Eligible ($<400$) | Macro-F1 (%) | $\Delta$ vs. `Model A` (pp) | $\Delta$ vs. `SLAMSS-IFS` (pp) | $p$ vs. `SLAMSS-IFS` |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| `W050` | 0.50 | 43,108 | 81.00 | Yes | $65.61 \pm 7.48$ | $+0.82$ $[+0.27, +1.36]$ | $-1.76$ $[-2.67, -0.88]$ | 0.00049 |
| `W075` | 0.75 | 93,796 | 176.96 | Yes | $66.26 \pm 7.10$ | $+1.47$ $[+0.86, +2.06]$ | $-1.11$ $[-1.96, -0.30]$ | 0.00672 |
| `A` | 1.00 | 163,940 | 309.91 | Yes | $64.79 \pm 7.45$ | $0.00$ | $-2.58$ $[-3.53, -1.68]$ | $1.98 \times 10^{-6}$ |
| **`P190`** *(Sec. 9)* | **1.08** | **190,940** | **375.16** | **Yes** | **$66.60 \pm 6.85$** | **$+1.80$ $[+1.14, +2.50]$** | **$-0.77$ $[-1.59, +0.07]$** | **0.07266** |
| `B` | 1.17 | 223,268 | 368.04 | Yes | $66.61 \pm 7.06$ | $+1.82$ $[+1.25, +2.43]$ | $-0.76$ $[-1.51, +0.01]$ | 0.04901 |
| `W125` | 1.25 | 253,540 | 479.86 | No | $66.70 \pm 6.80$ | $+1.91$ $[+1.20, +2.63]$ | $-0.67$ $[-1.39, +0.13]$ | 0.02160 |
| `W150` | 1.50 | 362,596 | 686.82 | No | $66.31 \pm 6.82$ | $+1.52$ $[+0.84, +2.21]$ | $-1.06$ $[-1.78, -0.33]$ | 0.00566 |
| `W200` | 2.00 | 639,076 | 1211.72 | No | $66.01 \pm 6.90$ | $+1.22$ $[+0.51, +1.92]$ | $-1.36$ $[-2.26, -0.43]$ | 0.00236 |
| `W250` | 2.50 | 993,380 | 1884.63 | No | $66.34 \pm 7.05$ | $+1.55$ $[+0.83, +2.26]$ | $-1.03$ $[-1.91, -0.11]$ | 0.02098 |

### 9C. `P190` Section 10 Downstream Summary
- **W0-B Dynamic INT8:** `375.16 KiB` in-memory (`126,532 / 190,940` quantized), `984.2 MMACs`, `73.30 ms` 1T / `53.07 ms` 4T, OOF FP32 `66.5940%` vs. INT8 `66.6184%` ($\bar{\Delta}=+0.0244\text{ pp}$ $[+0.0017, +0.0480]$, $p=0.0836$).
- **SleepAccel (`em_off`):** `qc_clean` ($n=25$): `61.52%` (`mean_single_model`, $+1.01\text{ pp}$ vs. `Model A` $p=0.0755$, $-0.25\text{ pp}$ vs. `full_v5` $p=0.5602$, $+0.44\text{ pp}$ vs. `SLAMSS-IFS` $p=0.5424$) / `63.42%` (`prob_ensemble`, $+0.48\text{ pp}$ vs. `Model A`, $+0.64\text{ pp}$ vs. `SLAMSS-IFS`); `all` ($N=31$): `61.77%` / `63.94%`.
- **`EXP-020` (`N=46`):** Nested `GATE_tau` ($\tau^*=0.05$ Folds 1–2, $0.01$ Folds 3–5; `3/46` abstained): `ZS = 66.42%`, `FT-2 = 66.09%` ($-0.33\text{ pp}$, $p=0.5882$), `GATE_tau = 66.55%` ($+0.13\text{ pp}$, $p=0.7720$).
