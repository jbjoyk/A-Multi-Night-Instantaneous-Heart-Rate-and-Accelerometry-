# EXP-029 Post-Execution `/egm-drift` Audit Report

**Date:** 2026-11-24  
**Design Spec:** `docs/designs/exp029_ultradian_pyramidal_coupler.md`  
**Pre-Execution Report:** `docs/audits/EXP029_PRE_EXECUTION_DRIFT_REPORT.md`  
**Audit Verdict:** **`PASS — 0 BLOCKING / 0 MAJOR / 2 NITS`**

---

## 1. Execution Completeness & Artifact Integrity

| Arm / Run Tag | Params | Checkpoints | Run JSONs | Subject CSV (`N=47`) | Summary JSON | Downstream Suite |
|---|---:|---:|---:|---|---|---|
| `cnn_bigru_200k` (Arm `b`) | 223,268 | 15 / 15 | 15 / 15 | `cnn_bigru_200k_subject_metrics.csv` | `cnn_bigru_200k_summary.json` | INT8, SleepAccel, `GATE_tau` |
| `film_only` (Arm `c`) | 172,260 | 15 / 15 | 15 / 15 | `film_only_subject_metrics.csv` | `film_only_summary.json` | INT8, SleepAccel, `GATE_tau` |
| `pyramid_only` (Arm `d`) | 182,468 | 15 / 15 | 15 / 15 | `pyramid_only_subject_metrics.csv` | `pyramid_only_summary.json` | INT8, SleepAccel, `GATE_tau` |
| `full_v5` (Arm `e`, `RCCLoss`) | 190,788 | 15 / 15 | 15 / 15 | `full_v5_subject_metrics.csv` | `full_v5_summary.json` | Paired Stats, INT8, SleepAccel, `GATE_tau` |
| `full_v5_ce` (Arm `e`, pure `CE`) | 190,788 | 15 / 15 | 15 / 15 | `full_v5_ce_subject_metrics.csv` | `full_v5_ce_summary.json` | Paired Sensitivity vs. `full_v5` |

- **Historical Immutability:** `git diff --name-only` across `experiments/EXP-018` through `experiments/EXP-028` returned **0 modified files**.
- **SleepAccel Isolation:** All `EXP-029` SleepAccel prediction bundles were written strictly to `experiments/EXP-029/sleepaccel_cache/predictions/` (`experiments/EXP-027/cache/predictions/` untouched).

---

## 2. BIDSleep 5-Fold CV Results & Holm-Bonferroni Paired Statistics (`N=47`)

| Arm | Params | Macro-F1 ($\pm$ SD) | Acc | $\kappa$ | Wake F1 | Light F1 | Deep F1 | REM F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **(a) `Model A` (Locked)** | 163,940 | 64.79% ± 7.45% | 66.18% | 0.5078 | — | — | — | — |
| **(b) `cnn_bigru_200k`** | 223,268 | 66.61% ± 7.06% | 68.07% | 0.5308 | 63.48% | 68.64% | 67.87% | 66.43% |
| **(c) `film_only`** | 172,260 | 66.67% ± 7.02% | 67.98% | 0.5318 | 63.82% | 67.99% | 68.11% | 66.76% |
| **(d) `pyramid_only`** | 182,468 | 66.92% ± 7.05% | 68.47% | 0.5336 | 64.04% | 69.15% | 67.53% | 66.97% |
| **(e) `full_v5` (`RCCLoss`)** | 190,788 | **66.74% ± 6.81%** | 68.17% | 0.5295 | 64.37% | 68.66% | 67.71% | 66.21% |
| **(e') `full_v5_ce` (pure `CE`)** | 190,788 | **66.95% ± 6.84%** | 68.35% | 0.5324 | 64.56% | 68.70% | 67.85% | 66.70% |
| **(f) `Official SLAMSS-IFS`** | 5,465,768 | 67.37% ± 6.73% | 68.75% | 0.5378 | — | — | — | — |

**Paired Comparisons (`full_v5_bidsleep_paired_stats.json`, 10,000-resample bootstrap, two-sided Wilcoxon, Holm-Bonferroni):**
- `full_v5` vs. `arm_a_modelA_164k`: **`+1.95 pp`** `[+1.31, +2.61 pp]`, `p_raw = 2.39e-7`, **`p_Holm = 1.43e-6`**, `d_z = 0.858`, wins **38/47**.
- `full_v5` vs. `arm_f_slamss_ifs`: **`-0.63 pp`** `[-1.36, +0.16 pp]`, `p_raw = 0.0242`, **`p_Holm = 0.1212`** (statistical parity at **28.65x** fewer parameters).
- `full_v5` vs. `cnn_bigru_200k`: `+0.13 pp` `[-0.35, +0.63 pp]`, `p_Holm = 1.0000` (achieves parity with `+36.19%` capacity control using `-14.5%` fewer parameters).
- `full_v5` vs. `film_only`: `+0.07 pp` `[-0.42, +0.56 pp]`, `p_Holm = 1.0000`.
- `full_v5` vs. `pyramid_only`: `-0.18 pp` `[-0.53, +0.18 pp]`, `p_Holm = 1.0000`.
- `full_v5` vs. `full_v5_ce`: `-0.21 pp` `[-0.35, -0.09 pp]`, `p_raw = 0.0272`, `p_Holm = 0.1212`.

---

## 3. W0-B Dynamic INT8 Edge Quantization (`*_w0b_int8_profile.json`)

| Arm | Total Params | Quantized (`Linear`+`GRU`) | In-Memory Payload | 1T CPU Latency | 4T CPU Latency | FP32 F1 | INT8 F1 | $\Delta$ (pp) | Wilcoxon $p$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cnn_bigru_200k` | 223,268 | 172,068 | 368.04 KiB | 75.72 ms | 55.72 ms | 66.61% | 66.64% | +0.0348 | 0.0264 |
| `film_only` | 172,260 | 112,804 | 342.41 KiB | 73.98 ms | 55.32 ms | 66.67% | 66.68% | +0.0111 | 0.4984 |
| `pyramid_only` | 182,468 | 131,332 | 328.00 KiB | 46.12 ms | 33.63 ms | 66.92% | 66.90% | -0.0240 | 0.2670 |
| **`full_v5`** | **190,788** | **131,332** | **360.50 KiB** | **67.31 ms** | **44.02 ms** | **66.74%** | **66.73%** | **-0.0113** | **0.9041** |

All arms satisfy the `< 400 KiB` payload and `< 0.10 pp` INT8 parity targets.

---

## 4. PhysioNet SleepAccel External Validation & `EXP-020` `GATE_tau`

### SleepAccel Zero-Shot (`n = 25 qc_clean`, `*_sleepaccel_summary.json`)
| Arm | `em_off` Single | `em_off` Ensemble | `em_on` Single | `em_on` Ensemble |
|---|---:|---:|---:|---:|
| `cnn_bigru_200k` | 61.69% | 63.25% | 59.21% | 60.99% |
| `film_only` | 61.39% | 62.97% | 58.92% | 60.42% |
| `pyramid_only` | 62.32% | 63.96% | 60.21% | 61.93% |
| **`full_v5`** | **61.78%** | **63.51%** | 59.73% | 61.30% |

- **Bland-Altman (`full_v5`, `em_off`, `prob_ensemble`):** `tst_min` bias `+14.16 min`, `sol_min` bias `+4.06 min`, `waso_min` bias `-17.16 min`, `sleep_efficiency_pct` bias `+3.13%`, `deep_pct_tst` bias `+7.59%` (improved vs. Model A `+8.67%` and SLAMSS-IFS `+10.65%`), `rem_pct_tst` bias `+8.80%`.
- **Sensitivity Finding (`DampedEMCalibrator`):** Unsupervised `em_on` (`alpha=0.5, n_iter=3`) degraded `full_v5` SleepAccel Macro-F1 (`63.51% -> 61.30%`) and widened Deep%/REM% bias (`+7.59% -> +8.45%`, `+8.80% -> +10.83%`). `em_off` is retained as the primary zero-shot operating mode.

### `EXP-020` `FT-2` / `GATE_tau` Safety (`N = 46`, `tau* = 0.01`)
| Arm | `ZS` Mean | `FT-2` Mean | `GATE_tau` Mean | `GATE_tau` vs. `ZS` | Abstained |
|---|---:|---:|---:|---:|---:|
| `cnn_bigru_200k` | 66.74% | 65.55% | 66.06% | -0.67 pp | 1 (`Bidslab34`) |
| `film_only` | 66.68% | 66.11% | 66.66% | -0.02 pp | 1 (`Bidslab34`) |
| `pyramid_only` | 66.69% | 67.26% | 67.53% | +0.84 pp | 1 (`Bidslab34`) |
| **`full_v5`** | **66.67%** | **67.59%** | **67.86%** | **`+1.19 pp`** | **1 (`Bidslab34`)** |

---

## 5. Drift Summary

- **Blocking:** `0`
- **Major:** `0`
- **Nits:** `2` (both pre-registered in `EXP029_PRE_EXECUTION_DRIFT_REPORT.md`: subclassing `SlidingWindowPredictor` in `exp029_calibration.py` instead of mutating `exp018_inference.py`, and `549.0 MB` peak VRAM vs. `< 450 MB` design estimate).
