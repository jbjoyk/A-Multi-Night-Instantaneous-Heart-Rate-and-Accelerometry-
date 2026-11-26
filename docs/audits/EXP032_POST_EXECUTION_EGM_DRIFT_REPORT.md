# EXP-032 Post-Execution EGM Drift Report

* **Audit Date:** `2026-11-26`
* **Specification Anchor:** [`docs/designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md`](../designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md)
* **Pre-Execution Clearance:** [`docs/audits/EXP032_PRE_EXECUTION_DRIFT_REPORT.md`](./EXP032_PRE_EXECUTION_DRIFT_REPORT.md) (`GO`, `10/10` unit tests in `tests/test_exp032_preflight.py`)
* **Post-Execution Verdict:** **`PASS`** (**0 Blocking, 0 Major, 0 Drift Detected**)

---

## 1. Executed Artifact & Integrity Verification

| Check | Scope | Result |
|---|---|---|
| **Stage A Tuning** | `experiments/EXP-032/stage_a_screening.csv`, `locked_configs.json` | **PASS** — `12` configs (`4/arm × 5 folds = 60` runs at `seed=42`, `N=25` `(fold, subject_id)` pooled `inner_val`); locked `M1: cfg#3` (`dropout=0.3, wd=0.01`, `66.97%`), `M2: cfg#3` (`dropout=0.3, wd=0.01`, `67.06%`), `M3: cfg#1` (`dropout=0.3, wd=0.01, tau=2.0, alpha_kd=0.50`, `69.51%`). |
| **Stage B Outer Runs** | `experiments/EXP-032/run_results/`, `checkpoints/`, `frozen_checkpoint_manifest.json` | **PASS** — `45/45` run JSONs and `45/45` checkpoints verified; `45/45` SHA-256 digests match `frozen_checkpoint_manifest.json` (`fold_manifests.csv` SHA-256 `36d84020...`). |
| **In-Cohort & External Summaries** | `exp032_summary.json`, `exp032_sleepaccel_summary.json`, `exp032_cpu_latency.json`, `{arm}_oof_subject_metrics.csv`, `{arm}_sleepaccel_metrics.csv` | **PASS** — `N=47` BIDSleep OOF, `n=25 qc_clean` & `N=31 all` SleepAccel (`em_off` only), and interleaved `1T`/`4T` FP32 CPU latency (`W100`, `M1`, `M2`, `M3`, `SLAMSS-IFS`) all verified. |
| **Protected File Immutability** | `PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md`, `exp013_models.py`, `experiments/EXP-018`–`EXP-031` | **PASS** — `git diff --name-only` confirmed **0 modifications**. |

---

## 2. Locked §5.2 Verdict & Scientific Attribution Check

- **Overall Verdict:** **`NOT SUPPORTED`** (`K_qualifying = 0`, all 3 arms `ARM_NOT_SUPPORTED`) — **No drift detected** against §5.2.
- **Standalone From-Scratch (`M1`, `M2`):**
  - `M1` (`1,402,468` params): `65.21%` Macro-F1 (`-2.16 pp` vs. `SLAMSS-IFS`, $p_{\text{Holm}} = 3.14 \times 10^{-5}$; `-1.47 pp` vs. `W100`, $p = 0.00153$), `67.45%` Accuracy.
  - `M2` (`1,402,492` params): `65.18%` Macro-F1 (`-2.19 pp` vs. `SLAMSS-IFS`, $p_{\text{Holm}} = 3.14 \times 10^{-5}$; `M2 - M1 = -0.03 pp`, $p = 0.2809$), `67.44%` Accuracy.
- **Teacher-Distilled (`M3`, `1,402,492` params, explicitly labeled teacher-distilled):**
  - **BIDSleep (`N=47`):** `67.38%` Macro-F1 (`+0.02 pp` vs. `SLAMSS-IFS` `67.37%`, 95% CI `[-0.64, +0.69] pp`, $p_{\text{Holm}} = 0.8627$ — statistical parity; `+2.21 pp` vs. `M2`, $p = 5.07 \times 10^{-10}$, `43/47` wins), **`69.75%` Accuracy** (**`+1.00 pp`** vs. `SLAMSS-IFS` `68.75%`, 95% CI `[+0.38, +1.61] pp`, $p = 0.00397$, `33/47` wins), $\kappa = 0.5548$.
  - **SleepAccel (`n=25 qc_clean`, `em_off`):** `61.43%` single (`+0.34 pp` vs. `SLAMSS-IFS` `61.08%`, $p_{\text{NI}} = 0.00032$), `62.63%` ensemble (`-0.15 pp` vs. `62.79%`, $p_{\text{NI}} = 0.00254$); `N=31 all`: `61.76%` / `63.01%`.
  - **Interleaved CPU Latency (`AMD Ryzen 7 7700`):** `129.14 ms` (`1T`) / `87.75 ms` (`4T`), **`28.1x` (`1T`) / `35.7x` (`4T`)** faster than `Official SLAMSS-IFS` (`3,631.24 ms` / `3,131.18 ms`) at `3.90x` fewer parameters.
