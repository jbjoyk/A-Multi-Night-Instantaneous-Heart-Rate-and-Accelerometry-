# EXP-030 Post-Execution EGM Drift Report

* **Date:** 2026-11-24
* **Audit Targets:** `docs/designs/exp030_cnn_bigru_capacity_pareto.md` & `docs/audits/EXP030_PRE_EXECUTION_DRIFT_REPORT.md` vs. executed artifacts in `experiments/EXP-030/` and `logs/exp030_gpu{0,1}.log`
* **Verdict:** **`PASS`** (`0` Blocking, `0` Major, `0` New Drift)

---

## 1. Post-Execution Drift Axis Verification

| Axis | Verification Target | Drift Class | Verified Evidence |
|---|---|---|---|
| **1. 75-Run Sweep & Anchor Completeness** | `15/15` checkpoints, `15/15` run JSONs, `N=47` CSVs (`0` NaNs) for `p1_100k`, `p3_268k`, `p4_404k`, `p5_666k`, `p6_1051k`; plus `15/15` `p2_164k` stitcher-parity JSONs | `NONE` | All 5 new grid points: `15` `.pt`, `15` `.json`, `47` CSV rows, `0` NaNs. `p2_164k`: `64.7907% ± 7.45%` (`0.0000 pp` diff vs. `DEC-037`). |
| **2. Deterministic Elbow Selection** | Zero SleepAccel look-ahead; exact `<= 0.20 pp` (`>= 90%` gain) $T$ and `<= 0.50 pp` degenerate guard | `NONE` | `elbow_selection.json`: **`p1_100k`** (`98,940` params), `m_max = 66.6726%`, `delta_max = +1.8819 pp`, `threshold_T = 66.4844%`, `guard_triggered = False`. |
| **3. Downstream Pillars 1–4 (`p1_100k` Only)** | Restricted to `star_tag = p1_100k`; `em_off`-only SleepAccel; nested per-fold `GATE_tau` | `NONE` | Pillar 1 (`+1.88 pp` vs. Model A, `p_Holm=1.19e-6`; `-0.69 pp` vs. SLAMSS-IFS, `p_Holm=0.1498`); Pillar 2 (`182.00 KiB`, `46.35 ms` 4T, `-0.013 pp` `p=0.4656`); Pillar 3 (`60.88%`/`62.51%`); Pillar 4 (`66.74%`, `3/46` abstained). |
| **4. Protected File Immutability** | `0` edits to `exp013_models.py`, `sleep_adaptive_net.py`, `PREREG_EXP-030_WIDTH_SWEEP.md`, `EXP-018`–`EXP-029` | `NONE` | Verified via `git status --short`. |
| **5. Log & Resource Hygiene** | Clean convergence, finite losses, RTX 3060 VRAM within budget | `NONE` | Peak VRAM `344.5 MB` (`p1_100k`) to `999.3 MB` (`p6_1051k`); all 75 runs exited cleanly. |

---

## 2. 6-Point Capacity–Accuracy–Edge Pareto Grid (`N = 47`, 5-Fold × 3-Seed CV)

| Grid Tag | Params | $\Delta\%$ vs. 164k | BIDSleep Macro-F1 | $\Delta$ vs. `p2_164k` | Dynamic INT8 (KiB) | MMACs (`L=1200`) | 4T INT8 Latency | Peak VRAM | Elbow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`p1_100k`** | **`98,940`** | `-39.65%` | **`66.67% ± 7.57%`** | **`+1.88 pp`** | **`182.00 KiB`** | **`456.1`** | **`46.35 ms`** | `344.5 MB` | **SELECTED (`CNN-BiGRU-Star`)** |
| `p2_164k` (Anchor) | `163,940` | `0.00%` | `64.79% ± 7.45%` | `0.00 pp` | `309.91 KiB` | `794.1` | `54.89 ms` | — (frozen) | Reference |
| `p3_268k` | `268,260` | `+63.63%` | `66.43% ± 7.08%` | `+1.64 pp` | `494.19 KiB` | `1,252.8` | `54.99 ms` | `523.7 MB` | Below $T$ (`66.48%`) |
| `p4_404k` | `403,524` | `+146.14%` | `66.50% ± 7.16%` | `+1.71 pp` | `726.88 KiB` | `1,822.3` | `68.65 ms` | `619.4 MB` | Meets $T$, larger |
| `p5_666k` | `665,556` | `+305.98%` | `66.17% ± 7.02%` | `+1.38 pp` | `1,237.63 KiB` | `3,172.9` | `87.33 ms` | `837.4 MB` | Below $T$ |
| `p6_1051k` | `1,051,492` | `+541.39%` | `66.25% ± 6.84%` | `+1.46 pp` | `1,941.29 KiB` | `4,966.8` | `102.85 ms` | `999.3 MB` | Below $T$ |

---

## 3. Downstream Pillar Summary (`CNN-BiGRU-Star = p1_100k`)

1. **Pillar 1 — BIDSleep Paired Statistics (`N = 47`):**
   - `p1_100k` (`66.67%`) vs. `Model A-164k` (`64.79%`): **`+1.88 pp`** [95% CI: `+1.25, +2.49 pp`], Wilcoxon $p = 5.95 \times 10^{-7}$, **$p_{\text{Holm}} = 1.19 \times 10^{-6}$**, $d_z = +0.875$, **`40/47` wins** (REM F1 `+3.56 pp`, $p = 2.40 \times 10^{-10}$).
   - `p1_100k` (`66.67%`) vs. `Official SLAMSS-IFS` (`67.37%`, `5,465,768` params): **`-0.69 pp`** [95% CI: `-1.57, +0.15 pp`], **$p_{\text{Holm}} = 0.1498$**, $d_z = -0.229$, `17/47` wins — **statistical parity at `55.24x` parameter compression** (gap narrowed from `2.58 pp` to `0.69 pp`; residual difference not ruled out).
2. **Pillar 2 — `W0-B` Dynamic INT8 Edge Profile:**
   - In-memory payload: **`182.00 KiB`** (`70.54%` params quantized; serialized `207.97 KiB` vs. `404.09 KiB` FP32).
   - CPU latency (`(1, 1200, 30)`): **`56.58 ms`** (1T INT8) / **`46.35 ms`** (4T INT8).
   - Full `N = 47` OOF parity: FP32 `66.6750%` vs. INT8 `66.6616%`, $\Delta = \mathbf{-0.0134\text{ pp}}$ [95% CI: `-0.0489, +0.0210 pp`], Wilcoxon $p = 0.4656$.
3. **Pillar 3 — Strictly Zero-Shot SleepAccel (`em_off` only):**
   - `qc_clean` (`n = 25`): `mean_single_model` **`60.88%`** (`+0.36 pp` vs. `Model A` $p=0.1645$; `-0.21 pp` vs. `SLAMSS-IFS` $p=1.0000$); `prob_ensemble` **`62.51%`** (`-0.43 pp` vs. `Model A` $p=0.2411$; `-0.28 pp` vs. `SLAMSS-IFS` $p=0.6150$).
   - `all` (`N = 31`): `mean_single_model` **`61.10%`**; `prob_ensemble` **`62.51%`**.
   - Bland-Altman (`qc_clean`, `prob_ensemble`): TST bias `+12.66 min` (`[-19.81, +45.13]`), SOL `+3.36 min` (`[-7.09, +13.81]`), WASO `-15.12 min` (`[-43.93, +13.69]`), SE `+2.79%` (`[-4.60, +10.18]`), Deep% `+8.81%` (`[-4.22, +21.83]`), REM% `+10.80%` (`[-3.30, +24.90]`).
4. **Pillar 4 — `EXP-020` `FT-2` & Nested Per-Fold `GATE_tau` (`N = 46`):**
   - Per-fold $\tau^*$: Fold 1 `0.05` (`1` abstained), Fold 2 `0.05` (`2` abstained), Folds 3–5 `0.01` (`0` abstained); total `3/46` abstained.
   - `ZS`: `66.84%`, `FT-2`: `66.27%` (`-0.57 pp`, $p = 0.6571$), `GATE_tau`: **`66.74%`** (`-0.10 pp` vs. `ZS`, $p = 0.9519$; `+0.47 pp` vs. `FT-2`, $p = 0.2850$).
