# EXP-031 Post-Execution EGM Drift Report

* **Date:** 2026-11-24
* **Audit Targets:** `PREREG_EXP-031_WIDTH_SWEEP.md` (locked commit `24ca187`) & `docs/audits/EXP031_PRE_EXECUTION_DRIFT_REPORT.md` vs. executed artifacts in `experiments/EXP-031/` and `logs/exp031_gpu{0,1}.log`
* **Verdict:** **`PASS`** (`0` Blocking, `0` Major, `0` New Drift)

---

## 1. Post-Execution Drift Axis Verification

| Axis | Verification Target | Drift Class | Verified Evidence |
|---|---|---|---|
| **1. 105-Run Sweep & Reused Arms Completeness** | `15/15` checkpoints, `15/15` run JSONs, `N=47` CSVs (`0` NaNs) for `W050`, `W075`, `P190`, `W125`, `W150`, `W200`, `W250`; exact provenance of reused arms `A`, `B`, `V5`, `S` | `NONE` | All 7 new arms: `15` `.pt`, `15` `.json`, `47` CSV rows, `0` NaNs. Reused arms verified: `A` (`experiments/EXP-030/p2_164k_subject_metrics.csv`, `64.7907%`), `B` (`experiments/EXP-029/cnn_bigru_200k_subject_metrics.csv`, `66.61%`), `V5` (`experiments/EXP-029/full_v5_subject_metrics.csv`, `66.74%`), `S` (`experiments/EXP-021/slamss_ifs_matched_subject_metrics.csv`, `67.37%`). |
| **2. Confirmatory Verdicts (`H-ARCH`, `H-SCALE`) & Sec. 9 Selection** | `H-ARCH` (`full_v5 - P190`), `H-SCALE` (Page's $L$, Holm two-sided $p$), and Section 9 `1-SE` selection (`< 400 KiB` INT8, excluding `V5`) strictly before downstream eval | `NONE` | `H-ARCH`: `+0.14 pp` `[-0.34, +0.61]`, `p_Holm=0.4037` -> **`NOT SUPPORTED (practically)`**. `H-SCALE`: Page's `L=10938.0`, `z=+2.493`, positive direction, `p_Holm,2s=0.0254` -> **`SUPPORTED`**. Sec. 9 (`m*=B` `66.61%`): **`P190`** (`190,940` params, `66.60%`, `375.16 KiB` INT8, gap `0.012 pp <= SE_boot 0.155 pp`) selected in `selected_backbone.json`. |
| **3. Section 10 Downstream Evaluation (`P190` Only)** | `W0-B` INT8 parity, zero-shot SleepAccel (`em_off` primary single-model & secondary ensemble alongside `A`, `full_v5`, `SLAMSS-IFS`), `EXP-020` `FT-2` / nested `GATE_tau` | `NONE` | `W0-B`: `375.16 KiB` INT8, `53.07 ms` 4T, `+0.0244 pp` `p=0.0836`. SleepAccel `qc_clean` (`n=25`): `61.52%` single / `63.42%` ensemble (`+0.44`/`+0.64 pp` vs. `SLAMSS-IFS`, `p >= 0.54`). `EXP-020` (`N=46`): `ZS=66.42%`, `FT-2=66.09%`, `GATE_tau=66.55%` (`3/46` abstained). |
| **4. Protected File Immutability** | `0` modifications to `exp013_models.py`, `sleep_adaptive_net.py`, `PREREG_EXP-031_WIDTH_SWEEP.md`, and historical `EXP-018`–`EXP-030` files | `NONE` | Verified via `git diff --name-only HEAD` (0 protected files modified). Non-substantive 2-line column-alias fix (`subject_id` vs. `subject`) applied to `eval_exp031_downstream.py` to read `EXP-029` SleepAccel CSV. |
| **5. Log & Resource Hygiene** | Clean convergence, finite losses, RTX 3060 VRAM within 12 GB budget across all 105 runs | `NONE` | `logs/exp031_gpu0.log`: `60/60` runs (`242.3`–`614.3 MB` VRAM); `logs/exp031_gpu1.log`: `45/45` runs (`520.4`–`997.4 MB` VRAM); all 105 runs exited cleanly. |

---

## 2. Single-Factor Width Sweep Table (`N = 47`, 5-Fold × 3-Seed CV)

| Arm | `w` | Params | FP32 (KiB) | Dynamic INT8 (KiB) | MMACs (`L=1200`) | 1T / 4T INT8 (ms) | BIDSleep Macro-F1 | $\Delta$ vs. `A` (95% CI) | $\Delta$ vs. `SLAMSS-IFS` (95% CI) | Sec. 9 (`< 400 KiB`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `W050` | `0.50` | `43,108` | `168.39` | `81.00` | `202.9` | `43.04 / 40.89` | `65.61% ± 7.48%` | `+0.82 pp` `[+0.27, +1.36]` | `-1.76 pp` `[-2.67, -0.88]` | Eligible |
| `W075` | `0.75` | `93,796` | `366.39` | `176.96` | `450.0` | `55.36 / 46.66` | `66.26% ± 7.10%` | `+1.47 pp` `[+0.86, +2.06]` | `-1.11 pp` `[-1.96, -0.30]` | Eligible |
| `A` (reused) | `1.00` | `163,940` | `640.39` | `309.91` | `794.1` | `77.24 / 56.58` | `64.79% ± 7.45%` | `0.00 pp` (Anchor) | `-2.58 pp` `[-3.53, -1.68]` | Eligible |
| **`P190`** | `1.08` | **`190,940`** | `745.86` | **`375.16`** | `984.2` | `71.46 / 54.56` | **`66.60% ± 6.85%`** | **`+1.80 pp` `[+1.14, +2.50]`** | **`-0.77 pp` `[-1.59, +0.07]`** | **SELECTED** |
| `B` (reused) | `1.17` | `223,268` | `872.14` | `368.04` | `864.7` | `65.98 / 50.94` | `66.61% ± 7.06%` | `+1.82 pp` `[+1.25, +2.43]` | `-0.76 pp` `[-1.51, +0.01]` | Eligible ($m^*$) |
| `W125` | `1.25` | `253,540` | `990.39` | `479.86` | `1,235.2` | `80.42 / 57.14` | `66.70% ± 6.80%` | `+1.91 pp` `[+1.20, +2.63]` | `-0.67 pp` `[-1.39, +0.13]` | Ineligible |
| `W150` | `1.50` | `362,596` | `1,416.39` | `686.82` | `1,773.5` | `111.93 / 71.07` | `66.31% ± 6.82%` | `+1.52 pp` `[+0.84, +2.21]` | `-1.06 pp` `[-1.78, -0.33]` | Ineligible |
| `W200` | `2.00` | `639,076` | `2,496.39` | `1,211.72` | `3,141.2` | `150.91 / 86.39` | `66.01% ± 6.90%` | `+1.22 pp` `[+0.51, +1.92]` | `-1.36 pp` `[-2.26, -0.43]` | Ineligible |
| `W250` | `2.50` | `993,380` | `3,880.39` | `1,884.63` | `4,897.2` | `195.69 / 106.26` | `66.34% ± 7.05%` | `+1.55 pp` `[+0.83, +2.26]` | `-1.03 pp` `[-1.91, -0.11]` | Ineligible |

---

## 3. Confirmatory Verdicts & Section 10 Downstream Summary (`P190`)

1. **`H-ARCH` (`full_v5` vs. `P190`, `N = 47`):**
   - `full_v5` (`190,788` params): `66.74%` (`66.7394%`) vs. `P190` (`190,940` params): `66.60%` (`66.5952%`).
   - $\bar{\Delta} = \mathbf{+0.14\text{ pp}}$ (`+0.1442 pp`) [95% CI: **`-0.34, +0.61 pp`**], Wilcoxon $p = 0.4037$, **$p_{\text{Holm}} = 0.4037$**, $d_z = +0.086$, `26W/21L`.
   - **Verdict:** **`NOT SUPPORTED (practically)`** (CI upper bound `+0.61 pp < +0.75 pp`). Per Section 8, architectural novelty/benefit is not claimed for `full_v5`.
2. **`H-SCALE` (Page's $L$-Trend across 9 Plain Widths, `N = 47`):**
   - $L = 10938.0$ ($\mu_L = 10575.0$, $\sigma_L = 145.43$), $z = \mathbf{+2.493}$ (`+2.4926`), positive direction (`L > mu_L`), $p_{\text{one-sided}} = 0.00634$, $p_{\text{two-sided}} = 0.01268$, **$p_{\text{Holm, two-sided}} = 0.0254$** (`0.02536 < 0.05`).
   - **Verdict:** **`SUPPORTED`**.
3. **Section 9 Selection (`selected_backbone.json`):**
   - $m^* = \text{B}$ (`66.61%`). `P190` (`66.60%`, `375.16 KiB` INT8) has gap `0.0120 pp <= SE_boot = 0.1545 pp` (`W075` gap `0.3470 pp > 0.2571 pp`, `W050` gap `0.9968 pp > 0.3482 pp`, `A` gap `1.8165 pp > 0.3025 pp`). Selected backbone: **`P190` (`190,940` params)** (BIDSleep estimate labeled optimistic per Section 13).
4. **Section 10 Downstream Evaluation (`P190` Only):**
   - **`W0-B` Dynamic INT8 (`P190_w0b_int8_profile.json`):** `375.16 KiB` in-memory (`126,532 / 190,940` quantized; serialized `402.91 KiB`), `984.2 MMACs`, `73.30 ms` 1T / `53.07 ms` 4T INT8 latency; OOF FP32 `66.5940%` vs. INT8 `66.6184%`, $\bar{\Delta} = +0.0244\text{ pp}$ `[+0.0017, +0.0480]`, $p = 0.0836$.
   - **Zero-Shot SleepAccel (`em_off` only):**
     - `qc_clean` (`n = 25`): `mean_single_model` **`61.52%`** (`+1.01 pp` vs. `Model A` $p=0.0755$; `-0.25 pp` vs. `full_v5` $p=0.5602$; `+0.44 pp` vs. `SLAMSS-IFS` $p=0.5424$); `prob_ensemble` **`63.42%`** (`+0.48 pp` vs. `Model A` $p=0.2872$; `-0.09 pp` vs. `full_v5` $p=1.0000$; `+0.64 pp` vs. `SLAMSS-IFS` $p=0.7112$).
     - `all` (`N = 31`): `mean_single_model` **`61.77%`**; `prob_ensemble` **`63.94%`**.
     - Bland-Altman (`qc_clean`, `prob_ensemble`): TST `+10.14 min` (`[-25.29, +45.57]`), SOL `+4.48 min` (`[-6.09, +15.05]`), WASO `-13.80 min` (`[-43.78, +16.18]`), SE `+2.24%` (`[-5.78, +10.27]`), Deep% `+7.24%` (`[-6.68, +21.17]`), REM% `+10.98%` (`[-4.70, +26.65]`).
   - **`EXP-020` `FT-2` & Nested `GATE_tau` (`N = 46`):**
     - Per-fold $\tau^*$: Fold 1 `0.05` (`1` abstained), Fold 2 `0.05` (`2` abstained), Folds 3–5 `0.01` (`0` abstained); total `3/46` abstained.
     - `ZS`: `66.42%`, `FT-2`: `66.09%` (`-0.33 pp`, $p = 0.5882$), `GATE_tau`: **`66.55%`** (`+0.13 pp` vs. `ZS`, $p = 0.7720$; `+0.46 pp` vs. `FT-2`, $p = 0.5930$).
