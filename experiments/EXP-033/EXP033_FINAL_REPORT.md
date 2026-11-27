# EXP-033 Final Report: Personalization and External Evaluation on Recipe-Consistent W100

- **Pre-registration:** `docs/designs/PREREG_EXP-033_W100_PERSONALIZATION_EXTERNAL.md` (lock commit `c1b784305812d3c7208ed73c24a4b08a85c1ef3c`)
- **H-A2 Verdict (`W100` `FT-2 - ZS`, N=46):** **NOT SUPPORTED (practically)** (`Mean Δ = +0.58 pp`, `Median Δ = +0.75 pp`, 95% CI `[-0.06, +1.22]`, `W/L/T = 30/16/0`, Wilcoxon raw two-sided `p = 2.6344e-02`, Holm-adjusted two-sided `p_Holm = 2.6344e-02`, `d_z = 0.259`)
- **S-SPEC Verdict (`W100` `FT-2 - FT-other-2`, N=46):** **INCONCLUSIVE** (`Mean Δ = +1.45 pp`, `Median Δ = +1.31 pp`, 95% CI `[+0.89, +2.02]`, `W/L/T = 37/9/0`, Wilcoxon raw two-sided `p = 1.2342e-05`, Holm-adjusted two-sided `p_Holm = 2.4683e-05`, `d_z = 0.730`)
- **Calibration-Depth Claim (`W100` E4 N=33):** **NOT CLAIMABLE** (Page's `L = 835.0`, `z = +0.573`, raw two-sided `p = 5.6673e-01`, raw one-sided `p = 2.8337e-01`; E4 `k=4` `Δ = +0.73 pp` `[-0.22, +1.77]`)

## 1. Backbone Summary Table (FT-2 vs ZS & FT-other-2, N=46)

| Backbone | Role | Mean Δ (pp) | Median Δ (pp) | 95% CI | W/L/T | p (raw, 2-sided) | d_z | FT-2 - FT-other-2 (pp) | 95% CI | FT-other-2 - ZS (pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `W100` | Primary | `+0.58` | `+0.75` | `[-0.06, +1.22]` | `30/16/0` | `2.6344e-02` | `0.259` | `+1.45` | `[+0.89, +2.02]` | `-0.87` |
| `p1_100k` | Descriptive | `+0.51` | `+0.70` | `[+0.03, +0.95]` | `31/15/0` | `2.7923e-02` | `0.314` | `+1.09` | `[+0.62, +1.56]` | `-0.58` |
| `P190` | Descriptive | `-0.02` | `+0.69` | `[-1.09, +0.85]` | `31/15/0` | `4.0247e-02` | `-0.007` | `+0.64` | `[-0.29, +1.47]` | `-0.66` |
| `full_v5` | Descriptive | `+0.45` | `+1.03` | `[-0.57, +1.41]` | `28/18/0` | `1.1124e-01` | `0.131` | `+1.27` | `[+0.41, +2.06]` | `-0.82` |
| `Model A` (EXP-019) | Older recipe | `+1.01` | `+1.55` | `[-0.17, +1.99]` | `34/12/0` | `2.6126e-03` (`1s: 0.00131`) | `0.264` | — | — | — |

## 2. W100 Calibration-Depth Curve (Descriptive)

| Cohort | k | N | ZS (%) | FT-k (%) | Mean Δ (pp) | Median Δ (pp) | 95% CI | W/L/T | p (raw, 2-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `E4` | 1 | 33 | `66.48` | `66.83` | `+0.35` | `+0.25` | `[-0.54, +1.29]` | `21/12/0` | `2.5653e-01` |
| `E4` | 2 | 33 | `66.48` | `67.01` | `+0.53` | `+0.53` | `[-0.27, +1.43]` | `23/10/0` | `5.0586e-02` |
| `E4` | 3 | 33 | `66.48` | `67.22` | `+0.74` | `+0.80` | `[-0.23, +1.80]` | `22/11/0` | `1.0069e-01` |
| `E4` | 4 | 33 | `66.48` | `67.21` | `+0.73` | `+0.63` | `[-0.22, +1.77]` | `19/14/0` | `1.9481e-01` |
| `E_1` | 1 | 46 | `65.83` | `66.11` | `+0.28` | `+0.03` | `[-0.29, +0.90]` | `23/23/0` | `5.2271e-01` |
| `E_2` | 2 | 46 | `66.74` | `67.32` | `+0.58` | `+0.75` | `[-0.06, +1.22]` | `30/16/0` | `2.6344e-02` |
| `E_3` | 3 | 42 | `66.73` | `67.27` | `+0.54` | `+0.63` | `[-0.14, +1.21]` | `28/14/0` | `8.9169e-02` |
| `E_4` | 4 | 33 | `66.48` | `67.21` | `+0.73` | `+0.63` | `[-0.22, +1.77]` | `19/14/0` | `1.9481e-01` |

## 3. GATE-τ Sweep (Descriptive, N=46)

| Backbone | τ (%) | Abstained | Mean Δ vs ZS (pp) | Median Δ (pp) | Min Δ (pp) | 10th-pct Δ (pp) | N(Δ ≤ -5 pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `W100` | 1% | 1 | `+0.65` | `+0.75` | `-5.18` | `-1.54` | 1 |
| `W100` | 2% | 2 | `+0.64` | `+0.75` | `-5.18` | `-1.54` | 1 |
| `W100` | 3% | 2 | `+0.64` | `+0.75` | `-5.18` | `-1.54` | 1 |
| `W100` | 5% | 8 | `+0.34` | `+0.26` | `-5.18` | `-1.50` | 1 |
| `p1_100k` | 1% | 1 | `+0.58` | `+0.70` | `-3.25` | `-1.54` | 0 |
| `p1_100k` | 2% | 2 | `+0.56` | `+0.64` | `-3.25` | `-1.54` | 0 |
| `p1_100k` | 3% | 2 | `+0.56` | `+0.64` | `-3.25` | `-1.54` | 0 |
| `p1_100k` | 5% | 8 | `+0.46` | `+0.30` | `-3.25` | `-1.47` | 0 |
| `P190` | 1% | 1 | `+0.34` | `+0.69` | `-7.88` | `-2.72` | 2 |
| `P190` | 2% | 2 | `+0.31` | `+0.63` | `-7.88` | `-2.72` | 2 |
| `P190` | 3% | 2 | `+0.31` | `+0.63` | `-7.88` | `-2.72` | 2 |
| `P190` | 5% | 8 | `+0.12` | `+0.12` | `-7.88` | `-2.72` | 2 |
| `full_v5` | 1% | 1 | `+0.71` | `+1.03` | `-7.71` | `-2.13` | 1 |
| `full_v5` | 2% | 2 | `+0.67` | `+0.73` | `-7.71` | `-2.13` | 1 |
| `full_v5` | 3% | 2 | `+0.67` | `+0.73` | `-7.71` | `-2.13` | 1 |
| `full_v5` | 5% | 8 | `+0.24` | `+0.00` | `-7.71` | `-2.13` | 1 |

## 4. Part B: W100 Zero-Shot on SleepAccel (em_off, Descriptive)

| Cohort | Eval Mode | Reference | N | W100 (%) | Ref (%) | Δ (pp) | 95% CI | p (raw, 2-sided) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `qc_clean` | `mean_single_model` | `ModelA` | 25 | `61.23` | `60.51` | `+0.71` | `[-0.13, +1.56]` | `1.0140e-01` |
| `qc_clean` | `mean_single_model` | `SLAMSS_IFS` | 25 | `61.23` | `61.08` | `+0.14` | `[-1.28, +1.56]` | `9.5785e-01` |
| `qc_clean` | `mean_single_model` | `p1_100k` | 25 | `61.23` | `60.88` | `+0.35` | `[-0.59, +1.20]` | `2.5210e-01` |
| `qc_clean` | `mean_single_model` | `P190` | 25 | `61.23` | `61.52` | `-0.30` | `[-0.93, +0.39]` | `3.5252e-01` |
| `qc_clean` | `prob_ensemble` | `ModelA` | 25 | `62.93` | `62.94` | `-0.01` | `[-1.03, +1.03]` | `8.9486e-01` |
| `qc_clean` | `prob_ensemble` | `SLAMSS_IFS` | 25 | `62.93` | `62.79` | `+0.15` | `[-1.45, +1.76]` | `8.9486e-01` |
| `qc_clean` | `prob_ensemble` | `p1_100k` | 25 | `62.93` | `62.51` | `+0.43` | `[-0.49, +1.24]` | `1.7296e-01` |
| `qc_clean` | `prob_ensemble` | `P190` | 25 | `62.93` | `63.42` | `-0.49` | `[-1.20, +0.19]` | `3.6659e-01` |
| `all` | `mean_single_model` | `ModelA` | 31 | `61.58` | `60.65` | `+0.93` | `[+0.07, +1.83]` | `5.0248e-02` |
| `all` | `mean_single_model` | `SLAMSS_IFS` | 31 | `61.58` | `62.27` | `-0.69` | `[-2.99, +1.17]` | `1.0000e+00` |
| `all` | `mean_single_model` | `p1_100k` | 31 | `61.58` | `61.10` | `+0.47` | `[-0.35, +1.26]` | `1.2032e-01` |
| `all` | `mean_single_model` | `P190` | 31 | `61.58` | `61.77` | `-0.20` | `[-0.77, +0.39]` | `4.3272e-01` |
| `all` | `prob_ensemble` | `ModelA` | 31 | `63.43` | `62.94` | `+0.49` | `[-0.57, +1.66]` | `5.9481e-01` |
| `all` | `prob_ensemble` | `SLAMSS_IFS` | 31 | `63.43` | `64.14` | `-0.72` | `[-3.51, +1.49]` | `8.0917e-01` |
| `all` | `prob_ensemble` | `p1_100k` | 31 | `63.43` | `62.51` | `+0.91` | `[-0.14, +2.11]` | `5.5240e-02` |
| `all` | `prob_ensemble` | `P190` | 31 | `63.43` | `63.94` | `-0.51` | `[-1.76, +0.60]` | `5.2941e-01` |

*Note: `ModelA` (`62.9385%` on `n=25` vs. `62.9366%` on `N=31`) and `p1_100k` (`62.5057%` on `n=25` vs. `62.5105%` on `N=31`) `prob_ensemble` values are independently computed over `25` and `31` subjects and coincide when rounded to two decimal places (`62.94%` and `62.51%`). Power limit: paired SD ~`4 pp` yields 80% MDE ~`2.3 pp` at `n=25`.*

### W100 SleepAccel Bland-Altman (qc_clean n=25, mean_single_model; margins: none; descriptive only)

| Metric | Bias | SD | LoA Lower | LoA Upper |
|---|---:|---:|---:|---:|
| `tst_min` | `+12.12` | `17.57` | `-22.31` | `+46.55` |
| `sol_min` | `+3.24` | `5.66` | `-7.86` | `+14.34` |
| `waso_min` | `-14.48` | `14.61` | `-43.11` | `+14.15` |
| `sleep_efficiency_pct` | `+2.67` | `4.04` | `-5.25` | `+10.60` |
| `deep_pct_tst` | `+7.10` | `6.26` | `-5.18` | `+19.38` |
| `rem_pct_tst` | `+12.23` | `7.57` | `-2.60` | `+27.06` |

## 5. Preregistration Governance & Safeguard Audit

- **(a) `FT-other-2` Donor Draws:** Confirmed `5` independent donor draws per `(subject, seed)` from the same fold's `inner_train_subjects` (`46 subjects × 3 seeds × 5 draws = 690` donor runs per backbone; fixed RNG seed per `(fold, seed, subject)`).
- **(b) State-Dict SHA-256 Hash Log:** Verified `0` mismatches across all `4,764` fine-tuning runs (`1,191` runs per backbone × 4 backbones); every `hash_before` matched the frozen checkpoint's `state_dict_sha256` and every `hash_after` differed after optimization.
- **(c) Safeguard Unit Tests & `ZS` OOF Parity:** `pytest tests/test_exp033_safeguards.py` passed `2/2` both **before** (`4.24s`) and **after** (`3.51s`) the full sweep, confirming all 15 `W100` checkpoint SHA-256 digests, `review_package/fold_manifests.csv` (`36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`), bitwise-identical `ZS` predictions before/after other-subject fine-tuning, and frozen `ZS` OOF Macro-F1 parity (`max abs diff = 1.11e-16` for `W100`/`P190`, `2.22e-16` for `p1_100k`/`full_v5`).
- **(d) Nested Inner-Validation Selected Configurations (per Fold & Backbone):**

| Backbone | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| `W100` | `lr=3e-05, ep=15, head_gru` (72.64%) | `lr=3e-05, ep=15, head_gru` (72.36%) | `lr=3e-05, ep=15, head_gru` (69.08%) | `lr=3e-05, ep=5, head_gru` (69.23%) | `lr=3e-05, ep=15, head_gru` (62.08%) |
| `p1_100k` | `lr=3e-05, ep=5, head_gru` (72.46%) | `lr=3e-05, ep=15, head_gru` (71.87%) | `lr=3e-05, ep=15, head_gru` (68.51%) | `lr=3e-05, ep=5, head_gru` (68.65%) | `lr=3e-05, ep=15, head_gru` (63.47%) |
| `P190` | `lr=3e-05, ep=5, head_gru` (72.40%) | `lr=0.0001, ep=15, head_gru` (72.03%) | `lr=3e-05, ep=5, head_gru` (68.95%) | `lr=3e-05, ep=5, head_gru` (69.34%) | `lr=3e-05, ep=5, head_gru` (63.27%) |
| `full_v5` | `lr=3e-05, ep=5, head_gru` (72.56%) | `lr=3e-05, ep=15, head_gru` (73.06%) | `lr=3e-05, ep=5, head_gru` (68.76%) | `lr=3e-05, ep=5, head_gru` (69.59%) | `lr=3e-05, ep=5, head_gru` (62.84%) |

- **(e) Section 2.2 Backbone-Selection Rule Verification:** Across the `EXP-031` `< 400 KiB` INT8 eligible plain CNN-BiGRU grid (`W050` `65.61%`, `W075` `66.26%`, `W100` `66.68%`, `P190` `66.60%`, `B` `66.61%`), `W100` (`163,940` params, `66.68%` OOF Macro-F1) is the highest-scoring eligible width (`gap = 0.00 pp <= 1 SE`) and is therefore selected under the smallest-width-within-1-SE rule.
- **(f) Confirmatory Family Holm-Bonferroni Adjustment (`{H-A2, S-SPEC}` on `W100`):** `H-A2` raw two-sided `p = 2.6344e-02`, Holm-adjusted two-sided `p_Holm = 2.6344e-02`; `S-SPEC` raw two-sided `p = 1.2342e-05`, Holm-adjusted two-sided `p_Holm = 2.4683e-05`.
