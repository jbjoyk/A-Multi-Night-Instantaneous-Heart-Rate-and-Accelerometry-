# Supplementary Appendix

**Manuscript:** *Training-Recipe Sensitivity, Edge Capacity Scaling, and Personalization Boundaries in Consumer-Smartwatch Sleep Staging: A Pre-Registered, Leakage-Free Benchmark*

---

## S1. Governance, Preregistration Lock Mapping, and Quarantine Log

| Experiment / Phase | Preregistration Document | Lock Commit | Primary Outcome / Governance Status |
|---|---|---|---|
| `EXP-018` / Stage 2 | `experiments/EXP-018/design_draft.md` | `4dff5b7` | Frozen `Model A` (`163,940` params, 3,200-step older recipe, `64.79%` OOF Macro-F1); **QUARANTINED** as historical anchor |
| `EXP-019` | `PREREG_EXP-019_H-A.md` | `4dff5b7` | Locked legacy `Model A` `FT-2`: `+1.0054 pp` (`[-0.17, +1.99]`, 1s $p=0.00131$, 2s $p=0.00261$, `34/46`); **ERRATUM APPLIED** (`INCONCLUSIVE`) |
| `EXP-021` | `experiments/EXP-021/` | `945c7c1` | Matched `Official SLAMSS-IFS` (`5,465,768` params, `67.37%` OOF Macro-F1) under unified recipe |
| `EXP-027` / `EXP-028` | `docs/designs/exp027_exp028_sleepaccel_external_validation.md` | `945c7c1` | Zero-shot external `SleepAccel v1.0.0` (`n=25` / `N=31`) & Bland-Altman (`em_off`) |
| `EXP-029` | `docs/designs/exp029_ultradian_pyramidal_coupler.md` | `24ca187` | `full_v5` (`190,788` params, `66.74%`) & `B` (`223,268` params, `66.61%`) |
| `EXP-030` | `docs/designs/exp030_cnn_bigru_capacity_pareto.md` | `b00e9e8` | 6-point Pareto grid (`p1_100k` `98,940` params, `66.67%`, `182.00 KiB` INT8) |
| `EXP-031` + A1 | `PREREG_EXP-031_WIDTH_SWEEP.md` & `AMENDMENT_EXP-031_A1_W100_RETRAIN.md` | `dbcdf5e` / `1c7fcf7` | 9-width sweep + `W100` retrained control (`66.68%`, `+1.89 pp` vs. frozen `Model A`), `H-ARCH` (`NOT SUPPORTED`) |
| `EXP-032` | `docs/designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md` | `906d594` | `M1` (`65.21%`), `M2` (`65.18%`), teacher-distilled `M3` (`1,402,492`, `67.38%` F1 / `69.75%` Acc, `87.75 ms` 4T FP32) |
| `EXP-033` | `docs/designs/PREREG_EXP-033_W100_PERSONALIZATION_EXTERNAL.md` | `c1b7843` | `W100` `FT-2` (`+0.58 pp`, med `+0.75 pp`, `30W/16L`, `NOT SUPPORTED practically`), `S-SPEC` (`+1.45 pp`, `INCONCLUSIVE`), reconciled `SleepAccel` |
| `EXP-034` | `docs/designs/PREREG_EXP-034_500EPOCH_CONVERGENCE.md` | `c7d2d19` | 500-epoch audit (`60/60` early-stopped): `W050` `+0.63 pp`, `W075` `+0.14 pp`, `p1_100k` `+0.01 pp`, `W100` `+0.02 pp` |

---

## S2. Subject-Disjoint 5-Fold Cross-Validation Manifests

- **Authoritative Manifest File:** `review_package/fold_manifests.csv` (SHA-256: `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`).
- **Partition Structure ($N = 47$ total subjects, 246 usable nights):**
  - Each fold assigns **5 inner-validation subjects** (`inner_val_subjects`), **9–10 strictly held-out outer-test subjects** (`outer_test_subjects`), and the remaining **32–33 inner-training subjects** (`inner_train_subjects`).
  - Zero subjects, nights, or 30-s epochs overlap across `inner_train`, `inner_val`, and `outer_test` within any fold.

---

## S3. Verified Class-Mix Audit Table

| Cohort / Subset | Scope | Total Epochs | Wake (%) | Light (%) | Deep (%) | REM (%) |
|---|---|---:|---:|---:|---:|---:|
| **BIDSleep v1.0.1 (`N=47`, 246 usable nights)** | Masked usable epoch pool | 203,338 | 10.83 | 44.20 | 18.91 | 26.06 |
| **BIDSleep v1.0.1 (`N=47`, 246 usable nights)** | Raw unmasked pooled labels | 209,482 | 11.06 | 44.60 | 18.54 | 25.79 |
| **BIDSleep v1.0.1 (`N=47`)** | Participant-averaged shares | 209,482 | 10.96 | 44.39 | 18.63 | 26.02 |
| **BIDSleep Legacy Single Split (`N=31` train)** | Historical training partition | 137,801 | 11.90 | 46.20 | 18.60 | 23.20 |
| **SleepAccel v1.0.0 (`qc_clean n=25`)** | Masked PSG epochs/checkpoint | 22,633 | 8.24 | 57.74 | 11.08 | 22.94 |

---

## S4. Full Recipe Comparison & 500-Epoch Convergence Details

- Full line-cited hyperparameter comparison across frozen `Model A`, the unified `EXP-031`/`EXP-033` recipe, and `Official SLAMSS-IFS` (both matched benchmark and official repository defaults) is archived in `experiments/EXP-031/recipe_comparison_table.md` and `.csv`.
- Full 60-run 500-epoch convergence logs and diagnostic mixed-budget Page's $L$ checks (`9-width`: $L=10395.0$, $z_L=-1.238$, $p_{\text{two}}=0.2158$; `10-width`: $L=13931.0$, $z_L=-1.520$, $p_{\text{two}}=0.1286$) are archived in `experiments/EXP-034/EXP034_FINAL_REPORT.md` and `exp034_summary.json`.

---

## S5. Completed TRIPOD+AI Checklist

The complete 22-item (including sub-items) TRIPOD+AI checklist is provided in [`TRIPOD_AI_CHECKLIST.md`](./TRIPOD_AI_CHECKLIST.md).
