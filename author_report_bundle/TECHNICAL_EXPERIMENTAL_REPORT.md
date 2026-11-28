# Technical & Experimental Report (Code- & Artifact-Verified)

- **Format:** Factual tables, verified splits, code-extracted architecture specs, and artifact-verified metrics only (no manuscript prose).
- **Figures:**
  - Figure 1: [fig1_baseline_vs_proposed_architectures.pdf](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig1_baseline_vs_proposed_architectures.pdf) / [.png](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig1_baseline_vs_proposed_architectures.png) (individual: [1a Model A](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig1a_model_a.png), [1b SLAMSS-IFS](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig1b_official_slamss_ifs.png), [1c full_v5](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig1c_full_v5.png), [1d p1_100k](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig1d_p1_100k.png), [1e P190](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig1e_p190.png), [specs JSON](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/data/architecture_specs.json))
  - Figure 2: [fig2_dataset_and_splits.pdf](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig2_dataset_and_splits.pdf) / [.png](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig2_dataset_and_splits.png)
  - Figure 3: [fig3_confusion_matrices.pdf](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig3_confusion_matrices.pdf) / [.png](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig3_confusion_matrices.png)
  - Figure 4: [fig4_latency_memory_pareto.pdf](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig4_latency_memory_pareto.pdf) / [.png](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/fig4_latency_memory_pareto.png)
- **Abbreviations & Conventions:** `OOF` = out-of-fold; `PSG` = polysomnography; `QC` = quality control; `cov_mo` = accelerometry coverage; `(B, L, 30)` = batch × epochs × 30s@1Hz; `RCCLoss` = `CE + 0.01 * RCC` class-imbalance loss; `FiLM` = Feature-wise Linear Modulation; `ZS` = zero-shot; `FT-2` / `FT-other-2` = 2-night own / donor fine-tuning; `GATE_tau` = abstention gate on min calibration stage fraction (`tau*`); `em_off`/`em_on` = post-hoc `DampedEMCalibrator`; Bland–Altman `bias = pred - ref` (`tst_min`, `sol_min`, `waso_min`, `sleep_efficiency_pct`, `deep_pct_tst`, `rem_pct_tst`).

---

## Section 1: Dataset Descriptions & Exact Train / Validation / Test Splits

- **Primary In-Cohort Dataset (`BIDSleep v1.0.1`):** PhysioNet BIDSleep v1.0.1 (Song et al., *IEEE TBME* 2026, DOI: `10.1109/TBME.2025.3612158`; PhysioNet dataset DOI: `10.13026/rees-1092`, ODC-By v1.0; local archive v1.0.0; University of Massachusetts Lowell Institutional Review Board; Dreem 2 ambulatory EEG ground truth).
  - **Cohort:** `47` healthy adults, `253` recorded nights, `246` usable nights (`7` excluded by `>= 60%` signal-coverage QC filter).
  - **Raw-Recomputed 4-Class Stage Mixes (five AASM stages W, N1, N2, N3, R grouped into four classes: Wake, Light = N1+N2, Deep = N3, REM):**
    - Masked usable pooled (`N = 47`, `246` nights, `203,338` valid 30-s epochs, `~826.6`/night): Wake `10.83%`, Light `44.20%`, Deep `18.91%`, REM `26.06%`.
    - Raw unmasked pooled (`209,482` epochs): Wake `11.06%`, Light `44.60%`, Deep `18.54%`, REM `25.79%`.
    - Raw unmasked participant-averaged (`N = 47`): Wake `10.96%`, Light `44.39%`, Deep `18.63%`, REM `26.02%`.
    - Legacy 31-subject single training split (`EXP-005`, `137,801` epochs): Wake `11.9%`, Light `46.2%`, Deep `18.6%`, REM `23.2%`.
  - **Signals & Sampling:** Apple Watch HealthKit instantaneous heart rate (IHR, median `0.197 Hz`, IQR `0.073 Hz`, resampled to `1 Hz`) + 3-axis wrist accelerometry (`1 Hz` magnitude), segmented into 30-s epochs (`x_ihr`: `(B, L, 30)`, `x_accel`: `(B, L, 30)`, `x_meta`: `(B, L, 5)` with `[freq, 0, 0, hr_mean, hr_std]` where the 2 circadian clock channels are zeroed).
  - **Context Horizon:** Full-night `L = 1200` epochs (10 hours) with `LogitStitchingSlidingWindowPredictor` (overlapping `L=1200` logit windows averaged across night).
  - **Cross-Validation:** Subject-disjoint 5-fold nested CV x 3 seeds (`42, 123, 2024`).
- **External Zero-Shot Dataset (`SleepAccel v1.0.0`):** PhysioNet SleepAccel v1.0.0 (Walch et al., *SLEEP* 2019, DOI: `10.1093/sleep/zsz180`; PhysioNet dataset DOI: `10.13026/hmhs-py35`, ODC-By v1.0; University of Michigan Institutional Review Board; Apple Watch + single-night in-lab PSG).
  - **Cohort:** `N = 31` recordings; primary `qc_clean` `n = 25` (`n_epochs >= 600` & ACC `cov_mo >= 50%`; `22,633` masked epochs per checkpoint: Wake `8.24%`, Light `57.74%`, Deep `11.08%`, REM `22.94%`); `n = 6` excluded (`5` with `< 600` epochs [`417–567`], `1` with ACC dropout `cov_mo = 0.1365`), retained in `all` (`N = 31`) sensitivity.
  - **Zero-Shot Rule:** Zero retraining, zero fine-tuning, zero threshold tuning.
- **Multi-Night Personalization Denominators (`BIDSleep`; `EXP-033` `W100` recipe-consistent & legacy frozen `Model A` older recipe):**
  - `k = 1, 2` (available cohort): `N = 46` (`>= 3` usable nights; `Bidslab49` has 2 nights).
  - `k = 3` (available cohort): `N = 42` (`>= 4` usable nights).
  - `k = 1..4` (common cohort on fixed `E_4`, 64 nights): `N = 33` (`>= 5` usable nights).

### Table 1: BIDSleep 5-Fold Subject-Disjoint Splits ([table1_dataset_splits.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/tables/table1_dataset_splits.csv))

| fold | role | subjects | nights |
| --- | --- | --- | --- |
| 1 | inner_train | 32 | 168 |
| 1 | inner_val | 5 | 24 |
| 1 | outer_test | 10 | 54 |
| 2 | inner_train | 32 | 168 |
| 2 | inner_val | 5 | 25 |
| 2 | outer_test | 10 | 53 |
| 3 | inner_train | 33 | 175 |
| 3 | inner_val | 5 | 26 |
| 3 | outer_test | 9 | 45 |
| 4 | inner_train | 33 | 174 |
| 4 | inner_val | 5 | 25 |
| 4 | outer_test | 9 | 47 |
| 5 | inner_train | 33 | 171 |
| 5 | inner_val | 5 | 28 |
| 5 | outer_test | 9 | 47 |

---

## Section 2: Exact Layer-by-Layer Architecture Tables & Parameter Breakdowns

- Extracted directly via PyTorch `named_children()` and forward hooks at full-night context `L = 1200` (`scripts/build_report_architectures.py`).

### Table 2: Submodule Parameter & Tensor Shape Breakdown ([table2_architecture_layer_breakdown.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/tables/table2_architecture_layer_breakdown.csv))

| model | architecture_family | total_params | submodule | module_type | params | pct_of_total | in_shape | out_shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model A / W100 | ConventionalCNNBiGRU | 163940 | hr_encoder | HREncoder | 29568 | 18.04 | (1, 1200, 30) | (1, 1200, 64) |
| Model A / W100 | ConventionalCNNBiGRU | 163940 | acc_encoder | ACCEncoder | 29568 | 18.04 | (1, 1200, 30) | (1, 1200, 64) |
| Model A / W100 | ConventionalCNNBiGRU | 163940 | meta_proj | Sequential | 96 | 0.06 | (1, 1200, 5) | (1, 1200, 16) |
| Model A / W100 | ConventionalCNNBiGRU | 163940 | temporal_gru | GRU | 97920 | 59.73 | Packed((1200, 144)) | Packed((1200, 96)); (4, 1, 48) |
| Model A / W100 | ConventionalCNNBiGRU | 163940 | temp_norm | LayerNorm | 192 | 0.12 | (1, 1200, 96) | (1, 1200, 96) |
| Model A / W100 | ConventionalCNNBiGRU | 163940 | classifier_head | Sequential | 6596 | 4.02 | (1, 1200, 96) | (1, 1200, 4) |
| Official SLAMSS-IFS | OfficialSLAMSSIFSWrapper | 5465768 | cnn | CNN | 5166272 | 94.52 | (1, 36000, 2) | (1, 1200, 517) |
| Official SLAMSS-IFS | OfficialSLAMSSIFSWrapper | 5465768 | s2s | Seq2Seq | 299496 | 5.48 | (1, 1200, 517) | (1, 1200, 4) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | hr_encoder | PrePoolEncoder | 29568 | 15.5 | (1, 1200, 30) | (1200, 64, 7) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | acc_encoder | PrePoolEncoder | 29568 | 15.5 | (1, 1200, 30) | (1200, 64, 7) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | meta_proj | Sequential | 96 | 0.05 | (1, 1200, 5) | (1, 1200, 16) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | intra_film | IntraEpochFiLM | 8320 | 4.36 | (1200, 64, 7) | (1200, 64, 7) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | fine_gru | GRU | 55872 | 29.28 | Packed((1200, 144)) | Packed((1200, 96)); (2, 1, 48) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | mask_pool | MaskWeightedPool1d | 0 | 0.0 | (1, 1200, 96) | (1, 300, 96) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | coarse_gru | GRU | 42048 | 22.04 | Packed((300, 96)) | Packed((300, 96)); (2, 1, 48) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | fusion_proj | Linear | 18528 | 9.71 | (1, 1200, 192) | (1, 1200, 96) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | temp_norm | LayerNorm | 192 | 0.1 | (1, 1200, 96) | (1, 1200, 96) |
| full_v5 | 2-scale multi-resolution FiLM+pooling | 190788 | classifier_head | Sequential | 6596 | 3.46 | (1, 1200, 96) | (1, 1200, 4) |
| p1_100k | ScaledConventionalCNNBiGRU | 98940 | hr_encoder | ScaledModalityEncoder | 16800 | 16.98 | (1, 1200, 30) | (1, 1200, 48) |
| p1_100k | ScaledConventionalCNNBiGRU | 98940 | acc_encoder | ScaledModalityEncoder | 16800 | 16.98 | (1, 1200, 30) | (1, 1200, 48) |
| p1_100k | ScaledConventionalCNNBiGRU | 98940 | meta_proj | Sequential | 96 | 0.1 | (1, 1200, 5) | (1, 1200, 16) |
| p1_100k | ScaledConventionalCNNBiGRU | 98940 | temporal_gru | GRU | 61104 | 61.76 | Packed((1200, 112)) | Packed((1200, 76)); (4, 1, 38) |
| p1_100k | ScaledConventionalCNNBiGRU | 98940 | temp_norm | LayerNorm | 152 | 0.15 | (1, 1200, 76) | (1, 1200, 76) |
| p1_100k | ScaledConventionalCNNBiGRU | 98940 | classifier_head | Sequential | 3988 | 4.03 | (1, 1200, 76) | (1, 1200, 4) |
| P190 | ScaledConventionalCNNBiGRU | 190940 | hr_encoder | ScaledModalityEncoder | 37296 | 19.53 | (1, 1200, 30) | (1, 1200, 72) |
| P190 | ScaledConventionalCNNBiGRU | 190940 | acc_encoder | ScaledModalityEncoder | 37296 | 19.53 | (1, 1200, 30) | (1, 1200, 72) |
| P190 | ScaledConventionalCNNBiGRU | 190940 | meta_proj | Sequential | 96 | 0.05 | (1, 1200, 5) | (1, 1200, 16) |
| P190 | ScaledConventionalCNNBiGRU | 190940 | temporal_gru | GRU | 109200 | 57.19 | Packed((1200, 160)) | Packed((1200, 100)); (4, 1, 50) |
| P190 | ScaledConventionalCNNBiGRU | 190940 | temp_norm | LayerNorm | 200 | 0.1 | (1, 1200, 100) | (1, 1200, 100) |
| P190 | ScaledConventionalCNNBiGRU | 190940 | classifier_head | Sequential | 6852 | 3.59 | (1, 1200, 100) | (1, 1200, 4) |

---

## Section 3: Master Performance Comparison Across All Models & Experiments

### Table 3: BIDSleep 5-Fold OOF Performance (`N = 47`, 3-Seed Participant Means, `ddof = 1`) ([table3_bidsleep_master_comparison.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/tables/table3_bidsleep_master_comparison.csv))

| experiment | model | params | macro_f1_pct | macro_f1_sd_pp | accuracy_pct | kappa | wake_f1_pct | light_f1_pct | deep_f1_pct | rem_f1_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-018 / DEC-037 | Model A (ConventionalCNNBiGRU, frozen) | 163940 | 64.79 | 7.45 | 66.18 | 0.5078 | 62.70 | 66.61 | 66.76 | 63.09 |
| EXP-031 A1 | W100 (ConventionalCNNBiGRU, retrained) | 163940 | 66.68 | 7.11 | 67.97 | 0.5309 | 63.67 | 68.25 | 68.14 | 66.66 |
| EXP-021 / DEC-038 | Official SLAMSS-IFS | 5465768 | 67.37 | 6.73 | 68.75 | 0.5378 | 64.25 | 69.44 | 70.27 | 65.51 |
| EXP-029 / DEC-042 | cnn_bigru_200k (Arm b) | 223268 | 66.61 | 7.06 | 68.07 | 0.5308 | 63.48 | 68.64 | 67.87 | 66.43 |
| EXP-029 / DEC-042 | film_only (Arm c) | 172260 | 66.67 | 7.02 | 67.98 | 0.5318 | 63.82 | 67.99 | 68.11 | 66.76 |
| EXP-029 / DEC-042 | pyramid_only (Arm d) | 182468 | 66.92 | 7.05 | 68.47 | 0.5336 | 64.04 | 69.15 | 67.53 | 66.97 |
| EXP-029 / DEC-042 | full_v5 (RCCLoss) | 190788 | 66.74 | 6.81 | 68.17 | 0.5295 | 64.37 | 68.66 | 67.71 | 66.21 |
| EXP-029 / DEC-042 | full_v5_ce (Pure CE) | 190788 | 66.95 | 6.84 | 68.35 | 0.5324 | 64.56 | 68.70 | 67.85 | 66.70 |
| EXP-030 / DEC-043 | p1_100k (Selected Elbow) | 98940 | 66.67 | 7.57 | 67.89 | 0.5303 | 64.21 | 68.02 | 67.81 | 66.65 |
| EXP-030 / DEC-043 | p3_268k | 268260 | 66.43 | 7.08 | 67.79 | 0.5251 | 64.47 | 68.26 | 67.11 | 65.89 |
| EXP-030 / DEC-043 | p4_404k | 403524 | 66.50 | 7.16 | 67.81 | 0.5275 | 63.99 | 68.17 | 67.53 | 66.32 |
| EXP-030 / DEC-043 | p5_666k | 665556 | 66.17 | 7.02 | 67.46 | 0.5216 | 63.84 | 67.97 | 67.45 | 65.43 |
| EXP-030 / DEC-043 | p6_1051k | 1051492 | 66.25 | 6.84 | 67.51 | 0.5244 | 63.63 | 67.76 | 67.75 | 65.87 |
| EXP-031 / DEC-044 | P190 (Selected Sec.9) | 190940 | 66.60 | 6.85 | 67.92 | 0.5293 | 64.14 | 68.23 | 67.47 | 66.54 |
| EXP-032 / DEC-045 | M1 (MultiScaleTCNBiGRU) | 1402468 | 65.21 | 6.86 | 67.45 | 0.5244 | 61.58 | 68.02 | 66.16 | 65.07 |
| EXP-032 / DEC-045 | M2 (+ LinearChainCRF4) | 1402492 | 65.18 | 6.84 | 67.44 | 0.5232 | 61.55 | 68.18 | 65.72 | 65.27 |
| EXP-032 / DEC-045 | M3 (Teacher-Distilled + CRF) | 1402492 | 67.38 | 6.53 | 69.75 | 0.5548 | 64.08 | 70.66 | 68.42 | 66.37 |

### Table 4: EXP-031 Single-Factor Width Sweep (`W050`–`W250`) ([table4_exp031_width_sweep.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/tables/table4_exp031_width_sweep.csv))

| arm | w | n_params | int8_in_memory_kib | eligible_int8_400kib | macro_f1_mean_pct | delta_vs_A_pp | wilcoxon_p_vs_A | delta_vs_SLAMSS_pp | wilcoxon_p_vs_SLAMSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W050 | 0.5 | 43108 | 81.00 | True | 65.61 | 0.82 | 0.0061 | -1.76 | 4.90e-04 |
| W075 | 0.75 | 93796 | 176.96 | True | 66.26 | 1.47 | 6.02e-06 | -1.11 | 0.0067 |
| A | 1.0 | 163940 | 309.91 | True | 64.79 | 0.00 | 1.0000 | -2.58 | 1.98e-06 |
| P190 | 1.08 | 190940 | 375.16 | True | 66.60 | 1.80 | 2.96e-06 | -0.77 | 0.0727 |
| B | 1.17 | 223268 | 368.04 | True | 66.61 | 1.82 | 4.18e-08 | -0.76 | 0.0490 |
| W125 | 1.25 | 253540 | 479.86 | False | 66.70 | 1.91 | 1.22e-06 | -0.67 | 0.0216 |
| W150 | 1.5 | 362596 | 686.82 | False | 66.31 | 1.52 | 4.16e-05 | -1.06 | 0.0057 |
| W200 | 2.0 | 639076 | 1211.72 | False | 66.01 | 1.22 | 6.96e-04 | -1.36 | 0.0024 |
| W250 | 2.5 | 993380 | 1884.63 | False | 66.34 | 1.55 | 1.85e-04 | -1.03 | 0.0210 |

- **Table 4 Notes:** `P190` (`72,100,64`) and `B` (`64,128,64`) are non-uniform width points ordered by `n_params` for `H-SCALE`; `wilcoxon_p_vs_A` and `wilcoxon_p_vs_SLAMSS` are unadjusted pairwise tests (`P190` `int8_in_memory_kib = 375.16 KiB < 400 KiB`, serialized `402.91 KiB`).
- **EXP-031 Confirmatory Hypothesis Verdicts (`N = 47`, Holm-corrected across `{H-ARCH, H-SCALE}`):**
  - **`H-ARCH` (`full_v5` vs. `P190`):** `+0.14 pp` `[95% CI: -0.34, +0.61 pp]`, raw Wilcoxon `p = 0.4037`, `p_Holm = 0.4037`, `d_z = +0.086`, `26W/21L` -> **`NOT SUPPORTED (practically)`**.
  - **`H-SCALE` (Page's L across 9 widths, original with frozen `Model A`):** `L = 10938.0`, `z = +2.493`, raw two-sided `p = 0.01268`, Holm-adjusted two-sided `p_Holm = 0.0254` -> **`SUPPORTED`**.
  - **Amendment A1 (`W100` Retrained Control, `R1: recipe effect`):** `W100` (`163,940` params under `EXP-031` recipe) achieved `66.68% ± 7.11%` (`Δ_R = +1.89 pp` `[+1.23, +2.56]`, raw `p = 2.12e-6` vs. frozen `Model A`; `Δ_P = +0.08 pp` `[-0.29, +0.46]`, `p_Holm = 1.0000` vs. `P190`; `W100 - SLAMSS-IFS`: `-0.69 pp`, 95% CI `[-1.44, +0.09]`, raw Wilcoxon `p = 0.04777`, `p_Holm = 0.1433` — raw `p < 0.05` but Holm-adjusted `p` is not significant, so a difference cannot be excluded; `H-SCALE-R` Page's `L = 10614.0`, `z = +0.265`, raw two-sided `p = 0.7912`).

### Table 5: External Zero-Shot PhysioNet SleepAccel v1.0.0 (`n = 25` `qc_clean` & `N = 31` `all`) ([table5_sleepaccel_zeroshot.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/tables/table5_sleepaccel_zeroshot.csv))

| model | em_mode | qc_clean_n25_single_f1_pct | qc_clean_n25_ensemble_f1_pct | all_n31_single_f1_pct | all_n31_ensemble_f1_pct |
| --- | --- | --- | --- | --- | --- |
| Model A (163,940p) | em_off | 60.51 | 62.94 | 60.65 | 62.94 |
| Official SLAMSS-IFS (5,465,768p) | em_off | 61.08 | 62.79 | 62.27 | 64.14 |
| full_v5 (190,788p) | em_off | 61.78 | 63.51 | — | — |
| full_v5 (190,788p) | em_on (DampedEMCalibrator) | 59.73 | 61.3 | — | — |
| p1_100k (98,940p) | em_off | 60.88 | 62.51 | 61.10 | 62.51 |
| P190 (190,940p) | em_off | 61.52 | 63.42 | 61.77 | 63.94 |
| W100 (163,940p, EXP-033) | em_off | 61.23 | 62.93 | 61.58 | 63.43 |
| M3 (1,402,492p, distilled, EXP-032) | em_off | 61.43 | 62.63 | 61.76 | 63.01 |

- **Notes:** `full_v5` was evaluated on `qc_clean` (`n = 25`) only (`—` on `all` `N = 31`). On `qc_clean` (`n = 25`) single-model `Model A` vs. `SLAMSS-IFS`, accuracy (`10/15` wins) and Wake F1 (`15/10` wins) share identical Wilcoxon `W = 106.0` (`p = 0.1336`) due to an exact mirrored signed-rank sum pattern (`W+ = 106.0, W- = 219.0` vs. `W+ = 219.0, W- = 106.0`).

### Table 6: SleepAccel v1.0.0 Single-Night Bland-Altman Agreement (`qc_clean` `n = 25`, `mean_single_model`) ([table6_sleepaccel_bland_altman.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/tables/table6_sleepaccel_bland_altman.csv))

| arm | metric | bias | loa_low | loa_high | mae | pearson_r |
| --- | --- | --- | --- | --- | --- | --- |
| ModelA | deep_pct_tst | 10.65 | -2.39 | 23.69 | 10.72 | 0.49 |
| ModelA | rem_pct_tst | 10.13 | -5.92 | 26.18 | 11.12 | 0.48 |
| ModelA | sleep_efficiency_pct | 2.62 | -5.6 | 10.83 | 3.58 | 0.38 |
| ModelA | sol_min | 2.71 | -10.91 | 16.33 | 5.73 | 0.66 |
| ModelA | tst_min | 11.81 | -23.79 | 47.4 | 15.87 | 0.91 |
| ModelA | waso_min | -13.57 | -43.05 | 15.9 | 15.09 | 0.4 |
| SLAMSS_IFS | deep_pct_tst | 8.67 | -3.09 | 20.42 | 8.81 | 0.5 |
| SLAMSS_IFS | rem_pct_tst | 7.21 | -10.3 | 24.73 | 9.64 | 0.2 |
| SLAMSS_IFS | sleep_efficiency_pct | 3.24 | -5.02 | 11.49 | 4.26 | 0.26 |
| SLAMSS_IFS | sol_min | 0.53 | -14.72 | 15.78 | 5.53 | 0.54 |
| SLAMSS_IFS | tst_min | 14.59 | -21.68 | 50.87 | 19.11 | 0.9 |
| SLAMSS_IFS | waso_min | -14.16 | -44.39 | 16.07 | 16.34 | 0.28 |
| W100 (EXP-033) | deep_pct_tst | 7.10 | -5.18 | 19.38 | 7.65 | 0.54 |
| W100 (EXP-033) | rem_pct_tst | 12.23 | -2.60 | 27.06 | 12.24 | 0.50 |
| W100 (EXP-033) | sleep_efficiency_pct | 2.67 | -5.25 | 10.60 | 3.68 | 0.42 |
| W100 (EXP-033) | sol_min | 3.24 | -7.86 | 14.34 | 5.03 | 0.78 |
| W100 (EXP-033) | tst_min | 12.12 | -22.31 | 46.55 | 16.37 | 0.92 |
| W100 (EXP-033) | waso_min | -14.48 | -43.11 | 14.15 | 15.76 | 0.42 |
| p1_100k | tst_min | 12.07 | -21.42 | 45.55 | 16.22 | 0.92 |
| p1_100k | sol_min | 3.21 | -8.27 | 14.68 | 5.19 | 0.76 |
| p1_100k | waso_min | -14.33 | -43.13 | 14.47 | 15.85 | 0.41 |
| p1_100k | sleep_efficiency_pct | 2.65 | -4.98 | 10.29 | 3.63 | 0.46 |
| p1_100k | deep_pct_tst | 9.03 | -3.65 | 21.71 | 9.36 | 0.44 |
| p1_100k | rem_pct_tst | 11.08 | -2.81 | 24.98 | 11.36 | 0.55 |
| P190 | tst_min | 8.69 | -26.18 | 43.57 | 15.19 | 0.92 |
| P190 | sol_min | 4.57 | -7.66 | 16.8 | 5.99 | 0.74 |
| P190 | waso_min | -12.47 | -42.05 | 17.11 | 14.66 | 0.41 |
| P190 | sleep_efficiency_pct | 1.9 | -6.08 | 9.87 | 3.42 | 0.45 |
| P190 | deep_pct_tst | 7.53 | -6.53 | 21.59 | 8.22 | 0.43 |
| P190 | rem_pct_tst | 11.17 | -3.98 | 26.32 | 11.55 | 0.44 |

### Table 7: Multi-Night Personalization (`EXP-033` `W100` Recipe-Consistent & Legacy `EXP-019-K` Clean-ZS Frozen `Model A` Older Recipe) ([table7_personalization_k_curve.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/tables/table7_personalization_k_curve.csv))

| Cohort | k | N | Mean_ZS_Clean | Mean_FT | Delta_pp | CI_lower | CI_upper | Wilcoxon_W | p_wilcoxon_two_sided_exact | p_holm_5test_E4_allk | Holm_Sig_5test_E4_allk | p_holm_5test_driver | Holm_Sig_5test_driver | p_holm_5test_prereg | Holm_Sig_5test_prereg | Wins | Losses | Ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Common (N=33, E4) | 1 | 33 | 63.54% | 65.28% | +1.75 | 0.630755566958631 | 2.92591062425216 | 136.0 | 0.008788360049948 | 0.0264 | True | 0.0352 | True | — | — | 24 | 9 | 0 |
| Common (N=33, E4) | 2 | 33 | 63.54% | 65.41% | +1.88 | 0.5881222105592135 | 3.247307834469871 | 123.0 | 0.0040232583414763 | 0.0161 | True | — | — | — | — | 24 | 9 | 0 |
| Common (N=33, E4) | 3 | 33 | 63.54% | 65.82% | +2.29 | 0.6881757142479258 | 3.9742096368935846 | 144.0 | 0.0136903915554285 | 0.0264 | True | 0.0356 | True | 0.0593 | False | 24 | 9 | 0 |
| Common (N=33, E4) | 4 | 33 | 63.54% | 66.25% | +2.71 | 1.1669350853626916 | 4.328169813251022 | 110.0 | 0.0016955372411757 | 0.0085 | True | 0.0085 | True | — | — | 24 | 9 | 0 |
| Available (Ek, k=1) | 1 | 46 | 63.95% | 64.18% | +0.23 | -0.5887402165645828 | 1.0479528387121155 | 491.0 | 0.5956819514196638 | — | — | — | — | — | — | 25 | 21 | 0 |
| Available (Ek, k=2) | 2 | 46 | 64.58% | 65.58% | +0.99 | -0.1838552314685274 | 1.9920103650070424 | 271.0 | 0.0027093214328317 | — | — | — | — | — | — | 33 | 13 | 0 |
| Available (Ek, k=3) | 3 | 42 | 64.25% | 65.67% | +1.42 | 0.1459722440429792 | 2.5794376071624185 | 213.0 | 0.0023189899457065 | — | — | — | — | — | — | 29 | 13 | 0 |
| Available (Ek, k=4) | 4 | 33 | 63.54% | 66.25% | +2.71 | 1.1669350853626916 | 4.328169813251022 | 110.0 | 0.0016955372411757 | — | — | — | — | — | — | 24 | 9 | 0 |

- **`EXP-033` (`DEC-046`, Recipe-Consistent `W100` & Multi-Backbone, `N = 46` / `E_4 N = 33`):**
  - **`W100` (`H-A2` `FT-2 - ZS`, `N=46`):** mean `+0.58 pp`, median `+0.75 pp`, 95% CI `[-0.06, +1.22]`, `30W/16L/0T` (raw two-sided `p = 0.0263`, `p_Holm = 0.0263`, `d_z = 0.259`) -> **`NOT SUPPORTED (practically)`**. Interpretation: typical participants gained about `+0.6` to `+0.75 pp` (`W100` median `+0.75 pp`, 95% CI `[+0.11, +1.28]`, 30 wins to 16 losses), which is statistically detectable but below the preregistered `+1.5 pp` practical threshold; the mean CI (`+0.58 pp`, 95% CI `[-0.06, +1.22]`) includes zero because a few participants lost more, and every fold selected the boundary configuration (`lr = 3e-5`, head + last GRU), consistent with a small effect.
  - **`W100` (`S-SPEC` `FT-2 - FT-other-2`, 5 donor draws, `N=46`):** mean `+1.45 pp`, median `+1.31 pp`, 95% CI `[+0.89, +2.02]`, `38W/8L/0T` (raw two-sided `p = 1.23e-5`, `p_Holm = 2.47e-5`, `d_z = 0.730`) -> **`INCONCLUSIVE`** (`FT-other-2 - ZS = -0.87 pp`).
  - **`W100` Calibration-Depth (`E_4`, `N=33`, `k=1..4`):** mean `+0.35`, `+0.53`, `+0.74`, `+0.73 pp` (`[-0.22, +1.77]`, median `+0.47 pp`, `20W/13L/0T`); Page's `L = 835.0`, `z = +0.573`, raw two-sided `p = 0.5667`, raw one-sided `p = 0.2834` -> **NOT CLAIMABLE**.
  - **Descriptive Backbones (`FT-2 - ZS`, `N=46`):** `p1_100k` mean `+0.51 pp` (median `+0.70 pp`, `[+0.03, +0.95]`, `31W/15L/0T`), `P190` mean `-0.02 pp` (median `+0.69 pp`, `[-1.09, +0.85]`, `31W/15L/0T`, left-skewed), `full_v5` mean `+0.45 pp` (median `+1.03 pp`, `[-0.57, +1.41]`, `28W/18L/0T`).
  - **Legacy `Model A` (`EXP-019` older recipe, `N=46`):** mean `+1.01 pp` (`+1.0054 pp`), median `+1.55 pp`, 95% CI `[-0.17, +1.99]`, `34W/12L/0T`, two-sided `p = 0.00261` (locked one-sided `p = 0.00131`).
  - **`GATE_tau` (Descriptive, no gating benefit claimed):** On `W100` (`τ = 1%..5%`, `1..8` abstained), mean `Δ = +0.65` to `+0.34 pp`, `min Δ = -5.18 pp`.
  - **`W100` Zero-Shot SleepAccel (`em_off`, descriptive, 2.36 pp 80% MDE):** `qc_clean (n=25)` `61.23%` single / `62.93%` ensemble (`+0.14 pp [-1.28, +1.56]` vs. `SLAMSS-IFS`, `p = 0.9578` — no statistically significant difference; a difference is not excluded); `all (N=31)` `61.58%` / `63.43%`. No clinical-agreement claim on Bland-Altman (`REM% bias +12.23%`, `Deep% bias +7.10%`).
- **Exploratory Page's L (`N = 33`, `E_4`, frozen `Model A` older recipe):** `L = 863.0`, `z = 2.261`, raw one-sided `p = 0.01187` (raw two-sided `p = 0.02374`; 4-test Holm-adjusted one-sided `p = 0.04748`; 5-test Holm-adjusted one-sided `p = 0.05935`).

### Table 7b: `GATE_tau` Safety Guardrail Across Models (`N = 46`; `EXP-033` Recipe-Consistent & Legacy `Model A` Older Recipe) ([table7b_exp020_gate_tau_comparison.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/author_report_bundle/tables/table7b_exp020_gate_tau_comparison.csv))

| experiment | model | n_subjects | zs_f1_pct | ft2_f1_pct | gate_tau_f1_pct | gate_vs_zs_pp | n_abstained |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-020 / DEC-040 | Model A (163,940p, frozen) | 46 | 64.58 | 65.60 | 65.91 | 1.33 | 1 |
| EXP-029 / DEC-042 | full_v5 (190,788p) | 46 | 66.67 | 67.59 | 67.86 | 1.19 | 1 |
| EXP-030 / DEC-043 | p1_100k (98,940p) | 46 | 66.84 | 66.27 | 66.74 | -0.10 | 3 |
| EXP-031 / DEC-044 | P190 (190,940p) | 46 | 66.42 | 66.09 | 66.55 | 0.13 | 3 |

- **Exploratory Donor Control (`S1`, `EXP-019`, frozen `Model A` older recipe, `N = 46`):** `FT-2` (`65.59%`) vs. `FT-other-2` (`62.86%`) -> `+2.73 pp` `[+1.05, +4.32 pp]`, `p = 9.0e-6` (`FT-other-2` is `-1.72 pp` below `ZS = 64.58%`).
- **Exploratory Unsupervised InfoMax (`S3` / `UA-2`, `DEC-036`, frozen `Model A` older recipe, `N = 46`):** `64.58% -> 64.79%`, `+0.21 pp` `[+0.04, +0.39 pp]`, `p_sup = 0.0322` (`SUB-BENCHMARK GAIN < +1.00 pp`).

---

## Section 4: Inference Time, Memory Footprint & Dynamic INT8 Quantization Difference

### Table 8: Dynamic INT8 Edge Profile & Interleaved CPU Latency (`L = 1200`, `B = 1`, `AMD Ryzen 7 7700 8-Core Processor`, All INT8 CVs `<= 5%`)

| model | total_params | quantized_params | fp32_weight_payload | int8_in_memory_kib | mmacs_L1200 | interleaved_1t_int8_ms | interleaved_4t_int8_ms | interleaved_4t_fp32_ms | oof_int8_delta_pp (95% CI) | oof_int8_wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p1_100k | 98940 | 69796 | 386.48 KiB | 182.00 | 456.1 | 47.07 | 41.02 | 40.33 (CV 4.83%) | -0.0134 [-0.0489, +0.0210] | 0.4656 |
| W100 / Model A | 163940 | 112932 | 640.39 KiB | 309.54 | 794.1 | 53.63 | 43.38 | not reported (CV > 5%) | +0.0011 [-0.0277, +0.0312] | 0.7892 |
| full_v5 | 190788 | 131332 | 745.27 KiB | 360.50 | 847.7 | 43.73 | 31.53 | not reported (CV > 5%) | -0.0113 [-0.0566, +0.0339] | 0.9041 |
| P190 | 190940 | 126532 | 745.86 KiB | 375.16 | 984.2 | 59.41 | 46.46 | not reported (CV > 5%) | +0.0244 [+0.0017, +0.0480] | 0.0836 |
| Official SLAMSS-IFS | 5465768 | — | 20.85 MiB | — | 40773.2 | 3292.75 | 3186.52 | 3261.39 (CV 3.21%) | — | — |

- **Amendment A1 Section 7 Latency Deviation Table (`docs/verification/interleaved_cpu_latency_fp32_4t_rerun.json`):**
  - Interleaved Dynamic INT8 (4T and 1T) and 1T FP32 timings all satisfied `CV <= 5%` and serve as the primary latency figures.
  - Under 4-thread FP32 pinned to 4 dedicated physical cores (`taskset -c 0,2,4,6`, `powersave` governor, up to 3 attempts of 30 warm-up + 200 timed passes), `p1_100k` passed on attempt 3 (`40.33 ms`, `CV = 4.83%`) and `Official SLAMSS-IFS` passed on attempt 1 (`3261.39 ms`, `CV = 3.21%`), whereas `W100` (best `CV = 6.29%`), `full_v5` (best `CV = 31.92%`), and `P190` (best `CV = 11.56%`) remained above `5%` due to multi-thread frequency scaling jitter under `powersave` and are marked `not reported (CV > 5%)` per Amendment A1 Section 7.
  - Process peak runtime RAM (`peak_rss_mib`: `800.59 MiB` 1T, `799.33 MiB` 4T) is reported strictly separately from serialized/in-memory weight payload (`KiB`).

---

## Section 5: Direct Responses to Supervisor Questions & Methodological Critiques

1. **Capacity vs. Architectural Novelty (`full_v5` vs. `P190` & `p1_100k` / `W100`):**
   - Matched-parameter plain control `P190` (`190,940` params) vs. `full_v5` (`190,788` params) yielded `+0.14 pp` (`95% CI [-0.34, +0.61]`, `p_Holm = 0.4037`, `NOT SUPPORTED`).
   - Plain `W100` (`163,940` params, `66.68% ± 7.11%`) and `p1_100k` (`98,940` params, `66.67% ± 7.57%`) narrowed the gap against `SLAMSS-IFS` (`67.37% ± 6.73%`): `W100` vs. `SLAMSS-IFS` was `-0.69 pp` (`95% CI [-1.44, +0.09]`, raw Wilcoxon `p = 0.04777`, `p_Holm = 0.1433`; raw `p < 0.05` but Holm-adjusted `p` is not significant, so a difference cannot be excluded).
   - **Conclusion:** Gains over frozen `Model A` (`64.79% ± 7.45%`) are driven by training recipe (`R1: recipe effect`) rather than FiLM or 2-scale multi-resolution coupling; architectural novelty is explicitly not claimed for `full_v5`.
2. **Why `em_on` (`DampedEMCalibrator`) Failed on SleepAccel (`EXP-029`):**
   - `em_on` (`alpha=0.5, n_iter=3, T=1.0`) dropped `full_v5` SleepAccel `qc_clean` Macro-F1 from `61.78% / 63.51%` (`em_off`) to `59.73% / 61.30%`, which is consistent with cross-modality scoring/covariate shift (BIDSleep masked `44.20%` Light / `18.91%` Deep vs. SleepAccel `qc_clean` `57.74%` Light / `11.08%` Deep).
3. **Hardware Deployment Boundaries (`W0-B` Caveats, `DEC-039`):**
   - Dynamic INT8 quantizes `nn.Linear` + `nn.GRU` (`~68.9%–70.5%` of params), leaving `Conv1d` + normalization in FP32 (`182.00 KiB` for `p1_100k` [`52.9%` reduction], `309.54 KiB` for `Model A`/`W100` [`51.7%`], `360.50 KiB` for `full_v5` [`51.6%`], `375.16 KiB` for `P190` [`49.7%`]).
   - Because all models use **bidirectional** GRUs over `L = 1200` epochs, they require **full-night sequence buffering on a smartwatch SoC or companion smartphone**, not causal 1-epoch streaming on a 512 KB SRAM bare-metal MCU.
