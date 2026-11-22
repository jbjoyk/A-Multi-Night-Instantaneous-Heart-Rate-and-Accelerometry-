# Execution Specification: EXP-019B & EXP-020 Exploratory Closure

- **Status:** Revised Execution Spec (Post-Goldfish Resolution; Strict NO-CODE / Pre-Execution Gate)
- **Date:** 2026-11-22
- **Governance Authority:** `RESEARCH_STATE.md` (DEC-029, DEC-037, DEC-038, DEC-039)
- **Classification:** Design-Informed Exploratory Closure (CPU-only `EXP-019B`; `cuda:0` `EXP-020`)
- **Execution Guard:** No training/analysis scripts written and no GPU jobs launched until explicit user authorization.
- **Target Scope & File Boundary:** New files created only under `experiments/EXP-019B/` and `experiments/EXP-020/`. **Modified/Deleted existing files: None.**

---

## 1. Mandatory Prior-Access Disclosure, Terminology & Claim Boundaries

> [!WARNING]
> **PRIOR-ACCESS DISCLOSURE (EXP-019B & EXP-020):**
> Both protocols are finalized **after** inspection of locked EXP-019 Stage C (outer-test evaluation) results ($\bar{\Delta}_{\text{FT-2}-\text{ZS}} = +1.01\text{ pp}$, 95% CI $[-0.17, +1.99]\text{ pp}$, 34/46 responders, worst-case outlier `Bidslab34` at $-16.34\text{ pp}$ with 0.0% Deep sleep on Night 2 and 1.3% on Night 3), the Phase A heterogeneity audit (`experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md`, where `min_stage_frac_cal` was already tested against EXP-019 delta at $\rho = +0.1063, p = 0.4819, q = 0.9367$), and locked EXP-021 paired benchmark results ($\bar{d}_{\text{Model A}-\text{SLAMSS}} = -2.58\text{ pp}$, $N=47$).
>
> Accordingly, **neither EXP-019B nor EXP-020 is a prospective confirmatory study**. Confirmatory verdict labels (`SUPPORTED`, `NOT SUPPORTED`, `CONFIRMED`) are strictly prohibited. All outputs must be reported as *design-informed exploratory findings* on the BIDSleep cohort; independent confirmation is deferred to external validation (`EXP-027`).

### Inline Terminology
- **IHR:** Instantaneous Heart Rate (HealthKit optical PPG heart rate in bpm).
- **Model A / `ZS`:** Frozen `SleepStagingFinalModel_v1.0` (`ConventionalCNNBiGRU`, 163,940 params, trained in `EXP-018`) evaluated zero-shot without subject adaptation.
- **`FT-2`:** Supervised fine-tuning on a subject's first $k=2$ usable calibration nights using the locked per-fold `EXP-019` Stage D (inner-val hyperparameter selection) schedules.
- **`SLAMSS-IFS`:** Zero-shot baseline benchmark model evaluated in `EXP-021` (`experiments/EXP-005/slamss_ifs_baseline.py`).
- **`TSK-06` (`DEC-039`):** Prior bug in `run_tsk06_k_curve.py` where in-place mutation of a shared `base_model` across subjects contaminated zero-shot (`ZS`) evaluations.

### Scope & Non-Goals
- **In Scope:** (1) CPU-only subject-level correlation closure (`EXP-019B`) across `EXP-019` and `EXP-021` deltas; (2) strictly nested-CV evaluation of KL-regularized and stage-coverage-gated adaptation (`EXP-020`) on the 15 frozen `SleepStagingFinalModel_v1.0` checkpoints.
- **Out of Scope:** Modifying locked `EXP-019`/`EXP-021` metrics or checkpoints; changing model architecture (`ConventionalCNNBiGRU`); re-tuning base `FT-2` learning rates, epochs, or layer scopes; running external dataset validation (`EXP-027`).

---

## 2. Reused Helpers & Frozen Assets (No Duplication)

All execution scripts must import existing verified helpers and read locked manifests/artifacts directly:

| Asset / Helper | Exact Path & Symbol | Role in Closure |
|---|---|---|
| Frozen Model Interface | `experiments/STAGE2_FINAL_MODEL/final_model_pipeline.py` (`FinalSleepStagingSystem`, `SCOPE_REGEX_MAP`, `load_frozen_checkpoint`, `_clone_frozen_backbone`, `make_predictor`, `audit_calibration_stage_coverage`, `verify_wearable_quality_records`) | SHA-256 checkpoint loading, authoritative `SCOPE_REGEX_MAP`, clean per-subject backbone cloning, predictor setup (`L=1200`, `neutralize_circadian=True`), stage coverage audit |
| Adaptation Primitives | `experiments/EXP-019/exp019_adaptation_core.py` (`compute_wearable_quality_mask`, `prepare_calibration_batch`, `adapt_supervised_ft`) | Quality mask definition, calibration batching (`B=2`, `L=1200`), reference `FT-2` training loop |
| Inner-Val Selection Pattern | `experiments/EXP-019/run_exp019_stage_d.py` (`evaluate_records_macro_f1`, `select_best_config`) | 4-decimal rounding convention (`round(mean_macro_f1, 4)`) and inner-val evaluation interface |
| Locked Stage D Schedules | `experiments/EXP-019/exp019_stage_d_selected_configs.json` | Exact per-fold `FT-2` `(lr, epochs, scope)` configurations (`folds.{1..5}.FT-2`) |
| Statistical Utilities | `experiments/EXP-019/audit_heterogeneity_mechanisms.py` (`compute_bh_fdr`); `experiments/EXP-021/analyze_tsk08_paired_benchmark.py` (`participant_bootstrap_ci`, `validate_subject_seed_completeness`, `load_model_a_oof_dataframe`) | Benjamini-Hochberg FDR, 10,000-resample subject bootstrap 95% CIs, 3-seed completeness checks |
| Audit Deliverables & Raw HR | `research_knowledge/updates/bidsleep_subject_diagnostic_table.csv`; `research_knowledge/updates/bidsleep_ihr_sampling_quality_summary.csv`; `research_knowledge/updates/bidsleep_calibration_class_coverage_summary.csv`; `cache/prep_57738d534b/{subject_id}__{night}.json`; `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/{subject_id}/{night}/hr.csv` | Cohort eligibility (`qualifies_exp019_eval`), IHR sampling metadata (`hr_file.n_rows_valid`, `hr_first_offset_s`, `hr_last_offset_s`, `rec_start_unix`), and raw unstandardized bpm values |
| Manifests & Locked Inputs | `experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json`; `experiments/EXP-018/checkpoints/modelA_L1200_fold{1..5}_seed{42,123,2024}.pt`; `experiments/EXP-014/EXP014_MANIFESTS/fold_{1..5}.json`; `review_package/fold_manifests.csv`; `review_package/night_inventory.csv`; `experiments/EXP-019/exp019_subject_metrics.csv`; `experiments/EXP-021/slamss_ifs_matched_subject_metrics.csv`; `experiments/EXP-021/slamss_ifs_paired_subject_deltas.csv` | 15 frozen checkpoints + SHA-256 digests; sole authoritative fold splits (`fold_{1..5}.json`); usable night ordering; immutable outer-test `ZS`/`FT-2` and `SLAMSS-IFS` subject metrics |

---

## 3. Part 1: EXP-019B — Subject-Level Heterogeneity Closure (CPU-Only)

### 3.1 Outcomes & Cohort Denominators ($N=46$ vs. $N=47$)

| Primary Family | Target Delta ($\Delta_i$) | Cohort & Night Accounting | Source | Feature Extraction Window (`W`) |
|---|---|---|---|---|
| **Family 1: EXP-019 Personalization** | $\text{Macro-F1}_{\text{FT-2}, i} - \text{Macro-F1}_{\text{ZS}, i}$ (pp, 3-seed mean) | **$N = 46$** (`Bidslab49` excluded because $n_{\text{usable}} = 2 \implies n_{\text{eval}} = 0$). **244 usable nights** = 92 cal (`[:2]`) + 152 eval (`[2:]`). | `experiments/EXP-019/exp019_subject_metrics.csv` | Strictly calibration nights (chronological usable nights `[:2]` from `review_package/night_inventory.csv`), blind to eval nights |
| **Family 2: EXP-021 Model Gap** | $\text{Macro-F1}_{\text{Model A}, i} - \text{Macro-F1}_{\text{SLAMSS-IFS}, i}$ (pp, 3-seed mean) | **$N = 47$** (`Bidslab49` included with 2 usable nights). **246 usable nights** total. | `experiments/EXP-021/slamss_ifs_paired_subject_deltas.csv` | All usable nights per subject (zero-shot benchmark has no cal/eval split) |

*Descriptive Sensitivity & Subgroup Checks (Excluded from BH-FDR):*
1. **Window & Cohort Isolation for Family 2:** (i) $N=46$ overlap (excluding `Bidslab49`) using **all usable nights** features (isolates cohort exclusion $N=47 \to 46$); (ii) $N=46$ overlap using **calibration-only (`[:2]`)** features (isolates window truncation `all_nights` $\to$ `[:2]`).
2. **Descriptive Responder vs. Non-Responder Subgroup Check (Family 1, $N=46$):** Compare feature distributions between responders ($\Delta_i > 0$, $N=33$ or locked responder count from `exp019_subject_metrics.csv`, verified at runtime; note prior Phase A split $N=33$ vs. $N=13$ / $34$ vs. $12$ depending on $>0$ vs. $\ge 0$ boundary) using descriptive Mann-Whitney $U$ / Hodges-Lehmann medians (no confirmatory claims).

### 3.2 Subject-Level Feature Specifications (6 Primary Predictors)

Computed deterministically per subject over window `W`. **Never pull raw BPM from z-scored `rec["ihr"]` cache tensors.**

| # | Feature Name | Exact Data Source & Formula | Units / Bounds | Integrity Check |
|---|---|---|---|---|
| 1 | `min_stage_frac` | From `cache/prep_57738d534b` (`prepare_dataset_records()`), restricted to `y in {0,1,2,3}`: $\min_{c \in \{0,1,2,3\}} N_c / N_{\text{valid}}$ | $[0, 0.25]$ | Exact match to `audit_calibration_stage_coverage()` (e.g., `Bidslab34` = `0.0068` cal Deep) and `k2_min_stage_pct / 100` in `bidsleep_subject_diagnostic_table.csv` |
| 2 | `stage_entropy` | Base-2 Shannon entropy over `y in {0,1,2,3}` in `W`: $H(P_W) = -\sum_{c: P_W(c) > 0} P_W(c)\log_2 P_W(c)$ | bits, $[0, 2.0]$ | $0\log_2 0 := 0$; finite for all subjects |
| 3 | `eff_ihr_hz` | From `cache/prep_57738d534b/{sub}__{night}.json` summed across nights $n \in W$: $\frac{\sum_{n \in W} \text{hr\_file.n\_rows\_valid}_n}{\sum_{n \in W} (\text{hr\_last\_offset\_s}_n - \text{hr\_first\_offset\_s}_n)}$ | Hz ($> 0$) | Matches `raw_hr_rate_hz` in `bidsleep_ihr_sampling_quality_summary.csv` (median `0.197 Hz`); all-nights window cross-checked against `mean_ihr_hz` in `bidsleep_subject_diagnostic_table.csv` |
| 4 | `missingness_rate` | From `cache/prep_57738d534b`: fraction of total 30-s epochs in `W` satisfying the boolean union `(~compute_wearable_quality_mask(rec["ihr"], rec["accel"])) \| (rec["y"] < 0) \| (rec["y"] >= 4)` | $[0, 1]$ | Strictly $< 1.0$ for all 47 subjects; no double-counting |
| 5 | `baseline_hr_mean` | From raw `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/{sub}/{night}/hr.csv` (col 0 = Unix timestamp s, col 1 = raw bpm): arithmetic mean of finite col 1 values falling inside valid unmasked 30-s epochs `[rec_start_unix + 30*e, rec_start_unix + 30*(e+1))` (where `rec_start_unix` is from `{sub}__{night}.json`) across nights in `W` | bpm | Physiological sanity check $[35, 120]$ bpm; zero NaNs |
| 6 | `baseline_hr_iqr` | Interquartile range ($Q_{75} - Q_{25}$) of the exact same valid unmasked raw bpm samples from `hr.csv` across nights in `W` | bpm ($\ge 0$) | Non-negative; finite for all 47 subjects |

### 3.3 Statistical Plan & Closed BH-FDR Multiplicity Control

1. **Point Estimate & Rank Test:** Two-sided Spearman rank correlation $\rho$ and two-sided $p$-value for each of the 6 features.
2. **Closed Multiplicity Families:** Benjamini-Hochberg FDR $q$-values computed via `compute_bh_fdr()` strictly over two closed primary families:
   - **Primary Family 1 (`EXP-019`, $N=46$):** $m = 6$ tests (`fdr_q_family`). Note prior Phase A audit tested `min_stage_frac_cal` ($q = 0.9367$ across a larger feature set); both $q$-values are reported transparently.
   - **Primary Family 2 (`EXP-021`, $N=47$):** $m = 6$ tests (`fdr_q_family`).
   - **Pooled Sensitivity:** $m = 12$ across Family 1 + Family 2 (`fdr_q_pooled`).
   - **Descriptive Exclusion:** Sensitivity rows ($N=46$ Family 2 checks), $N=33$ vs. $N=13$ subgroup checks, Oldham headroom check, and leave-`Bidslab34`-out checks are **excluded** from BH-FDR (`fdr_q_family = NaN`, `fdr_q_pooled = NaN`).
3. **Uncertainty:** 10,000-resample subject-level percentile bootstrap 95% CIs on Spearman $\rho$ (RNG seed `42`), using `np.nanpercentile(..., [2.5, 97.5])` and explicitly logging `n_nan_resamples` for any degenerate constant-vector draws.
4. **Mandatory Diagnostic Controls:**
   - **Oldham Headroom Check (Family 1):** Spearman $\rho$ of $(\text{ZS} + \text{FT-2})/2$ vs. $\Delta$ (logged in `exp019b_diagnostics.json`).
   - **Outlier Leverage Check:** Leave-`Bidslab34`-out Spearman $\rho$ for both Family 1 ($N=45$) and Family 2 ($N=46$), recorded in `loo_bidslab34_rho`.
5. **Interpretation Rule:** Features with $q \ge 0.05$ are null; features with $q < 0.05$ are labeled *exploratory correlation candidates* (never causal or confirmed).

### 3.4 Mandatory Pre-Execution Unit Tests for EXP-019B

Create `experiments/EXP-019B/test_exp019b_features.py` (CPU-fast, `< 15 s`, 100% pass required before running closure):
1. **`test_cohort_subject_id_sets`**: Assert Family 1 subject set has exact length `46` and excludes `Bidslab49`, while Family 2 subject set has exact length `47` and includes `Bidslab49`, matching `bidsleep_subject_diagnostic_table.csv` (`qualifies_exp019_eval`).
2. **`test_raw_hr_features_physiological_bounds_and_no_nans`**: Run feature extraction across all 47 subjects (`all_nights`) and 46 subjects (`cal_[:2]`); assert zero NaNs/Infs, `baseline_hr_mean` $\in [35, 120]$ bpm, `baseline_hr_iqr` $\ge 0$, `eff_ihr_hz` $> 0$, `missingness_rate` $\in [0, 1)$, and `min_stage_frac` parity within `1e-6` against `audit_calibration_stage_coverage()` on `Bidslab34` (`0.0068`).
3. **`test_synthetic_spearman_bh_fdr_and_bootstrap`**: Verify `compute_bh_fdr()` monotonicity and exact values on a synthetic 6-test p-value vector, verify `missingness_rate` boolean union does not double-count epochs that fail both `wearable_mask` and `y == -100`, and verify bootstrap `n_nan_resamples` counting + `np.nanpercentile` on a constant-vector edge case.

### 3.5 Planned EXP-019B Artifacts
- `experiments/EXP-019B/test_exp019b_features.py`
- `experiments/EXP-019B/run_exp019b_heterogeneity_closure.py`
- `experiments/EXP-019B/exp019b_subject_features.csv` (47-row table with `cal_` and `all_nights_` columns)
- `experiments/EXP-019B/exp019b_correlation_summary.csv` (12 primary rows + descriptive sensitivity rows: `outcome`, `analysis_type`, `feature`, `n`, `spearman_rho`, `p_value`, `fdr_q_family`, `fdr_q_pooled`, `ci95_low`, `ci95_high`, `n_nan_resamples`, `loo_bidslab34_rho`)
- `experiments/EXP-019B/exp019b_diagnostics.json` (Oldham headroom check, `N=33` vs. `N=13` subgroup medians/Mann-Whitney $p$, sanity-check diffs vs. `bidsleep_subject_diagnostic_table.csv`)
- `experiments/EXP-019B/EXP019B_CLOSURE_REPORT.md`

---

## 4. Part 2: EXP-020 — Regularized & Gated Adaptation Closure (`cuda:0`)

### 4.1 Frozen Backbone, Manifests & Exact Schedule Inheritance

- **Checkpoints:** All 15 frozen `SleepStagingFinalModel_v1.0` checkpoints (`fold{1..5}_seed{42,123,2024}`, 163,940 params each), verified against `experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json` SHA-256 digests before execution.
- **Authoritative Split Manifest:** `experiments/EXP-014/EXP014_MANIFESTS/fold_{1..5}.json` is the **sole source of truth** for `inner_val_subjects` (5 per fold) and `outer_test_subjects` ($N=46$ total across 5 folds; `Bidslab49` excluded from all `inner_val_subjects` and `outer_test_subjects`), cross-checked for 100% outer-test parity against `review_package/fold_manifests.csv`.
- **Chronological Night Split & Inference:** `cal = usable_nights[:2]`, `eval = usable_nights[2:]` (152 eval nights across $N=46$); full-night sliding-window stitching (`L=1200`, `neutralize_circadian=True`) identical to `run_exp019_stage_c.py`.
- **Exact `FT-2` Schedule Inheritance:** `FT-2-KL` inherits 100% of `adapt_supervised_ft()` and `experiments/EXP-019/exp019_stage_d_selected_configs.json` (`folds.{f}.FT-2`):
  - **Optimizer & Clipping:** `torch.optim.Adam(trainable_params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)`, `torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)`.
  - **Per-Fold `(lr, epochs, scope)`:** Fold 1 `(1e-4, 5, all)`, Fold 2 `(3e-5, 15, all)`, Fold 3 `(3e-5, 15, all)`, Fold 4 `(3e-5, 5, head_gru)`, Fold 5 `(3e-5, 5, all)`, using `final_model_pipeline.SCOPE_REGEX_MAP[scope]`.
  - **Batching & Modes:** `prepare_calibration_batch(cal_records, context_len=1200, device=device)` (`B=2`, `L=1200`, `neutralize_circadian=True`); `model.train()` (`dropout=0.2`) during adaptation steps, `model.eval()` for `ZS` precomputation and final evaluation.
  - **Deterministic RNG Seeding:** Immediately before every `(fold, seed, subject, arm)` adaptation call, seed `random`, `np.random`, `torch.manual_seed(seed)`, and `torch.cuda.manual_seed_all(seed)` so dropout masks are deterministic and $\lambda_{\text{KL}} = 0.0$ reproduces locked `FT-2` bit-for-bit.

### 4.2 Experimental Arms & Strictly Nested Inner-Val Selection (Zero Cross-Fold Leakage)

> [!IMPORTANT]
> **STRICT ANTI-LEAKAGE RULE:** `experiments/EXP-019/exp019_subject_metrics.csv` contains outer-test metrics only and is **strictly banned** during inner-val selection. For every fold $f \in \{1..5\}$, inner-val `ZS`, `FT-2`, `FT-2-KL`, `GATE_tau`, and `FT-2-KL + GATE_tau` scores are computed fresh on fold $f$'s 5 `inner_val_subjects` using fold $f$'s 3 checkpoints (`seed={42,123,2024}`). `exp019_subject_metrics.csv` is used solely for outer-test ($N=46$) baseline parity verification and reporting.

1. **Arm (a) — `FT-2-KL`:**
   - **Exact Objective (100% `FT-2` Parity at $\lambda_{\text{KL}} = 0.0$):**
     $$\mathcal{L}_{\text{total}}(\theta; \mathcal{D}_{\text{cal}}) = \mathcal{L}_{\text{FT-2}}(\theta; \mathcal{D}_{\text{cal}}) + \lambda_{\text{KL}} \cdot \frac{\sum_{b,t} M_{b,t} \, D_{\text{KL}}\!\left(\text{softmax}(z_{\text{ZS}, b, t}) \,\|\, \text{softmax}(z_{\theta, b, t})\right)}{\max(1, \sum_{b,t} M_{b,t})}$$
     where $\mathcal{L}_{\text{FT-2}}$ is `out["total_loss"]` (`ConventionalCNNBiGRU(..., use_rcc_loss=True)`, i.e. CE + 0.01 RCC loss) over valid-label mask $M_{b,t} = \text{batch["mask"]}_{b,t} = (y_{b,t} \in \{0,1,2,3\})$ (**no** `wearable_mask` on supervised or KL terms, preserving exact `FT-2` parity), and $p_{\text{ZS}} = \text{softmax}(z_{\text{ZS}}, \text{dim}=-1)$ is precomputed once per `(fold, seed, subject)` from the pristine frozen checkpoint in `.eval()` under `torch.no_grad()`. Implemented via `F.kl_div(F.log_softmax(z_theta, dim=-1), p_ZS, reduction="none").sum(dim=-1)`.
   - **Grid & Selection:** $\lambda_{\text{KL}} \in \{0.1, 1.0, 10.0\}$. For each fold $f \in \{1..5\}$, select $\lambda_{\text{KL}, f}^*$ on fold $f$'s 5 `inner_val_subjects` (3-seed mean) by maximizing `round(mean_inner_macro_f1, 4)` (matching `select_best_config` in `run_exp019_stage_d.py`). **Tie-breaker:** smallest $\lambda_{\text{KL}}$.
2. **Arm (b) — `GATE_tau`:**
   - **Rule:** Compute $m_i = \min_{c \in \{0,1,2,3\}} P_{C,i}(y=c)$ strictly on calibration nights `[:2]` (unrounded `min_stage_frac`). If $m_i < \tau$, deploy frozen `ZS`; otherwise deploy `FT-2`.
   - **Grid & Selection:** $\tau \in \{0.01, 0.02, 0.03, 0.05\}$. For each fold $f \in \{1..5\}$, select $\tau_f^*$ on fold $f$'s 5 `inner_val_subjects` by maximizing `round(mean_inner_macro_f1, 4)`. **Tie-breaker:** smallest $\tau$ (`0.01`).
   - **Secondary Fixed-Grid Diagnostic:** Report outer-test results for all four fixed $\tau \in \{0.01, 0.02, 0.03, 0.05\}$ strictly as an untuned secondary diagnostic table.
3. **Arm (c) — Combined `FT-2-KL + GATE_tau` (Strictly Per-Fold Nested):**
   - **Per-Fold Selection & Gate (Zero Cross-Fold Pooling):** For each fold $f \in \{1..5\}$ independently, evaluate $(\lambda_{\text{KL}, f}^*, \tau)$ over $\tau \in \{0.01, 0.02, 0.03, 0.05\}$ strictly on fold $f$'s 5 `inner_val_subjects` to select $\tau_{f, \text{comb}}^*$ (tie-breaker: smallest $\tau$). Deploy `FT-2-KL + GATE_tau` on fold $f$'s **outer-test** subjects **if and only if** on fold $f$'s 5 `inner_val_subjects`:
     1. Inner-val `FT-2-KL` achieves non-negative mean gain vs. `ZS` ($\bar{\Delta}_{\text{inner}}(\text{FT-2-KL} - \text{ZS}) \ge 0$); **AND**
     2. Inner-val `GATE_tau` triggers $\ge 1$ abstention on fold $f$'s 5 inner-val subjects and $\bar{\Delta}_{\text{inner}}(\text{FT-2-KL + GATE}_\tau) > \bar{\Delta}_{\text{inner}}(\text{FT-2-KL})$.
   - If fold $f$'s gate fails, record `FOLD_{f}_COMBINED_SKIPPED_BY_INNER_VAL_GATE` (for outer-test reporting, if any fold skips, fallback for that fold is `FT-2-KL` or `Arm c` is marked skipped for that fold and excluded from pooled Arm c outer-test claims).

### 4.3 W0-A Weight-Immutability & State-Reset Invariant (`restore_pristine_model()`)

Per `DEC-039` (`TSK-06`), never share a mutable `model` across subjects, folds, arms, or hyperparameters:
- Define `restore_pristine_model(system: FinalSleepStagingSystem, scope_regex: str) -> nn.Module`:
  1. Instantiate a **fresh** `ConventionalCNNBiGRU` clone via `system._clone_frozen_backbone()` (which deep-copies `system._frozen_state` on CPU and moves to `device`), guaranteeing zero shared tensor storage or optimizer state.
  2. Explicitly reset `p.requires_grad = True` for **all** parameters before applying `re.compile(scope_regex)` (preventing Fold 4 `head_gru` freeze leakage into any subsequent fold/arm).
  3. Set `.eval()` before any `ZS` forward pass, and keep the `z_ZS` teacher strictly frozen (`requires_grad=False`, `.eval()`).
- Call `restore_pristine_model()` before **every** `(fold, seed, subject, arm, hyperparam)` adaptation and evaluation pass.

### 4.4 Primary Comparison Family (Holm-Bonferroni) & Secondary Endpoints (Outer $N=46$)

All outer metrics are computed on per-subject 3-seed averaged Macro-F1 across $N=46$ (152 eval nights):
- **Closed Primary Comparison Family (Holm-Bonferroni adjusted $p$-values via paired subject-level permutation/Wilcoxon test, $m =$ number of executed arms among `{FT-2-KL, GATE_tau, FT-2-KL + GATE_tau}`):** Mean paired $\Delta$ vs. `ZS` (pp) + 10,000-resample subject bootstrap 95% CIs (seed `42`, unadjusted exploratory intervals).
- **Secondary Descriptive Endpoints (Unadjusted, No Confirmatory Claims):**
  - Mean paired $\Delta$ vs. `FT-2` (pp) + 95% bootstrap CI.
  - **Tail Risk:** 5th-percentile subject $\Delta$, max negative transfer ($\min_i \Delta_i$), 10th-percentile $\Delta$, catastrophic failure count $N_{\text{harm}} = \sum \mathbb{I}(\Delta_i \le -5.0\text{ pp})$.
  - **Outlier Audit:** `Bidslab34` Macro-F1, $\Delta$ vs. `ZS`, $\Delta$ vs. `FT-2` (locked `FT-2` = $-16.34\text{ pp}$), abstain flag.
  - **Abstention & Selection:** Per-fold $\lambda_{\text{KL}, f}^*$, $\tau_f^*$, $\tau_{f, \text{comb}}^*$, tie-breaker flags, per-fold Arm (c) gate decisions, abstained subject IDs, and the fixed-grid `GATE_tau` sensitivity table.

### 4.5 Mandatory Pre-Flight Unit Tests (`experiments/EXP-020/test_exp020_preflight.py`)

CPU-fast (`< 45 s`), 100% pass required before `cuda:0` launch:
1. **`test_zero_outer_test_hyperparameter_leakage`**: Using `EXP014_MANIFESTS/fold_{1..5}.json`, assert per-fold `set(inner_val_subs).isdisjoint(set(outer_test_subs))`, `Bidslab49` absent from all `inner_val_subs` and `outer_test_subs`, `len(all_outer_test_subs) == 46`, outer-test parity with `review_package/fold_manifests.csv`, `set(cal_nights).isdisjoint(set(eval_nights))`, and wrap inner-val selection so accessing any outer-test subject ID or `exp019_subject_metrics.csv` raises `AssertionError`.
2. **`test_deterministic_checkpoint_sha256_and_param_count`**: Verify all 15 checkpoints match `experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json` SHA-256 digests and have `163_940` parameters.
3. **`test_w0a_pristine_reset_and_multi_sweep_immutability`**: Hash `system._frozen_state`, `system.backbone.state_dict()`, and initial cloned weights across consecutive `(subject, fold, lambda_KL, arm)` sweeps—explicitly testing Fold 4 (`scope=head_gru`) followed by Fold 5 (`scope=all`) and consecutive `lambda_KL` values `{0.1, 1.0, 10.0}` on 2 subjects. Assert: (i) zero mutation of `_frozen_state` and `system.backbone`; (ii) all parameters have `requires_grad=True` before scope regex; (iii) Fold 4 `head_gru` leaves CNN weights identical to `ZS`; (iv) bit-identical `ZS` predictions on Subject 1 before/after sweeps and 4-decimal parity with `exp019_subject_metrics.csv` on Fold 1's first 2 subjects.
4. **`test_kl_loss_boundary_and_mask_invariance`**: Assert (i) $D_{\text{KL}}(\text{softmax}(z_{\text{ZS}}) \| \text{softmax}(z_\theta)) < 10^{-7}$ at initialization ($\theta = \theta_{\text{ZS}}$, both in `.eval()`); (ii) with deterministic RNG seeding, `FT-2-KL` at $\lambda_{\text{KL}} = 0.0$ reproduces `adapt_supervised_ft` (`FT-2`) adapted weights within `1e-5` (and matches locked `FT-2` Macro-F1 on Fold 1's first 2 subjects); (iii) perturbing logits at `-100` positions produces `0.0` change in total loss and zero gradient.
5. **`test_gate_tau_and_per_fold_arm_c_gate`**: Verify synthetic `min_stage_frac` $\in \{0.0, \tau - 10^{-6}, \tau, \tau + 10^{-6}\}$ abstains iff `< tau`, deterministic tie-breaking when inner-val scores tie at 4 decimals, and both pass and `FOLD_{f}_COMBINED_SKIPPED_BY_INNER_VAL_GATE` branches of the per-fold Arm (c) unlock gate.

### 4.6 Planned EXP-020 Artifacts & Compute Budget
- `experiments/EXP-020/test_exp020_preflight.py`
- `experiments/EXP-020/run_exp020_closure.py` (`cuda:0`, single RTX 3060; 15 `(fold, seed)` checkpoints $\times$ 5 inner-val subjects $\times$ (1 `ZS` + 1 `FT-2` + 3 `lambda_KL` values) = 75 inner-val subject-seed adaptations per `lambda_KL`, plus outer-test `FT-2-KL` on 46 subjects $\times$ 3 seeds = 138 adaptations; estimated wall time ~20–30 min)
- `experiments/EXP-020/exp020_inner_val_selection.json`
- `experiments/EXP-020/exp020_subject_metrics.csv`
- `experiments/EXP-020/exp020_summary_metrics.json` & `experiments/EXP-020/EXP020_CLOSURE_REPORT.md`

---

## 5. Key Audit Risks & Mitigations

1. **Inner-val sample size ($n=5$/fold) for `GATE_tau`:** Rare stage scarcity (`< 0.05`) may not appear in every fold's 5 inner-val subjects, forcing `tau` selection to fall back to `0.01`. *Mitigation:* Tie-breaker is locked pre-execution and the untuned fixed-grid sensitivity table over `{0.01, 0.02, 0.03, 0.05}` is reported as a secondary diagnostic.
2. **Double-dipping / prior-access inflation:** Both `min_stage_frac_cal` (EXP-019B) and `Bidslab34` (EXP-020) were observed prior to this spec. *Mitigation:* Explicit prior-access banner, no confirmatory labels, leave-`Bidslab34`-out sensitivity, strict per-fold nested-CV isolation, and closed BH-FDR / Holm-Bonferroni families.
3. **Device / environment drift:** Log Python, PyTorch, CUDA versions, and `cuda:0` GPU model in `exp020_summary_metrics.json` at startup.

---

## 6. Alternatives Considered

1. **Logit-Space KL (`FT-2-KL`) vs. L2-SP / EWC Parameter Anchoring:** Rejected L2-SP/EWC because isotropic weight penalties do not directly preserve output class probabilities on severely underrepresented classes (e.g., `Bidslab34` 0.68% Deep sleep) when `scope=all` adapts the full backbone; logit-space KL directly anchors predicted probabilities to `ZS` on scarce-stage epochs while allowing feature adaptation.
2. **Static `GATE_tau` vs. Soft Entropy/Confidence Weighting:** `audit_heterogeneity_mechanisms.py` already showed static `min_stage_frac_cal < 0.03` misses non-stage-scarcity failures (`Bidslab65`, `Bidslab10`). Rather than inventing an untried soft-weighting heuristic, `GATE_tau` is retained solely as a simple, auditable abstention comparator (using the existing `FinalModelConfig.min_stage_fraction_warn` concept) against continuous `FT-2-KL`.
3. **Fixed Stage D Schedules vs. Joint `(lr, epochs, scope, lambda_KL)` Re-Tuning:** Rejected joint re-tuning per `DEC-038`/`DEC-039` to avoid inflating inner-val search multiplicity on $n=5$ subjects/fold and to isolate the marginal effect of KL regularization on the locked `FT-2` baseline.
4. **Spearman Rank Correlation vs. Partial Correlation / Mixed-Effects (`EXP-019B`):** Rejected parametric/linear models because $N=46/47$ with extreme outliers (`Bidslab34` at $-16.34\text{ pp}$) violates normality assumptions; rank-based Spearman + leave-`Bidslab34`-out sensitivity is distribution-free and matches prior `EXP-019` Phase A audits.
