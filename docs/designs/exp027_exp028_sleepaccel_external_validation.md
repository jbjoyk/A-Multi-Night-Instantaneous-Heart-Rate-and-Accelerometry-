# EXP-027 & EXP-028: Strictly Zero-Shot External Validation on PhysioNet SleepAccel v1.0.0

*   **Author:** jbjoyk
*   **Status:** In Review (EGM Blockers & Majors Resolved)
*   **Date:** 2026-11-22

--------------------------------------------------------------------------------

## Overview

This document specifies the end-to-end execution protocol for **EXP-027** (zero-shot 4-class sleep-stage classification benchmark) and **EXP-028** (external polysomnography [PSG] sleep-summary Bland-Altman agreement analysis) on the **PhysioNet SleepAccel v1.0.0** dataset (`DOI: 10.13026/hmhs-py35`, N=31 participants, Apple Watch triaxial accelerometry + optical instantaneous heart rate [IHR] + in-lab PSG).

Both **Model A** (`ConventionalCNNBiGRU`, 163,940 parameters, `SleepStagingFinalModel_v1.0` frozen under `DEC-037`) and the comparator **Official SLAMSS-IFS** (`OfficialSLAMSSIFSWrapper`, 5,465,768 parameters, `EXP-021 / TSK-08`) are evaluated **strictly zero-shot**: zero retraining, zero fine-tuning, zero threshold tuning, and zero cohort-level recalibration.

## The Problem

All published and internal benchmarks for Model A and Official SLAMSS-IFS to date have been conducted within the single-cohort **PhysioNet BIDSleep** dataset (47 healthy adults, 246 usable nights). Within-cohort 5-fold cross-validation cannot rule out cohort-specific overfitting to BIDSleep's wearable hardware, preprocessing artifacts, or recording environment.

To establish external generalizability under a genuinely independent wearable device (Apple Watch) and independent in-lab PSG scoring protocol, we require a locked, pre-registered external validation protocol that:
1. Guarantees **100% sensor, unit, resampling, scaling, and metadata parity** with BIDSleep's `prep_57738d534b` feature cache.
2. Enforces a **pre-flight distribution gate** that aborts before inference if any unit mismatch (e.g., $m/s^2$ vs. $g$, RR-interval ms vs. bpm) or channel corruption is present.
3. Evaluates all 15 frozen checkpoints per model family on `cuda:1` under `torch.no_grad()` and reports both 15-checkpoint mean single-model performance (primary) and 15-checkpoint probability ensemble performance (secondary).
4. Remediates `FIND-B-02` by persisting raw subject-night paired tuples before computing Bland-Altman (1999) agreement statistics, standard errors on Limits of Agreement (`SE(LoA)`), and proportional bias slopes.

## Goals

*   **G1 (Zero-Shot External Benchmark — EXP-027):** Measure paired subject-level performance difference `Model A - Official SLAMSS-IFS` across N=31 SleepAccel participants on Macro-F1 (primary), Accuracy, Weighted-F1, Cohen's Kappa, and per-stage F1 (Wake, Light, Deep, REM) with 10,000-resample bootstrap 95% CIs, Wilcoxon signed-rank p-values, paired Cohen's $d_z$, and win/loss counts.
*   **G2 (External PSG Agreement — EXP-028):** Quantify clinical sleep-summary agreement (TST, SOL, WASO, SE, Deep%, REM%) against in-lab PSG for both models, saving raw subject-night paired tuples and reporting Bias (95% CI), SD, 95% LoA with Bland-Altman (1999) `SE(LoA)` and 95% CI on LoA bounds, MAE, Pearson $r$, and proportional bias slope.
*   **G3 (Strict Feature Parity & Safety):** Enforce exact parity with BIDSleep's `prep_57738d534b` representation, `SlidingWindowPredictor`, and `SLAMSSSlidingWindowPredictor`, gated by an automated unit/distribution sanity check and unit test suite before any GPU inference runs.

## Non-Goals

*   **No Training or Adaptation:** Zero retraining, zero Tier-1 InfoMax adaptation, zero Tier-2 supervised fine-tuning, and zero post-hoc decision-threshold or temperature tuning.
*   **No Circadian Feature Usage:** Circadian/clock features remain strictly neutralized (`clock=0`, `time_feat=0`) to match `final_model_spec.json` (`neutralize_circadian: true`).
*   **No Modification of Frozen Checkpoints:** All 15 Model A checkpoints (`experiments/EXP-018/checkpoints/modelA_L1200_fold{fold}_seed{seed}.pt`) and all 15 EXP-021 Official SLAMSS-IFS checkpoints remain read-only.

## Open Questions

*   *(All prior open questions resolved via `experiments/EXP-001/sleepchannelnet_protocol_patch_v2_UPDATED.ipynb` inspection and `evaluator.py::ClinicalHypnogramEvaluator` verification: `meta[:, 0]` is the per-epoch sample-density proxy `np.log1p(raw_counts.reshape(n_ep, 30).sum(1) / 30.0 * 10.0)`, and `n_obs` / `df_denom = n_obs - 1` are parameterized dynamically for QC-passing subjects alongside an all-`N=31` sensitivity run.)*

## Critical User Journeys (CUJs)

### CUJ 1: Governed Dataset Download, Integrity Verification, and QC

*   **Given:** Clean git workspace with `.gitignore` protecting `data/external/` (per `DEC-002`) and AWS CLI / HTTPS access to PhysioNet.
*   **When:** Operator runs the download and SHA-256 verification step for PhysioNet SleepAccel v1.0.0.
*   **Then:** All N=31 participant files (`motion/*_acceleration.txt`, `heart_rate/*_heartrate.txt`, `labels/*_labeled_sleep.txt`) are downloaded to `data/external/sleepaccel_v1.0.0/`, verified against `SHA256SUMS.txt`, recorded in `experiments/EXP-027/sleepaccel_sha256_manifest.json`, and audited for `>= 60%` sensor coverage.

### CUJ 2: Pre-Flight Feature Parity & Distribution Gate

*   **Given:** Raw SleepAccel text files and BIDSleep `prep_57738d534b` cache.
*   **When:** The preprocessing pipeline converts SleepAccel to 1 Hz `(T, 30)` arrays and runs the pre-flight distribution comparison gate.
*   **Then:** Accel magnitude and IHR raw/standardized medians, IQRs, and physiological ranges are compared against BIDSleep `prep_57738d534b`. If any unit mismatch (e.g., $m/s^2$ vs. $g$) or NaN/Inf violation is detected, execution aborts before GPU inference.

### CUJ 3: Zero-Shot Paired Inference (`EXP-027`) & External Bland-Altman (`EXP-028`)

*   **Given:** Pre-flight gate passed and `cuda:1` available.
*   **When:** Operator runs `EXP-027` paired inference across all 15 Model A and 15 SLAMSS-IFS checkpoints, followed by `EXP-028` sleep-summary agreement analysis.
*   **Then:** Per-subject metrics, 10,000-resample paired bootstrap CIs, raw subject-night paired tuples CSV, and Bland-Altman (1999) agreement tables are generated and saved under `experiments/EXP-027/` and `experiments/EXP-028/`.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### 1. Download, `.gitignore` Governance, and Coverage QC

*   **Download Command:**
    ```bash
    mkdir -p data/external/sleepaccel_v1.0.0
    aws s3 sync --no-sign-request s3://physionet-open/sleep-accel/1.0.0/ data/external/sleepaccel_v1.0.0/
    ```
    *Fallback (PhysioNet HTTPS):*
    ```bash
    wget -r -N -c -np -nH --cut-dirs=3 -P data/external/sleepaccel_v1.0.0/ https://physionet.org/files/sleep-accel/1.0.0/
    ```
*   **`.gitignore` Protection (`DEC-002`):** Add `data/external/` and `experiments/EXP-027/cache/` to `.gitignore` before downloading so no raw physiological recordings or derived `.npz` caches enter git history.
*   **SHA-256 Manifest:** Verify all downloaded files against upstream `SHA256SUMS.txt` (if provided) and write `experiments/EXP-027/sleepaccel_sha256_manifest.json` mapping every `motion/*_acceleration.txt`, `heart_rate/*_heartrate.txt`, and `labels/*_labeled_sleep.txt` file to its SHA-256 digest and byte size.
*   **Coverage QC (`>= 60%` Rule Matching BIDSleep `Config` in `experiments/EXP-001/sleepchannelnet_protocol_patch_v2_UPDATED.ipynb` Cell 9 L26–29 & Cell 16 L108–116):**
    *   For each 30 s epoch $k \in \{0, \dots, T-1\}$ on the 1 Hz grid `grid = np.arange(30 * T)`:
        *   An epoch has per-epoch HR coverage `cov_hr[k] = ok_hr[30k:30(k+1)].mean()`, where `ok_hr = nearest_gap(t_hr, grid) <= 120.0` (`max_gap_hr_s = 120.0`) after physiological `[30, 200]` bpm + `std3` outlier filtering.
        *   An epoch has per-epoch ACC coverage `cov_mo[k] = ok_mo[30k:30(k+1)].mean()`, where `ok_mo = nearest_gap(t_mo, grid) <= 10.0` (`max_gap_motion_s = 10.0`), plus an off-wrist flatline check (`intra-epoch variance >= 1e-6` on raw magnitude per `final_model_spec.json:L113`).
        *   An epoch is **sensor-valid** if `(cov_hr[k] >= 0.5) & (cov_mo[k] >= 0.5)` (`min_epoch_coverage = 0.5`); invalid epochs have `X[~valid] = 0` and `y[~valid] = -100`.
    *   **QC Threshold:** Require `usable_frac = (y >= 0).mean() >= 0.60` (`min_usable_frac = 0.6`). Any participant failing `< 60%` usable epochs (or `< 600` total epochs per `slamss_ifs_train.py:L162`) is flagged `qc_pass=False` in the QC manifest and excluded from the QC-clean primary cohort (`n_obs = sum(qc_pass)`, `df_denom = n_obs - 1`), with an all-`N=31` intent-to-evaluate sensitivity check reported alongside.

### 2. Exact Sensor & Feature Parity Contract

Graphify call-chain tracing confirms the authoritative BIDSleep feature and inference path:

```text
prep_57738d534b/*.npz
  └── slamss_ifs_train.py::load_night_features() [L105–L141]
        ├── X[:, 0, :] -> ihr (T, 30)
        ├── X[:, 6, :] -> accel (T, 30)
        ├── meta[:, 0:1] -> freq (T, 1)
        ├── meta[:, 2:3] -> time_feat (T, 1)
        └── y_raw -> 4-class y (T,)
        │
        ├──[Model A]──> experiments/EXP-018/exp018_inference.py::SlidingWindowPredictor [L20–L182]
        │                 ├── _prepare_window_tensors() [L38–L64] (clock=0, time_feat=0, hr_mean, hr_std)
        │                 └── exp013_models.py::ConventionalCNNBiGRU.forward() [L100–L183]
        │
        └──[SLAMSS]───> experiments/EXP-021/official_slamss_ifs_adapter.py::SLAMSSSlidingWindowPredictor [L264–L441]
                          ├── build_slamss_window_arrays() [L166–L193] (x_signal=(L*30,2), x_meta=(L,5))
                          └── OfficialSLAMSSIFSWrapper.forward() [L84–L118] (trg_zeros, teacher_forcing_ratio=0.0)
```

#### Signal & Label Reconciliation Table (Verified Against `experiments/EXP-001/sleepchannelnet_protocol_patch_v2_UPDATED.ipynb` Cell 9 & Cell 16)

1.  **Pre-PSG `t < 0` Boundary & Trim-Before-Scale Order:** SleepAccel recordings may contain hours of pre-PSG ambulatory data at `t < 0`. Restrict any `t < 0` samples strictly to a narrow boundary anchor window (`[-120.0, 0)` s for IHR matching `max_gap_hr_s=120.0`; `[-10.0, 0)` s for ACC matching `max_gap_motion_s=10.0`) solely for `t = 0` interpolation continuity, and trim to the labeled PSG span `[0, 30*T)` **before** computing `[0.1, 99.9]` percentiles, `std3` HR filtering, per-night median/IQR scaling, and `[-10.0, 10.0]` clipping.

| Component | SleepAccel v1.0.0 Raw Format | BIDSleep `prep_57738d534b` Contract | Exact Verified Reconciliation Rule |
|---|---|---|---|
| **Triaxial ACC** | `motion/*_acceleration.txt`: `(t_sec, x, y, z)` in **$g$** (~50 Hz) | `X[:, 6, :]`: 1 Hz triaxial magnitude `(T, 30)` | Clip each axis `x, y, z` to `[0.1, 99.9]` percentiles (`Cell 16 L92-93`), linearly interpolate onto 1 Hz `grid = np.arange(30*T)`, compute raw Euclidean norm $a(t) = \sqrt{ax^2 + ay^2 + az^2}$ in $g$ (gravity included, no high-pass/ENMO; `Cell 16 L94`), and compute `ok_mo = nearest_gap(t_mo, grid) <= 10.0` (`Cell 9 L25`). |
| **Heart Rate (IHR)** | `heart_rate/*_heartrate.txt`: `(t_sec, hr_bpm)` in **bpm** (~0.2–1 Hz) | `X[:, 0, :]`: 1 Hz IHR `(T, 30)` | Filter `np.isfinite(h) & (h >= 30) & (h <= 200)` (`Cell 16 L72`) followed by `std3` outlier filter `abs((h - mean)/std) <= 3` (`Cell 16 L74-76`). Linearly interpolate `hr_i = np.interp(grid, t_hr, hv)` onto 1 Hz `grid`, with gap mask `ok_hr = nearest_gap(t_hr, grid) <= 120.0` (`Cell 9 L23`, `Cell 16 L90`). Note: Gaps `> 15 s` are flagged in QC diagnostics; gaps `> 120 s` are masked (`ok_hr = False`, zero-filled post-scaling). |
| **Per-Night Scaling & Clipping** | Raw physical units ($g$ and bpm) | Robust per-night standardized `(T, 30)` clipped to `[-10.0, 10.0]` | Apply `robust_scale(v, floor)` (`Cell 16 L1-3`): `med = np.nanmedian(v); iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25); z = np.clip(np.nan_to_num((v - med) / max(float(iqr), floor)), -10.0, 10.0)` with `floor=2.0` for IHR (`L99`) and `floor=0.005` for ACC (`L100`). Multiply by `ok_hr` / `ok_mo` (`L106`) and zero-fill invalid epochs (`cov_hr < 0.5` or `cov_mo < 0.5`; `L109-110`). |
| **Metadata `x_meta`** | N/A (derived) | `(T, 5)`: `[freq, clock, time_feat, hr_mean, hr_std]` | `edges = np.linspace(0, 30*T, 30*T + 1)`; `raw_counts = np.histogram(t_hr_raw, bins=edges)[0]`; **`freq = np.log1p(raw_counts.reshape(T, 30).sum(1) / 30.0 * 10.0)`** (`Cell 16 L67-68, L112`). Set `clock = 0.0`, `time_feat = 0.0` (`neutralize_circadian=True`), `hr_mean = ihr.mean(axis=1, keepdims=True)`, `hr_std = ihr.std(axis=1, keepdims=True)` on the scaled `(T, 30)` `ihr` array (`exp018_inference.py:L48-57`, `official_slamss_ifs_adapter.py:L185-192`). |
| **Sleep Stage Labels** | `labels/*_labeled_sleep.txt`: `(t_sec, stage)` with `0=W, 1=N1, 2=N2, 3=N3, 5=REM` | `y_4class`: `0=Wake, 1=Light, 2=Deep, 3=REM, -100=Ignore` | Determine `T` from the contiguous 30 s label grid `t = 30 * k` (`k = 0..T-1`, filling any missing `k` with `-100`). Map `0 -> 0`, `{1, 2} -> 1`, `3 -> 2`, `5 -> 3`, all else (`-1`, `4`, `9`, `NaN`) -> `-100`. Note: BIDSleep cache uses `4=REM` (`slamss_ifs_train.py:L122`), whereas SleepAccel uses `5=REM`. |

### 3. Pre-Flight Distribution Comparison Gate

Before running any model forward pass, a mandatory gate compares SleepAccel's preprocessed 1 Hz arrays against BIDSleep's `prep_57738d534b` cohort summary statistics:
1. **Raw Physical Unit Check:**
   *   SleepAccel raw ACC magnitude median must fall within `[0.90, 1.10] g` (gravity baseline; if median is `~9.81`, units are $m/s^2$ -> **ABORT**).
   *   SleepAccel raw IHR median must fall within `[45.0, 95.0] bpm` (if median is `~600–1200`, units are RR ms -> **ABORT**).
2. **Standardized Feature & `meta[:, 0]` Distribution Check:**
   *   After per-night median/IQR scaling and `[-10.0, 10.0]` clipping, verify cohort-wide median of per-night standardized IHR/ACC medians $\approx 0.0$ (`|med| < 0.25`), standardized IQRs $\approx 1.0$ (`[0.5, 2.0]`), and **100%** of finite values within `[-10.0, 10.0]`.
   *   Verify `meta[:, 0]` (`freq`) is finite, non-negative, and within `[0.0, 5.0]` (matching BIDSleep's `np.log1p(raw_counts_30s / 30 * 10)` range without falsely aborting on hardware sampling-rate differences).
3. **Failure Mode:** Any gate failure raises `ExternalValidationGateError` and halts before GPU allocation.

### 4. Paired Zero-Shot Inference (`EXP-027` on `cuda:1`)

*   **Execution Environment:** `device = torch.device("cuda:1")` explicitly passed to both predictors (overriding `SLAMSSSlidingWindowPredictor`'s `cuda:0` default), `torch.no_grad()`, `model.eval()`. Model A runs in FP32 (`exp018_inference.py`); Official SLAMSS-IFS runs with `use_amp=True` FP16 autocast (`official_slamss_ifs_adapter.py:L311-313`).
*   **Model Arms (15 Checkpoints Each, `folds=[1, 2, 3, 4, 5]`, `seeds=[42, 123, 2024]`):**
    *   **Model A (`ConventionalCNNBiGRU`):** `experiments/EXP-018/checkpoints/modelA_L1200_fold{fold}_seed{seed}.pt`, run via `SlidingWindowPredictor(model, context_len=1200, neutralize_circadian=True, device=torch.device("cuda:1"))`.
    *   **Official SLAMSS-IFS (`OfficialSLAMSSIFSWrapper`):** `experiments/EXP-021/checkpoints/slamss_ifs_official_L1200_fold{fold}_seed{seed}.pt`, run via `SLAMSSSlidingWindowPredictor(model, context_len=1200, neutralize_circadian=True, device=torch.device("cuda:1"))` (which enforces `trg_zeros` and `teacher_forcing_ratio=0.0` inside `OfficialSLAMSSIFSWrapper.forward()`).
*   **Sliding-Window Invariants (`L=1200`):**
    *   `N <= 1200`: right-pad to `1200`, `m=0`, slice `[0:N]` (with `seq_lens=N` passed to `pack_padded_sequence` for Model A).
    *   `N > 1200`: `W=1200`, `stride=600`, `margin=300`, boundary anchors `w=0` and `w=N-1200`, emission probability averaging, and `np.all(counts > 0)` assertion.
*   **Classification Metric Reuse & Reporting Tiers:**
    *   Reuse `evaluator.py::SleepEvaluator.evaluate_predictions()` (`L84-L220`) for all 4-class metrics (`y_true != -100` masking).
    *   **(a) Primary — `mean_single_model`:** For each participant $i$ and checkpoint $c \in \{1..15\}$, compute $M_{i,c}$ and average $\bar{M}_i = \frac{1}{15} \sum_{c=1}^{15} M_{i,c}$.
    *   **(b) Secondary — `prob_ensemble`:** Average softmax probabilities $\bar{P}_i = \frac{1}{15} \sum_{c=1}^{15} P_{i,c}$ `(N_i, 4)`, take $\hat{y}_i = \arg\max \bar{P}_i$, and compute $M_i^{\text{ens}}$.
*   **Prediction Handoff Schema (`EXP-027 -> EXP-028`):**
    *   Persist per-subject prediction bundles to gitignored `experiments/EXP-027/cache/predictions/{subject_id}__{arm}.npz` containing `y_true` `(N_i,)`, `ckpt_preds` `(15, N_i)` int8, and `ens_probs` `(N_i, 4)` float32.
*   **Paired Estimand (`Model A - Official SLAMSS-IFS` across QC-passing `n_obs`, plus all-`N=31` sensitivity):**
    *   Primary metric: **Macro-F1**. Secondary (unadjusted/exploratory): Accuracy, Weighted-F1, Cohen's Kappa, Wake F1, Light F1, Deep F1, REM F1.
    *   Statistics: Mean paired delta $\bar{\Delta}$, 10,000-resample subject-level percentile bootstrap 95% CI (fixed seed `20261122`), two-sided Wilcoxon signed-rank test $p$-value, paired Cohen's $d_z = \bar{\Delta} / s_{\Delta}$, and win/loss/tie counts.

### 5. External PSG Bland-Altman Agreement (`EXP-028`)

*   **Sleep-Summary Metrics (Direct Reuse of `evaluator.py::ClinicalHypnogramEvaluator.compute_night_metrics()`, `L338-L412`):**
    *   Mask unscored PSG epochs consistently (`valid = (y_true != -100)`) across `y_true` and `y_pred`, then call `ClinicalHypnogramEvaluator.compute_night_metrics()` for `tst_min`, `sol_min` (first 3 consecutive sleep epochs = 1.5 min, defaulting to `tib`), `waso_min` (wake epochs in `[t_onset_idx : t_final_idx + 1]`), `sleep_efficiency_pct` (`100 * tst / tib`), `deep_pct_tst`, and `rem_pct_tst`.
*   **`eval_mode` Aggregation:**
    *   **`mean_single_model` (primary):** Compute `ClinicalHypnogramEvaluator.compute_night_metrics(ckpt_preds[c])` for each checkpoint $c \in \{1..15\}$ and average the 15 continuous summary values per participant (`pred_val`).
    *   **`prob_ensemble` (secondary):** Compute `ClinicalHypnogramEvaluator.compute_night_metrics(argmax(ens_probs))` per participant.
*   **Remediation of `FIND-B-02` (Raw Paired Tuples Persistence):**
    *   Save every raw subject-night paired tuple (`subject_id`, `arm`, `eval_mode`, `metric`, `ref_val`, `pred_val`, `diff = pred_val - ref_val`, `mean = 0.5 * (ref_val + pred_val)`) to `experiments/EXP-028/exp028_sleepaccel_raw_paired_tuples.csv` before any aggregation.
*   **Bland-Altman (1999) Statistics (Dynamic `n_obs`, `df_denom = n_obs - 1`, $t_{0.975, \text{df}}$):**
    *   Compute `bias`, `95% CI(bias)`, `sd_diff`, `loa_low/high`, `se_loa`, and `95% CI(LoA)` matching `research_engine/bidsleep_within_cohort_agreement_analysis.py:L64-L81`, plus `mae`, `rmse`, `pearson_r`, OLS proportional bias slope (`scipy.stats.linregress(mean, diff)`), and **Shapiro-Wilk normality test** on `diff` (reporting median and 2.5th–97.5th percentile LoA as non-parametric fallback whenever Shapiro-Wilk $p < 0.05$).

--------------------------------------------------------------------------------

## Alternatives Considered

### Alternative A: Cohort-Level Z-Scoring or Fine-Tuning on SleepAccel

*   **Pros:** Could absorb device-level differences between Apple Watch PPG/ACC and BIDSleep's Empatica/ActiGraph sensors, potentially raising raw Macro-F1.
*   **Cons:** Leaks cohort-level distribution statistics across the N=31 test participants and violates the strictly zero-shot external validation claim.
*   **Reason for Rejection:** Rejected by author: per-night median/IQR scaling and zero threshold/weight updates are required to test true out-of-the-box zero-shot generalization.

### Alternative B: Evaluating Only the 15-Checkpoint Probability Ensemble

*   **Pros:** Reduces computation and simplifies reporting to a single prediction vector per participant.
*   **Cons:** Conflates single-model generalization with 15-model ensembling variance reduction, making direct comparison to single-model BIDSleep OOF metrics ambiguous.
*   **Reason for Rejection:** Rejected by author: reporting 15-checkpoint mean single-model performance as primary and 15-checkpoint probability ensemble as secondary cleanly separates architecture generalization from ensemble variance reduction.

### Alternative C: Storing Only Summary Bland-Altman Tables Without Raw Paired Tuples

*   **Pros:** Smaller output artifact footprint.
*   **Cons:** Reproduces `FIND-B-02` (inability to audit proportional bias, non-normality of differences, or per-subject outliers from summary rows alone).
*   **Reason for Rejection:** Rejected by author: persisting `exp028_sleepaccel_raw_paired_tuples.csv` is mandatory to remediate `FIND-B-02` and enable independent verification.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Data Governance (`DEC-002`):** Raw SleepAccel files and derived `.npz` caches must remain outside Git via `.gitignore`. Only SHA-256 manifests, metrics JSON/CSV summaries, raw paired sleep-summary tuples, and markdown reports are committed.
*   **Compute & Determinism:** `cuda:1` only, `torch.no_grad()`, fixed bootstrap seed (`20261122`). Estimated runtime for 30 checkpoints $\times$ 31 nights ($\sim 930$ single-night passes) is under 5 minutes on GPU.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Target Files to Create/Modify

#### 1. `.gitignore` (Modify)
*   **Type / Scope:** Git governance configuration.
*   **Proposed Changes:** Add `data/external/` and `experiments/EXP-027/cache/` to prevent accidental commits of PhysioNet SleepAccel raw data or derived `.npz` feature caches (`DEC-002`).

#### 2. `experiments/EXP-027/sleepaccel_preprocessing.py` (Create)
*   **Type / Scope:** Python preprocessing & pre-flight gate module.
*   **Proposed Changes:**
    *   Implement `verify_sleepaccel_manifest(data_dir, out_json)` (SHA-256 verification and `sleepaccel_sha256_manifest.json` generation).
    *   Implement `preprocess_sleepaccel_subject(subject_id, data_dir) -> Dict[str, Any]` (boundary-only `t < 0` anchoring, trim-before-scale, `[30, 200]` bpm + `std3`, `robust_scale` with `[-10.0, 10.0]` clip, `meta[:, 0] = np.log1p(raw_counts_30s / 30.0 * 10.0)`, `>= 60%` usable QC, and `0->0, {1,2}->1, 3->2, 5->3, else -> -100` label map).
    *   Implement `run_preflight_distribution_gate(records, bidsleep_cache_dir)` verifying physical units, standardized `[-10.0, 10.0]` bounds, and `meta[:, 0]` finiteness/range, raising `ExternalValidationGateError` on mismatch.

#### 3. `experiments/EXP-027/run_exp027_sleepaccel_inference.py` (Create)
*   **Type / Scope:** Zero-shot paired inference & statistical comparison driver (`cuda:1`).
*   **Proposed Changes:**
    *   Load all 15 Model A checkpoints (`experiments/EXP-018/checkpoints/modelA_L1200_fold{fold}_seed{seed}.pt`) via `SlidingWindowPredictor(..., device=torch.device("cuda:1"))` and all 15 `EXP-021` Official SLAMSS-IFS checkpoints (`experiments/EXP-021/checkpoints/slamss_ifs_official_L1200_fold{fold}_seed{seed}.pt`) via `SLAMSSSlidingWindowPredictor(..., device=torch.device("cuda:1"))`.
    *   Reuse `evaluator.py::SleepEvaluator.evaluate_predictions()` for `mean_single_model` (primary) and `prob_ensemble` (secondary) across QC-passing `n_obs` (plus all-`N=31` sensitivity).
    *   Persist prediction bundles to `experiments/EXP-027/cache/predictions/{subject_id}__{arm}.npz`, export `exp027_per_subject_metrics.csv` and `exp027_summary.json` with 10,000-resample paired bootstrap CIs, Wilcoxon tests, Cohen's $d_z$, and win/loss counts.

#### 4. `experiments/EXP-028/run_exp028_sleepaccel_bland_altman.py` (Create)
*   **Type / Scope:** External PSG Bland-Altman agreement analysis driver.
*   **Proposed Changes:**
    *   Load `experiments/EXP-027/cache/predictions/*.npz` and call `evaluator.py::ClinicalHypnogramEvaluator.compute_night_metrics()` after consistent `y_true != -100` masking for both `mean_single_model` and `prob_ensemble`.
    *   Persist raw subject-night paired tuples to `experiments/EXP-028/exp028_sleepaccel_raw_paired_tuples.csv` (`FIND-B-02` remediation).
    *   Compute dynamic `n_obs` / `df_denom = n_obs - 1` Bias (95% CI), SD, 95% LoA with Bland-Altman (1999) `SE(LoA)` and 95% CI on LoA bounds matching `research_engine/bidsleep_within_cohort_agreement_analysis.py:L64-L81`, `scipy.stats.linregress` proportional bias slope, and Shapiro-Wilk normality check (with 2.5th–97.5th percentile LoA fallback when $p < 0.05$).

#### 5. `tests/test_exp027_exp028_sleepaccel.py` (Create)
*   **Type / Scope:** Unit test suite.
*   **Proposed Changes:**
    *   `test_sleepaccel_4class_label_mapping()`: Verifies `0->0`, `{1,2}->1`, `3->2`, `5->3`, and `-1/4/9 -> -100`.
    *   `test_1hz_aggregation_clipping_and_t_neg_trim_before_scale()`: Verifies triaxial magnitude, `robust_scale` `[-10.0, 10.0]` clipping, and that hours of pre-PSG `t < 0` ambulatory data do not pollute night `med`/`iqr` scaling.
    *   `test_meta_freq_formula_and_circadian_neutralization()`: Verifies `meta[:, 0] == np.log1p(raw_counts_30s / 30.0 * 10.0)` and `clock=0`, `time_feat=0`.
    *   `test_coverage_qc_60pct_gate()`: Verifies `< 60%` usable coverage sets `qc_pass=False` while `>= 60%` passes.
    *   `test_preflight_unit_mismatch_aborts()`: Verifies `run_preflight_distribution_gate()` aborts on ACC $m/s^2$ (`~9.81`), IHR RR-ms (`~800`), or unclipped `|z| > 10.0`.
    *   `test_predictors_smoke_short_and_long_night()`: Verifies `SlidingWindowPredictor` and `SLAMSSSlidingWindowPredictor` produce finite outputs for `N <= 1200` and `N > 1200`.
    *   `test_clinical_hypnogram_evaluator_parity_and_bland_altman()`: Verifies `ClinicalHypnogramEvaluator` 3-epoch SOL / `-100` masking / zero-sleep edge cases, `SE(LoA)`, proportional bias, and Shapiro-Wilk fallback.

### Verification Plan

*   **Unit Tests:** `.venv/bin/python -m pytest tests/test_exp027_exp028_sleepaccel.py -v` (must pass 100% before any dataset download or GPU run).
*   **Pre-Flight Gate:** Execute `run_preflight_distribution_gate()` on downloaded SleepAccel data and verify PASS before running `EXP-027`.
*   **E2E / Integration:** Run `EXP-027` on `cuda:1` and `EXP-028` on CPU; verify all 15+15 checkpoints execute without NaN/Inf and `exp028_sleepaccel_raw_paired_tuples.csv` row count equals `n_obs * 2 * 6 * 2` (`744` rows when `n_obs=31` across 2 eval modes).

--------------------------------------------------------------------------------

*Drafted with EGM*
