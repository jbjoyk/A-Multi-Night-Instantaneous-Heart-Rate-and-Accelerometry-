# EXP-030: Single-Factor Capacity–Accuracy–Edge Pareto Sweep of Plain `ConventionalCNNBiGRU`

*   **Author:** jbjoyk
*   **Status:** Remediated (`EGM M1–M6`, `N1–N6` Resolved; Pending User Verification)
*   **Date:** 2026-11-24

--------------------------------------------------------------------------------

## Overview

`EXP-030` is a single-factor capacity–accuracy–edge Pareto sweep of the plain [`ConventionalCNNBiGRU`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L27-L183) architecture across 6 width points spanning `98,940` to `1,051,492` parameters (`~100k` to `~1M`). Following the advisor audit of `EXP-029`—which showed that `Ultradian-Pyramidal-Coupler` (`full_v5`, `190,788` params, `66.74%` Macro-F1) did not significantly outperform the width-matched plain `cnn_bigru_200k` control (`223,268` params, `66.61%`, `+0.13 pp`, $p_{\text{Holm}} = 1.0000$)—we report `EXP-029` honestly as a negative architectural ablation and isolate width scaling in `ConventionalCNNBiGRU` as the sole independent variable.

To eliminate post-hoc selection bias, `EXP-030` pre-registers an exact, deterministic in-cohort elbow rule that selects a single final backbone (`CNN-BiGRU-Star`) strictly from BIDSleep `N = 47` 5-fold cross-validation Macro-F1 vs. parameter count *before* any downstream evaluation (W0-B Dynamic INT8, PhysioNet SleepAccel zero-shot transfer, or `EXP-020` personalization) is executed.

## The Problem

1. **Capacity Knee vs. Architectural Novelty (`EXP-029` Negative Ablation):** In `EXP-029`, `full_v5` (`66.74%`), `film_only` (`66.67%`), `pyramid_only` (`66.92%`), and the plain `cnn_bigru_200k` width control (`66.61%`) all clustered within `0.31 pp` of one another, while jumping `+1.82` to `+2.13 pp` above `Model A-164k` (`64.79%`). That pattern points to a capacity knee above `163,940` parameters rather than a structural breakthrough from sub-epoch FiLM or 2-minute pyramidal pooling.
2. **Unmapped Capacity–Accuracy–Edge Frontier:** We know `ConventionalCNNBiGRU` at `163,940` params (`64.79%`, `309.54 KiB` Dynamic INT8) and `223,268` params (`66.61%`, `368.04 KiB` Dynamic INT8), and `Official SLAMSS-IFS` at `5,465,768` params (`67.37%`). We do not yet know where plain `ConventionalCNNBiGRU` saturates between `~100k` and `~1M` parameters, nor what edge memory/compute cost that saturation requires.
3. **Downstream Comparator & Protocol Gaps in `EXP-029`:**
   - `EXP029_POST_EXECUTION_EGM_DRIFT_REPORT.md` inadvertently swapped the `Model A` (`+10.65%`) and `SLAMSS-IFS` (`+8.67%`) SleepAccel `deep_pct_tst` biases and did not compute like-for-like paired bootstrap CIs and Wilcoxon $p$-values (`single-model vs. single-model`, `ensemble vs. ensemble`) across 4-class F1 and all 6 Bland-Altman metrics.
   - `eval_exp029_downstream.py` hardcoded `tau_star = 0.01` across all 5 folds instead of running `EXP-020`'s nested per-fold inner-validation `tau*` selection (`TAU_GRID = [0.01, 0.02, 0.03, 0.05]`), where `EXP-020` selected `tau* = 0.05` on Fold 1 and `0.01` on Folds 2–5.
   - Unsupervised `DampedEMCalibrator` (`em_on`) degraded SleepAccel Macro-F1 across every `EXP-029` arm (`63.51% -> 61.30%` on `full_v5`) and widened Deep%/REM% bias, so it should be dropped from primary evaluation.

## Goals

*   **G1 (Single-Factor Capacity Sweep):** Train and evaluate the 5 new width points (`p1_100k`, `p3_268k`, `p4_404k`, `p5_666k`, `p6_1051k`) of plain `ConventionalCNNBiGRU` across 5 subject-disjoint folds x 3 seeds (`75` training runs total, reusing locked `Model A-164k` as Anchor `p2_164k`), using the exact `EXP-021`/`EXP-029` training protocol (`prep_57738d534b` cache, `36d84020` manifest gate, `L = 1200`, `RCCLoss`). Run a ~30s inference-only `LogitStitchingSlidingWindowPredictor` pass over the 15 frozen `DEC-037` checkpoints (`p2_164k`) for exact stitcher parity.
*   **G2 (Zero-Bias Elbow Selection):** Apply the pre-registered in-cohort BIDSleep elbow formula to select `CNN-BiGRU-Star` strictly from `N = 47` OOF Macro-F1 and parameter count before running any downstream evaluation script.
*   **G3 (Honest Downstream Evaluation of `CNN-BiGRU-Star` Only):** Evaluate `CNN-BiGRU-Star` on:
    1. BIDSleep paired statistics (`N = 47`, 10,000-resample bootstrap CIs, two-sided Wilcoxon, Holm-Bonferroni) vs. `Model A-164k` (`p2_164k`, guarded against self-comparison) and `Official SLAMSS-IFS` (`67.37%`), framed honestly ("gap narrowed from `2.58 pp` to `X pp`; residual difference not ruled out").
    2. `W0-B` Dynamic INT8 profiling (`FP32` vs. `INT8` OOF parity, in-memory KiB payload, 1T/4T CPU latency) across `CNN-BiGRU-Star` (plus analytical/empirical edge memory curve across `p1_100k`–`p6_1051k`).
    3. Strictly zero-shot PhysioNet SleepAccel (`n = 25` QC-clean and `N = 31` all; `em_off` only) with like-for-like paired CIs and unadjusted Wilcoxon $p$-values against `Model A-164k` and `Official SLAMSS-IFS` for 4-class Macro-F1 and all 6 Bland-Altman metrics (`tst_min`: Total Sleep Time [TST], `sol_min`: Sleep Onset Latency [SOL], `waso_min`: Wake After Sleep Onset [WASO], `sleep_efficiency_pct`, `deep_pct_tst`, `rem_pct_tst`), correcting the swapped Deep% comparator.
    4. `EXP-020` `FT-2` (2-night supervised fine-tuning of `classifier_head|temp_norm|temporal_gru`) and nested per-fold `GATE_tau` (min-stage-fraction abstention gate) personalization on `N = 46` multi-night subjects (`>= 3` usable nights; 1 subject of `N = 47` with `< 3` nights excluded) using the exact `EXP-020` inner-validation `tau*` selection rule.

## Non-Goals

*   **NG1:** Re-testing `Ultradian-Pyramidal-Coupler`, FiLM, pyramidal pooling, GRL, boundary loss, or transition heads.
*   **NG2:** Running downstream evaluations (SleepAccel, `EXP-020` `FT-2`/`GATE_tau`) on all 6 grid points—only `CNN-BiGRU-Star` is evaluated downstream to prevent multiple-testing/selection bias. (Analytical/static `W0-B` payload and MMACs are computed for all 6 points to plot the edge memory curve; full OOF INT8 parity is run on `CNN-BiGRU-Star`.)
*   **NG3:** Running `DampedEMCalibrator` (`em_on`) on SleepAccel (falsified in `EXP-029`).
*   **NG4:** Re-extracting raw `Bidslab*` or SleepAccel data, or modifying any historical artifacts in `experiments/EXP-018` through `experiments/EXP-029`.
*   **NG5:** Writing or executing Python code during this design phase (strict NO-CODE).

## Open Questions

*   **OQ1:** None blocking execution. The width grid, elbow rule, and downstream statistical tests are fully specified below.

## Critical User Journeys (CUJs)

### CUJ 1: 75-Run Width Sweep & Pre-Registered Elbow Selection

*   **Given**: Locked `prep_57738d534b` cache (`246` usable nights, `47` subjects), `experiments/EXP-014/EXP014_MANIFESTS/fold_1..5.json` verified against `review_package/fold_manifests.csv` (SHA-256 prefix `36d84020`), and dual RTX 3060 12GB GPUs (`cuda:0`, `cuda:1`).
*   **When**: The CV driver trains the 5 new width points (`p1_100k`, `p3_268k`, `p4_404k`, `p5_666k`, `p6_1051k`) across 5 folds x 3 seeds (`42, 123, 2024`; 75 runs total, `~2 hours` wall time across 2 GPUs), runs an inference-only ~30s `LogitStitchingSlidingWindowPredictor` pass over the 15 frozen `DEC-037` checkpoints (`experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json`) to establish an apples-to-apples `M(p2_164k)` baseline (or reconciles against `experiments/STAGE2_FINAL_MODEL/final_model_oof_subject_metrics.csv` normalized by `* 100.0`), and runs the standalone elbow selector script.
*   **Then**: The selector outputs `experiments/EXP-030/elbow_selection.json` naming `CNN-BiGRU-Star` strictly from BIDSleep `N = 47` Macro-F1 vs. parameter count before any downstream script is invoked.

### CUJ 2: Downstream Evaluation of `CNN-BiGRU-Star` Only

*   **Given**: `experiments/EXP-030/elbow_selection.json` has locked `CNN-BiGRU-Star` and its 15 checkpoints are saved under `experiments/EXP-030/checkpoints/<star_tag>/`.
*   **When**: The downstream evaluation suite runs the 4 contracted pillars exclusively on `CNN-BiGRU-Star`: (1) BIDSleep paired stats vs. `Model A-164k` and `SLAMSS-IFS`, (2) `W0-B` Dynamic INT8 OOF parity + edge memory curve, (3) zero-shot SleepAccel (`em_off` only, `n = 25` and `N = 31`) with like-for-like paired F1 and Bland-Altman tests, and (4) `EXP-020` `FT-2` + nested per-fold `GATE_tau`.
*   **Then**: All outputs are written to `experiments/EXP-030/` (with SleepAccel prediction bundles isolated in `experiments/EXP-030/sleepaccel_cache/predictions/`), ready for manuscript integration.

## Technical Plan (Architecture)

**Planned Script Paths:** `experiments/EXP-030/exp030_models.py`, `experiments/EXP-030/run_exp030_cv.py`, `experiments/EXP-030/select_exp030_star.py`, `experiments/EXP-030/eval_exp030_downstream.py`, `experiments/EXP-030/test_exp030_unit.py`.

### 1. Exact 6-Point Width Grid (`~100k` to `~1M` Parameters)

Because [`ConventionalCNNBiGRU`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L27-L183) hardcodes `head_hidden = 64` and [`HREncoder`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/sleep_adaptive_net.py#L79-L108)/[`ACCEncoder`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/sleep_adaptive_net.py#L110-L139) hardcode `32` and `64` channels in their first two conv blocks, we leave `exp013_models.py` and `sleep_adaptive_net.py` untouched and implement `ScaledConventionalCNNBiGRU` + `ScaledModalityEncoder(in_features=1, d_enc=d_enc, dropout=0.2)` (`c1 = d_enc // 2`, `c2 = d_enc`, preserving `res = Identity()` in the second `TemporalConvBlock`) inside `experiments/EXP-030/exp030_models.py`, exposed via `build_exp030_model(tag: str, loss_type: str = "rccloss") -> nn.Module`. It preserves the exact 14-key `forward()` return dictionary contract in [`exp013_models.py#L168-L183`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L168-L183), fixed `meta_proj = Linear(5, 16) + GELU` (`96` params), 2-layer bidirectional `GRU(input_size=2*d_enc+16, hidden_size=d_temp//2)`, `LayerNorm(d_temp)`, and `classifier_head = Linear(d_temp, head_hidden) -> LayerNorm(head_hidden) -> GELU -> Dropout(0.2) -> Linear(head_hidden, 4)`.

At `p2_164k` (`d_enc = 64, d_temp = 96, head_hidden = 64`), `c1 = 32` and `c2 = 64`, which is bit-for-bit identical in `state_dict` keys, shapes, and parameter count (`163,940` params) to locked `Model A-164k` (verified by a mandatory pre-flight unit test). Analytical counts below use `L = 1200` epochs (`30` samples/epoch) and the `W0-B` Dynamic INT8 payload formula (`1 B/param` for quantized `nn.Linear` + `nn.GRU` weights, `4 B/param` for FP32 `Conv1d` + `GroupNorm` + `LayerNorm` weights and biases):

| Grid Tag | `d_enc` | `d_temp` (`h`/dir) | `head_hidden` | Total Params | $\Delta\%$ vs. 164k | Quantized Params (`Linear`+`GRU`) | Dynamic INT8 Payload (KiB) | MMACs (`L=1200`) | Training Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`p1_100k`** | `48` | `76` (`38`) | `48` | **`98,940`** | `-39.65%` | `69,796` (`70.5%`) | `182.00 KiB` | `456.1` (`0.57x`) | `15` (new) |
| **`p2_164k` (Anchor)** | `64` | `96` (`48`) | `64` | **`163,940`** | `0.00%` | `112,804` (`68.8%`) | `309.54 KiB` (`309.91` anal.) | `794.1` (`1.00x`) | **`0` (locked `DEC-037` + ~30s logit-stitch eval)** |
| **`p3_268k`** | `80` | `128` (`64`) | `64` | **`268,260`** | `+63.63%` | `188,996` (`70.5%`) | `494.19 KiB` | `1,252.8` (`1.58x`) | `15` (new) |
| **`p4_404k`** | `96` | `160` (`80`) | `96` | **`403,524`** | `+146.14%` | `289,924` (`71.8%`) | `726.88 KiB` | `1,822.3` (`2.29x`) | `15` (new) |
| **`p5_666k`** | `128` | `200` (`100`) | `128` | **`665,556`** | `+305.98%` | `464,964` (`69.9%`) | `1,237.63 KiB` (`1.21 MiB`) | `3,172.9` (`4.00x`) | `15` (new) |
| **`p6_1051k`** | `160` | `256` (`128`) | `128` | **`1,051,492`** | `+541.39%` | `739,364` (`70.3%`) | `1,941.29 KiB` (`1.90 MiB`) | `4,966.8` (`6.25x`) | `15` (new) |

*Notes:*
- `p5_666k` (`665,556` params) matches the `666,060`-parameter scale reported in the SLAMSS-IFS paper text, while `p6_1051k` (`1,051,492` params) remains `5.20x` smaller than the official `5,465,768`-parameter SLAMSS-IFS repository model. All `d_enc` values are multiples of `8`, so `c1 = d_enc // 2` is always divisible by `4` for `GroupNorm(4, c1)`.
- **Off-Manifold Historical Pareto Reference:** `EXP-029` `cnn_bigru_200k` (`223,268` params, `66.61%` BIDSleep Macro-F1, `368.04 KiB` Dynamic INT8) is plotted/reported as a zero-cost off-manifold historical reference point (excluded from the `p1_100k`–`p6_1051k` elbow candidate set).

### 2. Training Protocol Parity (`EXP-021` / `EXP-029`)

*   **Optimizer & Schedule:** `Adam(lr=1e-4, betas=(0.9, 0.999))`, `batch_size = 4`, `val_batch_size = 16`, `max_epochs = 100`, `min_epochs = 40`, `patience = 25` on inner-validation Macro-F1, `grad_clip = 1.0`, `random_crop = True`, `L = 1200`.
*   **Loss:** `CE + 0.01 * RCCLoss` computed strictly in FP32 outside `torch.amp.autocast` (CUDA AMP float16 forward/backward).
*   **Inference:** [`LogitStitchingSlidingWindowPredictor`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-029/exp029_calibration.py) (`L = 1200`, pre-softmax logit stitching).
*   **`p2_164k` Stitcher-Parity Baseline:** Before elbow selection, run a ~30s inference-only pass (`0` training runs) of `LogitStitchingSlidingWindowPredictor` over the 15 frozen `DEC-037` checkpoints listed in [`experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json) to establish an apples-to-apples `M(p2_164k)` baseline (avoiding any stitcher mismatch vs. `SlidingWindowPredictor` in `final_model_oof_subject_metrics.csv`).
*   **Compute Budget:** `5` new widths x `5` folds x `3` seeds (`42, 123, 2024`) = `75` runs, split across `cuda:0` and `cuda:1` (`~2 hours` total on 2x RTX 3060).

### 3. Pre-Registered In-Cohort Elbow Selection Rule (Zero Selection Bias)

Let $M(p)$ be the BIDSleep `N = 47` 5-fold CV 3-seed participant-mean Macro-F1 (in percentage points, `%`) for grid point $p \in \{\text{p1\_100k}, \text{p2\_164k}, \text{p3\_268k}, \text{p4\_404k}, \text{p5\_666k}, \text{p6\_1051k}\}$, ordered strictly by parameter count $N(\text{p1\_100k}) < N(\text{p2\_164k}) < \dots < N(\text{p6\_1051k})$, with $M(\text{p2\_164k})$ obtained from the `LogitStitchingSlidingWindowPredictor` inference-only pass on the 15 frozen `DEC-037` checkpoints (and cross-checked against `final_model_oof_subject_metrics.csv` normalized by `* 100.0`).

1. Compute $M_{\max} = \max_{p} M(p)$ and $\Delta_{\max} = M_{\max} - M(\text{p2\_164k})$.
2. **Degenerate / Flat-Sweep Guard:** If $\Delta_{\max} \le 0.50\text{ pp}$, select `p2_164k` (`163,940` params) by parsimony (`guard_triggered = true`).
3. **Elbow Threshold $T$:** Otherwise ($\Delta_{\max} > 0.50\text{ pp}$), compute:
   $$T = \max\big(M(\text{p2\_164k}) + 0.90 \cdot \Delta_{\max},\; M_{\max} - 0.20\text{ pp}\big)$$
   This requires an eligible point to capture **both** $\ge 90\%$ of the sweep's maximum gain over `163,940` **and** lie within `<= 0.20 pp` of the sweep maximum $M_{\max}$.
4. **Selection:**
   $$\text{CNN-BiGRU-Star} = \arg\min_{p \;:\; M(p) \ge T} N(p)$$
5. **Locking & Sensitivity Note:** This rule is executed once by `select_exp030_star.py` and written to `experiments/EXP-030/elbow_selection.json` *before* `eval_exp030_downstream.py` is invoked. If adjacent grid points near the elbow differ by `~0.2–0.3 pp`, we will discuss the trade-off against edge memory and MMACs transparently in the report without altering the locked selection.

### 4. Downstream Evaluation Contract (Run ONLY on `CNN-BiGRU-Star`)

**`p2_164k` Fallback & Self-Comparison Guard:** If the degenerate guard selects `star_tag == "p2_164k"`, `eval_exp030_downstream.py` loads the 15 frozen `DEC-037` checkpoints via [`experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json) and guards Pillars 1 and 3 against redundant zero-diff self-comparison with `Model A-164k` (reporting `0.00 pp` identity instead of running a degenerate Wilcoxon test).

1. **BIDSleep Paired Statistics (`N = 47`):**
   - Compare `CNN-BiGRU-Star` against locked `Model A-164k` (`p2_164k`, `64.79%` historical / logit-stitched baseline) and `Official SLAMSS-IFS` (`67.37%`) using 10,000-resample participant bootstrap 95% CIs (`seed = 42`), two-sided Wilcoxon signed-rank tests, Cohen's $d_z$, and Holm-Bonferroni FWER adjustment across the 2 primary paired comparisons (`vs. Model A-164k`, `vs. SLAMSS-IFS`), plus secondary per-stage F1 (Wake, Light, Deep, REM).
   - **Claim Framing:** Report the exact difference vs. `SLAMSS-IFS` honestly ("gap narrowed from `2.58 pp` to `X pp`; residual difference not ruled out" if $p \ge 0.05$, or report exact significant difference if $p < 0.05$).
2. **`W0-B` Dynamic INT8 Edge Profiling:**
   - Compute analytical + empirical Dynamic INT8 in-memory payload (`KiB`), serialized disk size, and 1T/4T CPU latency across `p1_100k`–`p6_1051k` (untrained architecture instantiation for the `p1_100k`..`p6_1051k` curve) plus full 15-checkpoint OOF FP32-vs-INT8 Macro-F1 parity (`N = 47`) on `CNN-BiGRU-Star`.
   - **Quantization Backend & Input Shape:** Use CPU `torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear, nn.GRU}, dtype=torch.qint8)` profiled at target input shape `(1, 1200, 30)`.
3. **Zero-Shot PhysioNet SleepAccel (`n = 25` QC-clean & `N = 31` all; `em_off` only):**
   - Run all 15 `CNN-BiGRU-Star` checkpoints on locked `EXP-027` preprocessed SleepAccel data, writing isolated prediction bundles to `experiments/EXP-030/sleepaccel_cache/predictions/`.
   - Evaluate both `mean_single_model` (mean across 15 checkpoints) and `prob_ensemble` (mean predicted probabilities across 15 checkpoints).
   - **Like-for-Like Paired Inference:** Pair `CNN-BiGRU-Star` against locked `Model A-164k` and `Official SLAMSS-IFS` (`experiments/EXP-027/exp027_per_subject_metrics.csv` and `experiments/EXP-028/exp028_sleepaccel_raw_paired_tuples.csv`) strictly within matching evaluation mode (`mean_single_model` vs. `mean_single_model`, `prob_ensemble` vs. `prob_ensemble`) using 10,000-resample bootstrap 95% CIs and **unadjusted** two-sided Wilcoxon $p$-values (reported as secondary/external-transfer diagnostics, whereas BIDSleep uses Holm-Bonferroni FWER) for:
     * 4-class Macro-F1, Kappa, and per-stage F1 (`n = 25` and `N = 31`);
     * All 6 Bland-Altman clinical summary metrics (`tst_min`: Total Sleep Time [TST], `sol_min`: Sleep Onset Latency [SOL], `waso_min`: Wake After Sleep Onset [WASO], `sleep_efficiency_pct`, `deep_pct_tst`, `rem_pct_tst`) — testing bias (`pred - ref`), absolute error (`|pred - ref|`), and Limits of Agreement (LoA), while explicitly correcting `EXP-029`'s swapped Deep% comparator (`Model A` = `+10.65%`, `SLAMSS-IFS` = `+8.67%` on `mean_single_model`).
4. **`EXP-020` `FT-2` & Nested Per-Fold `GATE_tau` Personalization (`N = 46`):**
   - Reconcile `eval_exp029_downstream.py`'s hardcoded `tau_star = 0.01` with `EXP-020`'s exact nested inner-validation selection rule in [`run_exp020_regularized_gated_adaptation.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-020/run_exp020_regularized_gated_adaptation.py#L300-L310) without coupling to `FinalSleepStagingSystem` or `exp019_stage_d_selected_configs.json` (use a lightweight `ScaledConventionalCNNBiGRU` state-dict clone helper):
     * **Definitions:** `ZS` = zero-shot evaluation without target-subject adaptation; `FT-2` = 2-night supervised fine-tuning of `classifier_head|temp_norm|temporal_gru`; `min_stage_frac` = minimum per-stage label fraction across a subject's calibration nights (`cal_recs = recs[:2]`); `GATE_tau` = abstention gate that falls back to `ZS` when `min_stage_frac < tau`; `N = 46` = BIDSleep subjects with `>= 3` usable nights (1 subject of `N = 47` with `< 3` nights excluded).
     * For each fold $f \in \{1..5\}$, run `FT-2` (`epochs = 20`, `Adam(lr=1e-4, weight_decay=0.0)`, `grad_clip = 1.0`, `context_len = 1200`) on `inner_val_subjects` (`5` subjects/fold) using `cal_recs = recs[:2]` and `eval_recs = recs[2:]`, evaluating both `inner_val_subjects` and `outer_test_subjects` with `LogitStitchingSlidingWindowPredictor` via `evaluate_records_macro_f1`.
     * Sweep `TAU_GRID = [0.01, 0.02, 0.03, 0.05]` on `inner_val_subjects`, selecting `tau_f*` that maximizes 4-decimal-rounded mean Macro-F1 (`inner_zs_mean[s]` if `min_stage_frac < tau` else `inner_ft2_mean[s]`), breaking ties by smallest `tau`.
     * Apply the fold-specific `tau_f*` (never a global hardcoded `0.01`) to `outer_test_subjects` in fold $f$, and report `ZS`, `FT-2`, `GATE_tau`, paired 95% CIs, Wilcoxon $p$-values, and exact per-fold `tau_f*` + abstention counts.

## Alternatives Considered

### Alternative A: Retrain `Model A-164k` (`P2`) inside `EXP-030`
*   **Pros:** Uniform code path for all 6 grid points.
*   **Cons:** Wastes 15 GPU runs (~20 min) and risks minor numerical drift against the SHA-256-sealed `DEC-037` checkpoints (`64.79%`).
*   **Reason for Rejection:** Rejected by author: `P2` is already frozen and sealed under `DEC-037` with identical architecture, loss, cache, and manifests; reusing it saves compute and preserves exact continuity.

### Alternative B: Run Downstream Benchmarks on All 6 Widths and Pick the Best
*   **Pros:** Shows SleepAccel and `GATE_tau` trends across all 6 capacities.
*   **Cons:** Introduces severe multiple-testing / post-hoc selection bias if SleepAccel or `GATE_tau` influences backbone choice, and multiplies CPU/GPU evaluation time by 6x.
*   **Reason for Rejection:** Rejected by author: violates zero-selection-bias preregistration; `CNN-BiGRU-Star` must be locked solely from in-cohort BIDSleep CV before touching external SleepAccel or personalization benchmarks.

### Alternative C: Keep `DampedEMCalibrator` (`em_on`) in SleepAccel Evaluation
*   **Pros:** Continuity with `EXP-029`.
*   **Cons:** Empirically falsified in `EXP-029` across all 4 arms (degraded `full_v5` by `-2.21 pp` and widened Deep%/REM% bias).
*   **Reason for Rejection:** Rejected by author: `em_off` is the sole valid zero-shot operating mode.

## Detailed Implementation Plan

### Reused Helpers (Do Not Duplicate)

*   `experiments/EXP-029/run_exp029_cv.py`: `verify_manifests_and_splits`, `load_all_records_by_subject`, `compute_stage_loss_fp32`.
*   `experiments/EXP-029/eval_exp029_downstream.py`: `holm_bonferroni`, `compute_int8_profile`.
*   `experiments/EXP-021/analyze_tsk08_paired_benchmark.py`: `participant_bootstrap_ci`.
*   `experiments/EXP-027/run_exp027_sleepaccel_inference.py`: `load_preprocessed_sleepaccel`, `extract_8_metrics`.
*   `experiments/EXP-027/run_exp028_sleepaccel_bland_altman.py`: `compute_bland_altman_stats`.
*   `experiments/EXP-019/exp019_adaptation_core.py`: `prepare_calibration_batch`; `experiments/EXP-019/run_exp019_stage_d.py`: `evaluate_records_macro_f1`.

### Target Files to Create/Modify (After Design Audit Approval)

#### 1. `experiments/EXP-030/exp030_models.py` (Create)
*   **Type / Scope:** Model factory for `ConventionalCNNBiGRU` width grid `p1_100k`–`p6_1051k` (leaving `exp013_models.py` and `sleep_adaptive_net.py` untouched).
*   **Proposed Changes:** Implement `ScaledModalityEncoder(in_features=1, d_enc=d_enc, dropout=0.2)` (`c1 = d_enc // 2, c2 = d_enc`), `ScaledConventionalCNNBiGRU`, `EXP030_GRID` registry (`p1_100k`, `p2_164k`, `p3_268k`, `p4_404k`, `p5_666k`, `p6_1051k`), and `build_exp030_model(tag: str, loss_type: str = "rccloss") -> nn.Module`.

#### 2. `experiments/EXP-030/run_exp030_cv.py` (Create)
*   **Type / Scope:** 5-fold x 3-seed CV training and OOF evaluation driver (`75` new runs for `p1_100k`, `p3_268k`, `p4_404k`, `p5_666k`, `p6_1051k`), plus `--eval-p2-manifest` inference-only `LogitStitchingSlidingWindowPredictor` pass over the 15 frozen `DEC-037` checkpoints (`experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json`).
*   **CLI & Output Templates:** Supports `--grid-points`, `--device`, `--folds`, `--seeds`, `--eval-p2-manifest`, and `--smoke` (2-epoch dry-run into `smoke_out/`). Reuses `verify_manifests_and_splits`, `load_all_records_by_subject`, and `compute_stage_loss_fp32` from `run_exp029_cv.py`. Saves checkpoints to `experiments/EXP-030/checkpoints/<tag>/<tag>_L1200_fold{fold}_seed{seed}.pt` and OOF metrics to `experiments/EXP-030/<tag>_subject_metrics.csv`.

#### 3. `experiments/EXP-030/select_exp030_star.py` (Create)
*   **Type / Scope:** Pre-registered elbow selector.
*   **Proposed Changes:** Reads `N = 47` BIDSleep OOF Macro-F1 (`<tag>_subject_metrics.csv`) for `p1_100k`..`p6_1051k` (using the logit-stitched `p2_164k` CSV, and normalizing `final_model_oof_subject_metrics.csv`'s 141-row `[0, 1]` values by `* 100.0` for cross-check), applies the exact threshold formula $T$, and writes `experiments/EXP-030/elbow_selection.json` with keys `star_tag`, `star_params`, `m_max`, `delta_max`, `threshold_T`, `guard_triggered`, and `grid_summary`.

#### 4. `experiments/EXP-030/eval_exp030_downstream.py` (Create)
*   **Type / Scope:** Downstream evaluation suite restricted to `CNN-BiGRU-Star` (plus `W0-B` payload/latency curve across `p1_100k`–`p6_1051k`), with `star_tag == "p2_164k"` checkpoint-loader fallback via `experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json` and self-comparison guard in Pillars 1 and 3.
*   **Proposed Changes:** Reuses `holm_bonferroni`, `compute_int8_profile`, `participant_bootstrap_ci`, `load_preprocessed_sleepaccel`, `extract_8_metrics`, `compute_bland_altman_stats`, `prepare_calibration_batch`, and `evaluate_records_macro_f1`. Implements `FT-2` / nested per-fold `GATE_tau` (`TAU_GRID = [0.01, 0.02, 0.03, 0.05]`, `weight_decay=0.0`, `grad_clip=1.0`, `context_len=1200`, `LogitStitchingSlidingWindowPredictor` on `inner_val` and `outer_test`) via a lightweight `ScaledConventionalCNNBiGRU` state-dict clone helper, and CPU `quantize_dynamic(..., {nn.Linear, nn.GRU}, dtype=torch.qint8)` at `(1, 1200, 30)`.

#### 5. `experiments/EXP-030/test_exp030_unit.py` (Create)
*   **Type / Scope:** Unit & contract tests.
*   **Proposed Changes:** Verify exact parameter counts (`98,940`, `163,940`, `268,260`, `403,524`, `665,556`, `1,051,492`), bit-for-bit `p2_164k` parameter and `state_dict` parity with `ConventionalCNNBiGRU(64, 96)`, `forward()` 14-key dict contract (`seq_lens=None` and packed), Dynamic INT8 quantization compatibility, elbow rule edge cases (including `guard_triggered`), and nested `tau*` tie-breaking.

### Verification Plan

*   **Unit Tests:** `pytest experiments/EXP-030/test_exp030_unit.py` (must pass 100% before any GPU run).
*   **Smoke Run:** `python experiments/EXP-030/run_exp030_cv.py --grid-points p1_100k --folds 1 --seeds 42 --device cuda:0 --smoke`
*   **`p2_164k` Stitcher-Parity Eval (~30s):** `python experiments/EXP-030/run_exp030_cv.py --eval-p2-manifest experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json --device cuda:0`
*   **75-Run Dual-GPU Sweep:**
    - `python experiments/EXP-030/run_exp030_cv.py --grid-points p1_100k p3_268k p4_404k --folds 1 2 3 4 5 --seeds 42 123 2024 --device cuda:0`
    - `python experiments/EXP-030/run_exp030_cv.py --grid-points p5_666k p6_1051k --folds 1 2 3 4 5 --seeds 42 123 2024 --device cuda:1`
*   **Elbow Selection & Downstream:**
    - `python experiments/EXP-030/select_exp030_star.py`
    - `python experiments/EXP-030/eval_exp030_downstream.py --device cuda:0`
    - Run `/egm-drift` post-execution audit.

--------------------------------------------------------------------------------

*Drafted with EGM*
