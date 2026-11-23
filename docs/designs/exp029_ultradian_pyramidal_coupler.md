# EXP-029: Ultradian-Pyramidal-Coupler (`v5`) Lightweight Sleep Staging Architecture

*   **Author:** jbjoyk
*   **Status:** Approved for Implementation (EGM Blockers & Majors Resolved)
*   **Date:** 2026-11-23

--------------------------------------------------------------------------------

## Overview

`EXP-029` defines `Ultradian-Pyramidal-Coupler` (`190,788` parameters), a lightweight neural architecture for full-night (`L = 1200` 30-second epochs) 4-class sleep staging (Wake, Light, Deep, REM) from Apple Watch instantaneous heart rate (`IHR`) and wrist accelerometry (`ACC`). It replaces the static 30-second pooling and single-scale `BiGRU` in [`ConventionalCNNBiGRU`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L27-L183) (`163,940` parameters) with two structural mechanisms—sub-epoch (`~0.23 Hz`) pre-pooling `ACC -> HR` residual FiLM conditioning and a mask-aware `2`-scale pyramidal `BiGRU` (`30s` fine + `2-min` coarse)—paired with a decoupled zero-parameter post-hoc prior-calibration toggle inside [`SlidingWindowPredictor`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp018_inference.py#L20-L182).

The design consumes the exact locked `prep_57738d534b` feature cache, locked 5-fold subject-disjoint BIDSleep splits (`experiments/EXP-014/EXP014_MANIFESTS/fold_1..5.json`, cryptographically cross-checked against `review_package/fold_manifests.csv` SHA-256 prefix `36d84020`), and `EXP-027` PhysioNet SleepAccel cache without re-extracting raw recordings, while preserving `ConventionalCNNBiGRU` (`163,940` params) and `Official SLAMSS-IFS` (`5,465,768` params) as untouched reference baselines.

## The Problem

1. **In-Cohort Performance Gap vs. Server Reference (`EXP-021`):** `ConventionalCNNBiGRU` (`64.79%` Macro-F1) trails `Official SLAMSS-IFS` (`67.37%` Macro-F1) by `-2.58 pp` (`p = 1.98e-6`), driven primarily by **Deep F1 (`-3.51 pp`)** and **Light F1 (`-2.83 pp`)**. Both [`ConventionalCNNBiGRU`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L64-L85) and [`SleepAdaptiveNet`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/sleep_adaptive_net.py#L79-L138) compress each 30-second epoch independently via `AdaptiveAvgPool1d(1)` before any HR–ACC interaction, discarding sub-epoch phase alignment between wrist movement bursts and heart-rate spikes, and rely on a single-scale `BiGRU` over `L = 1200` epochs without explicit `2–16 min` ultradian downsampling.
2. **Cross-Modality Clinical Bias (`EXP-027` / `EXP-028`):** On external single-night PSG (`SleepAccel`, `n = 25 qc_clean`), both Model A and SLAMSS-IFS exhibit `+8.67%` to `+10.65%` Deep% bias, `+7.21%` to `+10.13%` REM% bias, and `-13.57` to `-14.16` min WASO bias due to Dreem 2 ambulatory EEG (`18.9%` Deep, `44.2%` Light) to in-lab PSG (`11.1%` Deep, `57.7%` Light) label-prior shift.
3. **Prior Multi-Head Failure (`EXP-014`):** `SleepAdaptiveNet` (`66.06%` vs. `66.63%`) demonstrated that bolting five competing auxiliary heads/losses (GRL, boundary loss, transition head, uncertainty head, epoch-level gate) onto the same 30-second bottleneck adds gradient interference without improving staging.
4. **Publication & Governance Constraints (`DEC-031`, `DEC-039`):** Under `DEC-031`, `163,940` parameters is the reference baseline and any capacity expansion requires explicit scientific justification and relative $\Delta\%$ disclosure. The proposed architecture (`190,788` params, `+16.38%` vs. `163,940`) justifies this expansion by adding sub-epoch FiLM and 2-scale pyramidal recurrence, while testing against a larger `ConventionalCNNBiGRU(d_temp=128)` control (`223,268` params, `+36.19%`) to rule out raw parameter-count effects. It also maintains `W0-B` Dynamic INT8 compatibility (`nn.Linear` + `nn.GRU`) and full-night `L = 1200` offline buffering on a smartwatch SoC or companion phone.

## Goals

*   **G1 (In-Cohort Primary Endpoint — `DEC-033`/`DEC-041` Compliant):** Achieve statistically significant Macro-F1 superiority over `ConventionalCNNBiGRU-164k` (`64.79%`) and `ConventionalCNNBiGRU-200k` (`223,268` params), and establish empirical parity (no statistically significant difference under paired testing, without claiming formal TOST equivalence) or superiority against `Official SLAMSS-IFS` (`67.37%` Macro-F1 on BIDSleep 5-fold CV, `N = 47`, 3 seeds = 15 runs) at `190,788` parameters (`28.65x` smaller than SLAMSS-IFS), specifically targeting Deep (`-3.51 pp`) and Light (`-2.83 pp`) F1 deficits.
*   **G2 (Capacity Attribution under `DEC-031`):** Isolate the individual contribution of each proposed mechanism against both `ConventionalCNNBiGRU-164k` (`163,940` params) and `ConventionalCNNBiGRU-200k` (`223,268` params) via a 6-arm falsifiable ablation matrix with Holm-Bonferroni FWER control.
*   **G3 (External Zero-Shot & Clinical Agreement):** Maintain or improve zero-shot Macro-F1 on PhysioNet SleepAccel (`n = 25 qc_clean` / `N = 31 all`, currently `60.51%` single-model / `62.94%` ensemble) and evaluate Deep%, REM%, and WASO Bland-Altman bias reduction using a zero-tuning, a priori locked post-hoc `DampedEMCalibrator` (`alpha = 0.5, n_iter = 3, T = 1.0`).
*   **G4 (Quantization & Personalization Safety):** Preserve `W0-B` Dynamic INT8 parity (`< 0.10 pp` Macro-F1 delta vs. FP32, `< 400 KiB` in-memory payload) and safe multi-night `FT-2` / `GATE_tau` personalization on stage-starved subjects (`Bidslab34`, `N = 46` multi-night cohort).

## Non-Goals

*   **NG1:** Re-extracting raw `Bidslab*` or SleepAccel recordings (strictly locked to `prep_57738d534b` and `EXP-027` caches).
*   **NG2:** Modifying or retraining frozen `ConventionalCNNBiGRU` (`163,940` params) or `Official SLAMSS-IFS` (`5,465,768` params) checkpoints.
*   **NG3:** Adding auxiliary multi-task heads/losses (no GRL, boundary loss, transition head, or uncertainty head).
*   **NG4:** Bare-metal `512 KB` SRAM MCU 1-epoch causal streaming (`DEC-039` locks deployment to full-night `L = 1200` offline buffering on smartwatch SoC / companion phone).
*   **NG5:** Writing or executing Python code during this design phase (strict NO-CODE).

## Open Questions

*   **OQ1:** Does training `Ultradian-Pyramidal-Coupler` with pure `CrossEntropyLoss` (without `0.01 * RCCLoss`) reduce baseline SleepAccel Deep%/REM% bias and improve `pi_src` posterior calibration for post-hoc EM? *(Addressed by a dedicated secondary ablation on Arm `(e)`.)*

## Critical User Journeys (CUJs)

### CUJ 1: 5-Fold Nested CV & 6-Arm Ablation Execution on Dual RTX 3060 GPUs

*   **Given**: The locked `prep_57738d534b` cache (`246` usable nights, `47` subjects) and `experiments/EXP-014/EXP014_MANIFESTS/fold_1..5.json` manifests (verified against `review_package/fold_manifests.csv` SHA-256 prefix `36d84020`) are present on disk, and `cuda:0` + `cuda:1` (2x RTX 3060 12GB) are available.
*   **When**: The researcher launches the 15-run (5 folds x 3 seeds: `42, 123, 2024`; `7` runs on `cuda:0`, `8` runs on `cuda:1`) dual-GPU training and OOF evaluation pipeline for Arms `(b)`–`(e)` (plus pure-`CE` sensitivity on Arm `(e)`), consuming `(B, 1200, 30)` `x_ihr`, `(B, 1200, 30)` `x_accel`, and `(B, 1200, 5)` `x_meta`.
*   **Then**: All 15 checkpoints per arm complete within `~25–45 min`, per-subject 3-seed means (`N = 47`) are aggregated, and paired comparisons (10,000-resample participant bootstrap 95% CIs + two-sided Wilcoxon signed-rank tests with Holm-Bonferroni FWER correction) evaluate Arm `(e)` against `(a)`, `(b)`, `(c)`, `(d)`, and `(f)`.

### CUJ 2: Zero-Shot External SleepAccel Validation & Post-Hoc Prior Calibration

*   **Given**: The 15 trained checkpoints of `Ultradian-Pyramidal-Coupler` and the locked `EXP-027` PhysioNet SleepAccel cache (`n = 25 qc_clean`, `N = 31 all`) are loaded with zero retraining or threshold tuning on SleepAccel.
*   **When**: [`SlidingWindowPredictor`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp018_inference.py#L20-L182) stitches pre-softmax logits across overlapping windows (for both `N <= L` and `N > L`) and runs unified post-stitch softmax + `DampedEMCalibrator` (`off` vs. `on` with locked `alpha = 0.5, n_iter = 3, T = 1.0`, using per-fold `inner_val` mean predicted marginals `pi_src`), writing isolated outputs to `experiments/EXP-029/sleepaccel_cache/predictions/`.
*   **Then**: `EXP-027` 4-class Macro-F1 and `EXP-028` Bland-Altman clinical metrics (`tst_min`, `sol_min`, `waso_min`, `sleep_efficiency_pct`, `deep_pct_tst`, `rem_pct_tst`) are persisted for both `mean_single_model` and `prob_ensemble` modes and compared against Model A and SLAMSS-IFS.

### CUJ 3: `W0-B` Dynamic INT8 Edge Profiling & `FT-2` / `GATE_tau` Safety Check

*   **Given**: All 15 `Ultradian-Pyramidal-Coupler` checkpoints are frozen and serialized.
*   **When**: `experiments/STAGE2_FINAL_MODEL/benchmark_int8_edge.py` (`quantize_dynamic` on `nn.Linear` + `nn.GRU` -> `qint8`) profiles FP32 vs. Dynamic INT8 Macro-F1 parity, in-memory weight payload (`KiB`), and single-/4-thread CPU latency across all 47 BIDSleep subjects, and a separate `EXP-020` `FT-2` + `GATE_tau` safety script runs on the `N = 46` multi-night cohort (1 subject excluded due to having only 1 night).
*   **Then**: Dynamic INT8 preserves `< 0.10 pp` Macro-F1 parity with `131,332 / 190,788 (68.84%)` parameters quantized (`360.50 KiB < 400 KiB`), and `GATE_tau` protects `Bidslab34` from missing-stage negative transfer.

## Technical Plan (Architecture)

### 1. Input Tensor Contract, `forward()` Interface, & Sub-Epoch Pre-Pooling Encoders

All arms consume and return the exact [`ConventionalCNNBiGRU.forward`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L100-L183) signature and 14-key return dictionary (`"logits"`, `"emission_logits"`, `"probabilities"`, `"alpha_gate"`, `"confidence"`, `"entropy"`, `"total_loss"`, `"loss_stage"`, `"loss_trans"`, `"loss_boundary"`, `"loss_adv"`, `"loss_conf"`, `"boundary_mask"`, `"temp_features"`), supporting `seq_lens=None` (treating all `T` epochs as valid when `None`, as required by `benchmark_int8_edge.py:207`):
*   `x_ihr`: `(B, T, 30)` float32 (`T = 1200` during training/standard inference; supports `L in {1, 20, 60, 120, 300, 600, 1200}`).
*   `x_accel`: `(B, T, 30)` float32 (1-channel ACC magnitude).
*   `x_meta`: `(B, T, 5)` float32 (`[freq, 0, 0, hr_mean, hr_std]`, circadian-neutralized).
*   `seq_lens`: `Optional[torch.Tensor]` `(B,)` int64 (valid epoch count `N <= T`).

In [`sleep_adaptive_net.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/sleep_adaptive_net.py#L79-L138), `HREncoder` and `ACCEncoder` apply `Conv1d(1->32, k=5) -> MaxPool(2) -> TCB(32->64, k=5) -> MaxPool(2) -> TCB(64->64, k=3)` yielding `(B*T, 64, 7)` (`~0.233 Hz`, `~4.29s/step` sub-epoch resolution) before `AdaptiveAvgPool1d(1)`. We expose this pre-pooled `(B*T, 64, 7)` map (preserving identical `29,568` encoder params/stream and `C <= 32` stem width to avoid `30x` MMAC inflation).

### 2. Module 1: Sub-Epoch (`~0.23 Hz`) Pre-Pooling `ACC -> HR` Residual FiLM (`8,320` params)

Before `AdaptiveAvgPool1d(1)`, a pointwise `Conv1d(64, 128, kernel_size=1)` projects `h_acc_seq` `(B*T, 64, 7)` into zero-initialized residual scale/shift modulators (both `weight` and `bias` initialized to `0.0` so `gamma = 1.0, beta = 0.0` at step 0):
*   `gamma = 1.0 + tanh(d_gamma)`, `beta = d_beta` (where `d_gamma, d_beta = chunk(FiLM_proj(h_acc_seq), 2, dim=1)`).
*   `h_hr_mod = gamma * h_hr_seq + beta`.
*   Both `h_hr_mod` and `h_acc_seq` then pass through `AdaptiveAvgPool1d(1) -> Dropout(0.2) -> Linear(64, 64) -> LayerNorm(64)`, and are concatenated with `meta_proj(x_meta)` (`16`-dim) as `[h_hr_mod; h_acc; h_meta]` (`(B, T, 144)`). Keeping the unmodulated `h_acc` skip path ensures gross-movement Wake cues remain intact.

### 3. Module 2: Mask-Aware 2-Scale Pyramidal `BiGRU` (`116,448` params)

```text
  x_ihr (B,T,30) ──► HREncoder_prepool ──► (B*T,64,7) ──┐
                                                         ├─► Sub-Epoch FiLM (ACC->HR) ──► Pool+Proj ──► h_hr (B,T,64)
  x_accel (B,T,30) ─► ACCEncoder_prepool ─► (B*T,64,7) ──┴────────────────────────────────────────────► h_acc (B,T,64)
  x_meta (B,T,5)  ──► meta_proj ──────────────────────────────────────────────────────────────────────► h_meta (B,T,16)
                                                                                                             │
  ┌────────────────────────────────────────── Concat [h_hr; h_acc; h_meta] (B,T,144) ◄───────────────────────┘
  ▼
  Fine 1-Layer BiGRU (144 -> 48/dir = 96, dropout=0.0, pack_padded_sequence) ──► h_fine (B,T,96)
  │                                                                                   │
  ├─► Zero-mask [seq_lens:T] ─► Mask-Weighted 4x AvgPool ─► h_pool (B,ceil(T/4),96)
  │                                  │
  │                                  ▼
  │                         Coarse 1-Layer BiGRU (96 -> 48/dir = 96, dropout=0.0, seq_lens_coarse)
  │                                  │
  │                                  ▼
  │                         repeat_interleave(4)[:, :T, :] + Zero-mask ──► h_coarse_up (B,T,96)
  │                                                                                   │
  └──────────────────────► Concat [h_fine; h_coarse_up] (B,T,192) ◄───────────────────┘
                                     │
                                     ▼
                           Linear(192->96) + LayerNorm(96) ──► Classifier Head (96->64->4) ──► logits (B,T,4)
```

*   **Padding Protection & `seq_lens=None` Fallback:** When `seq_lens` is `None`, `seq_lens = torch.full((B,), T, dtype=torch.int64)` and `valid_mask` is all ones. When `seq_lens[b] < T` (including `L = 1` edge cases), `valid_mask` `(B, T, 1)` zeroes `[seq_lens:T]` before `4x` pooling. If `T % 4 != 0`, `h_fine` is right-padded with zeros to `4 * ceil(T / 4)` before `reshape(B, T_c, 4, 96)`. Mask-weighted pooling computes `sum(x * m) / clamp(sum(m), min=1.0)`, and `seq_lens_coarse = ((seq_lens + 3) // 4).clamp(min=1)` is passed to `pack_padded_sequence` in the coarse `BiGRU`. Both 1-layer `BiGRU`s use `dropout=0.0` (avoiding PyTorch's single-layer `nn.GRU` dropout warning), and the redundant bare `Linear(96->96)` projection before the coarse `BiGRU` is omitted.
*   **Phase Alignment Note:** `repeat_interleave(4)` produces 2-minute coarse blocks; for `N > 1200` (`Case 2`), the trailing anchor window `w = N - 1200` shifts the 4-epoch pooling grid by `(N - 1200) % 4` relative to `w = 0`, which is averaged smoothly during sliding-window logit stitching.

### 4. Module 3: Training Objective (FP32 Safe) & Unified Post-Stitch Prior Calibration (`0` params)

*   **Training Objective & AMP Safety:** Primary training uses `CE + 0.01 * RCCLoss` (exact parity with `ConventionalCNNBiGRU` and `SLAMSS-IFS`). To prevent `float16` `RCCLoss` clamping (`1.0 - 1e-5 -> 1.0 -> log(0) = -inf`) under CUDA AMP, loss computation inside `forward()` (and/or in `run_exp029_cv.py`) is wrapped strictly in `torch.amp.autocast(device_type=..., enabled=False)` with `logits.float()`. A secondary pure-`CE` ablation on Arm `(e)` evaluates baseline posterior calibration.
*   **Unified Logit-Stitching & Locked `DampedEMCalibrator` (`SlidingWindowPredictor`):**
    1. Modify `SlidingWindowPredictor.predict_night` to accumulate **pre-softmax logits** (`logit_sum / counts`) across overlapping windows (for `N > L`) and route **both** `N <= L` (Case 1, removing its early-return bypass) and `N > L` (Case 2) through a single unified post-stitch `F.softmax(stitched_logits / T, dim=-1)` step.
    2. When `prior_calibrator` is enabled, compute `pi_src` per fold as the model's mean predicted softmax marginal on that fold's `inner_val_subjects` (accounting for `RCCLoss` distortion). Apply Saerens et al. (2002) EM on the full-night `(N, 4)` stitched probability matrix (zero tuning on SleepAccel) with locked a priori parameters **`alpha = 0.5, n_iter = 3, T = 1.0`**:
       $$\hat{\pi}_{\text{EM}}^{(t+1)}(c) = \frac{1}{N} \sum_{i=1}^{N} \frac{p_i(c) \cdot \pi^{(t)}(c) / \pi_{\text{src}}(c)}{\sum_{k} p_i(k) \cdot \pi^{(t)}(k) / \pi_{\text{src}}(k)}, \quad \pi^{(t+1)} = (1 - \alpha)\pi^{(t)} + \alpha \hat{\pi}_{\text{EM}}^{(t+1)}$$
       initialized at $\pi^{(0)} = \pi_{\text{src}}$, emitting final calibrated probabilities $p_i^{\text{cal}}(c) \propto p_i(c) \cdot \pi^{(n_{\text{iter}})}(c) / \pi_{\text{src}}(c)$.
    3. For `prob_ensemble` mode on SleepAccel, evaluate `DampedEMCalibrator` on the ensemble-averaged pre-softmax logits (`ens_probs`) as well as per-checkpoint for `mean_single_model`.

### 5. Exact Parameter Breakdown & 6-Arm Ablation Matrix (`DEC-031` Compliant)

| Arm / Model | Key Configuration | Encoders + Meta | FiLM | Temporal Backbone | Norm + Cls Head | Total Params | $\Delta\%$ vs. `163,940` |
|---|---|---:|---:|---:|---:|---:|---:|
| **(a) `ConventionalCNNBiGRU-164k`** | Locked baseline (`d_temp=96`, 2L BiGRU) | 59,232 | 0 | 97,920 | 6,788 | **163,940** | `0.00%` |
| **(b) `ConventionalCNNBiGRU-200k`** | Capacity control (`d_temp=128`, 2L BiGRU) | 59,232 | 0 | 155,136 | 8,900 | **223,268** | `+36.19%` |
| **(c) `FiLM-Only`** | 2L BiGRU (`d_temp=96`) + Sub-Epoch FiLM | 59,232 | 8,320 | 97,920 | 6,788 | **172,260** | `+5.07%` |
| **(d) `Pyramid-Only`** | 2-Scale BiGRU (`hidden=48/dir`), no FiLM | 59,232 | 0 | 116,448 | 6,788 | **182,468** | `+11.30%` |
| **(e) `Ultradian-Pyramidal-Coupler`** | **Full v5 (Sub-Epoch FiLM + 2-Scale BiGRU)** | 59,232 | 8,320 | 116,448 | 6,788 | **190,788** | **`+16.38%`** |
| **(f) `Official SLAMSS-IFS`** | Locked server reference (untouched) | — | — | — | — | **5,465,768** | `+3234.0%` |

*(Exact Dynamic INT8 Accounting for Arm `(e)`: `nn.Linear + nn.GRU` quantizable parameters = `8,416` [`hr_encoder.proj` 4,160 + `acc_encoder.proj` 4,160 + `meta_proj.0` 96] + `116,448` [Fine BiGRU 55,872 + Coarse BiGRU 42,048 + Fusion Linear 18,528] + `6,468` [`classifier_head` Linear layers 6,208 + 260] = **`131,332 / 190,788 (68.84%)`**. Remaining FP32 parameters [`Conv1d`, `GroupNorm`, `LayerNorm`, `IntraEpochFiLM`] = **`59,456 (31.16%)`**. Empirical Dynamic INT8 in-memory payload = `(131,332 * 1 + 59,456 * 4) / 1024 = 360.50 KiB < 400 KiB`.)*

## Alternatives Considered

### Alternative A: `PhysioDecomp-MSNet` (DS-TCN + 1-Layer `BiGRU` + Post-Pooling `r_HR`)

*   **Pros:** Fast `Conv1d` + 1-layer `BiGRU` execution (`~20–40 min` for 15 runs); `~193.4k` parameters.
*   **Cons:** Post-pooling linear decomposition (`r_HR = h_HR - (W_m h_ACC + b_m)`) cannot recover sub-epoch motion–HR lag and is redundant with downstream linear layers; dilated convolutions shift parameters into FP32 `Conv1d` (reducing Dynamic INT8 coverage) and risk `[seq_len:T]` padding bleed across `62.5`-minute receptive fields.
*   **Reason for Rejection:** Rejected by author in favor of Rank 1 (`Ultradian-Pyramidal-Coupler`) after Goldfish review (`egm-design`, `egm-zoom-out`, `egm-codebase-details`).

### Alternative B: Custom `Diagonal-LRU` Coarse Branch (`UltradianSSM-Coupler` Initial Variant)

*   **Pros:** Explicit log-spaced state decay channels for `15–120 min` ultradian cycles.
*   **Cons:** PyTorch `2.14.1+cu130` lacks native `nn.LRU`, `pack_padded_sequence` support, and `quantize_dynamic` INT8 quantization for custom `nn.Parameter` recurrences; Python sequential loops would slow 15-run CV by `10x–30x`.
*   **Reason for Rejection:** Rejected by author per Goldfish `egm-implementation-readiness` and `egm-codebase-details` audit; replaced with a native `1`-layer `BiGRU` on the `4x`-downsampled stream.

### Alternative C: `ProtoStage-HierNet` (Conformer-Lite + Shrinkage Cosine Prototype Head)

*   **Pros:** Explicit cosine prototype metric space for `FT-2` shrinkage.
*   **Cons:** Macaron Conformer at `d = 96` exceeds `350k` params (`> 300k` hard ceiling); non-shifted windowed attention has zero cross-window ultradian context and triggers `softmax(-inf) = NaN` on all-padded trailing windows; `MultiheadAttention.in_proj_weight` bypasses `quantize_dynamic`; prototype shrinkage duplicates `GATE_tau` (`DEC-040`).
*   **Reason for Rejection:** Rejected by author per 5 of 6 Goldfish lenses.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Hardware & Dual-GPU Scheduling:** Executed across 2x NVIDIA RTX 3060 12GB GPUs (`cuda:0`: `7` runs; `cuda:1`: `8` runs, totaling 15 runs per arm using `--pairs` + `--aggregate`). Peak VRAM `< 450 MB` at `B = 4, L = 1200`; expected wall-clock `~25–45 min` per 15-run arm.
*   **Quantization & SoC Footprint (`W0-B`):** `68.84%` (`131,332 / 190,788`) of parameters quantize under `quantize_dynamic` (`nn.Linear` + `nn.GRU`), yielding a **`360.50 KiB`** Dynamic INT8 in-memory payload (strictly **`< 400 KiB`** ceiling) and `< 85 ms/night` single-thread CPU latency for smartwatch SoC / companion phone full-night buffering.
*   **Reproducibility & Governance:** Deterministic seeds `{42, 123, 2024}`, SHA-256 checkpoint manifests, `review_package/fold_manifests.csv` (`36d84020`) split verification, and zero outer-test or SleepAccel tuning.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Locked Training Protocol (`EXP-021` Parity)

*   **Optimizer & Schedule:** `torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999))`, no LR scheduler, `grad_clip = 1.0` (`clip_grad_norm_`).
*   **Batch & Epoch Budget:** `micro_batch_size = 4`, `effective_batch_size = 4` (`accum_steps = 1`), `val_batch_size = 16`, `max_epochs = 100`, `min_epochs = 40`, `patience = 25` (early stopping on `inner_val_subjects` pooled `macro_f1 > best + 1e-6`).
*   **Data & AMP:** `random_crop = True`, `neutralize_circadian = True`, CUDA AMP `float16` (`GradScaler`) for forward/backward with `CE + 0.01 * RCCLoss` computed strictly in FP32 outside `autocast`.
*   **Statistical Testing Plan:** Unit of analysis is the `N = 47` participant-level 3-seed mean (`42, 123, 2024`). Report 10,000-resample participant bootstrap 95% CIs (`seed = 42`) and two-sided Wilcoxon signed-rank `p`-values with **Holm-Bonferroni** step-down FWER adjustment across the primary comparison family (Arm `(e)` vs. `(a)`, `(b)`, `(c)`, `(d)`, `(f)` on Macro-F1) and secondary per-stage F1s.

### Target Files to Create/Modify

#### 1. `./exp029_models.py` (Create)

*   **Type / Scope:** PyTorch Model Definitions & Factory (`NO-CODE` in this doc; implement only after design approval).
*   **Proposed Changes:**
    *   Implement `PrePoolEncoder` (exposing `(B*T, 64, 7)` feature maps), `IntraEpochFiLM` (`Conv1d(64, 128, 1)` with `weight=0, bias=0` init), `MaskWeightedPool1d` (`4x` mask-weighted average pooling + `seq_lens_coarse` + `seq_lens=None` fallback), and `UltradianPyramidalCoupler` (preserving the 14-key `forward()` return dict and FP32 loss wrapper).
    *   Reuse [`ConventionalCNNBiGRU`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L27-L183)`(d_temp=128)` directly for Arm `(b)` without duplicating a class.
    *   Provide `build_exp029_model(arm_id: str, loss_type: str = "rccloss")` with `arm_id in {"b_200k", "c_film", "d_pyramid", "e_full"}`.

#### 2. `./exp018_inference.py` (Modify — Backward-Compatible Extension)

*   **Type / Scope:** Full-Night Inference Engine (`SlidingWindowPredictor`).
*   **Proposed Changes:**
    *   Accumulate pre-softmax logits (`logit_sum / counts`) in Case 2 (`N > L`) and route both Case 1 (`N <= L`, removing early return) and Case 2 through a unified post-stitch `F.softmax(stitched_logits / T, dim=-1)`.
    *   Add optional `prior_calibrator: Optional[DampedEMCalibrator] = None` (default `None`, preserving bit-exact parity for existing callers) applied to the unified `(N, 4)` `final_probs`.

#### 3. `./experiments/EXP-029/run_exp029_cv.py` & `./experiments/EXP-029/run_exp029_dual_gpu.sh` (Create)

*   **Type / Scope:** Dual-GPU 5-Fold Nested CV Runner & Paired Evaluator.
*   **Proposed Changes:**
    *   Implement `EXP029Dataset` (mapping `prep_57738d534b` records to `(x_ihr, x_accel, x_meta, seq_lens, y, mask)` rather than `SLAMSSMatchedDataset`'s `x_signal`), enforce `verify_manifests_and_splits` against `review_package/fold_manifests.csv` (`36d84020`), and train Arms `(b)`–`(e)` (+ Arm `(e)` `pure_ce`) across `cuda:0` (7 runs) and `cuda:1` (8 runs).
    *   Compute `N = 47` 3-seed participant means, 10,000-resample bootstrap CIs, two-sided Wilcoxon tests, and Holm-Bonferroni adjusted `p`-values vs. Arms `(a)` and `(f)`.

#### 4. `./experiments/EXP-029/run_exp029_eval_sleepaccel_and_edge.py` (Create)

*   **Type / Scope:** External Zero-Shot, `W0-B` INT8, & `FT-2`/`GATE_tau` Validation Driver.
*   **Proposed Changes:**
    *   Run zero-shot `EXP-027`/`EXP-028` evaluation on SleepAccel (`n = 25 qc_clean`, `N = 31 all`) writing isolated prediction bundles to `experiments/EXP-029/sleepaccel_cache/predictions/{sub}__{arm}.npz` (never touching `experiments/EXP-027/cache/predictions/`) with `DampedEMCalibrator` `off` vs. `on` (`alpha = 0.5, n_iter = 3, T = 1.0`, per-fold `inner_val` `pi_src`).
    *   Run `W0-B` `quantize_dynamic` INT8 parity/latency benchmark via `benchmark_int8_edge.py` logic on all 15 checkpoints.
    *   Run `EXP-020` `FT-2` / `GATE_tau` safety evaluation on `N = 46` multi-night subjects (reusing `experiments/EXP-020/run_exp020_regularized_gated_adaptation.py` protocol: adapting `classifier_head` + `temp_norm` + `coarse_gru` with `GATE_tau` `tau* = 0.01`).

#### 5. `./tests/test_exp029_contracts.py` (Create)

*   **Type / Scope:** Unit & Contract Test Suite (CPU + CUDA AMP).
*   **Proposed Changes:**
    *   Verify exact parameter counts (`163,940`, `223,268`, `172,260`, `182,468`, `190,788`), `L in {1, 20, 60, 120, 300, 600, 1200}` forward/backward shapes, `seq_lens=None` and `seq_lens` padding invariance, CUDA AMP `float16` finite `RCCLoss` gradients, `quantize_dynamic` INT8 execution (`131,332` quantized params, `360.50 KiB`), and `DampedEMCalibrator` stability on `0%`-Deep nights.

### Verification Plan

*   **Unit & Contract Tests:** `pytest tests/test_exp029_contracts.py` (must pass 100% on CPU and CUDA AMP before launching any GPU training).
*   **Dual-GPU Smoke Check:** 1-fold, 1-seed, 2-epoch dry run on `cuda:0` and `cuda:1` verifying `prep_57738d534b` cache loading, `36d84020` manifest check, `< 450 MB` peak VRAM, and checkpoint serialization.
*   **Full 15-Run CV & Paired Statistics:** 10,000-resample participant-level bootstrap 95% CIs, two-sided Wilcoxon signed-rank tests (`N = 47`), and Holm-Bonferroni FWER adjustment against Arms `(a)`–`(d)` and `(f)`.
*   **External & Edge Verification:** `W0-B` Dynamic INT8 parity (`|Delta| < 0.10 pp`, `< 400 KiB`) and `EXP-027`/`EXP-028` SleepAccel zero-shot paired metrics + Bland-Altman LoA summary.

--------------------------------------------------------------------------------

*Drafted with EGM*
