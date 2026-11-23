# EXP-029 / v5 Lightweight Full-Night Wearable Sleep Staging Architecture

* **Author:** jbjoyk
* **Date:** 2026-11-23

## Overview

This brief defines and ranks three candidate lightweight neural architectures (`< 250k` parameters target, `< 300k` hard ceiling) for full-night (`L = 1200` 30-second epochs) 4-class sleep staging (Wake, Light, Deep, REM) from Apple Watch instantaneous heart rate (`IHR`) and accelerometry (`ACC`). The goal is to close the `-2.58 pp` Macro-F1 gap between our compact baseline [`ConventionalCNNBiGRU`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L27-L183) (`163,940` params, `64.79%` Macro-F1) and the 33.34x larger server reference `Official SLAMSS-IFS` (`5,465,768` params, `67.37%` Macro-F1)—specifically targeting Deep (`-3.51 pp`) and Light (`-2.83 pp`) F1 deficits—while maintaining or improving zero-shot transfer and clinical agreement on PhysioNet SleepAccel (`EXP-027`/`EXP-028`) and keeping multi-night personalization (`FT-2` / `GATE_tau`) safe on stage-starved subjects (`Bidslab34`).

All candidates must consume the exact locked `prep_57738d534b` feature cache (`x_ihr`: `(B, 1200, 30)`, `x_accel`: `(B, 1200, 30)`, `x_meta`: `(B, 1200, 5)` circadian-neutralized `[freq, 0, 0, hr_m, hr_s]`), locked 5-fold BIDSleep splits (`EXP014_MANIFESTS/fold_1..5.json`), and `EXP-027` SleepAccel cache, running full-night offline buffering on a smartwatch SoC or companion smartphone with INT8/FP16 quantization-clean operators.

---

## Analysis

### 1. Six-Lens Goldfish Review Synthesis

Six independent Goldfish reviewers (`egm-design`, `egm-codebase-details`, `egm-zoom-out`, `egm-implementation-readiness`, `egm-structural`, `egm-inventive-catalyst`) stress-tested the three candidate architectures against `exp013_models.py`, `sleep_adaptive_net.py`, `DEC-031`, and `DEC-038` through `DEC-041`. Key cross-cutting findings:

1. **The 30-Second Epoch Bottleneck (`HREncoder` / `ACCEncoder`):** Both `ConventionalCNNBiGRU` and the failed `SleepAdaptiveNet` (`EXP-014`, `66.06%` vs. `66.63%`) pool `30`-sample epochs independently via `AdaptiveAvgPool1d(1)` before any HR–ACC interaction. Any post-pooling fusion cannot recover sub-30-second phase alignment between transient wrist movement bursts and sympathetic heart-rate spikes.
2. **PyTorch `W0-B` Dynamic INT8 Operator Constraint:** Per `DEC-039` (`INT8_EDGE_BENCHMARK_REPORT.md`), static PTQ (Eager/FX) is blocked by residual `+` operations and `pack_padded_sequence`, while `torch.ao.quantization.quantize_dynamic` only quantizes `nn.Linear` and `nn.GRU`/`nn.LSTM`. `nn.Conv1d` (including `1x1` convolutions), `nn.MultiheadAttention.in_proj_weight`, and custom `nn.Parameter` recurrences stay in FP32.
3. **Padding Bleed in Sequence Modules (`seq_lens < 1200`):** `ConventionalCNNBiGRU` only masks padding inside `temporal_gru` via `pack_padded_sequence`. Non-causal dilated convolutions, `4x` temporal pooling, or windowed attention placed before or without a `BiGRU` will leak non-zero `HREncoder(0)`/`ACCEncoder(0)` bias vectors from `[seq_len : T]` into valid tail epochs unless explicitly zero-masked at every stage.
4. **Prior-Shift & `RCCLoss` Interaction on SleepAccel:** `CE + 0.01 * RCCLoss` already applies inverse-frequency FN/FP reweighting (`test_rccloss_audit.py`), which violates the calibrated-posterior assumption of Saerens et al. (2002) EM and interacts adversely with training-time logit adjustment. Decoupling prior calibration as a zero-parameter post-hoc step inside `SlidingWindowPredictor` (plus ablating pure `CE` vs. `CE + 0.01 * RCCLoss`) avoids double-counting.
5. **Contradiction Resolution & Ideal Final Result (IFR):** The ideal architecture resolves three contradictions without multi-head gradient interference: (a) multi-scale ultradian context (`2–16 min`) under `< 250k` parameters using a single-head `2`-scale `BiGRU` pyramid; (b) asymmetric `ACC -> HR` coupling at `1 Hz` before epoch pooling; and (c) explicit separation of physiological emissions from target label priors at inference.

---

### 2. Ranked Candidate Architectures

#### Rank 1 (Primary Recommended Candidate): `Ultradian-Pyramidal-Coupler` (Modified `UltradianSSM-Coupler`)
*Consensus Rank #1 across 5 of 6 Goldfish lenses (`design`, `zoom-out`, `structural`, `inventive-catalyst`, `codebase-details` viability); Rank #2 on raw implementation simplicity.*

* **Core Mathematical Formulation (3 Tightly Coupled, Ablatable Modules):**
  1. **Module 1 — 1 Hz Pre-Pooling Asymmetric `ACC -> HR` Residual FiLM (`~8.3k` params):**
     Decompose `HREncoder` and `ACCEncoder` before `AdaptiveAvgPool1d(1)` to expose `(B*T, C, 7)` (or `(B*T, C, 30)` with `C <= 32` to prevent `30x` MMAC inflation). Compute zero-initialized residual FiLM parameters from the ACC feature map:
     $$\gamma_{\text{ACC}} = 1 + \tanh(W_\gamma h_{\text{ACC}}^{\text{1Hz}}), \quad \beta_{\text{ACC}} = W_\beta h_{\text{ACC}}^{\text{1Hz}}$$
     $$\tilde{h}_{\text{HR}}^{\text{1Hz}} = \gamma_{\text{ACC}} \odot h_{\text{HR}}^{\text{1Hz}} + \beta_{\text{ACC}}$$
     Pool $\tilde{h}_{\text{HR}}^{\text{1Hz}}$ and $h_{\text{ACC}}^{\text{1Hz}}$ to `(B, T, 64)` each, retaining a direct `h_ACC` skip path concatenated with `[h_HR_mod, h_ACC, h_meta]` (`144`-dim) so gross-movement Wake cues are preserved.
  2. **Module 2 — Mask-Aware 2-Scale Pyramidal `BiGRU` Backbone (`~130k` params):**
     * *Fine Branch (`30s–2m`, `L = 1200`):* 1-layer `BiGRU(144, hidden=48)` with `pack_padded_sequence` -> `h_fine` `(B, T, 96)`.
     * *Coarse Branch (`2m–16m`, `L = 300`):* Zero-mask `[seq_len : T]`, apply mask-weighted `4x` average pooling (`sum(x * m) / clamp(sum(m), min=1)`), compute `seq_lens_coarse = (seq_lens + 3) // 4`, and run a 1-layer `BiGRU(96, hidden=48)` on `proj(h_fine_pool)` -> `h_coarse` `(B, ceil(T/4), 96)`. Upsample (`repeat_interleave(4)` sliced to `T`), zero-mask, and fuse via linear projection `Linear(192, 96)` + `LayerNorm(96)`.
     * *(Goldfish Guardrail: Replaces `diagonal-LRU` with standard `nn.GRU` so 100% of recurrent weights quantize under `W0-B` Dynamic INT8 and execute in cuDNN without custom scan kernels.)*
  3. **Module 3 — Decoupled Post-Hoc Prior / Damped-EM Calibration (`0` trainable params):**
     Implemented strictly outside `model.forward()` inside `SlidingWindowPredictor` on the full-night `(N, 4)` stitched posterior (excluding padded `[N : 1200]`): damped Saerens et al. (2002) EM update with Dirichlet floor toward $\pi_{\text{train}}$ ($\pi^{(t+1)} = (1 - \alpha)\pi_{\text{train}} + \alpha \hat{\pi}_{\text{EM}}^{(t+1)}$, $\alpha \in [0.2, 0.5]$) and temperature scaling tuned on OOF validation logits. Retains `GATE_tau` (`DEC-040`) for safe `FT-2`.

* **Exact Parameter & MMAC Budget:**
  * `HREncoder` + `ACCEncoder` + `meta_proj`: `29,568 + 29,568 + 96 = 59,232`
  * Pre-pooling `ACC -> HR` FiLM (`64 -> 128` pointwise): `8,320`
  * Fine 1-layer `BiGRU(144 -> 48)`: `55,872`
  * Coarse bottleneck `Linear(96 -> 96)` (`9,312`) + Coarse 1-layer `BiGRU(96 -> 48)` (`42,048`) + Fusion `Linear(192 -> 96)` + `LayerNorm(96)` (`18,720`): `70,080`
  * `classifier_head` (`96 -> 64 -> 4`): `6,596`
  * **Total:** **`~200,100` parameters** (27.3x smaller than `SLAMSS-IFS`; well below `250k` target; **~67% of weights in `Linear`/`GRU`** for `W0-B` Dynamic INT8). Keeping the `1 Hz` conv stem at `C <= 32` keeps total inference under `~12–18 MMACs/night` (`~50–75 ms` single-thread CPU).
  * *Optional YAGNI Slim Variant (`d_temp=64`, `hidden=32` coarse):* `~148k` params (below `163,940` `DEC-031` reference).

* **Falsifiable Ablation Ladder (6-Arm Matrix back to `ConventionalCNNBiGRU`):**
  * Arm (a): `ConventionalCNNBiGRU-164k` (`163,940` params, locked baseline, `64.79%` BIDSleep).
  * Arm (b): `ConventionalCNNBiGRU-200k` (`223,268` params, `temporal_gru` `hidden=64/dir -> 128`, capacity-matched control).
  * Arm (c): `+ 1 Hz Pre-Pooling ACC -> HR Residual FiLM only` (`172,260` params).
  * Arm (d): `+ 2-Scale Mask-Aware BiGRU Pyramid only` (`191,780` params).
  * Arm (e): `Full Ultradian-Pyramidal-Coupler` (`200,100` params; FiLM + 2-Scale Pyramid).
  * Arm (f): `Official SLAMSS-IFS` (`5,465,768` params, locked server reference, `67.37%` BIDSleep).
  * *(Plus secondary pure-`CE` loss ablation on Arm (e) vs. primary `CE + 0.01 * RCCLoss`, and zero-parameter post-hoc damped-EM / LOSO prior calibration evaluated inside `SlidingWindowPredictor` on SleepAccel and BIDSleep `FT-2` / `GATE_tau`.)*

* **Training Time on 2x RTX 3060 12GB (`cuda:0`, `cuda:1`):**
  * `100%` cuDNN-native operators (`Conv1d` + `GRU`). At `B = 4, L = 1200`, peak VRAM `< 300 MB`. Full 15-run CV (7 runs `cuda:0`, 8 runs `cuda:1`): **~25–45 minutes total**.

---

#### Rank 2 (Strong Runner-Up / Pure Conv-RNN Alternative): `PhysioDecomp-MSNet`
*Ranked #1 on raw implementation readiness; Rank #2 across `design`, `zoom-out`, `structural`, and `codebase-details`.*

* **Core Mathematical Formulation:**
  1. **Epoch Encoders:** Standard `HREncoder` + `ACCEncoder` (`59,136` params) + `meta_proj` (`96` params). *(Goldfish fix: move motion-residual decomposition pre-pooling or drop `r_HR = h_HR - (W_m h_ACC + b_m)` as redundant with downstream `Linear` projections.)*
  2. **Bottlenecked Dilated DS-TCN (`d in {1, 2, 4, 8, 16}`, `C = 96`):** `Linear(144, 96)` bottleneck (`13,920` params) followed by 5 depthwise-separable dilated blocks (`k = 3`, receptive field `~62.5 min`), with **explicit zero-masking of `[seq_len : T]` before every block** to prevent padding bleed. Pointwise projections implemented as `nn.Linear` on `(B, T, C)` so `quantize_dynamic` quantizes them.
  3. **1-Layer `BiGRU(96 -> 64)` + `classifier_head`:** Captures global sequential order (`~62.2k` + `8.6k` params).
  4. **Post-Hoc Prior Calibration:** Inference-only temperature/prior alignment (dropping training-time logit adjustment to avoid double-counting `RCCLoss`).

* **Exact Parameter Budget:**
  * Encoders + `meta_proj` (`59,232`) + Bottleneck (`13,920`) + 5 DS-TCN blocks (`~49,440`) + 1-layer `BiGRU(96 -> 64)` (`62,208`) + `classifier_head` (`8,644`) = **`~193,444` parameters** (`~198k` with `r_HR` if retained).
* **Falsifiable Ablation Ladder:** `R0` (`ConventionalCNNBiGRU`) -> `+ DS-TCN (zero-masked) + 1-layer BiGRU` -> `+ Capacity-matched ~195k ConventionalCNNBiGRU control` -> `+ Post-hoc prior calibration`.
* **Training Time on 2x RTX 3060 12GB:** Peak VRAM `< 250 MB`; 15-run CV **~20–40 minutes total**.
* **Why Ranked #2:** While fast and simple, operating on `30s`-pooled vectors leaves the `1 Hz` motion–HR phase misalignment unsolved, and shifting parameters into depthwise `Conv1d` reduces the fraction of weights quantized by `W0-B` Dynamic INT8.

---

#### Rank 3 (Rejected / Deferred): `ProtoStage-HierNet`
*Ranked #3 across 5 of 6 Goldfish lenses; Rank #2 only on theoretical prior-factorization.*

* **Core Mathematical Formulation:** Low-rank asymmetric bilinear HR–ACC coupler + Conformer-Lite (`DS-Conv` + windowed local-global attention with explicit `nn.Linear` Q/K/V) + temperature-scaled cosine prototype classifier head (`logits_c = s * cos(z, mu_c)`) with support-weighted shrinkage during `FT-2`.
* **Exact Parameter Budget:** Standard Macaron Conformer at `d = 96` costs `~148k`/block (`> 350k` for 2 blocks, breaching `< 300k`). Fits `< 250k` **only** with single-FFN blocks at `d = 64`: encoders (`59.2k`) + bilinear (`6.4k`) + proj (`13k`) + 2–3 blocks (`106k–159k`) + proto head (`4.5k`) = **`~189k–242k` parameters**.
* **Why Rejected:** (1) Non-shifted windowed attention has zero cross-window ultradian receptive field; (2) all-padded trailing windows (`seq_len <= win_start`) trigger `softmax(-inf) = NaN` hazards; (3) attention `QK^T` + `Softmax` + `LayerNorm` degrades static/dynamic INT8 edge deployment; and (4) the shrinkage prototype head duplicates `GATE_tau` (`DEC-040`), which already mitigates `Bidslab34` at zero parameter cost.

---

## Goals vs. Non-Goals

* **In-Scope Goals:**
  * Close the `-2.58 pp` Macro-F1 gap to `Official SLAMSS-IFS` (`67.37%`) in-cohort on BIDSleep 5-fold CV (`N = 47`), specifically targeting Deep (`-3.51 pp`) and Light (`-2.83 pp`) F1.
  * Maintain or improve zero-shot Macro-F1 on PhysioNet SleepAccel (`n = 25 qc_clean` / `N = 31 all`) while reducing Deep% (`+10.65%`), REM% (`+10.13%`), and WASO (`-13.57 min`) clinical bias.
  * Stay strictly `<= 250k` parameters (`< 300k` hard ceiling) with `W0-B` Dynamic INT8 parity and `~25–45 min` 15-run CV runtime on `2x RTX 3060 12GB`.
  * Keep `ConventionalCNNBiGRU` (`163,940` params) and `Official SLAMSS-IFS` (`5,465,768` params) untouched as fixed comparison baselines.
* **Out-of-Scope Non-Goals:**
  * Re-extracting raw `Bidslab*` or SleepAccel features (must use `prep_57738d534b` and `EXP-027` caches).
  * Multi-task auxiliary heads/losses (no GRL, boundary loss, transition head, or uncertainty head per `EXP-014` lessons).
  * Bare-metal `512 KB` MCU 1-epoch causal streaming (deployment regime is locked to full-night `L = 1200` offline buffering on smartwatch SoC / companion smartphone per `DEC-039`).
  * Writing or executing training code during this brainstorm phase (strict NO-CODE).

---

## Alternatives Weighed

* **`SleepAdaptiveNet`-style 5-Head Multi-Task Architecture (`EXP-014`):** Rejected by author and Goldfish review; already tested in `EXP-014` (`66.06%` vs. `66.63%`) and caused gradient interference across competing auxiliary losses without fixing the `30s` pooling or single-scale `BiGRU` bottlenecks.
* **Custom `Diagonal-LRU` / Complex SSM Coarse Branch (`UltradianSSM-Coupler` original variant):** Rejected by `egm-codebase-details`, `egm-design`, `egm-zoom-out`, `egm-implementation-readiness`, and `egm-structural`; lacks native PyTorch `pack_padded_sequence` and `W0-B` Dynamic INT8 quantization, and pure-PyTorch sequential scans would slow 15-run CV by `10x–30x`. Replaced with a standard 1-layer `BiGRU` on the `4x`-downsampled stream.
* **Post-Pooling Linear Motion-Residual (`PhysioDecomp-MSNet` `r_HR`):** Rejected by `egm-design` and `egm-zoom-out`; subtracting a linear projection after `AdaptiveAvgPool1d(1)` cannot undo intra-epoch motion–HR lag and is algebraically redundant with downstream linear/GRU layers.
* **`ProtoStage-HierNet` (Conformer-Lite + Shrinkage Prototype Head):** Rejected for `EXP-029` primary track due to `d=96` parameter overflow (`> 350k`), `softmax(-inf) = NaN` window-masking hazard, INT8 attention degradation, and duplication of `GATE_tau` (`DEC-040`).
* **Training-Time Logit Adjustment with `RCCLoss`:** Rejected by `egm-design` and `egm-codebase-details`; double-counts `RCCLoss` inverse-frequency reweighting and risks amplifying SleepAccel Deep%/REM% overestimation.

---

## Key Decisions & Staging for Next Steps

1. **Primary `EXP-029 / v5` Architecture Selection (Locked):** Advance **`Ultradian-Pyramidal-Coupler`** (`200,100` params) as the primary `v5` architecture, using mask-weighted `4x` temporal pooling + 1-layer coarse `BiGRU` fused with the fine 1x `BiGRU` to preserve native `pack_padded_sequence` and PyTorch `quantize_dynamic` INT8 compatibility (`W0-B`).
2. **Capacity-Matched Control & 6-Arm Ablation Matrix (Locked under `DEC-031`):** Include **`ConventionalCNNBiGRU-200k`** (`temporal_gru` `hidden=64/dir -> 128`, `223,268` params) alongside `ConventionalCNNBiGRU-164k` (`163,940` params), FiLM-only (`172,260` params), Pyramid-only (`191,780` params), Full `Ultradian-Pyramidal-Coupler` (`200,100` params), and `Official SLAMSS-IFS` (`5,465,768` params) to eliminate any parameter-count confound.
3. **Training Loss & Decoupled Prior Calibration (Locked):** Keep `CE + 0.01 * RCCLoss` as the primary training objective for strict parity with `ConventionalCNNBiGRU` and `SLAMSS-IFS`, paired with a secondary pure-`CE` ablation on `Ultradian-Pyramidal-Coupler`. Keep damped-EM / LOSO label-prior calibration strictly decoupled as a zero-parameter post-hoc inference toggle inside [`SlidingWindowPredictor`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp018_inference.py#L20-L182).
4. **Handoff to `/egm-draft`:** Formalize the complete `EXP-029` specification in `docs/designs/exp029_ultradian_pyramidal_coupler.md` under the strict NO-CODE rule.
