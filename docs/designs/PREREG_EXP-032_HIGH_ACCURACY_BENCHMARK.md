# PREREG_EXP-032: High-Accuracy Benchmark (`<= 1.5M` Parameters) vs. Official SLAMSS-IFS

* **Experiment ID:** `EXP-032`
* **Author:** `jbjoyk`
* **Date:** `2026-11-26`
* **Status:** **Resolved Pre-Registered Design (Ready for Pre-Lock Git Commit)**
* **Governance Standing:** Pre-registered exception to the `Lightweight-First` preference in [`DEC-031`](../../DECISIONS.md) (which established `163,940` parameters as the baseline reference size rather than a hard cap). `EXP-032` is scoped as a **server / companion-phone high-accuracy benchmark** (unconstrained up to `1.5M` parameters, no `< 400 KiB` INT8 MCU deployment gate, non-causal full-night sequence inference at `L=1200`).
* **Prior-Access Disclosure:** `EXP-032` is an architecture-targeted follow-up conditioned on prior `EXP-021` ([`DEC-038`](../../DECISIONS.md)), `EXP-029` ([`DEC-042`](../../DECISIONS.md)), `EXP-030` ([`DEC-043`](../../DECISIONS.md)), and `EXP-031` / `EXP-031-A1` ([`DEC-044`](../../DECISIONS.md), commit `16e70ef`) findings on BIDSleep (`N=47`). To prevent overfitting to `outer_test`, all hyperparameter choices are locked **strictly on inner validation folds (`inner_val`, `N=5` subjects/fold, `N=25` pooled across 5 folds)** before a single locked `15`-run (`5 folds × 3 seeds`) outer evaluation per arm, and external SleepAccel (`n=25` `qc_clean`, `N=31` `all`) remains **100% untouched** until all `45` outer checkpoints are frozen.

---

## 1. Scientific Grounding & Hypothesis

### 1.1 Empirical Motivation
1. **Plain CNN-BiGRU Capacity Saturation (`EXP-030`, `EXP-031`, `EXP-031-A1`):**
   - Retraining `Model A` (`163,940` params) under the `EXP-031` training recipe (`W100`) yields **`66.68% ± 7.11%`** BIDSleep OOF Macro-F1 (`67.97%` Accuracy), confirming a `+1.89 pp` recipe effect over frozen `Model A` (`64.79%`, $p = 2.12 \times 10^{-6}$).
   - However, scaling plain `ConventionalCNNBiGRU` width from `98,940` (`p1_100k`: `66.67%`) to `190,940` (`P190`, the `EXP-031` `< 400 KiB` INT8-selected width: `66.60%`) and `993,380` (`W250`: `66.25%`) **saturates** around `66.68%`, leaving an unresolved **`0.69 pp` gap** behind `Official SLAMSS-IFS` (`5,465,768` params: **`67.37%`** Macro-F1, **`68.75%`** Accuracy on BIDSleep `N=47`).
   - Mechanistically, `Official SLAMSS-IFS` achieves this advantage via its wide-kernel multi-scale `cnn3` block (`k = 11, 15, 21` over inter-epoch features, capturing 5.5–10.5 min sleep-cycle context) and its autoregressive `s2s` decoder modeling stage-transition structure (concentrated in Light and Deep sleep).
2. **External Generalization Gap (`EXP-027`, [`DEC-041`](../../DECISIONS.md)):**
   - `Official SLAMSS-IFS` drops **`-6.29 pp`** on zero-shot external SleepAccel (`n=25` `qc_clean`: `67.37% -> 61.08%` single-model, `62.79%` 15-model `prob_ensemble`), whereas regularized `P190` (`61.52%` / `63.42%`) and `Model A` (`62.94%` ensemble) preserve external performance.
   - Therefore, closing the `0.69 pp` BIDSleep gap requires **multi-scale wide-kernel temporal context + explicit stage-transition modeling** paired with **regularization** (dropout, weight decay, residual connections, or teacher distillation) so external SleepAccel performance is preserved.

### 1.2 Anchored Working Hypothesis & Comparators
- **Primary Superiority Comparator:** `Official SLAMSS-IFS` (`5,465,768` params, `67.37%` Macro-F1, `68.75%` Accuracy on BIDSleep `N=47`; `61.08%` single / `62.79%` ensemble on SleepAccel `n=25` `qc_clean`).
- **Recipe-Matched Capacity Anchor:** `W100` (`163,940` params, trained under the `EXP-031` recipe, `66.68%` Macro-F1) serves as the lightweight capacity anchor alongside `Official SLAMSS-IFS` (noting in descriptive `M1 - W100` contrasts that `EXP-032` uses `AdamW` + `CosineAnnealingLR`, whereas `W100` used constant `Adam(lr=1e-4)`).
- **Hypothesis:** Adding multi-scale large-kernel residual TCN context (`k = 11, 15, 21`, dilations `1, 2`) before a 2-layer BiGRU (`M1`: `1,402,468` params), exact 4-state linear-chain CRF transition modeling (`M2`: `1,402,492` params), and fold-matched `EXP-021` `Official SLAMSS-IFS` teacher distillation (`M3`: `1,402,492` params, explicitly labeled **teacher-distilled**) will **outperform `Official SLAMSS-IFS` on BIDSleep (`N=47`)** while maintaining **non-inferiority (`> -2.50 pp`) on zero-shot external SleepAccel (`n=25` `qc_clean`)**, at `3.90x` fewer parameters (`<= 1.5M` vs. `5.47M`).

---

## 2. Candidate Architectures, Exact Losses & Compute Profile

All three candidate architectures share the exact **locked input contract** (`prep_57738d534b` feature cache, `L=1200` sequence windows, `neutralize_circadian=True`):
- **Inputs:** `x_ihr: (B, T, 30)`, `x_acc: (B, T, 30)`, `meta: (B, T, 5)` (`[freq, clock, time_feat=0.0, hr_mean, hr_std]`), `mask: (B, T)`, `seq_lens: (B,)`.
- **Targets:** `y: (B, T)` with 4-class stage mapping `{0: Wake, 1: Light (N1+N2), 2: Deep (N3), 3: REM}` and `-100` (`ignore_index`) for unknown/artifact epochs and trailing padding.

### 2.1 Shared Backbone (`M1` Base, `1,402,468` Params, `4,053.08` MMACs at `L=1200`)
1. **Intra-Epoch Modality Encoders:** Dual `ScaledModalityEncoder(in_features=1, d_enc=128)` streams (`116,480` params each) from `experiments/EXP-030/exp030_models.py`, plus `meta_proj`: `Linear(5, 32) -> LayerNorm(32) -> GELU` (`256` params).
2. **Modality Fusion:** `Linear(288, 224) -> LayerNorm(224) -> GELU -> Dropout(p)` (`65,184` params).
3. **Multi-Scale Large-Kernel Residual TCN (`2` Blocks, `501,824` Params):**
   - Tensor transposed `(B, T, 224) -> (B, 224, T)` before TCN and `(B, 224, T) -> (B, T, 224)` after TCN.
   - Each `MultiScaleTCNBlock(d_model=224, d_branch=64, kernels=(11, 15, 21))` applies:
     - Bottleneck `Conv1d(224, 64, k=1) -> GroupNorm(8, 64) -> GELU`
     - Three parallel dilated convolutions `Conv1d(64, 64, k)` for `k in (11, 15, 21)` with `padding=((k-1)*d)//2`
     - Concatenation (`192` ch) -> `Conv1d(192, 224, k=1) -> GroupNorm(8, 224) -> GELU -> Dropout(p)` + residual skip.
   - **Block 1 (`dilation=1`):** Receptive field `11, 15, 21` epochs (`5.5–10.5 min`, matching `SLAMSS-IFS` `cnn3`).
   - **Block 2 (`dilation=2`):** Effective receptive field `21, 29, 41` epochs (`10.5–20.5 min`, ultradian horizon).
4. **Inter-Epoch BiGRU + Head (`568,320 + 33,924` Params):**
   - `pack_padded_sequence(seq_lens)` around `nn.GRU(224, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=p)` (`256`-dim output) -> `pad_packed_sequence(..., total_length=T)`.
   - `LayerNorm(256) -> Linear(256, 128) -> GELU -> Dropout(p) -> Linear(128, 4)` emission logits `z_S`.

### 2.2 Locked Loss Formulations & Per-Fold Class Weighting
All losses are computed strictly in **FP32 outside `torch.amp.autocast`** over valid non-`-100` epochs (`valid = (y != -100) & mask`).
- **Per-Fold Class Weights (`w_c`):** Computed strictly from each fold's `inner_train` valid epoch counts $N_c$ ($c \in \{0, 1, 2, 3\}$):
  $$\tilde{w}_c = \left(\frac{N_{\text{valid}}}{4 N_c}\right)^{0.5}, \qquad w_c = \frac{\tilde{w}_c}{\frac{1}{4}\sum_{j=0}^3 \tilde{w}_j}$$
  so $\frac{1}{4}\sum_{c=0}^3 w_c = 1.0$.
- **`M1` Loss (Pure Class-Weighted Cross-Entropy):**
  $$\mathcal{L}_{M1} = \mathcal{L}_{\text{CE\_weighted}}(z_S, y, w_c) = \frac{\sum_{b,t \in \text{valid}} w_{y_{b,t}} \, \text{CE}(z_{S,b,t}, y_{b,t})}{\sum_{b,t \in \text{valid}} w_{y_{b,t}}}$$
- **`M2` Loss (Per-Epoch Mean-Normalized CRF + Weighted CE Anchor):**
  $$\mathcal{L}_{M2} = \mathcal{L}_{\text{CRF\_mean}}(z_S, y, \text{mask}, w_c) + 0.1 \cdot \mathcal{L}_{\text{CE\_weighted}}(z_S, y, w_c)$$
  where $\mathcal{L}_{\text{CRF\_mean}}$ applies per-epoch weight $w_{y_t}$ to the valid-epoch emission contributions in the CRF NLL (marginalizing interior `-100` epochs as specified in §3.1) and divides by $\sum_{b,t \in \text{valid}} w_{y_{b,t}}$, keeping $\mathcal{L}_{\text{CRF\_mean}}$ on an $O(1)$ per-epoch scale. (Prior `EXP-013`/`EXP-014` transition matrices were unregularized `4×4` biases on `BiGRU`; `M2` tests whether exact global CRF partition training + Viterbi decoding improves `M1`.)
- **`M3` Loss (Teacher-Distilled `M2` CRF):**
  $$\mathcal{L}_{M3} = (1 - \alpha_{\text{kd}}) \, \mathcal{L}_{M2} + \alpha_{\text{kd}} \, \tau^2 \, \mathcal{L}_{\text{KL\_masked}}(z_S, z_T, y, \text{mask}, \tau)$$
  where $\mathcal{L}_{\text{KL\_masked}}$ computes `F.kl_div(F.log_softmax(z_S / tau, dim=-1), F.softmax(z_T / tau, dim=-1), reduction='none').sum(dim=-1)` averaged strictly over valid epochs (`(y != -100) & mask`), ensuring $\mathcal{L}_{M2}$ and $\tau^2 \mathcal{L}_{\text{KL\_masked}}$ have matched $O(1)$ per-epoch scales.

### 2.3 Comparison Matrix (CPU-Verified via `eval_exp030_downstream.compute_model_mmacs`)

| Arm | Architecture | Exact Params | vs. `W100` (`163,940`) | vs. `SLAMSS-IFS` (`5,465,768`) | `MMACs` (`L=1200`) | Inductive Bias & Locked Loss |
|---|---|---|---|---|---|---|
| **`W100`** *(Anchor)* | `ScaledConventionalCNNBiGRU` (`d_enc=64, d_temp=96`) | `163,940` | `1.00x` | `33.34x` smaller | `794.05` | Local intra-epoch CNN + 2-layer BiGRU (`66.68%` Macro-F1) |
| **`M1`** | **Multi-Scale Residual TCN + BiGRU** | **`1,402,468`** | `8.55x` | `3.90x` smaller | **`4,053.08`** | Wide-kernel `(11, 15, 21)` dilated residual TCN + 2-layer BiGRU; $\mathcal{L}_{M1}$ |
| **`M2`** | **Multi-Scale TCN-BiGRU + Exact 4-State CRF** | **`1,402,492`** | `8.55x` | `3.90x` smaller | **`4,053.08`** *(+DP)* | `M1` + `LinearChainCRF4` (`24` params: `4` start, `4` end, `4×4` transitions); $\mathcal{L}_{M2}$ |
| **`M3`** *(Teacher-Distilled)* | **Fold-Matched Teacher Distillation + `M2` CRF** | **`1,402,492`** | `8.55x` | `3.90x` smaller | **`4,053.08`** *(+DP)* | `M2` inference architecture + fold-matched `EXP-021` `SLAMSS-IFS` soft logits $z_T$ on `inner_train`; $\mathcal{L}_{M3}$ |
| **`SLAMSS-IFS`** *(Comparator)* | Official `BIDSLabUMass/SLAMSS-IFS` | `5,465,768` | `33.34x` | `1.00x` | `40,773.20` | Intra-epoch CNN+GRU + `cnn_ACT` + `cnn3 (k=11,15,21)` + `Seq2Seq` GRU (`67.37%` Macro-F1) |

---

## 3. Technical Specifications & Anti-Leakage Contracts

1. **Interior `-100` (`ignore_index`) Handling in `LinearChainCRF4`:**
   - In the CRF log-partition (`forward`) and numerator scoring, interior epochs with `y_t == -100` (inside `mask_t == 1` valid night bounds) are **marginalized** by setting emission potentials to `0.0` across all 4 states at those positions (preserving sequence transition continuity without `NaN`/`Inf` gradients), while trailing padding (`mask_t == 0`) terminates cleanly via `end_transitions`.
2. **Stitched Full-Night Decoding & `prob_ensemble` Contract (`M2`/`M3`):**
   - For nights `T > 1200`, `CRFLogitStitchingPredictor` stitches **pre-CRF emission logits** `z_S` across overlapping windows (`stride=600`, analogous to `LogitStitchingSlidingWindowPredictor` in `EXP-030`/`EXP-031`; note `EXP-021` stitched probabilities).
   - `LinearChainCRF4.forward_backward_marginals(z_stitched)` runs once on the full-night stitched sequence to produce calibrated per-epoch posterior probabilities $P_{\text{crf}}(y_t \mid x_{1:T})$, and `LinearChainCRF4.viterbi_decode(z_stitched)` produces single-checkpoint hard hypnograms.
   - **SleepAccel (`n=25`) Evaluation:** `mean_single_model` averages the Macro-F1 of the 15 single-checkpoint hard predictions (`viterbi_decode` for `M2`/`M3`; `argmax` for `M1`), while `prob_ensemble` averages the 15 per-checkpoint posterior probability matrices ($P_{\text{crf}}$ for `M2`/`M3`; $\text{softmax}(z_S)$ for `M1`) before taking `argmax`.
   - **External Calibration Lock:** Per [`DEC-042`](../../DECISIONS.md) and [`DEC-043`](../../DECISIONS.md), `em_off` (no `DampedEMCalibrator` prior-shift EM post-processing) is locked as the sole SleepAccel evaluation mode.
3. **CRF `L=1200` Wall-Clock Optimization:**
   - Preflight MUST benchmark 1 training epoch on RTX 3060 using `torch.jit.script` / `torch.compile` (or parallel associative scan over `ceil(log2(1200)) = 11` steps) to confirm `<= 5 min`/run.
4. **Deterministic Teacher Pre-Caching & Zero-Leakage Contract (`M3`):**
   - Before Stage A/B `M3` training, a deterministic pre-cache script extracts `(x_ihr, x_acc, meta, y, mask, z_T)` on fixed `L=1200` windows (`stride=600`, matching `EXP-021`/`EXP-031` windowing) strictly for `inner_train` subjects of fold $f$ using frozen `EXP-021` checkpoint `(f, s)` (`"model_state_dict"`, `trg_zeros`, `teacher_forcing_ratio=0.0`, `neutralize_circadian=True`), saved to `experiments/EXP-032/cache/teacher_logits_f{f}_s{s}.pt` (excluded from Git via `.gitignore`).
   - Verified against `review_package/fold_manifests.csv` (SHA-256 `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`): `EXP-021` trained strictly on `inner_train`, used `inner_val` only for early stopping, and never touched `outer_test`. `M3` never computes distillation loss on `inner_val` or `outer_test`, and never uses cross-fold or ensemble teachers.
   - **Claim Hygiene:** `M3` is explicitly labeled **teacher-distilled** in all tables, figures, and manuscript text.

---

## 4. Two-Stage Protocol, Training Specs & Output Schemas

### Stage A: Inner-Validation Hyperparameter Tuning (Strictly `outer_test`-Blind)
- **Optimizer & Schedule (Locked):** `batch_size=4` (`val_batch_size=16`), `AdamW(lr=1e-4, weight_decay=wd)`, `CosineAnnealingLR(T_max=100, eta_min=1e-6)` stepped per epoch, `grad_clip=1.0`, `max_epochs=100, min_epochs=40, patience=25` (best-checkpoint tracking and `epochs_no_improve` start at `epoch >= 40`). Forward/backward uses `float16` AMP on TCN+BiGRU with **FP32 forced** for `LinearChainCRF4`, `L_CE_weighted`, and `L_KL_masked`.
- **Selection Metric (Locked):** Pooled **participant-level Macro-F1** across all `N=25` `inner_val` subjects (`5 folds × 1 seed [seed=42] = 5 runs/config`), computed by averaging each subject's nights first and then taking the unweighted mean across the 25 subjects.
- **Grid (`<= 4` configs/arm, `<= 60` screening runs):**
  - `M1` (4 configs): `(dropout, weight_decay)` in `{(0.2, 1e-4), (0.2, 1e-2), (0.3, 1e-4), (0.3, 1e-2)}`.
  - `M2` (4 configs): `(dropout, weight_decay)` in `{(0.2, 1e-4), (0.2, 1e-2), (0.3, 1e-4), (0.3, 1e-2)}`.
  - `M3` (4 configs): Inherits `M2`'s winning `(dropout, weight_decay)` and tunes `(tau, alpha_kd)` in `{(2.0, 0.25), (2.0, 0.50), (4.0, 0.25), (4.0, 0.50)}`.
- **Lock:** Winning per-arm configs are serialized to `experiments/EXP-032/locked_configs.json` before Stage B.

### Stage B: Locked Outer Evaluation (`45` Runs = `3 Arms × 5 Folds × 3 Seeds`)
- **Seeds:** `[42, 123, 2024]` across `folds 1..5`. Checkpoints selected strictly by `inner_val` participant Macro-F1 (`epoch >= 40`), evaluated **once** on `outer_test` (`N=47` pooled OOF participant means).
- **External & Hardware Evaluation:** After all 45 checkpoints are frozen and SHA-256 hashed (`frozen_checkpoint_manifest.json`), evaluate zero-shot on SleepAccel (`n=25` `qc_clean` primary, `N=31` `all` secondary, `em_off` only) and profile `1T`/`4T` FP32 CPU latency (interleaved round-robin across arms on `AMD Ryzen 7 7700` to eliminate thermal drift) alongside `W100` and `SLAMSS-IFS`.
- **Output Schemas (`experiments/EXP-032/`):**
  - `locked_configs.json`, `stage_a_screening.csv`
  - `run_results/{arm}_fold{f}_seed{s}.json`, `checkpoints/{arm}_fold{f}_seed{s}.pt`, `frozen_checkpoint_manifest.json`
  - `{arm}_oof_subject_metrics.csv`, `{arm}_sleepaccel_metrics.csv`
  - `exp032_summary.json`, `exp032_sleepaccel_summary.json`, `exp032_cpu_latency.json`

---

## 5. Statistical Analysis & Exhaustive Verdict Rule

All paired comparisons use **participant-level means** (`N=47` BIDSleep; `n=25` `qc_clean` SleepAccel) against `Official SLAMSS-IFS` (`EXP-021`) with **10,000-resample participant bootstrap 95% CIs (`seed=42`)** and **paired Wilcoxon signed-rank tests**.

### 5.1 Multiplicity Control
- **Primary Family (3 comparisons, Holm-Bonferroni $\alpha = 0.05$):** `{M1, M2, M3}` vs. `Official SLAMSS-IFS` (`67.37%`) on BIDSleep (`N=47`) **Macro-F1** (two-sided Wilcoxon $p_{\text{Holm}}$).
- **Secondary Family ($3 \times K_{\text{qualifying}}$ comparisons, Holm-Bonferroni $\alpha = 0.05$):** For the $K_{\text{qualifying}}$ arms that pass the primary BIDSleep Macro-F1 gate ($p_{\text{Holm}} < 0.05$ and $\bar{\Delta} > 0$), test 3 gates per arm (with Holm-Bonferroni applied across all $3 \times K_{\text{qualifying}}$ tests whenever $K_{\text{qualifying}} \ge 1$):
  1. BIDSleep (`N=47`) **Accuracy superiority** vs. `68.75%` ($\bar{\Delta} > 0$, two-sided $p_{\text{Holm}} < 0.05$).
  2. SleepAccel (`n=25` `qc_clean`, `mean_single_model`) **Macro-F1 non-inferiority** vs. `61.08%` at margin $\delta_{\text{NI}} = -2.50\text{ pp}$ (matched to `n=25`'s `80%` MDE of `2.36–2.59 pp` under `SD = 4.04–4.43 pp` while ruling out `-6.29 pp` collapse: 95% bootstrap CI lower bound $> -2.50\text{ pp}$ AND one-sided Wilcoxon $p_{\text{Holm,NI}} < 0.05$ on $\Delta + 2.50\text{ pp} > 0$).
  3. SleepAccel (`n=25` `qc_clean`, `prob_ensemble`) **Macro-F1 non-inferiority** vs. `62.79%` at margin $\delta_{\text{NI}} = -2.50\text{ pp}$ (95% bootstrap CI lower bound $> -2.50\text{ pp}$ AND one-sided Wilcoxon $p_{\text{Holm,NI}} < 0.05$).
- **Descriptive Contrasts:** `M1 - W100` (capacity + TCN + `AdamW`/cosine vs. `W100`), `M2 - M1` (CRF gain), `M3 - M2` (distillation gain).

### 5.2 Exhaustive, Mutually Exclusive Verdict Rule
- **Per-Arm Verdict (`A in {M1, M2, M3}`):**
  - **`ARM_SUPPORTED`:** Primary BIDSleep Macro-F1 beats `SLAMSS-IFS` ($\bar{\Delta} > 0$, $p_{\text{Holm}} < 0.05$) **AND** all 3 secondary gates pass under Holm-Bonferroni ($3 \times K_{\text{qualifying}}$).
  - **`ARM_INCONCLUSIVE`:** Primary BIDSleep Macro-F1 beats `SLAMSS-IFS` ($\bar{\Delta} > 0$, $p_{\text{Holm}} < 0.05$) while $\ge 1$ secondary gate fails, **OR** ($p_{\text{Holm}} \ge 0.05$ while unadjusted Wilcoxon $p < 0.05$ and $\bar{\Delta} \ge +0.50\text{ pp}$).
  - **`ARM_NOT_SUPPORTED`:** Otherwise (fails both $p_{\text{Holm}} < 0.05$ with $\bar{\Delta} > 0$ and the unadjusted $p < 0.05, \bar{\Delta} \ge +0.50\text{ pp}$ trend criterion).
- **Overall Experiment Verdict:**
  - **`SUPPORTED`:** $\ge 1$ arm achieves `ARM_SUPPORTED` (explicitly noting whether standalone `M1`/`M2` or teacher-distilled `M3` succeeded).
  - **`INCONCLUSIVE`:** $0$ arms achieve `ARM_SUPPORTED` and $\ge 1$ arm achieves `ARM_INCONCLUSIVE`.
  - **`NOT SUPPORTED`:** All 3 arms evaluate to `ARM_NOT_SUPPORTED`.

---

## 6. Preflight Gate Checklist (Before Any GPU Run)
1. Assert exact parameter counts (`M1 = 1,402,468`, `M2 = 1,402,492`, `M3 = 1,402,492`, all `<= 1,500,000`) and `L=1200` MMACs (`4,053.08`).
2. Unit-test `LinearChainCRF4` on synthetic sequences with interior `-100` spans, trailing padding, class weights $w_c$, and extreme logits: assert `O(1)` per-epoch normalized NLL, finite FP32 gradients, exact forward/Viterbi consistency, and zero `NaN`/`Inf`.
3. Verify `review_package/fold_manifests.csv` SHA-256 (`36d84020...`), `prep_57738d534b` cache integrity, and zero subject overlap across `inner_train`/`inner_val`/`outer_test`.
4. Verify `EXP-021` teacher pre-caching (`teacher_logits_f{f}_s{s}.pt`) covers strictly `inner_train` windows of fold $f$ with zero `inner_val`/`outer_test` leakage.
5. Benchmark 1 `M2` CRF epoch on RTX 3060 to confirm wall-clock feasibility (`<= 5 min`/run).
