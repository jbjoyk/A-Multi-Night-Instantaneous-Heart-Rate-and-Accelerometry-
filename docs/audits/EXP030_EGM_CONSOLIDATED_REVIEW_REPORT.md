# EXP-030 EGM Consolidated Review Report

*   **Target Document:** [`docs/designs/exp030_cnn_bigru_capacity_pareto.md`](../designs/exp030_cnn_bigru_capacity_pareto.md)
*   **Review Mode:** Uninterrupted Full Flow (Stages 5, 6, 7, 8, 9)
*   **Date:** 2026-11-24
*   **Overall Verdict:** **`0` 🔴 Blockers | `6` 🟡 Majors | `6` 🟢 Nits** — Ready for remediation edits before `EXP-030` implementation.

---

## 1. Consolidated Action Items (Deduplicated & Prioritized)

### 🔴 Blockers (`0`)
None.

### 🟡 Majors (`6`)

1. **M1 — Define `ScaledConventionalCNNBiGRU` Locally in `exp030_models.py` (Preserve Historical Code)**
   * **Issue:** `exp013_models.ConventionalCNNBiGRU` hardcodes `head_hidden=64`, and `sleep_adaptive_net.HREncoder`/`ACCEncoder` hardcode `32` and `64` channels in their first two conv blocks. Directly instantiating `ConventionalCNNBiGRU` for `P1, P3–P6` will fail or silently keep `32`/`64` encoder widths.
   * **Action:** Lock `experiments/EXP-030/exp030_models.py` to implement `ScaledConventionalCNNBiGRU` + `ScaledModalityEncoder(in_features=1, d_enc=d_enc, dropout=0.2)` (`c1 = d_enc // 2, c2 = d_enc`), keeping `exp013_models.py` and `sleep_adaptive_net.py` untouched. Specify `build_exp030_model(tag: str, loss_type: str = "rccloss") -> nn.Module` and require a unit test verifying exact `state_dict` / parameter parity (`163,940` params) at `p2_164k`.

2. **M2 — Align `P2` Inference Stitcher & Add `P2` Fallback + Self-Comparison Guard**
   * **Issue:** `P1, P3–P6` use `LogitStitchingSlidingWindowPredictor`, while `P2` (`64.79%`) comes from `final_model_oof_subject_metrics.csv` (`SlidingWindowPredictor`). Mixing stitchers inside the `0.20 pp` elbow rule confounds `M(P2)` and `Delta_max`. Also, if the degenerate guard selects `P2`, `eval_exp030_downstream.py` has no fallback path to load `DEC-037` checkpoints (`frozen_checkpoint_manifest.json`) and would run a zero-diff self-comparison against `Model A-164k`.
   * **Action:** Run `LogitStitchingSlidingWindowPredictor` inference-only on the 15 frozen `DEC-037` checkpoints (~30s, 0 training runs) to establish an apples-to-apples `M(P2)` baseline; add an explicit `star_tag == "p2_164k"` checkpoint-loader fallback via `experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json` and a self-comparison guard in Pillars 1 and 3.

3. **M3 — Reuse `EXP-029` / `EXP-020` Helpers Instead of Duplicating Drivers or Coupling to `FinalSleepStagingSystem`**
   * **Issue:** Creating `run_exp030_cv.py` and `eval_exp030_downstream.py` from scratch risks duplicating ~1,000 lines of manifest verification, data loading, `RCCLoss`, `W0-B` quantization, Bland-Altman stats, and `holm_bonferroni`. Also, `run_exp020_regularized_gated_adaptation.py` couples to `FinalSleepStagingSystem` (which hardcodes `P2` paths) and reads per-fold configs from `exp019_stage_d_selected_configs.json`.
   * **Action:** Explicitly list reused imports (`verify_manifests_and_splits`, `load_all_records_by_subject`, `compute_stage_loss_fp32`, `holm_bonferroni`, `compute_int8_profile`, `participant_bootstrap_ci`, `load_preprocessed_sleepaccel`, `extract_8_metrics`, `compute_bland_altman_stats`, `prepare_calibration_batch`). Implement `FT-2` / nested `GATE_tau` with a lightweight `ScaledConventionalCNNBiGRU` state-dict clone helper.

4. **M4 — Pin `EXP-020` `FT-2` Hyperparameters, Predictor, and `W0-B` Quantization Backend**
   * **Issue:** `FT-2` omits `weight_decay=0.0`, `grad_clip=1.0`, `context_len=1200`, `evaluate_records_macro_f1`, and whether `FT-2`/`GATE_tau` uses `LogitStitchingSlidingWindowPredictor`. `W0-B` omits the quantization backend (`quantize_dynamic(..., {nn.Linear, nn.GRU}, dtype=torch.qint8)` on CPU) and target input shape `(1, 1200, 30)`.
   * **Action:** Pin all `FT-2` hyperparameters, `LogitStitchingSlidingWindowPredictor` usage across both `inner_val_subjects` and `outer_test_subjects`, and the exact CPU Dynamic INT8 profiling contract.

5. **M5 — Specify CLI Interfaces, `elbow_selection.json` Schema, Unit Normalization, and Verification Commands**
   * **Issue:** Multi-GPU dispatch flags (`--grid-points`, `--device`, `--folds`, `--seeds`), checkpoint/CSV naming (`<tag>_L1200_fold{fold}_seed{seed}.pt`, `<tag>_subject_metrics.csv`), `* 100.0` aggregation of `P2`'s 141-row `[0, 1]` CSV, `elbow_selection.json` keys, and exact Verification Plan commands are omitted.
   * **Action:** Add exact CLI flags, file templates, `elbow_selection.json` schema (`star_tag`, `star_params`, `m_max`, `delta_max`, `threshold_T`, `guard_triggered`, `grid_summary`), `* 100.0` normalization note, and exact shell commands to `Detailed Implementation Plan`.

6. **M6 — Add Concise Inline Clarifications for Study-Specific Terms (Without Glossary Bloat)**
   * **Issue:** Comprehension review flagged undefined terms (`FT-2`, `GATE_tau`, `ZS`, `min_stage_frac`, `N = 46` vs. `N = 47` exclusion, 14-key `forward()` dict, `W0-B` payload formula, `SLAMSS-IFS`, `RCCLoss`, `EGM`). Per Stage 8 Zoom Out, avoid turning the design doc into a full glossary, but clarify study-specific mechanics inline.
   * **Action:** Add 1-line inline definitions for `ZS` (zero-shot), `FT-2` (2-night supervised fine-tuning), `GATE_tau` (min-stage-fraction abstention gate), `min_stage_frac` (minimum per-stage label fraction across `cal_recs`), `N = 46` (`>= 3` usable nights), `W0-B` payload formula, and link `exp013_models.py#L168-L183` for the 14-key `forward()` contract.

### 🟢 Nits (`6`)

1. **N1:** Include `EXP-029` `cnn_bigru_200k` (`223,268` params, `66.61%` Macro-F1, `368.04 KiB` INT8) as a zero-cost off-manifold historical reference point on the Pareto curve/narrative.
2. **N2:** Clarify that SleepAccel paired CIs and Wilcoxon $p$-values are reported unadjusted as secondary/external-transfer diagnostics (whereas BIDSleep uses Holm-Bonferroni).
3. **N3:** Note transparently that if adjacent grid points near the elbow differ by `~0.2–0.3 pp`, the trade-off against edge memory/MMACs will be discussed without altering the locked selection.
4. **N4:** Fix `< 0.20 pp (<= 0.20 pp)` in Line 94 to `<= 0.20 pp`, and `probability/logit stitching` in Line 83 to `pre-softmax logit stitching`.
5. **N5:** Keep `p1_100k`..`p6_1051k` as the single canonical tag set and list the 5 planned script paths under `Technical Plan` as well as `Detailed Implementation Plan`.
6. **N6:** Expand `TST`, `SOL`, `WASO`, and `LoA` on first use.

---

## 2. Traceability Matrix

| Consolidated ID | Severity | Stage 5 (Zoom Out) | Stage 6 (Comprehension) | Stage 7 (Critic) | Stage 8 (Int. Zoom Out) | Stage 9 (Impl. Readiness) |
|---|---|---|---|---|---|---|
| **M1** (`ScaledConventionalCNNBiGRU`) | 🟡 Major | — | — | #1 | #2 | #1 |
| **M2** (`P2` Stitcher Parity & Fallback) | 🟡 Major | #1 | #1 | #2, #4 | #1 | #2 |
| **M3** (Reuse `EXP-029`/`EXP-020`) | 🟡 Major | #2 | — | — | #2 | #4 |
| **M4** (Pin `FT-2` & `W0-B` Contract) | 🟡 Major | #5.2 | #4 | #3, #4 | #2 | #4 |
| **M5** (CLI, Schemas, Commands) | 🟡 Major | — | — | — | — | #3 |
| **M6** (Concise Inline Definitions) | 🟡 Major | — | #2, #3, #4 | #9 | #3 | — |
| **N1** (`cnn_bigru_200k` Reference) | 🟢 Nit | #3 | — | #5 | #4 | — |
| **N2** (SleepAccel Unadjusted Note) | 🟢 Nit | — | — | #6 | #5 | — |
| **N3** (Elbow Sensitivity Note) | 🟢 Nit | #4 | — | — | — | — |
| **N4** (`<= 0.20 pp` & Pre-Softmax Text) | 🟢 Nit | #5.1 | #5 | #8, #10 | #5 | #5 |
| **N5** (Canonical Tags & File List) | 🟢 Nit | — | — | #7 | #6 | #5 |
| **N6** (Expand Sleep Abbreviations) | 🟢 Nit | — | #6 | — | — | — |
