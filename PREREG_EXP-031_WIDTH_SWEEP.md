# PRE-REGISTRATION (PROPOSAL): EXP-031, Single-Factor Width Sweep of the Plain CNN-BiGRU

**Status:** DRAFT until committed. Commit this file, record the hash in `EXPERIMENTS.md`, and only then train any new configuration. After the lock commit, sections 3 to 9 are immutable; corrections go in section 14 as dated amendments.
**Suggested ID:** EXP-031.
**Items marked `TO FILL` or `TO VERIFY`:** resolved from the repository and hardware environment prior to lock commit.
**Does not modify:** EXP-018 to EXP-030 artifacts, the frozen Model A (`DEC-037`) checkpoints, `EXP-029` `cnn_bigru_200k` / `full_v5` checkpoints, or any locked preregistration.

---

## 1. Purpose

EXP-029 found that all four new arms beat frozen Model A by about +1.8 to +2.1 pp Macro-F1, but a plain 223,268-parameter CNN-BiGRU (arm b) scored within 0.13 pp of the full architecture (`full_v5`, 190,788 parameters; difference +0.13 pp, CI [-0.35, +0.63], Holm p = 1.0). The most economical explanation is that Model A (163,940 parameters) was slightly too small, not that FiLM or the pyramid helps. This experiment tests that explanation directly.

**Questions:**
- **Q-SCALE:** how does subject-independent Macro-F1 scale with parameter count for a plain CNN-BiGRU under an identical training recipe?
- **Q-ARCH:** does `full_v5` beat a plain CNN-BiGRU of matched parameter count?
- **Q-SELECT:** which width should be the final backbone under a pre-declared deployment constraint?

This is a single-factor scaling study, not architecture search. No component other than width changes.

---

## 2. Required disclosures (before any result is interpreted)

1. **Design-informed:** this experiment was designed after seeing EXP-029 (and EXP-030) results on the same 47 subjects, so it is labeled "design-informed" and cannot confirm generalization.
2. **Iteration history (`v1`–`v4` development lineage and BIDSleep results):** `full_v5` is the fifth version in the project's architectural development lineage (`DEC-037`–`DEC-042`, `EXPERIMENTS.md`, `docs/designs/exp029_ultradian_pyramidal_coupler.md`):
   - **v1 (`ConventionalCNNBiGRU` / Model A, 163,940 params):** Matched-capacity baseline (`exp013_models.py`, frozen under `DEC-037` as `SleepStagingFinalModel_v1.0`). BIDSleep 5-fold CV ($N=47$, $L=1200$): `64.7907%` Macro-F1 (stitcher-aligned OOF, `DEC-043`; `64.79% ± 7.45%` in `EXP-021`/`DEC-038`).
   - **v2 (`SleepAdaptiveNet`, 162,325 params):** Auxiliary multi-head/multi-loss architecture (reliability gate, transition head, GRL, boundary loss, uncertainty head; `EXP-013`/`EXP-014`). BIDSleep 5-fold CV ($N=47$, $T=600$): `65.96%` Macro-F1 (`66.06%` Acc) vs. Model A `66.58%` Macro-F1 (`66.63%` Acc), yielding parity/negative attribution ($\Delta = -0.69\text{ pp}$); auxiliary heads were closed.
   - **v3 (`MultiScaleConventionalCNNBiGRU`, 192,692 params):** Pre-recurrent $k \in \{1, 3, 5\}$ multi-scale temporal conv fusion (`EXP-014` NAS Iteration 1 / `EXP-019-EXPLORATORY`). Exploratory Fold-1 smoke gains were not replicated on independent re-run (`DEC-026` erratum); reclassified as exploratory and closed (`DEC-029`).
   - **v4 (`EXP-029` pre-`full_v5` structural ablations: `film_only` 172,260 params; `pyramid_only` 182,468 params; `cnn_bigru_200k` 223,268 params):** Evaluated on BIDSleep 5-fold CV ($N=47$, $L=1200$, `DEC-042`): `film_only` = `66.67% ± 7.02%`, `pyramid_only` = `66.92% ± 7.05%`, `cnn_bigru_200k` = `66.61% ± 7.06%`, compared with `full_v5` (`Ultradian-Pyramidal-Coupler`, 190,788 params) = `66.74% ± 6.81%` (`full_v5_ce` = `66.95% ± 6.84%`).
   - *(Note: `EXP-030` [`DEC-043`] subsequently evaluated a 6-point non-uniform plain `ScaledConventionalCNNBiGRU` grid `p1_100k`..`p6_1051k` on BIDSleep, selecting `p1_100k` [98,940 params, `66.67%`]; `EXP-031` isolates a uniform `w`-multiplier sweep with matched-parameter control `P190`.)*
3. **SleepAccel isolation:** Confirmed: **zero PhysioNet SleepAccel v1.0.0 data or outputs informed any `v1`–`v5` architectural design, hyperparameter choice, or width selection.** Under `DEC-041` (`EXP-027`/`EXP-028`), SleepAccel ($N=31$, `n_qc_pass=25`) was evaluated strictly zero-shot on `Model A` and `Official SLAMSS-IFS` only *after* `Model A` was permanently frozen (`DEC-037`). Under `DEC-042` (`EXP-029`), all `v5` architectures (`film_only`, `pyramid_only`, `full_v5`, `cnn_bigru_200k`) were locked in `docs/designs/exp029_ultradian_pyramidal_coupler.md` prior to any SleepAccel inference, and SleepAccel predictions were persisted strictly post-training in isolated `experiments/EXP-029/sleepaccel_cache/predictions/`. SleepAccel remains an independent zero-shot external test set for `v1`–`v5`.
4. **Winner's curse:** the same BIDSleep cross-validation will be used to choose a width and to report its performance. The reported figure for the selected width is therefore optimistic. The selection rule in section 9 targets an elbow, not the maximum, to limit this. The external check in section 10 is the independent evidence.

---

## 3. Single factor and constants

**Single factor:** model width multiplier `w`. Confirmed against `exp013_models.py` (`ConventionalCNNBiGRU`, base `d_enc=64`, `d_temp=96`, `head_hidden=64`, `meta_proj=16`) and `experiments/EXP-030/exp030_models.py` (`ScaledConventionalCNNBiGRU` + `ScaledModalityEncoder` with `c1 = d_enc // 2`, `c2 = d_enc`): `w` scales `d_enc`, `d_temp`, and `head_hidden` uniformly via `round_to_8(base * w) = max(8, int(round(base * w / 8.0) * 8))`, while keeping `meta_proj = Linear(5, 16) + GELU` (`96` params) fixed across all widths.

**Held constant across all arms:**
- Architecture family: plain `ConventionalCNNBiGRU` (`ScaledConventionalCNNBiGRU` preserving the exact 14-key `forward()` contract, depth, kernel sizes `5, 5, 3`, `GroupNorm(4, c1)` / `GroupNorm(4, c2)`, `dropout=0.2`, `LayerNorm`, `meta_proj=16`, 2-layer bidirectional `GRU`, and `Linear(d_temp, head_hidden) -> LayerNorm(head_hidden) -> GELU -> Dropout(0.2) -> Linear(head_hidden, 4)`).
- Training recipe (identical to frozen Model A / `EXP-029` / `EXP-030` `run_exp030_cv.py`):
  - Optimizer: `Adam(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)`
  - Batch size: train `micro_batch_size = 4` (`RandomCropSleepDataset(context_len=1200, neutralize_circadian=True)`), validation `val_batch_size = 16`
  - Schedule & stopping: `max_epochs = 100`, `min_epochs = 40`, early stopping `patience = 25` on validation Macro-F1 (`> best + 1e-6`), gradient clipping `max_norm = 1.0`
  - Loss & precision: `CE + 0.01 * RCCLoss` computed in `FP32` outside `torch.amp.autocast(device_type="cuda", dtype=torch.float16)` (`compute_stage_loss_fp32`)
  - Inference: `LogitStitchingSlidingWindowPredictor(context_len=1200, neutralize_circadian=True)`
- Context: full night, `L = 1200`, sliding-window inference as in EXP-018 / EXP-029 / EXP-030.
- Task: 4-class (Wake/Light/Deep/REM), valid labels 0 to 3 (`-100` ignore_index).
- Splits: fixed EXP-018/EXP-029/EXP-030 5-fold subject-disjoint manifests (`experiments/EXP-014/EXP014_MANIFESTS/fold_1..5.json`, cross-checked against `review_package/fold_manifests.csv` SHA-256: `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`, prefix `36d84020`).
- Seeds: `42, 123, 2024`.
- Evaluation: subject-level Macro-F1 averaged over seeds within subject (`N = 47`).

---

## 4. Arms

All 9 plain widths were instantiated on CPU (`ScaledConventionalCNNBiGRU`) and profiled with `torch.ao.quantization.quantize_dynamic(..., {nn.Linear, nn.GRU}, dtype=torch.qint8)` and `compute_model_mmacs(..., seq_len=1200, epoch_len=30)`:

| Arm | Width `w` | `(d_enc, d_temp, head_hidden)` | Total Params | FP32 KiB | Dynamic INT8 KiB | MMACs (`L=1200`) | Sec. 9 (`< 400 KiB`) | Source / Isolation |
|---|---|---|---|---:|---:|---:|---|---|
| **W050** | 0.50 | `(32, 48, 32)` | 43,108 | 168.39 | 81.00 | 202.9 | **ELIGIBLE** | **New** (`experiments/EXP-031/`) |
| **W075** | 0.75 | `(48, 72, 48)` | 93,796 | 366.39 | 176.96 | 450.0 | **ELIGIBLE** | **New** (`experiments/EXP-031/`) |
| **A (reused)** | 1.00 | `(64, 96, 64)` | 163,940 | 640.39 | 309.91 | 794.1 | **ELIGIBLE** | Existing frozen `DEC-037` checkpoints (`64.7907%` OOF) |
| **P190** | ~1.08 | `(72, 100, 64)` | 190,940 | 745.86 | 375.16 | 984.2 | **ELIGIBLE** | **New** (`+0.080%` vs. `full_v5` `190,788`; `experiments/EXP-031/`) |
| **B (reused)** | ~1.17 | `(64, 128, 64)` | 223,268 | 872.14 | 368.04 | 864.7 | **ELIGIBLE** | Existing `EXP-029` `cnn_bigru_200k` checkpoints (`66.61%`) |
| **W125** | 1.25 | `(80, 120, 80)` | 253,540 | 990.39 | 479.86 | 1,235.2 | INELIGIBLE | **New** (`experiments/EXP-031/`) |
| **W150** | 1.50 | `(96, 144, 96)` | 362,596 | 1,416.39 | 686.82 | 1,773.5 | INELIGIBLE | **New** (`experiments/EXP-031/`) |
| **W200** | 2.00 | `(128, 192, 128)` | 639,076 | 2,496.39 | 1,211.72 | 3,141.2 | INELIGIBLE | **New** (`experiments/EXP-031/`) |
| **W250** | 2.50 | `(160, 240, 160)` | 993,380 | 3,880.39 | 1,884.63 | 4,897.2 | INELIGIBLE | **New** (`experiments/EXP-031/`) |
| **V5 (reused)** | n/a | `full_v5` | 190,788 | 745.27 | 360.50 | — | **ELIGIBLE** | Existing `EXP-029` `full_v5` (`66.74%`) |
| **S (reused)** | n/a | `Official SLAMSS-IFS` | 5,465,768 | 21,350.66 | — | — | INELIGIBLE | Existing `EXP-021` (`67.37%`) |

**Comparison to `EXP-030` (`DEC-043`) & Isolation Rule:**
- None of the 7 new `EXP-031` configurations match `EXP-030` (`p1_100k`..`p6_1051k`) bit-for-bit: under the uniform `round_to_8(base * w)` rule, `W075` has `(48, 72, 48)` (`93,796` vs. `p1_100k` `(48, 76, 48)` `98,940`), `W125` has `(80, 120, 80)` (`253,540` vs. `p3_268k` `(80, 128, 64)` `268,260`), `W150` has `(96, 144, 96)` (`362,596` vs. `p4_404k` `(96, 160, 96)` `403,524`), `W200` has `(128, 192, 128)` (`639,076` vs. `p5_666k` `(128, 200, 128)` `665,556`), and `W250` has `(160, 240, 160)` (`993,380` vs. `p6_1051k` `(160, 256, 128)` `1,051,492`).
- **New training:** All 7 new configurations (`W050`, `W075`, `P190`, `W125`, `W150`, `W200`, `W250`) × 5 folds × 3 seeds = **105 runs**, trained fresh and isolated under `experiments/EXP-031/`. Nothing is written to earlier experiment folders.

---

## 5. Training and profiling protocol

- Train every new configuration with the locked recipe in section 3. No per-width hyperparameter tuning, early stopping on test data, or loss changes.
- For every width, record: trainable parameters; FP32 weight KiB; dynamic-INT8 in-memory payload KiB (`quantize_dynamic` on `{nn.Linear, nn.GRU}`, `dtype=torch.qint8`, same method as `W0-B`); MMACs at `L = 1200`; CPU latency at `1` and `4` threads (`OMP_NUM_THREADS=1` / `4`, `torch.set_num_threads(1/4)`).
- **Verified hardware & software environment:**
  - CPU: `AMD Ryzen 7 7700 8-Core Processor`
  - GPUs: `2x NVIDIA GeForce RTX 3060 12GB` (`cuda:0`, `cuda:1`)
  - Python: `3.14.7` (`.venv/bin/python`)
  - PyTorch: `2.14.1+cu130` (CUDA `13.0`)
- Save checkpoints and per-subject metrics in `experiments/EXP-031/`, and write nothing to earlier experiment folders.

---

## 6. Outcomes and statistics

- **Primary outcome:** subject-level Macro-F1, N = 47, averaged over the three seeds within subject.
- **Secondary outcomes:** accuracy, kappa, per-stage F1.
- **Inference:** paired differences, 10,000-resample participant-level percentile bootstrap 95% CI, two-sided Wilcoxon signed-rank (robustness check), Cohen's d_z.
- **Multiplicity:** Holm correction across the two confirmatory tests in section 8.
- **Trend test:** Page's L across the ordered widths W050 to W250 plus A and B (ordered by parameter count), reporting both one-sided and two-sided p-values and stating which one decides the verdict. The two-sided p decides.
- **Power note:** EXP-029 gives a paired SD of about 1.7 pp for `full_v5` vs the plain 223k model (derived from its CI), so with N = 47 the minimum detectable effect at 80% power is about 0.7 pp. The practical threshold in section 8 is set just above it.

---

## 7. Hypotheses

**H-ARCH (confirmatory, primary):** `full_v5` outperforms the matched-parameter plain CNN-BiGRU (P190) in subject-level Macro-F1.

**H-SCALE (confirmatory, secondary):** Macro-F1 increases with parameter count across the plain-model widths (Page's L).

**Descriptive (no confirmatory claim):** the width at which marginal gains fall below 0.5 pp; the smallest plain width not significantly below SLAMSS-IFS (paired, Holm-adjusted; no equivalence claim); per-stage scaling; the Macro-F1 vs parameters and vs INT8 payload Pareto plots.

---

## 8. Decision rules (LOCKED at commit)

**H-ARCH** (Δ = `full_v5` minus P190; practical threshold +0.75 pp, proposed, edit before lock only):
- **SUPPORTED:** Δ ≥ +0.75 pp and the CI lower bound > 0.
- **NOT SUPPORTED (practically):** the CI upper bound < +0.75 pp.
- **INCONCLUSIVE:** anything else.
No rescue tuning or post hoc width selection for P190.

**H-SCALE:** SUPPORTED if Page's two-sided p (Holm-adjusted across the two confirmatory tests) < 0.05; otherwise NOT SUPPORTED.

**Allowed claims:**
- If H-ARCH is NOT SUPPORTED or INCONCLUSIVE, the paper may not claim architectural novelty or benefit. `full_v5` is reported as a capacity-matched ablation.
- If H-ARCH is SUPPORTED, the paper may report an architectural benefit at matched parameters, flagged "design-informed (v5 chosen after v1 to v4)", with the section 2 disclosures.

---

## 9. Backbone selection rule (LOCKED at commit)

1. **Deployment constraint (declared now):** dynamic-INT8 in-memory payload < 400 KiB, as in EXP-029's design targets. Only widths meeting it are eligible (`W050`, `W075`, `A`, `P190`, `B`). Wider widths (`W125`, `W150`, `W200`, `W250`) are still run to map the curve.
2. **Selection among eligible widths:** the smallest width whose mean Macro-F1 is within one paired standard error (bootstrap SE of the paired difference) of the best eligible width. Ties go to fewer parameters.
3. The selected backbone is the sole new backbone carried forward. No width is selected by external or personalization results.

---

## 10. Downstream use of the selected backbone

- **SleepAccel (once):** evaluate only a pre-declared set zero-shot: the selected width, A (w = 1.00), `full_v5` and SLAMSS-IFS. Report primary single-model mean and secondary 15-model ensemble, with paired bootstrap CIs. Do not evaluate other widths on SleepAccel.
- **Personalization:** FT-2, the k = 1 to 4 curve and GATE sensitivity for the selected backbone, with the existing locked rules and a separate preregistration. Report Model A, `full_v5` and the selected backbone together. Do not choose the "best" backbone for personalization after seeing results.
- **INT8 and edge profile:** as in section 5.

---

## 11. Outputs

Per-subject, per-seed metrics (CSV); width table (parameters, payload, MMACs, latency, Macro-F1 with paired CI vs A); the H-ARCH verdict; Page's L result; scaling and Pareto figures; v1 to v4 history table; written SleepAccel isolation statement; `EXP031_FINAL_REPORT.md` stating verdicts exactly as defined above.

---

## 12. Compute and timeline

Verified against `EXP-030` (`run_results/*/*.json`, same `ScaledConventionalCNNBiGRU` and `L=1200` recipe): single fold-seed runs averaged `324.3s–345.2s` (~`5.4–5.8 min`/run, `81–86 GPU-min` per 15-run arm), with ~`41–46 min` wall-clock per 15-run configuration when split concurrently across dual NVIDIA RTX 3060 12GB GPUs (`cuda:0`: 7 runs, `cuda:1`: 8 runs). Budget `~5.0 hours` total wall-clock for all 7 new configurations (105 runs) on dual RTX 3060 GPUs, plus `~10 min` for CPU INT8 profiling (`1T`/`4T`) and statistical aggregation.

---

## 13. Reporting language rules

- Say "capacity effect," not "architectural improvement," unless H-ARCH is supported.
- Do not write "parity" with SLAMSS-IFS without a pre-specified equivalence margin. Report differences with CIs.
- Report all widths and iterations; do not report only the best.
- Label the selected-width BIDSleep estimate as optimistic.

---

## 14. Amendments

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |
