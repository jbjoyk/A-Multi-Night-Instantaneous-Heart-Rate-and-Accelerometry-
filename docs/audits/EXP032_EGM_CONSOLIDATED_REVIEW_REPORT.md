# EXP-032 EGM Consolidated Goldfish Review Report

* **Target Document:** [`docs/designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md`](../designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md)
* **Audit Date:** `2026-11-26`
* **Overall Verdict:** **NO-GO (Pre-Lock Revisions Required)**
* **Goldfish Reviewers Executed:**
  - `egm-zoom-out` (Initial Zoom Out, Stage 5)
  - `egm-comprehension` (Zero-Context Comprehension Test, Stage 6)
  - `egm-design` (Hostile Critic Review, Stage 7)
  - `egm-implementation-readiness` (Implementation Readiness & Codebase Parity, Stage 9)
* **Codebase Parity Verified:**
  - `ScaledModalityEncoder` (`experiments/EXP-030/exp030_models.py`) tensor shapes `(B, T, 30)` + `meta (B, T, 5)` and exact parameter counts (`M1 = 1,402,468`; `M2`/`M3 = 1,402,492`) **PASS**.
  - `EXP-021` (`experiments/EXP-021/run_tsk08_slamss_matched.py`) fold/seed manifest alignment (`FOLDS=[1..5]`, `SEEDS=[42, 123, 2024]`, `review_package/fold_manifests.csv` SHA-256 `36d84020`) **PASS** (note: `EXP-021` checkpoints store weights under `"model_state_dict"`).

---

## 1. Consolidated Findings Summary (Blocker / Major / Nit)

| ID | Severity | Consolidated Finding | Affected Scope | Sources | Required Remediation |
|---|---|---|---|---|---|
| **BLK-01** | **Blocker** | **Contradictory & Non-Exhaustive Verdict Rule (`INCONCLUSIVE` vs. `NOT SUPPORTED`)** | §5.2 (L110–113) | `COMP`, `ZOOM`, `CRIT` | Remove the unadjusted CI rescue clause (`bar_Delta >= +0.50 pp with 95% CI excluding 0`) and undefined "uncertainty margin"; partition `SUPPORTED` / `INCONCLUSIVE` / `NOT SUPPORTED` strictly on primary $p_{\text{Holm}} < 0.05$ and secondary gate pass/fail, and define the Overall aggregation rule. |
| **BLK-02** | **Blocker** | **`M3` Loss Scale Mismatch (`L_CRF` vs. `KL`) & Unspecified `KL` Masking** | §2.2 (L53), §3.3 (L66) | `COMP`, `ZOOM`, `CRIT`, `IMPL` | Normalize `L_CRF` per valid epoch (`NLL / n_valid`) in FP32 outside AMP autocast so it matches epoch-averaged `KL` (avoiding ~`1,000x` scale domination), and explicitly lock `KL` masking over `valid = (y != -100) & mask`. |
| **BLK-03** | **Blocker** | **Unresolved Loss Forks & Undefined Class-Weight Formula (`w_c`)** | §2.2 (L51–53), §3.3 (L65–66) | `COMP`, `ZOOM`, `CRIT`, `IMPL` | Eliminate all `"or"` clauses (`CE` vs. `RCCLoss` in `M1`; emission-potential scaling vs. `+ 0.1 * L_CE_weighted` in `M2`/`M3`). Lock one exact `w_c` formula (computed per fold from `inner_train`, e.g., `(N / n_c)^0.5` normalized to `mean=1.0`) and exact loss placement. |
| **BLK-04** | **Blocker** | **Missing `EXP-021` Teacher Logits `z_T` Pre-Caching & Random-Crop Contract (`M3`)** | §2.2 (L53), §3.4 (L68), §3.5 (L69–72) | `CRIT`, `IMPL` | Specify pre-caching full-night stitched FP32 teacher logits `z_T` `(N, 4)` to disk per `(fold, seed)` strictly on `inner_train` nights using a logit-stitching `SLAMSSSlidingWindowPredictor`, slicing `z_T[s:s+L]` in `RandomCropSleepDataset` to avoid `5.47M` teacher forward-pass overhead inside training. |
| **BLK-05** | **Blocker** | **Ambiguous `M2`/`M3` CRF Decoding (`Viterbi` vs. `Marginal`) & `prob_ensemble` Input** | §2.2 (L52), §3.2 (L64), §4 (L93), §5.1 | `ZOOM`, `CRIT`, `IMPL` | Lock whether `mean_single_model` `y_pred` uses Viterbi or marginal `argmax`; define `prob_ensemble` for `M2`/`M3` as averaging 15 checkpoints' forward-backward marginal posteriors `P(y_t \| x)`; and specify `CRFLogitStitchingPredictor` for `T > 1200`. |
| **BLK-06** | **Blocker** | **Capacity + Recipe Confound & `M3` Circularity / Multiplicity Penalty** | §1.2, §2.1–2.2, §4, §5.1 | `ZOOM` | Either add a capacity- and recipe-matched plain `ScaledConventionalCNNBiGRU` control (`~1.40M`) to isolate TCN context from `8.55x` width + `AdamW`/cosine tuning, and separate `M3` (distillation) from the primary `M1`/`M2` Holm-Bonferroni family (or explicitly justify `M3` as a separate secondary/compression arm). |
| **BLK-07** | **Blocker** | **Underpowered & Policy-Inconsistent `-1.00 pp` SleepAccel (`n=25`) NI Gate** | §1.2 (L24), §5.1 (L106–107), §5.2 | `ZOOM`, `CRIT` | Given `DEC-033` (retracted NI margins) and `DEC-041` (`n=25` CI width `~3.1–3.4 pp`, `<20%` NI power at `Delta=0`), replace the `-1.00 pp` NI gate with paired estimation + 95% CIs (or justify power and align CI alpha with Holm-Bonferroni). |
| **MAJ-01** | **Major** | **`M1` `RCCLoss` Contradiction with Stage A 4-Config Grid** | §2.2 (L51), §4 (L84–85) | `COMP`, `ZOOM`, `CRIT`, `IMPL` | Lock `M1`'s exact training loss (`CE` or `CE + 0.01 * RCCLoss`) so it is not left to an unbudgeted Stage A choice. |
| **MAJ-02** | **Major** | **`inner_val` Macro-F1 Aggregation & `min_epochs=40` Checkpoint Rule** | §4 (L83, L85, L92) | `COMP`, `IMPL` | Lock `inner_val` Macro-F1 as unweighted participant-level mean (matching `outer_test`) and lock whether `best_val` tracking / `epochs_no_improve` starts at epoch 1 or `min_epochs=40`. |
| **MAJ-03** | **Major** | **Missing Optimizer, Scheduler, Batch Size, Grad Clip, and FP32/AMP Specs** | §4 (L85–93) | `IMPL` | Specify `CosineAnnealingLR(T_max=100, eta_min=...)`, `micro_batch_size=4`, `val_batch_size=16`, `grad_clip=1.0`, and `float16` AMP backbone with FP32 CRF/loss outside autocast. |
| **MAJ-04** | **Major** | **Cross-Arm Secondary Multiplicity & Missing Artifact Output Schemas** | §4, §5.1 | `CRIT`, `IMPL` | Clarify whether secondary Holm-Bonferroni applies per arm or across qualifying arms (or only to the best primary arm), and define output CSV/JSON paths/schemas (`run_results/`, `{arm}_subject_metrics.csv`, `{arm}_sleepaccel_metrics.csv`, `frozen_checkpoint_manifest.json`). |
| **MAJ-05** | **Major** | **`DEC-031` Framing & `M2` Redundancy vs. Prior `EXP-013`/`EXP-014` Evidence** | Header, §2.2 (`M2`) | `ZOOM` | Reframe `DEC-031` accurately (`163,940` is a baseline reference size, not a ceiling) and explicitly justify `M2` CRF given `EXP-013`/`EXP-014` null transition-matrix gains on BiGRU. |
| **MAJ-06** | **Major** | **Undefined Inline Terms (`em_off`, `DampedEMCalibrator`, `P190`, `DEC-*`)** | §1.1 (L20), §3.6 (L75), §4 (L93) | `COMP` | Add concise inline definitions and relative links to `DECISIONS.md` for `em_off`, `DampedEMCalibrator`, and `P190`. |
| **NIT-01** | **Nit** | **Premature Status Header** | Header (L6) | `COMP`, `ZOOM`, `CRIT` | Change status to `Draft — Revisions Required` until all Blockers/Majors are resolved. |
| **NIT-02** | **Nit** | **Stitching Terminology (`probability` vs. `LogitStitchingSlidingWindowPredictor`)** | §3.2 (L64) | `CRIT`, `IMPL` | Clarify that `EXP-030`/`EXP-031` used `LogitStitchingSlidingWindowPredictor` (pre-softmax logits), whereas `EXP-021` used probability stitching. |
| **NIT-03** | **Nit** | **Label Mapping, `meta` Semantics, `SLAMSS-IFS` MMACs, `M1 - W100` Confound & Packing** | §1.2, §2.1–2.2, §4, §5.1 | `COMP`, `ZOOM`, `CRIT`, `IMPL` | Document `{0: Wake, 1: Light, 2: Deep, 3: REM}`, 5 `meta` channels, `pack_padded_sequence(seq_lens)` + `(B,T,C)<->(B,C,T)` transpose, `M1 - W100` optimizer difference, and `interleaved 1T/4T` definition. |

---

## 2. Traceability Matrix

| Consolidated ID | `egm-zoom-out` | `egm-comprehension` | `egm-design` (Critic) | `egm-implementation-readiness` |
|---|---|---|---|---|
| **BLK-01** | Major #4.4 | Blocker #1 | Blocker #2 | — |
| **BLK-02** | Major #2 | Blocker #2 | Blocker #1.1 | Major #3 |
| **BLK-03** | Major #4.2 | Blocker #2 | Blocker #1.2 | Blocker #1 |
| **BLK-04** | — | — | Major #6.2 | Blocker #2 |
| **BLK-05** | Major #4.3 | Major #4.3 | Blocker #3 | Major #5 |
| **BLK-06** | Blocker #1, Blocker #2 | — | — | — |
| **BLK-07** | Blocker #3 | — | Major #4 | — |
| **MAJ-01** | Major #4.1 | Major #3 | Major #5 | Blocker #1 |
| **MAJ-02** | — | Major #4.1, #4.2 | — | Nit #7.1 |
| **MAJ-03** | — | — | — | Major #4 |
| **MAJ-04** | — | — | Major #6.1 | Major #6 |
| **MAJ-05** | Major #1.2, #1.3 | — | — | — |
| **MAJ-06** | — | Major #5 | — | — |
| **NIT-01** | Nit #3.1 | Nit #6.4 | Nit #7.1 | — |
| **NIT-02** | — | — | Nit #7.2 | Major #5 |
| **NIT-03** | Nit #3.2, #3.3 | Nit #6.1–6.3 | Nit #7.3 | Nit #7.2 |
