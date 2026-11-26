# EXP-032 Pre-Execution EGM Drift Audit Report

* **Audit Date:** `2026-11-26`
* **Specification Anchor:** [`docs/designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md`](../designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md)
* **Audited Artifacts:**
  - [`experiments/EXP-032/exp032_models.py`](../../experiments/EXP-032/exp032_models.py)
  - [`experiments/EXP-032/run_exp032_pipeline.py`](../../experiments/EXP-032/run_exp032_pipeline.py)
  - [`tests/test_exp032_preflight.py`](../../tests/test_exp032_preflight.py)
  - [`.gitignore`](../../.gitignore)
* **Overall Verdict:** **`GO`** (**0 Blockers, 0 Majors, 0 Nits Remaining** — All 4 Blockers `B1`–`B4`, 3 Majors `M1`–`M3`, and 4 Nits `N1`–`N4` resolved and re-verified)

---

## 1. Axis-by-Axis Final Verification

| Drift Axis | Status | Verification Evidence |
|---|---|---|
| **Axis 1: Exact Spec-to-Code Parity** | **PASS** | `M1 = 1,402,468`, `M2/M3 = 1,402,492` params; `4,053.08` MMACs (`L=1200`); `compute_fold_class_weights` (`mean=1.0`); FP32 parallel associative-scan `LinearChainCRF4`, `L_CE_weighted`, `L_KL_masked`; `AdamW(lr=1e-4)` + `CosineAnnealingLR(T_max=100, eta_min=1e-6)` + `grad_clip=1.0`, `min_epochs=40`; Stage A keyed by `(fold, subject_id)` (`assert len == 25`); `CRFLogitStitchingPredictor` (`viterbi_decode` / `forward_backward_marginals`). |
| **Axis 2: Zero-Leakage Teacher Caching** | **PASS** | `--stage cache_teachers` loads fold-matched `EXP-021` checkpoint `(f, s)` (`"model_state_dict"`), runs `OfficialSLAMSSIFSWrapper` (`trg_zeros`, `teacher_forcing_ratio=0.0`, `neutralize_circadian=True`), extracts `L=1200, stride=600` strictly on `inner_train`, enforces subset check in `TeacherCachedDataset`, and `experiments/EXP-032/cache/` is explicitly listed in `.gitignore`. |
| **Axis 3: Statistical & Multiplicity Parity** | **PASS** | `_paired_vs_ref` unpacks 2-tuple `(ci_lo, ci_hi)` with explicit `subject_id` alignment assertion and `-2.50 pp` one-sided NI Wilcoxon test; `stage_eval_external` passes `prep_dir="experiments/EXP-027/cache/preprocessed", qc_only=False` (reporting `n=25 qc_clean` + `N=31 all` under `em_off`), verifies all 45 SHA-256 digests in `frozen_checkpoint_manifest.json`, benchmarks `W100` & `Official SLAMSS-IFS` alongside `M1–M3` in interleaved `1T`/`4T` FP32 CPU latency (with warmup), and runs `evaluate_exp032_verdicts` (`3` primary Holm-Bonferroni, `3 × K_qualifying` secondary Holm-Bonferroni, descriptive `M1 - W100` / `M2 - M1` / `M3 - M2`, exhaustive §5.2 verdicts). |
| **Axis 4: Protected File Immutability** | **PASS** | `git diff --name-only` confirmed **0 modifications** to `PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md`, `exp013_models.py`, and `experiments/EXP-018`–`EXP-031`. |

---

## 2. Resolution Audit Trail

| ID | Original Severity | Resolution Applied | Verification |
|---|---|---|---|
| **B1** | Blocker | `_paired_vs_ref` unpacks 2-tuple `(ci_lo, ci_hi)` from `participant_bootstrap_ci(...)` and computes `mean_d = float(np.mean(diff))`. | `test_paired_vs_ref_bootstrap_ni_and_index_alignment` **PASS** |
| **B2** | Blocker | `stage_eval_external` passes `prep_dir=os.path.join(EXP027_DIR, "cache", "preprocessed"), qc_only=False` and reports `n=25 qc_clean` + `N=31 all` (`em_off`). | Code verified |
| **B3** | Blocker | Added `evaluate_exp032_verdicts(...)` implementing 3-arm primary Holm-Bonferroni, $3 \times K_{\text{qualifying}}$ secondary Holm-Bonferroni, SleepAccel `-2.50 pp` NI tests against `EXP-027` `Official SLAMSS-IFS`, and exhaustive §5.2 per-arm & overall verdicts. | `test_verdict_rule_exhaustive_and_multiplicity` **PASS** |
| **B4** | Blocker | Added `M1_minus_W100` (from `experiments/EXP-031/runs/W100/W100_subject_metrics.csv`), `M2_minus_M1`, and `M3_minus_M2` descriptive contrasts in `stage_aggregate`. | Code verified |
| **M1** | Major | Keyed Stage A `inner_val` pooling by `(fold, subject_id)` with `assert len(fold_sub_f1s) == 25`. | `test_stage_a_fold_subject_pooling_n25` **PASS** |
| **M2** | Major | Included `W100` and `Official SLAMSS-IFS` alongside `M1–M3` with 1 warmup round in interleaved `1T`/`4T` FP32 CPU latency (`exp032_cpu_latency.json`). | Code verified |
| **M3** | Major | Added `verify_frozen_checkpoint_manifest(args.output_dir)` asserting all 45 checkpoint SHA-256 digests before loading SleepAccel. | Code verified |
| **N1** | Nit | Added 1 warmup batch + `num_workers=2` in `stage_preflight` and asserted `est_70ep_min <= 5.0` (`4.12s/epoch`, `est. 70-epoch early-stop = 4.80 min`, `peak_vram = 876.1 MB`). | `--stage preflight --device cuda:0` **PASS** |
| **N2** | Nit | Added `experiments/EXP-032/cache/` explicitly to `.gitignore`. | File verified |
| **N3** | Nit | Added `assert list(a_s.index) == list(b_s.index)` in `_paired_vs_ref`. | `test_paired_vs_ref_bootstrap_ni_and_index_alignment` **PASS** |
| **N4** | Nit | Expanded `tests/test_exp032_preflight.py` to 10 unit tests covering class weights, `_paired_vs_ref`, `(fold, subject_id)` pooling, and exhaustive §5.2 verdicts. | `pytest -v tests/test_exp032_preflight.py` (**10/10 PASS**, `1.93s`) |
