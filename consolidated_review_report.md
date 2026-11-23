# Consolidated Goldfish Review Report: `docs/designs/exp029_ultradian_pyramidal_coupler.md`

*   **Target Document:** [`docs/designs/exp029_ultradian_pyramidal_coupler.md`](./docs/designs/exp029_ultradian_pyramidal_coupler.md)
*   **Status:** Resolved (All `5` Blockers, `5` Majors, and `6` Nits accepted and applied)
*   **Date:** 2026-11-23
*   **Reviewers:** `egm-zoom-out`, `egm-design`, `egm-structural`, `egm-codebase-details`, `egm-implementation-readiness`
*   **Overall Verdict:** **NOT READY FOR IMPLEMENTATION / Q1 PREREGISTRATION** (`5 Blockers`, `5 Majors`, `6 Nits`)

---

## 1. Blockers (Must Resolve Before Implementation)

### B1. `DampedEMCalibrator` Mechanics, Temperature Scaling $T$, & Source-Domain Tuning Incoherence
*   **Sources:** `egm-design` (B1, B2, M3), `egm-zoom-out` (B1), `egm-implementation-readiness` (B2), `egm-codebase-details` (M2), `egm-structural` (M2)
*   **Affected Scope:** `docs/designs/exp029_ultradian_pyramidal_coupler.md` (L11, L26, L40, L50–54, L109–114, L170–174, L187) vs. `exp018_inference.py` (L78–111, L139–140, L171–182) and `test_rccloss_audit.py` (L37–51).
*   **Findings:**
    1. **Post-Softmax Temperature Scaling Impossibility:** [`SlidingWindowPredictor`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp018_inference.py#L20-L182) applies `F.softmax(logits, dim=-1)` per window *before* stitching into `final_probs` (`exp018_inference.py:96, 140`). Standard logit temperature scaling `softmax(logits / T)` cannot be recovered from averaged window probabilities. Furthermore, `SlidingWindowPredictor.predict_night` has an early `return` for Case 1 (`N <= L`, `exp018_inference.py:103-111`), which would silently bypass `prior_calibrator` if applied only after Case 2 (`L171`).
    2. **Undefined $\pi_{\text{src}}$ & `RCCLoss` Miscalibration:** Equation (1) uses $\pi_{\text{src}}(c)$ without defining it. Because primary training uses `CE + 0.01 * RCCLoss`, softmax outputs are **not** calibrated to empirical `inner_train` class priors. Using empirical frequencies as $\pi_{\text{src}}$ violates Bayes' rule in Saerens EM.
    3. **EM Fixed-Point & Tuning Leakage:** Anchoring $\pi^{(t+1)} = (1 - \alpha)\pi_{\text{src}} + \alpha \hat{\pi}_{\text{EM}}^{(t+1)}$ prevents convergence to $\pi_{\text{tgt}}$. Tuning $(\alpha, T)$ on pooled BIDSleep `OOF` leaks outer test data; tuning on BIDSleep `inner_val` cannot optimize SleepAccel Dreem 2 -> PSG shift (since BIDSleep has no modality shift, $\alpha = 0$ is optimal on-domain); and tuning $\alpha$ around SleepAccel `1455390` (`0%` Deep) is implicit external test leakage.
    4. **YAGNI / Baseline Confound:** `DampedEMCalibrator` is a 0-parameter post-hoc wrapper. Bundling it into `EXP-029` without testing `Model A + EM` and `SLAMSS-IFS + EM` on existing `EXP-027` prediction caches conflates representation gains with prior adjustment.
*   **Required Fix:** Either decouple `DampedEMCalibrator` from `EXP-029` (evaluating it first on frozen `EXP-027` caches of `Model A` and `SLAMSS-IFS`), or explicitly define: (a) per-window `logits / T` vs. power-scaled `final_probs ** (1/T)`, (b) Case 1 (`N <= L`) + Case 2 (`N > L`) hook placement, (c) $\pi_{\text{src}}$ as the model's mean predicted `inner_val` marginal, (d) EM iteration count $K$ and final $p_i^{\text{cal}}(c)$, and (e) a single preregistered $\alpha$ (e.g., $\alpha = 0.35$) rather than on-domain tuning.

### B2. Unspecified Training Loop, Optimizer, AMP FP32 Loss Boundary, & Manifest Verification
*   **Sources:** `egm-implementation-readiness` (B1, M5), `egm-zoom-out` (M3), `egm-structural` (M1), `egm-codebase-details` (M1, M5)
*   **Affected Scope:** `docs/designs/exp029_ultradian_pyramidal_coupler.md` (L44–48, L111, L153, L168, L176–181, L192, L198) vs. `experiments/EXP-021/run_tsk08_slamss_matched.py` (L16–20, L88–151, L306–340, L390–393) and `test_rccloss_audit.py` (L21–33).
*   **Findings:**
    1. **AMP `float16` `NaN` Hazard in `RCCLoss`:** `RCCLoss.forward` (`test_rccloss_audit.py:21-29`) clamps probabilities to `[1e-5, 1 - 1e-5]` and casts weights to `x.data.type()`. Under CUDA AMP `float16`, `1.0 - 1e-5` rounds to `1.0`, causing `torch.log(0.0) = -inf` and `NaN` gradients. `EXP-021` avoids this by computing `CE + 0.01 * RCCLoss` strictly in FP32 outside `torch.amp.autocast`.
    2. **Missing Training Hyperparameters & Dataset Contract:** `run_exp029_cv.py` omits optimizer (`Adam(lr=1e-4, betas=(0.9, 0.999))`), LR scheduler (none), `max_epochs=100`, `min_epochs=40`, `patience=25`, `grad_clip=1.0`, `random_crop=True`, `val_batch_size=16`, checkpoint selection criterion (`inner_val` pooled `macro_f1 > best + 1e-6`), and cannot reuse `SLAMSSMatchedDataset` (`run_tsk08_slamss_matched.py:309`, which yields `x_signal` instead of separate `x_ihr`, `x_accel`, `x_meta`, `seq_lens`).
    3. **Missing Cryptographic Manifest Gate:** Omits `verify_manifests_and_splits` against `review_package/fold_manifests.csv` (SHA-256 prefix `36d84020`, `run_tsk08_slamss_matched.py:88-151`).
*   **Required Fix:** Lock all training hyperparameters to exact `EXP-018`/`EXP-021` parity, mandate FP32 loss computation outside `autocast`, and specify the `EXP-029` dataset contract and manifest SHA-256 verification.

### B3. Missing Holm-Bonferroni Multiplicity Control, Seed-Aggregation Unit, & Equivalence Framing
*   **Sources:** `egm-structural` (B2), `egm-design` (M4), `egm-zoom-out` (M4)
*   **Affected Scope:** `docs/designs/exp029_ultradian_pyramidal_coupler.md` (L24, L48, L200) vs. `experiments/EXP-021/analyze_tsk08_paired_benchmark.py` (L5–19, L136–195) and `DECISIONS.md` (`DEC-033`, `DEC-041`).
*   **Findings:**
    1. Proposes paired Wilcoxon tests (`N = 47`) and 10,000-resample bootstrap CIs across 6 arms and 5+ metrics without Holm-Bonferroni FWER control, primary vs. secondary hypothesis hierarchy, or joint significance criterion (`CI excludes 0 AND Wilcoxon p < 0.05`).
    2. Fails to state that the unit of analysis is the `N = 47` participant mean across 3 seeds (`42, 123, 2024`), and omits Arms `(c)` and `(d)` from CUJ 1 / Verification Plan comparisons.
    3. G1 ("Match or surpass Official SLAMSS-IFS") violates `DEC-033`/`DEC-041`, which bar claiming equivalence from a non-significant $p$-value without a pre-specified equivalence margin.
*   **Required Fix:** Preregister a strict hierarchical testing plan (Primary: Arm `(e)` vs. `(a)` and `(e)` vs. `(f)` on `N = 47` seed-averaged Macro-F1 with Holm-Bonferroni; Secondary: per-stage F1s and Arms `(b)`/`(c)`/`(d)` with Holm-Bonferroni) and adopt `DEC-041` parity/superiority framing.

### B4. Quantizable Weight Arithmetic Error (`132,356` vs. `140,644`) & Unreachable `~310–350 KiB` INT8 Payload Target
*   **Sources:** `egm-structural` (B1), `egm-codebase-details` (M4), `egm-implementation-readiness` (M4), `egm-design` (N9), `egm-zoom-out` (N6)
*   **Affected Scope:** `docs/designs/exp029_ultradian_pyramidal_coupler.md` (L127, L154) vs. `experiments/STAGE2_FINAL_MODEL/benchmark_int8_edge.py` (L155–166).
*   **Findings:** `132,356 / 200,100 = 66.15%` mistakenly included `LayerNorm(64)` (`128` FP32 params) in `classifier_head` and omitted `hr_encoder.proj` (`4,160`), `acc_encoder.proj` (`4,160`), and `meta_proj.0` (`96`). The exact `nn.Linear + nn.GRU` count is **`140,644 / 200,100 = 70.29%`** (`59,456` FP32 params). Under `benchmark_int8_edge.py:165`, the actual Dynamic INT8 in-memory payload is **`369.60 KiB`**, exceeding the `~310–350 KiB` NFR target.
*   **Required Fix:** Correct the count to `140,644 / 200,100 (70.29%)` and update the NFR Dynamic INT8 payload target to `~370 KiB` (`369.6 KiB`).

### B5. `DEC-031` Capacity-Count Disclosure (`Delta%` vs. `163,940`) & Loose `223,268`-Param Control
*   **Sources:** `egm-zoom-out` (B2)
*   **Affected Scope:** `docs/designs/exp029_ultradian_pyramidal_coupler.md` (L20, L25, L116–127, L145) vs. `DECISIONS.md` (`DEC-031`, L537–570).
*   **Findings:** `DEC-031` removed arbitrary ceilings (`< 250k / < 300k`) and requires explicit $\Delta\%$ disclosure relative to `163,940` (`+22.06%` for Arm `(e)`, `+36.19%` for Arm `(b)`). Furthermore, Arm `(b)` (`223,268` params) is `+11.58%` larger than Arm `(e)` (`200,100`), whereas `d_temp=120` (`hidden=60/dir`) yields `207,284` params (`+3.59%` vs. Arm `(e)`).
*   **Required Fix:** Add the mandatory $\Delta\%$ vs. `163,940` column to Section 5 and justify the `+22.06%` capacity increase under `DEC-031` (or tighten Arm `(b)` / slim Arm `(e)`).

---

## 2. Majors (Should Resolve Before Coding)

1. **M1 — Unscoped `EXP-020` `FT-2` / `GATE_tau` Adaptation Protocol (`CUJ 3`):** Missing script reference, adaptation LR/epochs/`tau`, frozen vs. unfrozen parameter groups (`FiLM`, `fine_gru`, `coarse_gru`, `classifier_head`), and runtime budget (`138` runs). *(`implementation-readiness` B3, `zoom-out` M5)*
2. **M2 — `UltradianPyramidalCoupler.forward()` Return Dict & `seq_lens=None` Contract:** Must specify the 14-key `Dict[str, torch.Tensor]` return format (`out["logits"]` for `exp018_inference.py:95, 139`) and the `seq_lens=None` fallback required by `benchmark_int8_edge.py:207`. *(`codebase-details` M3)*
3. **M3 — `EXP-027` / `EXP-028` Cache Isolation, Arm Naming, & Ensemble Calibration Order:** `run_exp028_sleepaccel_bland_altman.py` globs all `*.npz` in `exp027_dir/cache/predictions` and splits on `"__"`. Must isolate `EXP-029` cache under `experiments/EXP-029/sleepaccel_cache/`, specify `mean_single_model` vs. `prob_ensemble` EM order, and note unscored (`y == -100`) PSG boundary epochs. *(`implementation-readiness` M6, `design` M7/N12)*
4. **M4 — Redundant `Linear(96->96)` Projection Before Coarse `BiGRU`:** A bare `Linear(96->96)` without nonlinearity before `nn.GRU` is algebraically redundant (`9,312` params). Either insert `LayerNorm + GELU + Dropout` or remove it. *(`design` M6)*
5. **M5 — Reuse `ConventionalCNNBiGRU(d_temp=128)` & Specify `build_exp029_model` `arm_id`s:** `ConventionalCNNBiGRU(d_temp=128)` in `exp013_models.py:49-57` already produces `223,268` params without a duplicate class; document exact `arm_id` strings. *(`implementation-readiness` M7, `codebase-details` N7)*

---

## 3. Nits (Cleanup & Clarity)

1. **N1 — Module 2 Heading Parameter Mismatch (`125,952` vs. `125,760`):** Line 81 includes `temp_norm` (`192`); align to `125,760` to match Section 5. *(`structural` N1, `design` N10, `zoom-out` N6, `codebase-details` N6, `implementation-readiness` N8)*
2. **N2 — "`1 Hz` FiLM" Naming (`(B*T, 64, 7)` is `~0.23 Hz`):** Rename to "Sub-Epoch (`~0.23 Hz`) Pre-Pooling FiLM" or "Intra-Epoch Pre-Pooling FiLM" to avoid reviewer pushback. *(`design` N8)*
3. **N3 — `nn.GRU(num_layers=1)` Dropout & FiLM Zero-Init:** Set `dropout=0.0` on 1-layer `nn.GRU`s (avoiding PyTorch warnings) and explicitly zero-init both `weight` and `bias` in `IntraEpochFiLM`. *(`zoom-out` N7, `implementation-readiness` N9/N10)*
4. **N4 — Dual-GPU Split Typo (`9` vs. `7` runs) & Manifest Path Shorthand:** Fix `9 runs` typo in NFR to `7 runs` (`cuda:0`) + `8 runs` (`cuda:1`), and write full `experiments/EXP-014/EXP014_MANIFESTS/fold_1..5.json`. *(`structural` N3, `implementation-readiness` N11, `codebase-details` N7)*
5. **N5 — Unexplained `N = 46` vs. `N = 47` in `FT-2` / `GATE_tau`:** Note the 1-subject exclusion (single-night subject ineligible for multi-night `FT-2`). *(`structural` N2)*
6. **N6 — `repeat_interleave(4)` Phase Shift & Unbatched `inner_val` Throughput:** Note `w = N - 1200` 4-epoch phase shift and confirm `SlidingWindowPredictor` validation batching. *(`design` N11, `implementation-readiness` N12)*

---

## 4. Verified Exact Items
*   **Parameter Math (Arms `(a)`–`(f)`):** `163,940`, `223,268`, `172,260`, `191,780`, `200,100`, and `5,465,768` all verified exact.
*   **Pre-Pool Encoder Shapes:** `(B*T, 64, 7)` with `29,568` params/stream verified against `sleep_adaptive_net.py:79-138`.
*   **Dual-GPU Hardware Feasibility:** Peak VRAM `< 300–450 MB` at `B = 4, L = 1200` and `~25–45 min` per 15-run arm on 2x RTX 3060 12GB verified feasible.

---

## 5. Traceability Matrix

| Consolidated ID | Severity | Raw Reviewer Findings |
|---|---|---|
| **B1** | Blocker | `design` B1, B2, M3; `zoom-out` B1; `implementation-readiness` B2; `codebase-details` M2; `structural` M2 |
| **B2** | Blocker | `implementation-readiness` B1, M5; `zoom-out` M3; `structural` M1; `codebase-details` M1, M5 |
| **B3** | Blocker | `structural` B2; `design` M4; `zoom-out` M4 |
| **B4** | Blocker | `structural` B1; `codebase-details` M4; `implementation-readiness` M4; `design` N9; `zoom-out` N6 |
| **B5** | Blocker | `zoom-out` B2 |
| **M1** | Major | `implementation-readiness` B3; `zoom-out` M5 |
| **M2** | Major | `codebase-details` M3 |
| **M3** | Major | `implementation-readiness` M6; `design` M7, N12 |
| **M4** | Major | `design` M6 |
| **M5** | Major | `implementation-readiness` M7; `codebase-details` N7 |
| **N1–N6** | Nit | `structural` N1–N3; `design` N8, N10–N11; `zoom-out` N6–N7; `codebase-details` N6–N7; `implementation-readiness` N8–N12 |
