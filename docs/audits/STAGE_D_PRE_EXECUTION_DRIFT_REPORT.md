# Stage D Pre-Execution Drift Report (DEC-037 Gate)

* **Date:** 2026-11-23
* **Governance Standard:** `DEC-037` (0 `BLOCKING` / 0 `MAJOR` drift items required for `GO`)
* **Verdict:** **`GO`** (1 `BLOCKING` drift caught & fixed during Axis 2 audit; 0 remaining `BLOCKING` / 0 `MAJOR` items)

---

## AXIS 1: Spec-to-Code Drift Table

| Script | Locked Spec | Verification Checks | Drift Classification | Notes |
|---|---|---|---|---|
| [`run_exp019b_heterogeneity_audit.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-019B/run_exp019b_heterogeneity_audit.py) | `exp019b_exp020_closure_spec.md` | Family 1 (`N=46`, `[:2]`), Family 2 (`N=47`, all nights), raw `hr.csv` BPM, `compute_bh_fdr()`, 10k bootstrap, Oldham + LOO `Bidslab34`. | `NONE` | 3/3 unit tests pass. |
| [`run_exp020_regularized_gated_adaptation.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-020/run_exp020_regularized_gated_adaptation.py) | `exp019b_exp020_closure_spec.md` | (a) Splits loaded strictly from `EXP014_MANIFESTS/fold_{f}.json`; (b) `lambda_KL=0.0` parity `4.47e-07` vs `adapt_supervised_ft`; (c) `restore_pristine_model()` resets `requires_grad=True` on Fold 4 `head_gru`; (d) strictly per-fold nested inner-val selection. | `NONE` | 5/5 unit tests pass. |
| [`prepare_sleepaccel.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-027/prepare_sleepaccel.py) | `exp027_exp028_sleepaccel_external_validation.md` | **Caught & fixed during Axis 2 audit:** removed non-BIDSleep raw `acc_ep_var >= 1e-6` filter that was dropping motionless sleep epochs (`n_qc_pass` recovered `20 -> 25`). Matches `sleepchannelnet_protocol_patch_v2_UPDATED.ipynb` Cell 16 `L108-L112` verbatim (`meta[:, 0]`, `[-10, 10]` clip, `t < 0` trim-before-scale). | `NONE` (post-fix) | 7/7 unit tests pass; pre-flight gate `PASS`. |
| [`run_exp027_sleepaccel_inference.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-027/run_exp027_sleepaccel_inference.py) | `exp027_exp028_sleepaccel_external_validation.md` | (d) Respects `--device` (`cuda:0` under `CUDA_VISIBLE_DEVICES=1`); `torch.no_grad()`; `trg_zeros` + `teacher_forcing_ratio=0.0`; `neutralize_circadian=True`; 15+15 checkpoints; `mean_single_model` + `prob_ensemble`; handoff `.npz` bundles. | `NONE` | Smoke-verified on `cuda:0` (`CUDA_VISIBLE_DEVICES=1`). |
| [`run_exp028_sleepaccel_bland_altman.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-027/run_exp028_sleepaccel_bland_altman.py) | `exp027_exp028_sleepaccel_external_validation.md` | (e) Calls `ClinicalHypnogramEvaluator.compute_night_metrics()` directly; both `mean_single_model` & `prob_ensemble`; raw tuples CSV; Bland-Altman 1999 `SE(LoA)`; proportional bias; Shapiro-Wilk + 2.5th–97.5th percentile LoA fallback. Placed under `experiments/EXP-027/` per Phase 4A directive. | `NIT` (script dir) | Functional parity `NONE`. |

---

## AXIS 2: BIDSleep vs. SleepAccel Distribution & QC Audit

### Side-by-Side Feature & Label Distribution

| Metric | BIDSleep (`prep_57738d534b`, 246 nights) | SleepAccel (`qc_clean`, `n_obs=25`) | SleepAccel (`all`, `n_obs=31`) |
|---|---:|---:|---:|
| **Raw ACC Norm Median (`g`)** | `~1.0` (`g`, gravity incl.) | `0.9979` | `0.9973` |
| **Raw IHR Median (`bpm`)** | `64.0` | `61.0` | `64.0` |
| **Std IHR `[med, IQR, min, max]`** | `[0.0000, 0.9680, -4.0781, 10.0]` | `[0.0000, 0.9482, -2.8514, 10.0]` | `[0.0000, 0.9509, -3.1547, 10.0]` |
| **Std ACC `[med, IQR, min, max]`** | `[0.0012, 0.6113, -10.0, 10.0]` | `[0.0013, 0.6464, -10.0, 10.0]` | `[0.0024, 0.6470, -10.0, 10.0]` |
| **`meta[:, 0]` (`freq`) `[med, IQR, min, max]`** | `[1.0986, 0.0000, 0.0000, 2.2687]` | `[1.0986, 0.1178, 0.0000, 2.0794]` | `[1.0986, 0.1178, 0.0000, 2.0794]` |
| **PSG Stage % (`W / L / D / R`)** | `10.83 / 44.20 / 18.91 / 26.06` | `8.24 / 57.74 / 11.08 / 22.94` | `8.63 / 56.62 / 12.69 / 22.07` |

### Excluded SleepAccel Subjects (`6 / 31` Excluded from `qc_clean`)

*(Note: During initial Axis 2 inspection, 5 subjects with `cov_acc >= 0.927` were flagged as `usable_frac < 0.60` due to a raw `acc_ep_var >= 1e-6` check not present in BIDSleep `prep_57738d534b`. Removing that check recovered all 5 subjects, leaving only the 6 genuine exclusions below.)*

| Subject ID | `n_epochs` | `cov_hr_mean` | `cov_mo_mean` | `usable_frac` | Unknown PSG Epochs | Exact Failure Reason |
|---|---:|---:|---:|---:|---:|---|
| `3509524` | `417` | `1.000` | `1.000` | `1.000` | `0 / 417` | `n_epochs < 600` (3.48 h PSG) |
| `4018081` | `510` | `0.998` | `0.974` | `0.973` | `0 / 510` | `n_epochs < 600` (4.25 h PSG) |
| `46343` | `567` | `1.000` | `1.000` | `0.977` | `13 / 567` | `n_epochs < 600` (4.73 h PSG) |
| `5132496` | `465` | `1.000` | `1.000` | `1.000` | `0 / 465` | `n_epochs < 600` (3.88 h PSG) |
| `759667` | `475` | `1.000` | `0.957` | `0.956` | `0 / 475` | `n_epochs < 600` (3.96 h PSG) |
| `7749105` | `960` | `0.797` | `0.136` | `0.124` | `15 / 960` | `usable_frac < 0.60` (Apple Watch ACC dropout: `cov_mo=0.136`) |

---

## AXIS 3: Checkpoint & Governance Integrity

* **Model A Checkpoints:** `15 / 15` verified against `experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json` SHA-256 digests.
* **EXP-021 Official SLAMSS-IFS Checkpoints:** `15 / 15` (`experiments/EXP-021/checkpoints/slamss_ifs_official_L1200_fold{1..5}_seed{42,123,2024}.pt`) verified present and loaded cleanly into `OfficialSLAMSSIFSWrapper`.
* **`.gitignore` Protection (`DEC-002`):** `git check-ignore -v` confirmed both `data/external/sleepaccel_v1.0.0` and `experiments/EXP-027/cache` are ignored.
* **Locked Artifacts (`git status`):** `0` modifications to any locked `EXP-018`, `EXP-019`, or `EXP-021` checkpoints, manifests, or CSVs.

---

## AXIS 4: 1-Subject Smoke Dry-Runs & Telemetry

* **GPU Pre-Check (`nvidia-smi`):**
  * `GPU 0` (NVIDIA GeForce RTX 3060): `50°C`, `21%` util, `1482 / 12288 MiB` (< 80°C, healthy).
  * `GPU 1` (NVIDIA GeForce RTX 3060): `39°C`, `0%` util, `13 / 12288 MiB` (< 80°C, idle).
* **EXP-020 Dry-Run (`CUDA_VISIBLE_DEVICES=0`, `Bidslab08`, Fold 1, Seed 42):**
  * Runtime: `2.35 s` | Peak VRAM: `253.6 MiB`
  * `lambda_KL=0.0` max weight diff vs. `adapt_supervised_ft`: `4.47e-07` (`PASS`)
* **EXP-027 + EXP-028 Dry-Run (`CUDA_VISIBLE_DEVICES=1`, Subject `1066528`, Fold 1, Seed 42, output in `/tmp/exp027_smoke` & `/tmp/exp028_smoke`):**
  * Runtime: `1.86 s` | Peak VRAM: `215.0 MiB`
  * Output schema verified: `['subject_id', 'qc_pass', 'arm', 'eval_mode', 'metric', 'ref_val', 'pred_val', 'diff', 'mean']` (`72` rows for 3-copy test).

---

## Final Gate Verdict: **`GO`**

All 4 axes pass with **0 `BLOCKING`** and **0 `MAJOR`** remaining drift items. Authorized to launch full `EXP-019B`, `EXP-020`, and `EXP-027` + `EXP-028` runs.
