# EXP-031 Pre-Execution Drift & Readiness Report

* **Date:** 2026-11-24
* **Preregistration:** `PREREG_EXP-031_WIDTH_SWEEP.md` (Sections 3–9 locked and immutable)
* **Verdict:** **`GO`** (`0` Blocking, `0` Major drift)

---

## 1. Executive Summary

All 4 `EXP-031` modules and the pre-flight unit test suite have been implemented strictly matching `PREREG_EXP-031_WIDTH_SWEEP.md`. Pre-flight unit tests (`25/25` passed in `2.07s`), Section 5 CPU profiling across all 9 plain widths (`1T`/`4T` latency, Dynamic INT8 KiB, MMACs at `L=1200`), and 2-epoch GPU smoke dry-runs on `W050` (`cuda:0`, `242.4 MB` peak VRAM) and `W250` (`cuda:1`, `992.6 MB` peak VRAM) all succeeded with zero errors.

---

## 2. Spec-to-Code Parity & Drift Audit

| Component | Planned Path | Status | Drift Severity | Verification |
|---|---|---|---|---|
| Model Factory | `experiments/EXP-031/exp031_models.py` | Implemented | `0` Blocking / `0` Major | Reuses `ScaledConventionalCNNBiGRU` from `experiments/EXP-030/exp030_models.py` cleanly without mutating it; exact parameter counts for all 9 plain widths (`W050`..`W250`); `P190` = `190,940` (`+0.080%` vs. `full_v5` `190,788`); 5-arm `< 400 KiB` eligibility mask (`W050, W075, A, P190, B`). |
| Sweep CV Driver | `experiments/EXP-031/run_exp031_width_sweep.py` | Implemented | `0` Blocking / `0` Major | Locked recipe: `Adam(lr=1e-4, weight_decay=0.0)`, `B=4`, `val_B=16`, `max_epochs=100`, `min_epochs=40`, `patience=25`, `grad_clip=1.0`, FP32 `CE + 0.01*RCCLoss` outside autocast, `LogitStitchingSlidingWindowPredictor(L=1200)`; guards against retraining `A` or `B`; supports `--arms`, `--folds`, `--seeds`, `--device`, `--aggregate`, `--smoke`. |
| Confirmatory & Selection | `experiments/EXP-031/analyze_exp031_confirmatory.py` | Implemented | `0` Blocking / `0` Major | `H-ARCH` (`full_v5 - P190`, `seed=42` 10k bootstrap, `+0.75 pp` threshold); `H-SCALE` (Page's $L$ across 9 widths, one-sided & two-sided $p$, Holm-Bonferroni across `{H-ARCH, H-SCALE}`, directional verdict `L > mu_L` and Holm two-sided `p < 0.05`); Section 9 `1-SE` selection (`seed=42`) strictly over the 5 `< 400 KiB` eligible plain widths (excluding `V5`); Section 5 CPU profiling. |
| Downstream Eval | `experiments/EXP-031/eval_exp031_downstream.py` | Implemented | `0` Blocking / `0` Major | Reads `selected_backbone.json`; runs `W0-B` INT8 OOF parity, strictly zero-shot SleepAccel (`em_off` primary single-model & secondary 15-model ensemble with `seed=42` paired CIs alongside `A`, `full_v5`, `SLAMSS-IFS`), and `EXP-020` `FT-2` / nested `GATE_tau` (`N=46`) alongside `A` and `full_v5`. |
| Pre-Flight Tests | `tests/test_exp031_preflight.py` | Implemented | `0` Blocking / `0` Major | `25/25` tests passing (`pytest`). |

---

## 3. Section 5 CPU Profile & Parameter Verification (All 9 Plain Widths)

| Arm | `(d_enc, d_temp, head_hidden)` | Params (Spec / Actual) | FP32 KiB | Dynamic INT8 KiB | MMACs (`L=1200`) | 1T INT8 (ms) | 4T INT8 (ms) | Sec. 9 (`< 400 KiB`) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `W050` | `(32, 48, 32)` | `43,108` / **`43,108`** | `168.39` | `81.00` | `202.9` | `51.66` | `45.84` | **ELIGIBLE** |
| `W075` | `(48, 72, 48)` | `93,796` / **`93,796`** | `366.39` | `176.96` | `450.0` | `57.41` | `46.67` | **ELIGIBLE** |
| `A` | `(64, 96, 64)` | `163,940` / **`163,940`** | `640.39` | `309.91` | `794.1` | `79.25` | `57.31` | **ELIGIBLE** (reused) |
| `P190` | `(72, 100, 64)` | `190,940` / **`190,940`** | `745.86` | `375.16` | `984.2` | `74.20` | `54.97` | **ELIGIBLE** |
| `B` | `(64, 128, 64)` | `223,268` / **`223,268`** | `872.14` | `368.04` | `864.7` | `67.61` | `56.30` | **ELIGIBLE** (reused) |
| `W125` | `(80, 120, 80)` | `253,540` / **`253,540`** | `990.39` | `479.86` | `1,235.2` | `81.45` | `63.52` | INELIGIBLE |
| `W150` | `(96, 144, 96)` | `362,596` / **`362,596`** | `1,416.39` | `686.82` | `1,773.5` | `117.03` | `74.58` | INELIGIBLE |
| `W200` | `(128, 192, 128)` | `639,076` / **`639,076`** | `2,496.39` | `1,211.72` | `3,141.2` | `153.48` | `87.23` | INELIGIBLE |
| `W250` | `(160, 240, 160)` | `993,380` / **`993,380`** | `3,880.39` | `1,884.63` | `4,897.2` | `196.15` | `103.83` | INELIGIBLE |

---

## 4. Integrity & Pre-Flight Execution Results

- **Authoritative Manifest SHA-256 (`review_package/fold_manifests.csv`):** `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf` (prefix `36d84020`).
- **Historical & Protected Files Modified:** `0` (`exp013_models.py`, `sleep_adaptive_net.py`, `PREREG_EXP-031_WIDTH_SWEEP.md`, and `experiments/EXP-018`–`EXP-030` untouched).
- **Pytest (`tests/test_exp031_preflight.py`):** **`25 passed` in `2.07s`**.
- **2-Epoch Smoke Dry-Runs (`--smoke`):**
  - `W050` (`cuda:0`, Fold 1 Seed 42): Peak VRAM **`242.4 MB`**, `6.7s`, finite loss (`1.3503 -> 1.2560`), Test F1 `28.54%`, checkpoint saved to `experiments/EXP-031/smoke_out/checkpoints/W050/W050_L1200_fold1_seed42.pt`.
  - `W250` (`cuda:1`, Fold 1 Seed 42): Peak VRAM **`992.6 MB`** (~8.1% of 12 GB RTX 3060), `10.9s`, finite loss (`1.2294 -> 1.1301`), Test F1 `47.85%`, checkpoint saved to `experiments/EXP-031/smoke_out/checkpoints/W250/W250_L1200_fold1_seed42.pt`.

---

## 5. Dual-GPU Launch Commands (105 Runs — For User Execution)

```bash
# Terminal 1 (GPU 0: W050, W075, P190, W150 — 60 runs)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/EXP-031/run_exp031_width_sweep.py \
  --arms W050 W075 P190 W150 --folds 1 2 3 4 5 --seeds 42 123 2024 --device cuda:0

# Terminal 2 (GPU 1: W125, W200, W250 — 45 runs)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/EXP-031/run_exp031_width_sweep.py \
  --arms W125 W200 W250 --folds 1 2 3 4 5 --seeds 42 123 2024 --device cuda:0
```

*(Note: When `CUDA_VISIBLE_DEVICES=1` is set, PyTorch maps physical GPU 1 to logical `cuda:0`.)*
