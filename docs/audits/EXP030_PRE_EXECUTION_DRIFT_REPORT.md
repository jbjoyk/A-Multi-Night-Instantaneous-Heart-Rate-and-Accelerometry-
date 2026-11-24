# EXP-030 Pre-Execution Drift & Readiness Report

* **Date:** 2026-11-24
* **Design Spec:** `docs/designs/exp030_cnn_bigru_capacity_pareto.md`
* **Verdict:** **`GO`** (`0` Blocking, `0` Major drift)

---

## 1. Executive Summary

All 5 planned `EXP-030` modules have been implemented strictly matching `docs/designs/exp030_cnn_bigru_capacity_pareto.md`. Pre-flight unit tests (`17/17` passed), the `p2_164k` `LogitStitchingSlidingWindowPredictor` inference-only pass over the 15 frozen `DEC-037` checkpoints (`64.7907%`, `0.0000 pp` max subject diff vs. `DEC-037`), and 2-epoch GPU smoke dry-runs on both `p1_100k` (`343.1 MB` VRAM) and `p6_1051k` (`996.4 MB` VRAM) all succeeded with zero errors.

---

## 2. Spec-to-Code Parity & Drift Audit

| Component | Planned Path | Status | Drift Severity | Verification |
|---|---|---|---|---|
| Model Factory | `experiments/EXP-030/exp030_models.py` | Implemented | `0` Blocking / `0` Major | `ScaledModalityEncoder`, `ScaledConventionalCNNBiGRU`, `build_exp030_model`, exact 14-key `forward()` contract, bit-for-bit `p2_164k` parity with `ConventionalCNNBiGRU()`. |
| CV Driver | `experiments/EXP-030/run_exp030_cv.py` | Implemented | `0` Blocking / `0` Major | Reuses `EXP-029` helpers (`verify_manifests_and_splits`, `load_all_records_by_subject`, `compute_stage_loss_fp32`, `LogitStitchingSlidingWindowPredictor`); supports `--grid-points`, `--folds`, `--seeds`, `--device`, `--eval-p2-manifest`, `--aggregate`, `--smoke`. |
| Elbow Selector | `experiments/EXP-030/select_exp030_elbow.py` | Implemented | `0` Blocking / `0` Major | Exact `<= 0.20 pp` & `>= 90%` gain threshold $T$ + `<= 0.50 pp` degenerate guard; writes `elbow_selection.json`. |
| Downstream Suite | `experiments/EXP-030/eval_exp030_downstream.py` | Implemented | `0` Blocking / `0` Major | Restricted to `star_tag` with `p2_164k` fallback & self-comparison guard; Pillars 1–4 implemented (`em_off` only on SleepAccel, nested per-fold `TAU_GRID` via lightweight state-dict clone). |
| Pre-Flight Tests | `tests/test_exp030_preflight.py` | Implemented | `0` Blocking / `0` Major | `17/17` tests passing (`pytest`). |

---

## 3. Exact Parameter Count, Dynamic INT8 Payload & MMACs Verification

| Grid Tag | `d_enc` | `d_temp` | `head_hidden` | Total Params (Spec / Actual) | Quantizable (`Linear`+`GRU`) | Dynamic INT8 Payload (KiB) | MMACs (`L=1200`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `p1_100k` | 48 | 76 | 48 | `98,940` / **`98,940`** | `69,796` | `182.00` | `456.1` |
| `p2_164k` | 64 | 96 | 64 | `163,940` / **`163,940`** | `112,804` | `309.91` | `794.1` |
| `p3_268k` | 80 | 128 | 64 | `268,260` / **`268,260`** | `188,996` | `494.19` | `1,252.8` |
| `p4_404k` | 96 | 160 | 96 | `403,524` / **`403,524`** | `289,924` | `726.88` | `1,822.3` |
| `p5_666k` | 128 | 200 | 128 | `665,556` / **`665,556`** | `464,964` | `1,237.63` | `3,172.9` |
| `p6_1051k` | 160 | 256 | 128 | `1,051,492` / **`1,051,492`** | `739,364` | `1,941.29` | `4,966.8` |

---

## 4. Integrity & Pre-Flight Execution Results

- **Authoritative Manifest SHA-256 (`review_package/fold_manifests.csv`):** `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf` (prefix `36d84020` verified).
- **Historical & Protected Files Modified:** `0` (`exp013_models.py`, `sleep_adaptive_net.py`, `PREREG_EXP-030_WIDTH_SWEEP.md`, and `experiments/EXP-018`–`EXP-029` untouched).
- **Pytest (`tests/test_exp030_preflight.py`):** **`17 passed` in `2.09s`**.
- **`p2_164k` Stitcher-Parity Eval (`--eval-p2-manifest --device cuda:0`):**
  - `N = 47` subjects, Macro-F1 = **`64.7907% ± 7.45%`**, elapsed `22.1s`.
  - Cross-check vs. `DEC-037` (`final_model_oof_subject_metrics.csv`): `max_abs_diff = 0.0000 pp`.
- **2-Epoch Smoke Dry-Runs (`--smoke`):**
  - `p1_100k` (`cuda:0`, F1 S42): Peak VRAM **`343.1 MB`**, `7.3s`, finite loss (`1.2991 -> 1.2236`), Test F1 `30.98%`.
  - `p6_1051k` (`cuda:1`, F1 S42): Peak VRAM **`996.4 MB`** (< 8.5% of 12 GB RTX 3060), `11.0s`, finite loss (`1.2408 -> 1.1310`), Test F1 `47.22%`.
- **Estimated 75-Run Dual-GPU Runtime:**
  - `cuda:0` (`p1_100k`, `p3_268k`, `p4_404k` — 45 runs): ~`1.5–2.0 hours`.
  - `cuda:1` (`p5_666k`, `p6_1051k` — 30 runs): ~`1.5–2.0 hours`.
  - Total wall-clock time (concurrent): **`~2.0 hours`**.

---

## 5. Dual-GPU Launch Commands (For User Execution)

```bash
# Terminal 1 (GPU 0: p1_100k, p3_268k, p4_404k — 45 runs)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/EXP-030/run_exp030_cv.py \
  --grid-points p1_100k p3_268k p4_404k --folds 1 2 3 4 5 --seeds 42 123 2024 --device cuda:0

# Terminal 2 (GPU 1: p5_666k, p6_1051k — 30 runs)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/EXP-030/run_exp030_cv.py \
  --grid-points p5_666k p6_1051k --folds 1 2 3 4 5 --seeds 42 123 2024 --device cuda:0
```

*(Note: When `CUDA_VISIBLE_DEVICES=1` is set, PyTorch maps physical GPU 1 to logical `cuda:0`.)*
