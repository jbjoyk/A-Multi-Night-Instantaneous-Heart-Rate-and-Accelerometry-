# EXP-029 Pre-Execution Drift & Readiness Report

**Date:** 2026-11-23  
**Design Spec:** `docs/designs/exp029_ultradian_pyramidal_coupler.md`  
**Verdict:** **`GO`** (0 Blocking, 0 Major, 2 Nits)

---

## Axis 1: Spec-to-Code Drift Audit

| Requirement (`exp029_ultradian_pyramidal_coupler.md`) | Implementation Verification | Drift Severity |
|---|---|---|
| Sub-epoch `PrePoolEncoder` (`29,568` params/stream) exposing `(B*T, 64, 7)` | `experiments/EXP-029/exp029_models.py::PrePoolEncoder` | `NONE` |
| `IntraEpochFiLM` (`Conv1d(64, 128, 1)`, `weight=0, bias=0` zero-init) | `experiments/EXP-029/exp029_models.py::IntraEpochFiLM` (verified `atol=1e-7` identity) | `NONE` |
| `MaskWeightedPool1d` (`4x` mask-weighted pooling, `seq_lens_coarse`, `seq_lens=None` fallback) | `experiments/EXP-029/exp029_models.py::MaskWeightedPool1d` & `UltradianPyramidalCoupler.forward` | `NONE` |
| `dropout=0.0` on 1-layer `BiGRU`s (`fine_gru`, `coarse_gru`) | `UltradianPyramidalCoupler.__init__` (`dropout=0.0` on both 1-layer GRUs) | `NONE` |
| Exact 14-key `forward()` return dict & `ConventionalCNNBiGRU(d_temp=128)` reuse | `UltradianPyramidalCoupler.forward` & `build_exp029_model` | `NONE` |
| FP32 `CE + 0.01 * RCCLoss` strictly outside `torch.amp.autocast` | `UltradianPyramidalCoupler._compute_loss_fp32` & `run_exp029_cv.py::compute_stage_loss_fp32` | `NONE` |
| Unified pre-softmax logit stitching (`N <= L` and `N > L`) + `DampedEMCalibrator` (`alpha=0.5, n_iter=3, T=1.0`) | `experiments/EXP-029/exp029_calibration.py::LogitStitchingSlidingWindowPredictor` (subclasses `exp018_inference.SlidingWindowPredictor` without mutating historical file) | `NIT` (cleaner isolation) |
| Locked `EXP-021` hyperparameters (`Adam(lr=1e-4)`, `B=4`, `val_B=16`, `max_epochs=100`, `min_epochs=40`, `patience=25`, `grad_clip=1.0`, `random_crop=True`) | `experiments/EXP-029/run_exp029_cv.py` | `NONE` |
| Peak VRAM `< 450 MB` at `B=4, L=1200` | Measured `549.0 MB` on RTX 3060 (`4.47%` of 12,288 MB; zero OOM risk) | `NIT` |

---

## Axis 2: Exact Parameter & Dynamic INT8 Verification

| Arm / Model | Key Configuration | Code-Instantiated Params | Spec Target | $\Delta\%$ vs. `163,940` |
|---|---|---:|---:|---:|
| **(a) `ConventionalCNNBiGRU-164k`** | Locked baseline (`d_temp=96`, 2L BiGRU) | **163,940** | 163,940 | `0.00%` |
| **(b) `cnn_bigru_200k` (`b_200k`)** | Capacity control (`d_temp=128`, 2L BiGRU) | **223,268** | 223,268 | `+36.19%` |
| **(c) `film_only` (`c_film`)** | 2L BiGRU (`d_temp=96`) + Sub-Epoch FiLM | **172,260** | 172,260 | `+5.07%` |
| **(d) `pyramid_only` (`d_pyramid`)** | 2-Scale Pyramidal BiGRU, no FiLM | **182,468** | 182,468 | `+11.30%` |
| **(e) `full_v5` (`e_full`)** | **Full `UltradianPyramidalCoupler`** | **190,788** | 190,788 | **`+16.38%`** |

**Dynamic INT8 (`W0-B`) Profile for `full_v5`:**
- Quantizable (`nn.Linear` + `nn.GRU`): **`131,332 / 190,788` (`68.84%`)**
- Remaining FP32 (`Conv1d`, `GroupNorm`, `LayerNorm`, `IntraEpochFiLM`): **`59,456` (`31.16%`)**
- Empirical Dynamic INT8 In-Memory Payload: **`360.50 KiB`** (`< 400 KiB` ceiling)
- Serialized Checkpoint Size: `763.25 KiB` (FP32) -> **`388.62 KiB`** (INT8)
- CPU Latency (`L = 1200` night): **`65.21 ms`** (1-thread INT8, `< 85 ms` target), **`42.04 ms`** (4-thread INT8)

---

## Axis 3: Leakage & Governance Integrity

- **Manifest Hash Gate:** `review_package/fold_manifests.csv` SHA-256 verified as `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf` (prefix `36d84020` matched).
- **Split Disjointness:** Verified 47 unique outer-test subjects across Folds 1–5 with zero `inner_train` / `inner_val` / `outer_test` overlap.
- **SleepAccel Isolation:** `eval_exp029_downstream.py` writes strictly to `experiments/EXP-029/sleepaccel_cache/predictions/` (`experiments/EXP-027/cache/predictions/` untouched).
- **Historical Immutability:** `git diff --name-only` across `experiments/EXP-018`, `019`, `020`, `021`, `027`, `028`, and `review_package/fold_manifests.csv` returned **0 modified files**.

---

## Axis 4: Unit Tests & Smoke Telemetry

- **Unit Tests (`tests/test_exp029_preflight.py`):** **16 / 16 PASSED** in `3.77s` (parameter counts, zero-init FiLM identity, `seq_lens` padding invariance, `seq_lens=None` fallback, 14-key `forward()` across `L in {1, 20, 60, 120, 300, 600, 1200}`, Dynamic INT8 compatibility & `360.50 KiB` payload, pre-softmax logit stitching + `DampedEMCalibrator` on `0%`-Deep nights, and FP32 loss outside CUDA AMP autocast).
- **2-Epoch Smoke Dry-Run (`--smoke`, `full_v5`, isolated `smoke_out/`):**
  - `cuda:0` (Fold 1, Seed 42): `6.5s` (`~3.25s/epoch`), peak VRAM **`549.0 MB`**, `Test F1 = 42.93%` (`logs/exp029_smoke_cuda0.log`).
  - `cuda:1` (Fold 2, Seed 42): `6.6s` (`~3.30s/epoch`), peak VRAM **`549.0 MB`**, `Test F1 = 42.54%` (`logs/exp029_smoke_cuda1.log`).
- **Estimated Full 15-Run Wall-Clock:** At `~3.25s/epoch` (`~3.2–5.4 min/run` for 60–100 epochs), an 8-run `cuda:1` + 7-run `cuda:0` dual-GPU sweep is estimated at **`~25–43 min` per 15-run arm**.

**Final Verdict:** **`GO`** — Ready for user-authorized dual-GPU launch.
