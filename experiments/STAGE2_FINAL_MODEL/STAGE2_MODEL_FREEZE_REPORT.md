# Stage 2: Final Model Freeze Verification Report

- **Model Release:** `SleepStagingFinalModel_v1.0` (`ConventionalCNNBiGRU`)
- **Verification Timestamp (UTC):** `2026-11-21T02:02:16Z`
- **Exact Trainable Parameters:** `163,940`
- **Weight Footprint Only:** `640.39 KiB` (FP32) | `320.2 KiB` (FP16) | `160.1 KiB` (INT8)
- **Compression vs. SLAMSS-IFS:** `4.06x` (paper text 666k) | `33.34x` (official repo 5.47M)
- **Checkpoints Verified:** `15 / 15` (5 folds x 3 seeds)
- **Max Fold-Pooled Parity Diff vs. Locked EXP-018 JSON:** `0.000099` (Threshold: `<= 0.000500`)
- **Max Per-Subject Parity Diff vs. Locked EXP-018 JSON:** `0.000655` (Threshold: `<= 0.005000`)
- **Freeze Seal Status:** **FROZEN & CLEARED FOR STAGE 3 (TSK-08)**

---

## 1. 47-Subject Unweighted Participant-Mean OOF Reference Summary (N = 47)

> **Aggregation Note:** Metrics below are unweighted participant means ± between-participant SD across `N = 47` independent subjects (each averaged across seeds `42, 123, 2024`), matching the Stage 3 paired comparative estimand (`PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md` §4.1). Locked EXP-018 pooled-epoch 3-seed OOF Macro-F1 (`0.6479 ± 0.0053`) pools epochs across all subjects prior to averaging over `N = 3` seeds.

| Metric | Participant Mean | Between-Participant SD |
| :--- | :---: | :---: |
| **Macro-F1** | **64.79%** | 7.45 pp |
| **Accuracy** | 66.18% | 6.79 pp |
| **Weighted-F1** | 65.96% | 6.73 pp |
| **Cohen's Kappa** | 0.5078 | 0.1011 |
| **Wake F1** | 62.70% | 11.05 pp |
| **Light F1** | 66.61% | 5.25 pp |
| **Deep F1** | 66.76% | 12.31 pp |
| **REM F1** | 63.09% | 13.11 pp |

---

## 2. Cryptographic Checkpoint Registry (SHA-256)

| Fold | Seed | SHA-256 (First 16 Hex) | Fold Parity Diff | Max Subject Parity Diff | Status |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 42 | `1bbc65b1ffa67d01...` | `0.000016` | `0.000334` | `FROZEN_AND_PARITY_VERIFIED` |
| 1 | 123 | `4e4af4fa325d90ae...` | `0.000057` | `0.000349` | `FROZEN_AND_PARITY_VERIFIED` |
| 1 | 2024 | `43706d6efb84733e...` | `0.000070` | `0.000232` | `FROZEN_AND_PARITY_VERIFIED` |
| 2 | 42 | `6ff162e5f8749a69...` | `0.000028` | `0.000655` | `FROZEN_AND_PARITY_VERIFIED` |
| 2 | 123 | `8d147b36854823ac...` | `0.000005` | `0.000349` | `FROZEN_AND_PARITY_VERIFIED` |
| 2 | 2024 | `b73890666795c619...` | `0.000097` | `0.000277` | `FROZEN_AND_PARITY_VERIFIED` |
| 3 | 42 | `ed275dc8c48339c5...` | `0.000028` | `0.000323` | `FROZEN_AND_PARITY_VERIFIED` |
| 3 | 123 | `da15d322a74c27e0...` | `0.000025` | `0.000265` | `FROZEN_AND_PARITY_VERIFIED` |
| 3 | 2024 | `5c0de015a3f6d009...` | `0.000059` | `0.000350` | `FROZEN_AND_PARITY_VERIFIED` |
| 4 | 42 | `4b1efa83d67aa3f0...` | `0.000013` | `0.000319` | `FROZEN_AND_PARITY_VERIFIED` |
| 4 | 123 | `261bab7bdb16ca18...` | `0.000013` | `0.000496` | `FROZEN_AND_PARITY_VERIFIED` |
| 4 | 2024 | `4c94de9c45136410...` | `0.000003` | `0.000020` | `FROZEN_AND_PARITY_VERIFIED` |
| 5 | 42 | `c34cea492afb1221...` | `0.000099` | `0.000390` | `FROZEN_AND_PARITY_VERIFIED` |
| 5 | 123 | `cab7b0b9a14a65aa...` | `0.000025` | `0.000200` | `FROZEN_AND_PARITY_VERIFIED` |
| 5 | 2024 | `3630bc53c2412045...` | `0.000092` | `0.000282` | `FROZEN_AND_PARITY_VERIFIED` |
