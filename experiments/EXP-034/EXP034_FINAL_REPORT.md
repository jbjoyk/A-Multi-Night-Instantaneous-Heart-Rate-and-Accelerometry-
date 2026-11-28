# EXP-034 Final Report: 500-Epoch Convergence Audit on Compact CNN-BiGRU Widths

- **Preregistration:** `docs/designs/PREREG_EXP-034_500EPOCH_CONVERGENCE.md` (lock commit `c7d2d19e34563056a2d2fe07e6c53db879ab050e`)
- **Gating Rule (`P190`–`W250` 500-epoch extension required?):** **YES**

## 1. Paired Convergence Results (`500ep - 100ep`, $N = 47$)

| Arm | Params | 100-ep F1 (%) | 500-ep F1 (%) | Mean $\Delta$ (pp) | Median $\Delta$ (pp) | 95% CI | W/L/T | $p_{\text{raw}}$ (2s) | $p_{\text{Holm}}$ (2s) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `W050_500ep` | 43,108 | 65.61 | 66.24 | +0.63 | +0.88 | [+0.09, +1.09] | 34/13/0 | 6.9622e-04 | 2.7849e-03 | **STATISTICALLY DETECTABLE BUT BELOW PRACTICAL THRESHOLD** |
| `W075_500ep` | 93,796 | 66.26 | 66.40 | +0.14 | +0.12 | [-0.15, +0.44] | 25/22/0 | 1.9678e-01 | 5.9035e-01 | **NOT SUPPORTED** |
| `p1_100k_500ep` | 98,940 | 66.67 | 66.68 | +0.01 | +0.00 | [-0.17, +0.18] | 14/14/19 | 9.4554e-01 | 1.0000e+00 | **NOT SUPPORTED** |
| `W100_500ep` | 163,940 | 66.68 | 66.70 | +0.02 | +0.00 | [-0.11, +0.15] | 10/8/29 | 6.7907e-01 | 1.0000e+00 | **NOT SUPPORTED** |

## 2. Per-Arm Training Convergence Logs (15 runs/arm)

| Arm | Completed Epochs Mean [Range] | Best Epoch Mean [Range] | Early Stopped | Hit 500 Ceiling | Mean Wall (s) | Max VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| `W050_500ep` | 184.20 [113, 301] | 159.20 [88, 276] | 15/15 | 0/15 | 565.6 | 242.8 |
| `W075_500ep` | 130.27 [78, 211] | 105.27 [53, 186] | 15/15 | 0/15 | 430.0 | 344.5 |
| `p1_100k_500ep` | 118.87 [81, 154] | 93.87 [56, 129] | 15/15 | 0/15 | 414.0 | 344.5 |
| `W100_500ep` | 112.20 [65, 141] | 87.20 [40, 116] | 15/15 | 0/15 | 413.6 | 433.1 |

## 3. Diagnostic Mixed-Budget `H-SCALE` Sensitivity Check (Page's $L$, $N = 47$)

- **9-Width (`W050_500ep`, `W075_500ep`, `W100_500ep` + 100-ep `P190`–`W250`):** $L = 10395.0$, $z_L = -1.238$, $p_{\text{one}} = 8.9209e-01$, $p_{\text{two}} = 2.1583e-01$
- **10-Width (inserting `p1_100k_500ep`):** $L = 13931.0$, $z_L = -1.520$, $p_{\text{one}} = 9.3570e-01$, $p_{\text{two}} = 1.2860e-01$
