# EXP-018: Context-Horizon Scaling Curve — Phase 1 Final Report
- **Experiment ID**: EXP-018
- **Phase**: Phase 1 (Horizon Discovery & Gating)
- **Execution Timestamp**: 2026-11-14 17:43:22
- **Git Commit**: `17260f263380d64b80e414599253595b708402a5`
- **Execution Device**: `cuda:1`
- **Runs Completed**: 21 / 21 (100% success)

---
## 1. Executive Summary & Gating Decision

- **Primary Contrast**: $L=600$ vs $L=60$
- **Primary Estimand ($\Delta \text{Macro-F1}$)**: **+4.27 pp** [95% CI: **+2.53 pp**, **+6.46 pp**]
- **Frozen Decision Category**: `SUPPORTED (Context Benefit)`
- **Decision Interpretation**: Long context significantly improves wearable sleep staging beyond local windows. Phase 2 CV is AUTHORIZED.
- **Phase 2 Authorization**: **AUTHORIZED**
- **Phase 1b Circadian Ablation**: **PERMITTED**

---
## 2. Context-Horizon Performance Curve (Mean ± SD across 3 Seeds)

| Context Length ($L$) | Window (min) | Macro-F1 | Accuracy | Cohen's $\kappa$ | MCC | Wake F1 | Light F1 | Deep F1 | REM F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $L=1$ | 0.5m | **0.3654** $\pm$ 0.0057 | 48.56% | 0.1710 | 0.2066 | 0.3557 | 0.6269 | 0.2003 | 0.2786 |
| $L=20$ | 10.0m | **0.5688** $\pm$ 0.0150 | 58.93% | 0.3859 | 0.3949 | 0.5764 | 0.6376 | 0.5266 | 0.5346 |
| $L=60$ | 30.0m | **0.6117** $\pm$ 0.0025 | 62.30% | 0.4451 | 0.4498 | 0.6165 | 0.6527 | 0.5774 | 0.6002 |
| $L=120$ | 60.0m | **0.6100** $\pm$ 0.0055 | 62.34% | 0.4470 | 0.4506 | 0.5761 | 0.6457 | 0.6023 | 0.6160 |
| $L=300$ | 150.0m | **0.6321** $\pm$ 0.0089 | 64.60% | 0.4805 | 0.4842 | 0.5887 | 0.6632 | 0.6207 | 0.6557 |
| $L=600$ | 300.0m | **0.6529** $\pm$ 0.0078 | 66.49% | 0.5114 | 0.5194 | 0.6044 | 0.6773 | 0.6890 | 0.6410 |
| $L=1200$ | 600.0m | **0.6993** $\pm$ 0.0139 | 70.04% | 0.5693 | 0.5728 | 0.6990 | 0.7058 | 0.7354 | 0.6572 |

---
## 3. Parametric Saturation Curve Fit

Model formulation: $F_1(L) = A - B e^{-k L}$
- **Asymptotic Ceiling ($A$)**: `0.6435`
- **Dynamic Span ($B$)**: `0.2953`
- **Decay Rate ($k$)**: `0.064716`
- **95% Saturation Threshold ($L^*$)**: **46.3 epochs** (23.1 minutes)
- **Goodness of Fit ($R^2$)**: `0.9258`

---
## 4. Clinical Polysomnography Metrics

| Context Length ($L$) | TST MAE (min) | SOL MAE (min) | WASO MAE (min) | Sleep Efficiency MAE (%) |
| :--- | :---: | :---: | :---: | :---: |
| $L=1$ | 44.30m | 26.73m | 13.88m | 10.67% |
| $L=20$ | 33.02m | 18.85m | 13.35m | 8.02% |
| $L=60$ | 31.04m | 16.61m | 13.91m | 7.62% |
| $L=120$ | 38.12m | 18.68m | 14.22m | 9.27% |
| $L=300$ | 35.65m | 19.30m | 15.36m | 8.73% |
| $L=600$ | 36.61m | 17.14m | 14.11m | 8.91% |
| $L=1200$ | 27.14m | 12.45m | 13.02m | 6.66% |

---
## 5. All 21 Individual Run Artifacts

| Run ID | Context ($L$) | Seed | Macro-F1 | Accuracy | Cohen's $\kappa$ | Runtime (s) | Artifact Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `L1_s42` | 1 | 42 | 0.3730 | 48.66% | 0.1771 | 281.6s | [`phase1_results/run_L0001_seed42.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0001_seed42.json) |
| `L1_s123` | 1 | 123 | 0.3640 | 48.57% | 0.1701 | 274.7s | [`phase1_results/run_L0001_seed123.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0001_seed123.json) |
| `L1_s2024` | 1 | 2024 | 0.3592 | 48.45% | 0.1657 | 274.3s | [`phase1_results/run_L0001_seed2024.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0001_seed2024.json) |
| `L20_s42` | 20 | 42 | 0.5682 | 58.80% | 0.3819 | 200.1s | [`phase1_results/run_L0020_seed42.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0020_seed42.json) |
| `L20_s123` | 20 | 123 | 0.5508 | 57.84% | 0.3589 | 197.2s | [`phase1_results/run_L0020_seed123.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0020_seed123.json) |
| `L20_s2024` | 20 | 2024 | 0.5875 | 60.15% | 0.4170 | 196.1s | [`phase1_results/run_L0020_seed2024.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0020_seed2024.json) |
| `L60_s42` | 60 | 42 | 0.6135 | 62.34% | 0.4445 | 143.6s | [`phase1_results/run_L0060_seed42.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0060_seed42.json) |
| `L60_s123` | 60 | 123 | 0.6082 | 62.55% | 0.4434 | 145.5s | [`phase1_results/run_L0060_seed123.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0060_seed123.json) |
| `L60_s2024` | 60 | 2024 | 0.6135 | 62.01% | 0.4475 | 155.0s | [`phase1_results/run_L0060_seed2024.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0060_seed2024.json) |
| `L120_s42` | 120 | 42 | 0.6147 | 62.95% | 0.4524 | 194.5s | [`phase1_results/run_L0120_seed42.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0120_seed42.json) |
| `L120_s123` | 120 | 123 | 0.6023 | 61.17% | 0.4365 | 190.2s | [`phase1_results/run_L0120_seed123.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0120_seed123.json) |
| `L120_s2024` | 120 | 2024 | 0.6130 | 62.90% | 0.4520 | 185.7s | [`phase1_results/run_L0120_seed2024.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0120_seed2024.json) |
| `L300_s42` | 300 | 42 | 0.6402 | 65.52% | 0.4924 | 149.9s | [`phase1_results/run_L0300_seed42.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0300_seed42.json) |
| `L300_s123` | 300 | 123 | 0.6363 | 64.55% | 0.4848 | 149.7s | [`phase1_results/run_L0300_seed123.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0300_seed123.json) |
| `L300_s2024` | 300 | 2024 | 0.6197 | 63.73% | 0.4642 | 150.4s | [`phase1_results/run_L0300_seed2024.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0300_seed2024.json) |
| `L600_s42` | 600 | 42 | 0.6540 | 67.46% | 0.5126 | 138.4s | [`phase1_results/run_L0600_seed42.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0600_seed42.json) |
| `L600_s123` | 600 | 123 | 0.6618 | 67.39% | 0.5252 | 138.6s | [`phase1_results/run_L0600_seed123.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0600_seed123.json) |
| `L600_s2024` | 600 | 2024 | 0.6429 | 64.61% | 0.4964 | 138.6s | [`phase1_results/run_L0600_seed2024.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L0600_seed2024.json) |
| `L1200_s42` | 1200 | 42 | 0.7133 | 71.38% | 0.5841 | 184.0s | [`phase1_results/run_L1200_seed42.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L1200_seed42.json) |
| `L1200_s123` | 1200 | 123 | 0.7044 | 70.75% | 0.5766 | 184.2s | [`phase1_results/run_L1200_seed123.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L1200_seed123.json) |
| `L1200_s2024` | 1200 | 2024 | 0.6803 | 68.00% | 0.5472 | 183.8s | [`phase1_results/run_L1200_seed2024.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/run_L1200_seed2024.json) |

---
## 6. Scientific Governance & Compliance

- **Model A Parameter Count**: 163,940 parameters verified across all 21 runs.
- **Compute Invariant**: $B \times L \times N_{\text{accum}} = 2,400$ verified across all 21 runs (7,680,000 total tokens per run).
- **Held-Out Test Set**: 11 test subjects strictly LOCKED; zero test epochs evaluated.
- **Reproducibility**: All random seeds, splits, batch streams, and raw predictions recorded.
