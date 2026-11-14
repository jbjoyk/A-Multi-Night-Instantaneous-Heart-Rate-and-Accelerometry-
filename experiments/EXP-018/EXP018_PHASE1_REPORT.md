# EXP-018: Context-Horizon Scaling Curve — Phase 1 Final Report
- **Experiment ID**: EXP-018
- **Phase**: Phase 1 (Horizon Discovery & Gating)
- **Execution Timestamp**: 2026-11-14 17:43:22
- **Git Commit**: `17260f263380d64b80e414599253595b708402a5`
- **Execution Device**: `cuda:1`
- **Runs Completed**: 21 / 21 (100% success)

---
## 1. Executive Summary & Gating Decision

- **Evaluation Cohort**: Phase 1 Validation Partition ($N=5$ subjects, 28 nights; split hash `bf33573623f47f3f`)
- **Primary Contrast**: $L=600$ vs $L=60$
- **Primary Estimand ($\Delta \text{Macro-F1}$)**: **+4.27 pp** [95% CI: **+2.53 pp**, **+6.46 pp**]
  *(Note: $+4.27\text{ pp}$ represents the subject-paired mean delta with equal subject weighting via paired bootstrap; difference of pooled marginal means is $+4.12\text{ pp}$)*
- **Subject-Level Concordance (100% Positive)**:
  - `Bidslab13`: $+4.17\text{ pp}$
  - `Bidslab40`: $+3.39\text{ pp}$
  - `Bidslab55`: $+3.92\text{ pp}$
  - `Bidslab56`: $+8.33\text{ pp}$
  - `Bidslab64`: $+1.51\text{ pp}$
- **Pre-Registered Gating Rule**: Phase 2 Cross-Validation is authorized if $\Delta \text{Macro-F1} \ge +1.0\text{ pp}$ and the 95% bootstrap CI is strictly positive ($\text{CI}_{\text{lower}} > 0$).
- **Frozen Decision Category**: `SUPPORTED (Context Benefit)`
- **Decision Interpretation**: Long context significantly improves wearable sleep staging beyond local windows across all discovery subjects. Phase 2 5-fold cross-validation ($N=47$) is **AUTHORIZED**. Phase 1b circadian ablation is **PERMITTED**.

---
## 2. Context-Horizon Performance Curve (Mean ± SD across 3 Seeds)

| Context Length ($L$) | Window (min) | Macro-F1 | Accuracy (%) | Cohen's $\kappa$ | MCC | Wake F1 | Light F1 | Deep F1 | REM F1 |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $L=1$ | 0.5 | **0.3654** $\pm$ 0.0057 | 48.56 $\pm$ 0.09 | 0.1710 $\pm$ 0.0047 | 0.2066 $\pm$ 0.0023 | 0.3557 | 0.6269 | 0.2003 | 0.2786 |
| $L=20$ | 10.0 | **0.5688** $\pm$ 0.0150 | 58.93 $\pm$ 0.95 | 0.3859 $\pm$ 0.0239 | 0.3949 $\pm$ 0.0186 | 0.5764 | 0.6376 | 0.5266 | 0.5346 |
| $L=60$ | 30.0 | **0.6117** $\pm$ 0.0025 | 62.30 $\pm$ 0.22 | 0.4451 $\pm$ 0.0017 | 0.4498 $\pm$ 0.0007 | 0.6165 | 0.6527 | 0.5774 | 0.6002 |
| $L=120$ | 60.0 | **0.6100** $\pm$ 0.0055 | 62.34 $\pm$ 0.83 | 0.4470 $\pm$ 0.0074 | 0.4506 $\pm$ 0.0085 | 0.5761 | 0.6457 | 0.6023 | 0.6160 |
| $L=300$ | 150.0 | **0.6321** $\pm$ 0.0089 | 64.60 $\pm$ 0.73 | 0.4805 $\pm$ 0.0119 | 0.4842 $\pm$ 0.0119 | 0.5887 | 0.6632 | 0.6207 | 0.6557 |
| $L=600$ | 300.0 | **0.6529** $\pm$ 0.0078 | 66.49 $\pm$ 1.33 | 0.5114 $\pm$ 0.0118 | 0.5194 $\pm$ 0.0111 | 0.6044 | 0.6773 | 0.6890 | 0.6410 |
| $L=1200$ | 600.0 | **0.6993** $\pm$ 0.0139 | 70.04 $\pm$ 1.47 | 0.5693 $\pm$ 0.0159 | 0.5728 $\pm$ 0.0154 | 0.6990 | 0.7058 | 0.7354 | 0.6572 |

> [!NOTE]
> **Operational & Physiological Dynamics**:
> 1. **Intermediate Plateau**: Performance plateaus between $L=60$ (0.6117) and $L=120$ (0.6100), as 1-hour context captures local transition inertia without spanning full ultradian cycles.
> 2. **Inference Modality at $L=1200$**: For $L=1200$, recordings with $N \le 1200$ epochs evaluate under Case 1 (single whole-night forward pass without window trimming or stitching), whereas $L \le 600$ evaluate under Case 2 (overlapping sliding windows with margin trimming). Phase 2 cross-validation will explicitly analyze whether boundary stitching loss contributes to the performance difference between $L=600$ and $L=1200$.

---
## 3. Parametric Saturation Curve Fit (Single-Scale Baseline)

Model formulation: $F_1(L) = A - B e^{-k L}$
- **Asymptotic Ceiling ($A$)**: `0.6435`
- **Dynamic Span ($B$)**: `0.2953`
- **Decay Rate ($k$)**: `0.064716`
- **95% Saturation Threshold ($L^*$)**: **46.3 epochs** (23.1 minutes)
- **Goodness of Fit ($R^2$)**: `0.9258`

> [!IMPORTANT]
> **Physiological Reconciliation of the Saturation Paradox**:
> The single-exponential model fits the rapid initial rise ($L=1 \to 60$) and interprets the intermediate plateau ($L=60 \to 120$) as asymptotic saturation, yielding $A = 0.6435$ and $L^* \approx 46.3\text{ epochs}$ ($23.1\text{ min}$). This captures the **local transition inertia regime** (smoothing sensor noise and immediate epoch transitions).
> However, empirical performance breaks through this ceiling, reaching **0.6529** at $L=600$ and **0.6993** at $L=1200$ ($+5.58\text{ pp}$ above $A$). This occurs because human sleep is governed by **multi-hour ultradian NREM-REM cycles** ($\approx 90\text{--}120\text{ min}$). When context extends beyond $L=300$ epochs ($2.5\text{--}10\text{ hours}$), the recurrent BiGRU accesses macro-architectural progression across the night, initiating a second scaling regime. A dual-timescale model will be parameterized upon completion of the full 47-subject Phase 2 cross-validation.

---
## 4. Clinical Polysomnography Metrics

*Acronyms: Total Sleep Time (TST), Sleep Onset Latency (SOL), Wake After Sleep Onset (WASO), Sleep Efficiency (SE).*

| Context Length ($L$) | TST MAE (min) | SOL MAE (min) | WASO MAE (min) | Sleep Efficiency MAE (%) |
| :--- | ---: | ---: | ---: | ---: |
| $L=1$ | 44.30 | 26.73 | 13.88 | 10.67 |
| $L=20$ | 33.02 | 18.85 | 13.35 | 8.02 |
| $L=60$ | 31.04 | 16.61 | 13.91 | 7.62 |
| $L=120$ | 38.12 | 18.68 | 14.22 | 9.27 |
| $L=300$ | 35.65 | 19.30 | 15.36 | 8.73 |
| $L=600$ | 36.61 | 17.14 | 14.11 | 8.91 |
| $L=1200$ | 27.14 | 12.45 | 13.02 | 6.66 |

> [!NOTE]
> **Clinical Metric Dissociation (Micro-Arousal Low-Pass Filtering)**:
> While Macro-F1 improves from $L=60$ to $L=600$, TST MAE degrades from $31.04\text{ min}$ to $36.61\text{ min}$. Intermediate BiGRU horizons function as temporal low-pass filters that over-smooth brief nocturnal micro-arousals (Wake recall drops from $50.8\%$ at $L=60$ to $42.8\%$ at $L=120$ and $46.4\%$ at $L=600$), inflating estimated sleep duration. At $L=1200$, access to whole-night structural boundaries restores Wake discrimination ($60.8\%$ Wake recall), yielding the most accurate clinical PSG agreement ($27.14\text{ min}$ TST MAE, $12.45\text{ min}$ SOL MAE, $6.66\%$ SE MAE).

---
## 5. All 21 Individual Run Artifacts

| Run ID | Context ($L$) | Seed | Macro-F1 | Accuracy (%) | Cohen's $\kappa$ | Runtime (s) | Artifact Path |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `L1_s42` | 1 | 42 | 0.3730 | 48.66 | 0.1771 | 281.6 | [`phase1_results/run_L0001_seed42.json`](phase1_results/run_L0001_seed42.json) |
| `L1_s123` | 1 | 123 | 0.3640 | 48.57 | 0.1701 | 274.7 | [`phase1_results/run_L0001_seed123.json`](phase1_results/run_L0001_seed123.json) |
| `L1_s2024` | 1 | 2024 | 0.3592 | 48.45 | 0.1657 | 274.3 | [`phase1_results/run_L0001_seed2024.json`](phase1_results/run_L0001_seed2024.json) |
| `L20_s42` | 20 | 42 | 0.5682 | 58.80 | 0.3819 | 200.1 | [`phase1_results/run_L0020_seed42.json`](phase1_results/run_L0020_seed42.json) |
| `L20_s123` | 20 | 123 | 0.5508 | 57.84 | 0.3589 | 197.2 | [`phase1_results/run_L0020_seed123.json`](phase1_results/run_L0020_seed123.json) |
| `L20_s2024` | 20 | 2024 | 0.5875 | 60.15 | 0.4170 | 196.1 | [`phase1_results/run_L0020_seed2024.json`](phase1_results/run_L0020_seed2024.json) |
| `L60_s42` | 60 | 42 | 0.6135 | 62.34 | 0.4445 | 143.6 | [`phase1_results/run_L0060_seed42.json`](phase1_results/run_L0060_seed42.json) |
| `L60_s123` | 60 | 123 | 0.6082 | 62.55 | 0.4434 | 145.5 | [`phase1_results/run_L0060_seed123.json`](phase1_results/run_L0060_seed123.json) |
| `L60_s2024` | 60 | 2024 | 0.6135 | 62.01 | 0.4475 | 155.0 | [`phase1_results/run_L0060_seed2024.json`](phase1_results/run_L0060_seed2024.json) |
| `L120_s42` | 120 | 42 | 0.6147 | 62.95 | 0.4524 | 194.5 | [`phase1_results/run_L0120_seed42.json`](phase1_results/run_L0120_seed42.json) |
| `L120_s123` | 120 | 123 | 0.6023 | 61.17 | 0.4365 | 190.2 | [`phase1_results/run_L0120_seed123.json`](phase1_results/run_L0120_seed123.json) |
| `L120_s2024` | 120 | 2024 | 0.6130 | 62.90 | 0.4520 | 185.7 | [`phase1_results/run_L0120_seed2024.json`](phase1_results/run_L0120_seed2024.json) |
| `L300_s42` | 300 | 42 | 0.6402 | 65.52 | 0.4924 | 149.9 | [`phase1_results/run_L0300_seed42.json`](phase1_results/run_L0300_seed42.json) |
| `L300_s123` | 300 | 123 | 0.6363 | 64.55 | 0.4848 | 149.7 | [`phase1_results/run_L0300_seed123.json`](phase1_results/run_L0300_seed123.json) |
| `L300_s2024` | 300 | 2024 | 0.6197 | 63.73 | 0.4642 | 150.4 | [`phase1_results/run_L0300_seed2024.json`](phase1_results/run_L0300_seed2024.json) |
| `L600_s42` | 600 | 42 | 0.6540 | 67.46 | 0.5126 | 138.4 | [`phase1_results/run_L0600_seed42.json`](phase1_results/run_L0600_seed42.json) |
| `L600_s123` | 600 | 123 | 0.6618 | 67.39 | 0.5252 | 138.6 | [`phase1_results/run_L0600_seed123.json`](phase1_results/run_L0600_seed123.json) |
| `L600_s2024` | 600 | 2024 | 0.6429 | 64.61 | 0.4964 | 138.6 | [`phase1_results/run_L0600_seed2024.json`](phase1_results/run_L0600_seed2024.json) |
| `L1200_s42` | 1200 | 42 | 0.7133 | 71.38 | 0.5841 | 184.0 | [`phase1_results/run_L1200_seed42.json`](phase1_results/run_L1200_seed42.json) |
| `L1200_s123` | 1200 | 123 | 0.7044 | 70.75 | 0.5766 | 184.2 | [`phase1_results/run_L1200_seed123.json`](phase1_results/run_L1200_seed123.json) |
| `L1200_s2024` | 1200 | 2024 | 0.6803 | 68.00 | 0.5472 | 183.8 | [`phase1_results/run_L1200_seed2024.json`](phase1_results/run_L1200_seed2024.json) |

---
## 6. Scientific Governance & Architectural Specification

- **Model Architecture**: Model A is formally defined as `ConventionalCNNBiGRU` ([`exp013_models.py`](exp013_models.py#L69-L72); 163,940 parameters). Parameter count verified identically across all 21 runs.
- **Compute Invariant**: $B \times L \times N_{\text{accum}} = 2,400$ tokens/step ($3,200\text{ steps} \times 2,400 = 7,680,000\text{ total tokens/run}$) verified across all 21 runs.
- **Held-Out Test Set**: 11 held-out test subjects unaccessed; zero test evaluations performed.
- **Statistical Notation**: MCC denotes Matthews Correlation Coefficient; CV denotes Cross-Validation.
- **Reproducibility**: All random seeds, splits, batch streams, and raw predictions recorded in `phase1_results/`.
