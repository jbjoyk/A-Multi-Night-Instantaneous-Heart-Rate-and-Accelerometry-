# EXP-018: Context-Horizon Scaling Curve — Phase 2 Cross-Validation Final Report

- **Experiment ID**: EXP-018
- **Phase**: Phase 2 (Cohort-Wide 5-Fold Cross-Validation across 47 Subjects)
- **Execution Timestamp**: 2026-11-15 05:56:54
- **Git Commit**: `81b7fa1443b574db3d700f2b63956e83278cd0f0`
- **Execution Device**: `cuda:1`
- **Runs Completed**: 105 / 105 (100% success across 5 folds)
- **Evaluated Subjects**: All 47 subjects (246 usable nights, complete out-of-fold coverage)

---
## 1. Executive Summary & Cohort-Wide Estimand

- **Primary Contrast**: $L=600$ vs $L=60$ (Evaluated Out-Of-Fold across all 47 Subjects)
- **Primary Estimand ($\Delta \text{Macro-F1}$)**: **+5.33 pp** [95% CI: **+4.28 pp**, **+6.39 pp**]
- **Decision Category**: `SUPPORTED (Context Benefit)`
- **Interpretation**: Long context significantly improves wearable sleep staging beyond local windows. Phase 2 CV is AUTHORIZED.
- **Subject-Level Win/Loss**: 46 / 47 subjects positive (97.9%) for $L=600$ over $L=60$.

---
## 2. Cohort-Wide Out-Of-Fold Performance Curve (Mean ± SD across 3 Seeds)

| Context Length ($L$) | Window (min) | Out-Of-Fold Macro-F1 | Accuracy (%) | Cohen's $\kappa$ |
| :--- | ---: | :---: | ---: | ---: |
| $L=1$ | 0.5 | **0.3692** $\pm$ 0.0011 | 47.58 $\pm$ 0.09 | 0.1661 $\pm$ 0.0013 |
| $L=20$ | 10.0 | **0.5319** $\pm$ 0.0036 | 57.54 $\pm$ 0.19 | 0.3588 $\pm$ 0.0046 |
| $L=60$ | 30.0 | **0.5773** $\pm$ 0.0039 | 60.85 $\pm$ 0.55 | 0.4148 $\pm$ 0.0050 |
| $L=120$ | 60.0 | **0.5894** $\pm$ 0.0045 | 62.06 $\pm$ 0.59 | 0.4323 $\pm$ 0.0061 |
| $L=300$ | 150.0 | **0.6049** $\pm$ 0.0058 | 63.62 $\pm$ 0.50 | 0.4549 $\pm$ 0.0077 |
| $L=600$ | 300.0 | **0.6306** $\pm$ 0.0031 | 65.45 $\pm$ 0.14 | 0.4877 $\pm$ 0.0048 |
| $L=1200$ | 600.0 | **0.6479** $\pm$ 0.0053 | 66.18 $\pm$ 0.73 | 0.5078 $\pm$ 0.0076 |

---
## 3. Parametric Saturation Curve Fit (Full Cohort $N=47$)

Model formulation: $F_1(L) = A - B e^{-k L}$
- **Asymptotic Ceiling ($A$)**: `0.6143`
- **Dynamic Span ($B$)**: `0.2545`
- **Decay Rate ($k$)**: `0.049978`
- **95% Saturation Threshold ($L^*$)**: **59.9 epochs** (30.0 minutes)
- **Goodness of Fit ($R^2$)**: `0.9472`

---
## 4. Scientific Governance & Reproducibility

- **Model Architecture**: Model A (`ConventionalCNNBiGRU`, 163,940 parameters).
- **Compute Invariant**: $B \times L \times N_{\text{accum}} = 2,400$ tokens/step (3,200 steps = 7,680,000 total tokens) verified across all 105 runs.
- **Cross-Validation Manifests**: Exact manifests from `experiments/EXP-014/EXP014_MANIFESTS/fold_{1..5}.json` with strictly disjoint partitions.
- **Held-Out Test Coverage**: Out-of-fold evaluation covers all 47 subjects without data leakage.
