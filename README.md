# Wearable Sleep Staging Research Project

This repository contains reproducible research code, experimental benchmarks, cross-validation studies, and scientific reports for wearable four-class sleep staging (Wake, Light, Deep, REM) using instantaneous heart rate and accelerometry against EEG-derived polysomnography labels.

## Research Overview

The project investigates wearable-based sleep staging on the *Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels* (47 subjects, 253 recorded nights, 246 usable nights, ~145k valid 30-second windows).

### Key Architectures & Milestones
- **SLAMSS-IFS Baseline Reproduction (EXP-005)**: Independent scientific reproduction audit of the official IEEE TBME 2026 SLAMSS-IFS architecture (`BIDSLabUMass/SLAMSS-IFS`).
- **SleepAdaptiveNet (EXP-006 to EXP-012)**: Compact dual-branch architecture (~162k parameters, 33.7x smaller than official SLAMSS-IFS) incorporating dynamic reliability gating between HR and accelerometry, BiGRU temporal context, domain-adversarial subject alignment (GRL), and boundary-aware loss.
- **Factorial Component Attribution (EXP-013)**: Rigorous single-component ablation on 34,241 held-out test windows demonstrating the contribution of GRL and adaptive gating.
- **75-Run 5-Fold Cross-Validation (EXP-014)**: Definitive subject-level 5-fold cross-validation across all 47 subjects (5 models x 5 folds x 3 seeds = 75 runs) demonstrating equivalence/parity between matched-capacity conventional CNN-BiGRU and specialized architectures, and highlighting that inter-subject physiological variance dominates staging error.

---

## Repository Structure

```
├── README.md                              # This file
├── RESEARCH_STATE.md                      # Current research facts, objectives, and verified findings
├── EXPERIMENTS.md                         # Complete registry of all experiments (EXP-001 through EXP-014)
├── DECISIONS.md                           # Architectural and experimental decisions (DEC-001 through DEC-020)
├── HANDOFF.md                             # Continuity handoff notes and active task state
├── NOTEBOOK_STATE.md                      # Notebook execution and environment audit state
│
├── EXP014_FINAL_REPORT.md                 # Complete report on 75-run 5-fold cross-validation
├── EXP014_PROTOCOL.md                     # Frozen evaluation protocol for EXP-014
├── EXP014_SUMMARY.csv                     # Model performance summary across all 75 runs
├── EXP014_RESULTS.csv                     # Per-fold, per-seed aggregate results
├── EXP014_PER_CLASS.csv                   # Per-class metrics across folds and models
├── EXP014_SUBJECT_METRICS.csv             # Per-subject metrics across all 47 subjects
├── EXP014_PAIRED_BOOTSTRAP.csv            # Paired subject-level cluster bootstrap statistics (B=1000)
├── EXP014_CONFUSION_MATRICES/             # Normalized and raw confusion matrices
├── EXP014_MANIFESTS/                      # Subject partition manifests for folds 1–5
├── EXP014_TRAINING_HISTORY/               # Validation loss/Macro-F1 trajectories
│
├── EXP013_RESULTS.md                      # Component attribution report on test partition
├── EXP013_OUTPUTS/                        # CSVs and metrics for EXP-013 factorial ablations
│
├── EXP012_AUDIT/                          # Independent reproduction & benchmark audit
├── EXP012_BENCHMARK_REPORT.pdf            # 10-page publication benchmark report
├── SUPERVISOR_REVIEW_EXP012/              # Supervisor review dossiers and audit reports
├── report_assets/                         # High-resolution publication figures (fig1–fig5)
│
├── sleep_adaptive_net.py                  # Core PyTorch model: SleepAdaptiveNet (A0–A5)
├── exp013_models.py                       # Factorial ablation model variants (Models A–E)
├── exp014_models.py                       # Cross-validation model variants
├── run_exp014_cv.py                       # Cross-validation runner script
├── run_exp013_attribution.py              # Attribution study runner script
├── run_exp012_final_benchmark.py          # Benchmark test runner script
├── official_slamss_ifs/                   # Reference implementation of SLAMSS-IFS
└── requirements.txt                       # Python dependencies
```

---

## AI Agent & Claude Collaboration Guide

If you are an AI assistant (such as Claude or Antigravity) reviewing this repository to check experiment findings and suggest the next experiment:

1. **Check Research State & Trajectory**:
   - Read [RESEARCH_STATE.md](./RESEARCH_STATE.md) for current facts, metrics, and data parameters.
   - Read [EXPERIMENTS.md](./EXPERIMENTS.md) for full details, hypotheses, and results of EXP-001 through EXP-014.
   - Read [DECISIONS.md](./DECISIONS.md) to understand why specific design choices were locked.
2. **Review Recent Empirical Findings**:
   - Consult [EXP014_FINAL_REPORT.md](./EXP014_FINAL_REPORT.md): Note that the empirical training phase across all 47 subjects is **concluded and frozen**.
   - Note that across all 47 subjects, conventional matched-capacity CNN-BiGRU achieved parity ($66.63\%$ accuracy) with specialized gating and GRL ($66.06\%$, bootstrap CI spans zero).
3. **Formulating Next Recommendations**:
   - Since model architecture modifications under ~160k parameters have reached saturation/equivalence, future research avenues should focus on:
     - Multi-night longitudinal personalization (adapting per-subject over successive recording nights).
     - Signal quality index (SQI) filtering for noisy PPG/optical wrist readings.
     - Synthetic data augmentation for under-represented REM/Deep stages.
     - Manuscript drafting and synthesis of the comprehensive 75-run findings for journal submission.

---

## Dataset & Environment

- Dataset: PhysioNet *A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels* (v1.0.0).
- Raw recordings (`Bidslab*`, ~28 GB) are excluded from Git commits via `.gitignore`.
- Python 3.10+ / PyTorch 2.x with CUDA support. Run `pip install -r requirements.txt`.
