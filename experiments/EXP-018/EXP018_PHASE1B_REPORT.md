# EXP-018: Circadian Feature Ablation — Phase 1b Final Report

- **Experiment ID**: EXP-018
- **Phase**: Phase 1b (Circadian Feature Ablation)
- **Execution Timestamp**: 2026-11-14 23:38:38
- **Git Commit**: `1dda77bb427801bcfb1018284729a86e96f1998c`
- **Execution Device**: `cuda:1`
- **Runs Completed**: 9 / 9 (100% success)

---
## 1. Executive Summary & Gating Relevance

Phase 1b measures whether the observed context-horizon gains in Phase 1 stem from true sequence dynamics or if the BiGRU is merely learning a proxy clock for chronological time.

We compare models trained with active normalized time (`time_feat`, $x_{\text{meta}}[:, :, 2] \in [0, 1]$) versus the Phase 1 neutralized baseline (`time_feat = 0.0`) on the identical 31-train / 5-validation partition across seeds 42, 123, 2024.

---
## 2. Comparison Table: Active Circadian Time vs Neutralized Baseline

| Context Length ($L$) | Window (min) | Neutralized Macro-F1 | Active `time_feat` Macro-F1 | Circadian $\Delta$ (pp) | Neutralized Acc (%) | Active Acc (%) |
| :--- | ---: | :---: | :---: | ---: | ---: | ---: |
| $L=1$ | 0.5 | 0.3654 $\pm$ 0.0057 | **0.5131** $\pm$ 0.0035 | **+14.77 pp** | 48.56 | 55.32 |
| $L=60$ | 30.0 | 0.6117 $\pm$ 0.0025 | **0.6555** $\pm$ 0.0078 | **+4.38 pp** | 62.30 | 66.19 |
| $L=600$ | 300.0 | 0.6529 $\pm$ 0.0078 | **0.6665** $\pm$ 0.0018 | **+1.36 pp** | 66.49 | 68.04 |

---
## 3. Scientific Insights & Mechanistic Conclusions

- **Isolated Epoch ($L=1$)**: Circadian time delta = **+14.77 pp**.
- **Local Window ($L=60$)**: Circadian time delta = **+4.38 pp**.
- **Multi-Hour Context ($L=600$)**: Circadian time delta = **+1.36 pp**.

> [!NOTE]
> **Circadian Information Subsumption**:
> Circadian time provides substantial improvement when temporal context is absent ($L=1$) or short ($L=60$). As sequence context expands to multi-hour horizons ($L=600$), the recurrent BiGRU extracts nocturnal progression directly from underlying cardiorespiratory and motion dynamics, demonstrating that long-context gains represent genuine sequence modeling.

---
## 4. Individual Run Artifacts

| Run ID | Context ($L$) | Seed | Active Macro-F1 | Active Accuracy (%) | Cohen's $\kappa$ | Runtime (s) | Artifact Path |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `L1_s42_p1b` | 1 | 42 | 0.5139 | 55.21 | 0.3234 | 291.9 | [`phase1b_results/run_L0001_seed42.json`](phase1b_results/run_L0001_seed42.json) |
| `L1_s123_p1b` | 1 | 123 | 0.5169 | 55.50 | 0.3288 | 294.4 | [`phase1b_results/run_L0001_seed123.json`](phase1b_results/run_L0001_seed123.json) |
| `L1_s2024_p1b` | 1 | 2024 | 0.5085 | 55.24 | 0.3184 | 311.9 | [`phase1b_results/run_L0001_seed2024.json`](phase1b_results/run_L0001_seed2024.json) |
| `L60_s42_p1b` | 60 | 42 | 0.6618 | 66.37 | 0.5103 | 158.8 | [`phase1b_results/run_L0060_seed42.json`](phase1b_results/run_L0060_seed42.json) |
| `L60_s123_p1b` | 60 | 123 | 0.6446 | 65.84 | 0.4877 | 152.2 | [`phase1b_results/run_L0060_seed123.json`](phase1b_results/run_L0060_seed123.json) |
| `L60_s2024_p1b` | 60 | 2024 | 0.6603 | 66.37 | 0.5082 | 155.7 | [`phase1b_results/run_L0060_seed2024.json`](phase1b_results/run_L0060_seed2024.json) |
| `L600_s42_p1b` | 600 | 42 | 0.6649 | 68.69 | 0.5354 | 145.8 | [`phase1b_results/run_L0600_seed42.json`](phase1b_results/run_L0600_seed42.json) |
| `L600_s123_p1b` | 600 | 123 | 0.6690 | 68.25 | 0.5376 | 140.5 | [`phase1b_results/run_L0600_seed123.json`](phase1b_results/run_L0600_seed123.json) |
| `L600_s2024_p1b` | 600 | 2024 | 0.6656 | 67.17 | 0.5293 | 141.8 | [`phase1b_results/run_L0600_seed2024.json`](phase1b_results/run_L0600_seed2024.json) |

---
## 5. Governance & Compliance

- **Model Architecture**: Model A (`ConventionalCNNBiGRU`, 163,940 parameters).
- **Compute Invariant**: $B \times L \times N_{\text{accum}} = 2,400$ tokens/step (3,200 steps = 7,680,000 total tokens) verified across all 9 runs.
- **Held-Out Test Set**: 11 held-out test subjects strictly LOCKED; zero test evaluations performed.
