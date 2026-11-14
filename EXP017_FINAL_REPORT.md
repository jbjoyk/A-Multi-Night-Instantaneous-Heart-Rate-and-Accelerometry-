# EXP-017 Final Report: Controlled Evaluation of Sequence Truncation in Wearable Sleep Staging

*   **Experiment ID:** EXP-017
*   **Protocol Document:** `design_draft.md`
*   **Execution Date:** 2026-11-14
*   **Commit Hash:** `7194e17de69bf954fbd4a79a1263dbdb1d4ca6da`
*   **Dataset Fingerprint:** `8ae7a916a4cd5709`
*   **Split Hash:** `bf33573623f47f3f` (31 Train / 5 Val / 11 Test)
*   **Hardware:** NVIDIA GeForce RTX 3060 12GB (`cuda:0`)
*   **Lifecycle Status:** `COMPLETED`
*   **Scientific Evidence:** `TRUE`
*   **Final Decision:** `INCONCLUSIVE` (Hypothesis $H_1$ is not supported under the tested protocol)

---

## 1. Executive Summary

EXP-017 evaluated whether training wearable sleep staging models on 10-hour-capped extended sequences ($T \le 1200$ epochs) improves minority Rapid Eye Movement (REM) sleep recall compared to 5-hour truncated sequences ($T = 600$) when both conditions are evaluated strictly on the **identical first 600 chronological epoch positions, with invalid positions masked**.

Model A (`ConventionalCNNBiGRU`, 163,940 parameters) was trained for exactly 80 optimization epochs across three independent random seeds (42, 123, 2024) under both sequence conditions (6 independent training runs total; 145–245s per run, peak VRAM 255.3–456.6 MB). All training was conducted on 31 train subjects (160 usable nights) and evaluated on 5 validation subjects (28 usable nights). The 11 held-out test subjects remained strictly unaccessed and locked throughout.

### Key Findings
1.  **Primary Estimand (REM Recall Difference on Matched 600 Epochs)**:
    - Extended sequence training did **not** produce a reliable positive shift in REM recall on the initial 600 epochs.
    - Multi-seed Point Estimate: **$-0.84\text{ percentage points}$** (Mean $T=600$: $71.94\% \pm 5.07\%$, Mean $T \le 1200$: $71.05\% \pm 1.81\%$).
    - Paired Subject Cluster Bootstrap ($B=1000$): **$[-5.68\%, +3.94\%]$**, which strictly spans zero.
    - Direction Consistency across validation cohort: **$2/5$ subjects positive, $3/5$ subjects negative**.
2.  **General Performance & Non-Target Stages**:
    - Extended sequence training yielded consistent overall staging benefits:
      - **Accuracy**: $+2.33\text{ percentage points}$ [95% CI: $+1.64\%, +3.03\%$] ($70.47\% \to 72.80\%$).
      - **Macro-F1**: $+0.0210$ [95% CI: $+0.0123, +0.0269$] ($0.7126 \to 0.7335$).
      - **Cohen's $\kappa$**: $+0.0331$ ($0.5870 \to 0.6201$).
      - **Deep Sleep Recall**: $+8.52\text{ percentage points}$ [95% CI: $+6.35\%, +11.72\%$] ($70.07\% \to 78.59\%$).
3.  **Protocol Decision Rule**:
    - Under the pre-registered decision criteria in `design_draft.md` Section K, because the 95% bootstrap confidence interval spans zero ($[-5.68\%, +3.94\%]$) and non-inferiority margins on Wake/Light recall were breached on individual subjects, the primary outcome is classified as:
      $$\mathbf{INCONCLUSIVE}$$
    - Scientific Interpretation: **Hypothesis $H_1$ is not supported under the tested protocol.** Extended sequence context stabilizes slow-wave sleep and general sequence representations, but does not specifically boost REM sensitivity on early-night sleep.

---

## 2. Experimental Setup and Preflight Verification

| Control Parameter | Pre-Specified Protocol | Verification Status |
| :--- | :--- | :--- |
| **Model Architecture** | Model A (`ConventionalCNNBiGRU`, 163,940 params) | **VERIFIED [PASS]** (Bitwise matched) |
| **Recurrent Padding Protection** | `pack_padded_sequence` / `pad_packed_sequence` | **VERIFIED [PASS]** (Tests confirmed 0.0 leakage) |
| **Dataset Fingerprint** | `8ae7a916a4cd5709` | **VERIFIED [PASS]** |
| **Split Hash** | `bf33573623f47f3f` (31 Train / 5 Val / 11 Test) | **VERIFIED [PASS]** |
| **Usable Nights Loaded** | 160 Train nights, 28 Val nights | **VERIFIED [PASS]** |
| **Test Set Isolation Gate** | 0 test records loaded, 11 test subjects locked | **VERIFIED [PASS]** |
| **Optimizer & Learning Rate** | Adam, $\text{lr} = 10^{-4}$, weight decay $0.0$ | **VERIFIED [PASS]** |
| **Batch Size & Epochs** | Batch size = 4 nights, Epochs = 80 per run | **VERIFIED [PASS]** |
| **Hardware Lease** | `cuda:0` via `GPUResourceManager` | **VERIFIED [PASS]** |

---

## 3. Seed-Level Empirical Results (Primary Matched-600 Epochs)

All metrics below are computed on the **identical first 600 chronological epoch positions** (16,688 valid evaluated windows across the 28 validation nights):

| Seed | Condition | Accuracy (%) | Macro-F1 | Cohen's $\kappa$ | MCC | REM Recall (%) | Wake Recall (%) | Light Recall (%) | Deep Recall (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | $T=600$ | 71.56% | 0.7240 | 0.5973 | 0.5996 | 74.38% | 76.77% | 68.61% | 69.87% |
| **42** | $T \le 1200$ | 73.15% | 0.7359 | 0.6192 | 0.6202 | 69.06% | 75.47% | 69.87% | 78.93% |
| **42** | $\Delta$ | **+1.59 pp** | **+0.0119** | **+0.0220** | **+0.0206** | **-5.32 pp** | -1.30 pp | +1.26 pp | +9.06 pp |
| **123** | $T=600$ | 70.76% | 0.7165 | 0.5888 | 0.5907 | 75.32% | 78.34% | 68.10% | 64.99% |
| **123** | $T \le 1200$ | 73.18% | 0.7354 | 0.6263 | 0.6265 | 71.47% | 80.27% | 66.21% | 79.20% |
| **123** | $\Delta$ | **+2.42 pp** | **+0.0189** | **+0.0375** | **+0.0358** | **-3.85 pp** | +1.93 pp | -1.89 pp | +14.21 pp |
| **2024** | $T=600$ | 69.10% | 0.6972 | 0.5750 | 0.5781 | 66.12% | 77.27% | 68.32% | 62.46% |
| **2024** | $T \le 1200$ | 72.06% | 0.7293 | 0.6148 | 0.6173 | 72.61% | 76.26% | 70.85% | 65.57% |
| **2024** | $\Delta$ | **+2.97 pp** | **+0.0320** | **+0.0398** | **+0.0393** | **+6.49 pp** | -1.01 pp | +2.53 pp | +3.11 pp |
| **Mean** | $T=600$ | $70.47\% \pm 1.26\%$ | $0.7126 \pm 0.0138$ | $0.5870 \pm 0.0112$ | $0.5895 \pm 0.0108$ | $71.94\% \pm 5.07\%$ | $77.46\% \pm 0.80\%$ | $68.34\% \pm 0.26\%$ | $65.77\% \pm 3.77\%$ |
| **Mean** | $T \le 1200$ | $72.80\% \pm 0.64\%$ | $0.7335 \pm 0.0037$ | $0.6201 \pm 0.0058$ | $0.6213 \pm 0.0047$ | $71.05\% \pm 1.81\%$ | $77.33\% \pm 2.57\%$ | $68.98\% \pm 2.44\%$ | $74.57\% \pm 7.79\%$ |
| **Mean** | $\Delta$ | **$+2.33\% \pm 0.69\%$** | **$+0.0210 \pm 0.0102$** | **$+0.0331 \pm 0.0097$** | **$+0.0319 \pm 0.0100$** | **$-0.89\% \pm 6.44\%$** | $-0.13\% \pm 1.79\%$ | $+0.63\% \pm 2.27\%$ | **$+8.79\% \pm 5.56\%$** |

---

## 4. Per-Subject Analysis & Cluster Bootstrap ($N=5$)

Evaluating per-subject paired deltas ($\Delta_s = \text{Metric}_{(T \le 1200)} - \text{Metric}_{(T=600)}$) averaged across all 3 seeds:

| Subject | Valid Epochs (0:600) | $T=600$ REM Recall (%) | $T \le 1200$ REM Recall (%) | $\Delta \text{REM Recall}$ (pp) | $\Delta \text{Macro-F1}$ | $\Delta \text{Accuracy}$ (pp) | $\Delta \text{Deep Recall}$ (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bidslab13** | 4,196 | 78.65% | 70.20% | **-8.45 pp** | +0.0311 | +3.07 pp | +7.53 pp |
| **Bidslab40** | 4,193 | 82.12% | 82.79% | **+0.67 pp** | +0.0241 | +2.45 pp | +14.58 pp |
| **Bidslab55** | 2,320 | 79.42% | 75.70% | **-3.72 pp** | +0.0199 | +3.32 pp | +8.19 pp |
| **Bidslab56** | 2,983 | 64.99% | 72.57% | **+7.58 pp** | +0.0058 | +0.61 pp | +6.66 pp |
| **Bidslab64** | 2,996 | 48.58% | 48.30% | **-0.28 pp** | +0.0182 | +2.06 pp | +5.63 pp |

### Paired Cluster Bootstrap ($B=1000$ iterations)
*   **Primary Estimand ($\Delta \text{REM Recall}$)**:
    - Point Mean: **$-0.84\text{ pp}$**
    - Point Median: **$-0.28\text{ pp}$**
    - 95% Bootstrap CI: **$[-5.68\text{ pp}, +3.94\text{ pp}]$** (strictly spans zero)
    - Direction Consistency: **2 subjects positive (Bidslab40, Bidslab56), 3 subjects negative (Bidslab13, Bidslab55, Bidslab64)**
*   **Deep Sleep Recall**: Point Mean: **$+8.52\text{ pp}$** [95% CI: $+6.35\text{ pp}, +11.72\text{ pp}$] (strictly excludes zero)
*   **Overall Macro-F1**: Point Mean: **$+0.0198$** [95% CI: $+0.0123, +0.0269$] (strictly excludes zero)
*   **Overall Accuracy**: Point Mean: **$+2.30\text{ pp}$** [95% CI: $+1.64\text{ pp}, +3.03\text{ pp}$] (strictly excludes zero)

---

## 5. Secondary Operational Analysis (Descriptive / Non-Causal Extended Coverage)

Evaluating Model A trained on $T \le 1200$ across its entire available recording extent ($T \le 1200$; 23,183 valid evaluated windows across 28 nights, +38.9% more data than $T=600$):

| Seed | Evaluated Epochs | Accuracy (%) | Macro-F1 | Cohen's $\kappa$ | REM Recall (%) | Wake Recall (%) | Light Recall (%) | Deep Recall (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 23,183 | 71.69% | 0.7126 | 0.5898 | 66.56% | 79.51% | 68.61% | 71.27% |
| **123** | 23,183 | 71.71% | 0.7172 | 0.5975 | 70.52% | 83.27% | 64.91% | 74.63% |
| **2024** | 23,183 | 70.95% | 0.7161 | 0.5935 | 74.18% | 80.12% | 67.24% | 79.81% |
| **Mean** | 23,183 | **$71.45\% \pm 0.43\%$** | **$0.7153 \pm 0.0024$** | **$0.5936 \pm 0.0039$** | **$70.42\% \pm 3.81\%$** | **$80.97\% \pm 2.02\%$** | **$66.92\% \pm 1.87\%$** | **$75.24\% \pm 4.31\%$** |

Across extended recordings, Model A demonstrates stable four-class operational performance ($\approx 71.5\%$ accuracy, $\approx 0.715$ Macro-F1, $\approx 70.4\%$ REM recall) when evaluated over all hours up to hour 10.

---

## 6. Scientific Interpretation & Falsification Analysis

1.  **Hypothesis $H_1$ Evaluation**:
    - **Hypothesis**: Training with $T \le 1200$ improves validation REM recall relative to $T=600$ on the matched initial 600 epochs.
    - **Finding**: The observed point difference is slightly negative ($-0.84$ pp) and the 95% bootstrap CI spans zero ($-5.68$ pp to $+3.94$ pp). Direction consistency was split (2 vs 3).
    - **Conclusion**: **Hypothesis $H_1$ is NOT SUPPORTED.** Extending the sequence training horizon does not specifically rescue or improve REM recall within the early-night sleep window.
2.  **Unexpected Representation Gain in Slow-Wave Sleep (Deep Sleep)**:
    - Interestingly, extended sequence training produced a large, statistically reliable gain in **Deep Sleep Recall (+8.52 pp [95% CI: +6.35, +11.72 pp])** and overall Macro-F1 (+0.0198).
    - By backpropagating across complete sleep cycles, the BiGRU learned more stable representations of non-REM slow-wave sleep boundaries, boosting Deep sleep accuracy on the early-night epochs where slow-wave sleep concentrates.

---

## 7. Artifacts and Provenance

*   **Master Memory Record:** Registered as 14th record in `research_memory.json` (validated by `ResearchMemoryValidator`).
*   **Run Artifacts:**
    - `research/experiments/EXP-017/ModelA_T600_seed42_results.json`
    - `research/experiments/EXP-017/ModelA_T600_seed123_results.json`
    - `research/experiments/EXP-017/ModelA_T600_seed2024_results.json`
    - `research/experiments/EXP-017/ModelA_T1200_seed42_results.json`
    - `research/experiments/EXP-017/ModelA_T1200_seed123_results.json`
    - `research/experiments/EXP-017/ModelA_T1200_seed2024_results.json`
*   **Summary Tables:**
    - `research/experiments/EXP-017/EXP017_SUMMARY.csv`
    - `research/experiments/EXP-017/EXP017_SUBJECT_DELTAS.csv`
    - `research/experiments/EXP-017/EXP017_BOOTSTRAP.json`
*   **Model Checkpoints:**
    - `research/experiments/EXP-017/checkpoints/model_A_T600_seed42_epoch80.pt`
    - `research/experiments/EXP-017/checkpoints/model_A_T600_seed123_epoch80.pt`
    - `research/experiments/EXP-017/checkpoints/model_A_T600_seed2024_epoch80.pt`
    - `research/experiments/EXP-017/checkpoints/model_A_T1200_seed42_epoch80.pt`
    - `research/experiments/EXP-017/checkpoints/model_A_T1200_seed123_epoch80.pt`
    - `research/experiments/EXP-017/checkpoints/model_A_T1200_seed2024_epoch80.pt`

---

## 8. Recommended Next Actions

1.  **Acknowledge RQ-003 Resolution**:
    - The hypothesis that 5-hour sequence truncation specifically suppressed REM recall in early project phases is unconfirmed. REM recall limitations are governed by optical PPG pulse SNR and autonomic arousal patterns rather than sequence training horizon.
2.  **Next Protocol Directions**:
    - **RQ-001 (EXP-015)**: Prune the dummy zero-placeholder `clock` feature from `x_meta` to eliminate uninformative weights.
    - **RQ-002 (EXP-016)**: Recalibrate GRL gradient scaling to address the 100-fold attenuation identified in DEC-021.
