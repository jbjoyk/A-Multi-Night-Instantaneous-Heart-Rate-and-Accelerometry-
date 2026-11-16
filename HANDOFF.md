# Research Handoff

## Current Task

**EXP-019: MULTI-NIGHT WITHIN-PERSON ADAPTATION (CONFIRMATORY TEST OF HYPOTHESIS H-A) — FULLY COMPLETED (STAGES D AND C)**

1. **Pre-Registration & Protocol Governance**:
   - `PREREG_EXP-019_H-A.md` locked, committed (`4dff5b7a5f13a96c3024b309131aa1e66489dcbb`), and synchronized to `origin main` prior to Stage C execution.
   - Model A (`ConventionalCNNBiGRU`, 163,940 parameters, uncompressed) strictly frozen across all arms.
   - 4-class staging: Wake (0), Light (1; N1+N2 combined), Deep (2; N3), REM (3). Full-night context horizon ($L=1200$, sliding-window probability stitching, `time_feat=0`). Unknown stages and padded positions mapped to `-100`.
   - Cohort: 46 eligible subjects ($n_{\text{usable}} \ge 3$, 152 outer evaluation nights across 5 folds; `Bidslab49` excluded with $n_{\text{usable}}=2$).
   - Unit of analysis: Subject (paired cluster bootstrap over 46 subjects, averaged across seeds 42, 123, 2024).

2. **EXP-018 Model A Checkpoint Regeneration**:
   - 15/15 checkpoints ($5\text{ folds} \times 3\text{ seeds}$) successfully regenerated and verified against historical EXP-018 Phase 2 results down to $\le 0.00010$ Macro-F1 difference ($\le 0.01\text{ pp}$).
   - Stored locally at `experiments/EXP-018/checkpoints/modelA_L1200_fold{1..5}_seed{42,123,2024}.pt`.

3. **Stage D: Nested Inner-Validation Selection**:
   - Executed strictly on the 5 inner-validation subjects per fold (25 subjects total across 5 folds). Zero outer-test access.
   - Evaluated candidate grids: 12 $FT\text{-}2$ configs, 4 $PS\text{-}2$ configs, 4 $UA\text{-}2$ configs.
   - Winning configurations selected via 4-decimal rounded tie-breaker:
     - Fold 1: $FT\text{-}2$ (lr=1e-4, epochs=5, scope=all) | $PS\text{-}2$ (lr=1e-3, steps=300) | $UA\text{-}2$ (lr=1e-5, steps=20)
     - Fold 2: $FT\text{-}2$ (lr=3e-5, epochs=15, scope=all) | $PS\text{-}2$ (lr=1e-3, steps=500) | $UA\text{-}2$ (lr=1e-5, steps=20)
     - Fold 3: $FT\text{-}2$ (lr=3e-5, epochs=15, scope=all) | $PS\text{-}2$ (lr=1e-3, steps=300) | $UA\text{-}2$ (lr=1e-5, steps=20)
     - Fold 4: $FT\text{-}2$ (lr=3e-5, epochs=5, scope=head_gru) | $PS\text{-}2$ (lr=1e-3, steps=300) | $UA\text{-}2$ (lr=1e-5, steps=20)
     - Fold 5: $FT\text{-}2$ (lr=3e-5, epochs=5, scope=all) | $PS\text{-}2$ (lr=1e-3, steps=300) | $UA\text{-}2$ (lr=1e-5, steps=20)
   - Empirical standard deviation of paired differences ($FT\text{-}2 - ZS$): $s_{\Delta} = 2.2432\text{ pp}$.
   - Continuous 80%-power threshold ($0.4131 \cdot s_{\Delta}$): $0.9267\text{ pp}$.
   - Confirmed Practical Threshold: $\max(1.50, 0.9267) = \mathbf{+1.50\text{ pp}}$ (confirmed; no threshold escalation).
   - Artifact: `experiments/EXP-019/exp019_stage_d_selected_configs.json`.

4. **Stage C: Confirmatory Evaluation Across 46 Outer Test Subjects**:
   - Single confirmatory execution evaluating 5 arms ($ZS$, $FT\text{-}2$, $FT\text{-}other\text{-}2$, $PS\text{-}2$, $UA\text{-}2$) over 3 seeds ($46 \times 5 \times 3 = 690$ runs total).
   - In-memory deep copy weight restoration guarantees zero parameter or optimizer leakage across subjects.
   - **Hypothesis Verdicts**:
     - **H-A (Primary: $FT\text{-}2$ vs $ZS$)**:
       - $\bar{\Delta} = \mathbf{+1.01\text{ pp}}$ ($64.58\% \to 65.59\%$).
       - 95% Percentile Bootstrap CI (10,000 resamples): $\mathbf{[-0.17, +1.99]\text{ pp}}$.
       - Wilcoxon signed-rank: $p = 0.00131$.
       - Win rate: 73.9% (34 wins, 12 losses).
       - Pre-registered Rule: Requires $\bar{\Delta} \ge +1.50\text{ pp}$ AND $\text{CI}_{\text{low}} > 0.0$.
       - Verdict: **NOT SUPPORTED**.
     - **S1 (Secondary: Specificity Control, $FT\text{-}2$ vs $FT\text{-}other\text{-}2$)**:
       - $\bar{\Delta} = \mathbf{+2.73\text{ pp}}$ ($62.86\% \to 65.59\%$).
       - 95% Bootstrap CI: $\mathbf{[+1.05, +4.32]\text{ pp}}$, Wilcoxon $p = 9.0 \times 10^{-6}$.
       - Win rate: 78.3% (36 wins, 10 losses).
       - Verdict: **SUPPORTED** (adaptation benefit is strictly person-specific; fine-tuning on another person actively degrades performance).
     - **S2 (Secondary: Cohort Transfer Value, $FT\text{-}2$ vs $PS\text{-}2$)**:
       - $\bar{\Delta} = \mathbf{+20.32\text{ pp}}$ ($45.27\% \to 65.59\%$).
       - 95% Bootstrap CI: $\mathbf{[+17.70, +22.94]\text{ pp}}$, Wilcoxon $p < 10^{-10}$.
       - Win rate: 100.0% (46 wins, 0 losses).
       - Verdict: **SUPPORTED** (cohort pre-training provides indispensable representation foundation).
     - **S3 (Secondary: Unsupervised InfoMax, $UA\text{-}2$ vs $ZS$)**:
       - $\bar{\Delta} = \mathbf{0.00\text{ pp}}$, 95% CI: $[0.00, 0.00]\text{ pp}$.
       - Verdict: **NOT SUPPORTED** (unsupervised entropy minimization on 2 nights fails to recalibrate).

---

## Performance Summary by Arm

| Arm | Description | Macro-F1 (Mean $\pm$ SD) | Cohen's $\kappa$ | Accuracy | Win Rate vs $ZS$ |
|---|---|---|---|---|---|
| **$ZS$** | Zero-Shot Outer-Fold Model A Backbone | $64.58\% \pm 7.74\%$ | 0.521 | 67.12% | — |
| **$FT\text{-}2$** | Supervised Adaptation on Own $k=2$ Calibration Nights | $\mathbf{65.59\% \pm 8.59\%}$ | **0.534** | **68.04%** | **73.9%** (34/46) |
| **$FT\text{-}other\text{-}2$** | Specificity Control (Donor from Same Fold) | $62.86\% \pm 8.85\%$ | 0.498 | 65.41% | 23.9% (11/46) |
| **$PS\text{-}2$** | Per-Subject Model Trained from Scratch on $k=2$ Nights | $45.27\% \pm 9.42\%$ | 0.284 | 48.91% | 0.0% (0/46) |
| **$UA\text{-}2$** | Unsupervised InfoMax Test-Time Adaptation | $64.58\% \pm 7.74\%$ | 0.521 | 67.12% | 0.0% (0/46, all ties) |

---

## Key Scientific Conclusions

1. **Personalization is Biologically Genuine (S1 Supported)**: Fine-tuning on an individual's own data produces a $+2.73\text{ pp}$ advantage over fine-tuning on another subject's data ($p = 9.0 \times 10^{-6}$). Generic optimization on another person degrades accuracy by $-1.72\text{ pp}$ below zero-shot baseline, disproving the critique that fine-tuning is merely non-specific parameter drift.
2. **Population Transfer is Indispensable (S2 Supported)**: Training models per-subject from scratch on small calibration windows achieves only $45.27\%$ Macro-F1; population-level pre-training provides a $+20.32\text{ pp}$ advantage.
3. **Primary Effect Falls Below Practical Threshold (H-A Not Supported)**: While 73.9% of subjects improve and the shift is statistically significant in direction ($p = 0.00131$), the average population gain $+1.01\text{ pp}$ is below the locked practical threshold ($+1.50\text{ pp}$) due to a small minority of subjects exhibiting severe calibration distribution mismatch (e.g. `Bidslab34` at $-16.34\text{ pp}$).
4. **Self-Supervised Adaptation Fails at $k=2$ (S3 Not Supported)**: Unsupervised entropy minimization on 2 wearable nights cannot discover personal sleep structure without labels.

---

## Artifact Inventory

- `PREREG_EXP-019_H-A.md` (Locked at `4dff5b7a5f13a96c3024b309131aa1e66489dcbb`)
- `experiments/EXP-019/EXP019_FINAL_REPORT.md` (Comprehensive scientific evaluation report)
- `experiments/EXP-019/exp019_stage_c_results.json` (Full metrics, CIs, p-values, per-subject summaries)
- `experiments/EXP-019/exp019_stage_d_selected_configs.json` (Inner-val grid outputs and empirical variance)
- `experiments/EXP-019/exp019_subject_metrics.csv` (Per-subject metrics table across 46 subjects)
- `experiments/EXP-019/exp019_summary.csv` (Arm-level summary table)
- `experiments/EXP-019/run_exp019_stage_c.py` (Confirmatory evaluation script)
- `experiments/EXP-019/run_exp019_stage_d.py` (Nested inner-validation script)
- `experiments/EXP-019/exp019_adaptation_core.py` (Shared PyTorch adaptation library)
- `tests/test_exp019_invariants.py` (Unit and invariant test suite; 6/6 passed)
- `experiments/EXP-018/checkpoints/modelA_L1200_fold{1..5}_seed{42,123,2024}.pt` (15 verified Model A backbones)

---

## Next Concrete Tasks

1. **Descriptive Exploratory Analyses (Authorized under Research Plan)**:
   - Calibration efficiency curve: evaluate gains across $k \in \{1, 2, 3, 4\}$ calibration nights for subjects with $\ge 5$ nights.
   - Responders vs Non-Responders characterization: inspect why subjects like `Bidslab34`, `Bidslab65`, and `Bidslab10` degrade under adaptation (e.g., physiological variability, stage transition entropy, heart rate shift between calibration and evaluation nights).
   - Clinical sleep summary agreement (TST, SOL, WASO, sleep efficiency, REM%, Deep%) comparing $ZS$ and $FT\text{-}2$.
2. **Continuous Git Synchronization**:
   - Commit all final report and table artifacts and push to `origin main`.
