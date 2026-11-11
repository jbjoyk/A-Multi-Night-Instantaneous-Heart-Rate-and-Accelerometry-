# Research State

Last updated: 2026-11-11 (DEC-021: Experimental Phase Reopened)

## Research question

Can real Apple Watch instantaneous heart rate and accelerometry signals support four-class sleep
staging against EEG-derived labels for healthy-population longitudinal sleep monitoring, and do
opt-in channel-attention, HR-motion, transition-aware, and unsupervised-personalization experiments
improve on the recorded supervised baseline? The previous SleepJEPA experiment did not improve over
supervised learning and is a documented negative result, not an active contribution.

## Research objectives

- Investigate wearable-based sleep staging using heart-rate and accelerometry signals.
- Preserve and document reproducible research progress.
- Maintain a reliable handoff context for different coding agents.

## Current research direction

The active notebook retains the real-data supervised SleepChannelNet comparisons and adds opt-in
SE attention, HR-motion interactions, transition regularization/Viterbi, and unsupervised head
entropy adaptation. The primary task is four-class classification: Wake, Light, Deep, and REM.
Revised experiments have not been run. Exact SLAMSS-IFS reproduction remains unimplemented and
unverified; the notebook's GAT-free baseline is explicitly an architectural proxy.

## Dataset information

The notebook used the real dataset at the configured local path and discovered 253 nights from
47 subjects. Raw label counts were `{0: 23850, 1: 15414, 2: 80881, 3: 39908, 4: 55861,
5: 2363}`, with 1.08% unknown labels. The raw `Bidslab*` recordings are intentionally excluded
from Git because they are approximately 28 GB. See the dataset README and `SHA256SUMS.txt` for
the available documentation and integrity information.

## Model architecture

The recorded baseline is SleepChannelNet with channel attention and a complete graph configuration,
using 10 registered nodes/features, a bidirectional GRU temporal encoder, frequency/time
features, embedding dimension 64, 4 attention heads, 2 graph-attention layers, hidden size 96,
dropout 0.3, and 240-step sequences. The notebook also retains identity/no-graph, early-fusion
CNN-GRU, and GBM baselines. New model/configuration paths are unverified and opt-in. Existing
architectures use GroupNorm, so the unsupervised adaptation path entropy-minimizes the head and
cannot update BatchNorm running statistics.

## Current implementation

- Notebook: `sleepchannelnet_protocol_patch_v2.ipynb`
- Python dependencies: `requirements.txt`
- Notebook kernel: Python 3.14.7; PyTorch 2.14.1+cu130; CUDA device detected with 2 GPUs.
- Additional research documents: `Problem Statement.docx`, `Research Gaps.docx`, and
  `Datasets Informations.docx`

## Historical completed tasks (recorded 2026-10-23)

- Git repository initialized on the `main` branch.
- Initial project materials committed locally.
- Shared research-memory documentation created.
- Raw dataset recordings excluded from the normal Git commit.
- Notebook executed through its final banner with all 21 code cells executed.
- Seven real-data integrity checks passed; no synthetic data was generated.
- SleepJEPA pretraining and fine-tuning completed.

## Current problems

- Results are from one deterministic subject-level split and seed 42; they are not cross-validation
  results.
- QC reports 134/246 eligible nights with absolute motion-energy/wake cross-correlation lag >4
  epochs or undefined lag. This is a behavioral correlation diagnostic, not evidence of timestamp
  misalignment. A three-night raw timestamp spot check found no basis for a global shift; it is not
  an exhaustive audit. Seven nights were excluded for low usable signal fraction.
- Raw recordings are not included in the GitHub repository.

## Unverified assumptions

- Saved notebook outputs are evidence of the recorded run, but should be reproduced before making
  publication claims.
- The older run work directory was
  `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL`. The updated notebook source is configured to write new outputs in `sleepchannelnet_results_REAL` beside the notebook; the current notebook edits have not been executed yet.

## Current experiment and training status

The older recorded run completed, with historical metrics listed below and in
[EXPERIMENTS.md](./EXPERIMENTS.md). At the time of the initial implementation note, revised-source
experiments had not run. The latest revised-source validation search and one frozen test evaluation
are now recorded below and in EXP-004. Treat the historical values as results from their own saved
run, not as reproduced results from the revised source.

An independent scientific reproduction audit of the baseline SLAMSS-IFS model against the official `BIDSLabUMass/SLAMSS-IFS` repository was conducted (EXP-005). The architecture has been verified across two variants: the 666k parameter paper-text specification and the 5.47M parameter official repository model. Dynamic subject splits (31 train / 5 val / 11 test; 160/28/58 usable nights), exact composite RCCLoss formulation, 10/10 unit tests, 10-step profiling benchmark, and deterministic bitwise reproducibility have all been verified on GPU. The baseline reproduction is classified as Classification B (Faithful Reproduction with Documented Repository Adaptations). Full 500-epoch training is paused pending user selection between the two architecture variants.

The proposed final research architecture `SleepAdaptiveNet` has been implemented in `sleep_adaptive_net.py` and validated (EXP-006). It includes separate HR and ACC modality encoders, learned adaptive reliability fusion gate, temporal context BiGRU, transition-aware module, boundary-aware loss, subject-invariant representation learning via GRL, and a calibrated confidence/uncertainty head. Ablations A0 through A5 are independently switchable. All unit tests, tensor shapes, gradient flow, and smoke-training steps have passed without launching full training.

The first controlled scientific ablation experiment on seed 42 was completed (EXP-007). All 6 ablations were trained for 10 epochs on the 160 usable training nights and evaluated on the 28 usable validation nights (test set strictly excluded). Results indicate that GRL subject-invariance (A4) and the full model (A5) achieve the highest validation Macro-F1 (0.5411 vs 0.5282 baseline) and double REM F1 (0.361 vs 0.181). Boundary loss (A3) cuts boundary error rate by 7.0 percentage points, but requires GRL stabilization to prevent accuracy dips. Adaptive gating ($\alpha_t$) dynamically shifts from 0.567 during Wake to 0.782 during Deep sleep. Full 500/1000-epoch training remains paused.

EXP-008 conducted a controlled boundary weight sweep ($\lambda \in \{0.0, 0.10, 0.20, 0.50\}$) and unconfounded GRL evaluation on seed 42. Disentanglement confirms that GRL standalone (holding boundary loss at 0) yields a verified +3.21% accuracy ($56.11\% \to 59.32\%$) and +0.0353 Macro-F1 ($0.5256 \to 0.5610$) over A2. Boundary loss monotonically reduces boundary errors (from 54.47% to 49.59% at $\lambda=0.20$ and 45.85% at $\lambda=0.50$). $\lambda=0.20$ is established as the preferred Pareto boundary weighting point (-4.88% boundary error with 59.04% accuracy and 0.5543 Macro-F1), while $\lambda=0.50$ is rejected due to excessive Wake recall degradation. Full 500/1000-epoch training remains strictly paused.

EXP-009 evaluated multi-seed stability across seeds 42, 123, 2024 ($N=3$) comparing Candidate 1 (`B0_GRL`, $\lambda=0$) and Candidate 2 (`B2`, $\lambda=0.20$). Results verify that `B0_GRL` consistently maintains superior overall performance across all 3 seeds: Accuracy $59.61\% \pm 0.45\%$ vs $58.85\% \pm 0.21\%$ ($\Delta = -0.75\% \pm 0.48\%$), Macro-F1 $0.5606 \pm 0.0162$ vs $0.5491 \pm 0.0173$ ($\Delta = -0.0115 \pm 0.0043$), Cohen's $\kappa$ $0.4138 \pm 0.0128$ vs $0.4000 \pm 0.0104$ ($\Delta = -0.0137 \pm 0.0062$), and Wake recall $53.95\% \pm 3.29\%$ vs $47.75\% \pm 2.35\%$ ($\Delta = -6.20\% \pm 1.06\%$). Conversely, `B2` stably reduces boundary error rate across all 3 seeds ($49.76\% \pm 0.91\%$ vs $54.04\% \pm 1.51\%$, $\Delta = -4.28\% \pm 0.77\%$). Dynamic modality gating reliably converges to the same physiological weighting pattern across all seeds and models ($\alpha_{\text{Wake}} \approx 0.58$ vs $\alpha_{\text{Deep}} \approx 0.78$). `B0_GRL` is recommended as the default candidate for full training. Full 500/1000-epoch training remains strictly paused, and the held-out test set has never been loaded or used.

EXP-010 evaluated long-training convergence of `B0_GRL` across 500 epochs on seed 42. Empirical results demonstrate that validation performance peaks sharply at **Epoch 78**: Accuracy = **74.22%** (+13.9 pp over Epoch 10), Macro-F1 = **0.7441** (+0.188 over Epoch 10), Cohen's $\kappa$ = **0.6342** (+0.214 over Epoch 10), Val Loss = **0.7245**, and ECE = **0.0240**. Extended training completely resolves minority REM under-fitting, with REM F1 surging to **0.6882** (recall 63.92%, precision 74.54%) at Epoch 78. Beyond Epoch ~80–100, post-peak overfitting occurs under constant learning rate ($10^{-4}$): validation loss climbs to 1.1473 at Epoch 500 while training loss falls to 0.5047, with validation Macro-F1 drifting down to 0.6829 and ECE rising to 0.1949. The adaptive gate preserves learned stage-dependent modality weighting ($\alpha_{\text{Wake}} < \alpha_{\text{Deep}}$) throughout all 500 epochs. Held-out test set remains strictly unaccessed.

EXP-011 evaluated multi-seed long-training convergence across seeds 42, 123, 2024 ($N=3$, 500 epochs each) and audited the EXP-009 vs EXP-010 seed-42 discrepancy. Results resolve the discrepancy: inter-epoch `val_loader` iterations in PyTorch advance the CPU RNG state, causing a minor shift in batch shuffling order; under matched evaluation protocols, results replicate bitwise to the exact decimal place. Multi-seed results confirm that the optimal convergence window is robust across independent seeds: Peak Epoch = $80.7 \pm 10.3$ (range: 72–92), Peak Macro-F1 = $0.7296 \pm 0.0132$ (CV: 1.8%), Peak Accuracy = $72.17\% \pm 1.86\%$, Peak $\kappa$ = $0.6073 \pm 0.0243$. Minority REM F1 surges universally to $0.6886 \pm 0.0186$ (recall $71.1\% \pm 6.5\%$) at peak across all seeds. Overfitting post-peak is universal across 100% of seeds, with validation loss climbing to 1.15–1.29 at Epoch 500. Held-out test set remains strictly unaccessed.

EXP-012 executed the final frozen benchmark evaluation of `SleepAdaptiveNet` (`B0_GRL`) on the untouched official test set (11 subjects, 58 usable nights, 34,241 valid evaluated 30-second windows). Training was conducted for exactly 80 epochs per seed across 3 random seeds (42, 123, 2024) under frozen hyperparameters (Adam $\text{lr}=10^{-4}$, weight decay 0.0, batch size 4, $T=600$ with 0 padding) with zero early stopping, zero LR scheduling, and zero test leakage. Test performance across the 3 seeds yielded: Accuracy = **$68.44\% \pm 1.35\%$** (Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%), Macro-F1 = **$0.6859 \pm 0.0139$** (Seed 42: 0.6986, Seed 123: 0.6880, Seed 2024: 0.6711), Weighted-F1 = **$0.6835 \pm 0.0141$**, Cohen's $\kappa$ = **$0.5527 \pm 0.0188$**, MCC = **$0.5541 \pm 0.0187$**, and ECE = **$0.0257 \pm 0.0050$**. High Wake recall was achieved ($75.95\% \pm 1.20\%$). Boundary error rate was 59.08% vs 30.41% non-boundary. Compared to the published 5.47M parameter SLAMSS-IFS benchmark (Table V: 71.04% Acc, 0.7079 Weighted-F1), SleepAdaptiveNet uses 162,325 parameters (a **33.7x reduction**) and is classified as **Case B (Competitive / Approximate Parity under 33.7x Parameter Compression)**. All test benchmarks are complete and frozen.

EXP-013 conducted a rigorous component attribution experiment using an OFAT factorial ablation against the frozen EXP-012 B0_GRL reference (Model E) and an approximately matched-capacity conventional CNN-BiGRU baseline (Model A: 163,940 params, +0.99% capacity). Evaluated on 34,241 valid windows across 11 held-out test subjects and 3 random seeds ($B=1000$ paired subject-level cluster bootstrap):
1. **Full Proposed Model vs. Matched Baseline ($E - A$)**: SleepAdaptiveNet yields a statistically verified gain of **$+1.79\%$ Accuracy** [95% CI: $+0.52\%, +3.35\%$] and **$+0.0199$ Macro-F1** [95% CI: $+0.0074, +0.0342$], strictly excluding zero and proving that SleepAdaptiveNet's mechanisms provide benefits beyond parameter capacity on that specific partition.
2. **Subject-Invariant GRL ($E - B$)**: Standalone GRL ablation reveals that domain-adversarial subject alignment contributes **$+1.39\%$ Accuracy** [95% CI: $-0.01\%, +2.84\%$] and **$+0.0128$ Macro-F1** (97.4% bootstrap mass > 0).
3. **Adaptive Reliability Gating ($E - D$)**: Dynamic gating contributes **$+0.54\%$ Accuracy** [95% CI: $-0.53\%, +1.60\%$] over equal 0.5/0.5 fusion, with learned gate weights reflecting physiological signal reliability ($\alpha_{\text{Wake}} \approx 0.54$ vs $\alpha_{\text{Sleep}} \approx 0.80$).
4. **Transition Modeling ($E - C$)**: The transition module contributes **$+0.04\%$ Accuracy** [95% CI: $-0.29\%, +0.42\%$], proving that inter-epoch transition constraints are largely redundant in the presence of a bidirectional GRU sequence encoder.

EXP-014 executed an independent 75-run subject-level 5-fold cross-validation across all 47 subjects in the BIDSLab Apple Watch + Dreem 2 dataset (246 usable nights, 145,283 valid evaluated windows) across 5 models and 3 random seeds (42, 123, 2024).
Results demonstrate **Scientific Outcome B (Equivalence / Parity Across Subject Folds)**:
- Model A (Conventional Matched CNN-BiGRU, 163.9k params): Accuracy = **$66.63\% \pm 3.33\%$**, Macro-F1 = **$0.6658 \pm 0.0365$**, $\kappa$ = **$0.5251 \pm 0.0505$**.
- Model B (Minus GRL, 154.0k params): Accuracy = **$66.68\% \pm 3.31\%$**, Macro-F1 = **$0.6664 \pm 0.0345$**, $\kappa$ = **$0.5244 \pm 0.0505$**.
- Model C (Minus Transition, 162.3k params): Accuracy = **$66.07\% \pm 4.04\%$**, Macro-F1 = **$0.6596 \pm 0.0424$**, $\kappa$ = **$0.5154 \pm 0.0600$**.
- Model D (Minus Adaptive Gate, 153.9k params): Accuracy = **$66.68\% \pm 3.59\%$**, Macro-F1 = **$0.6684 \pm 0.0371$**, $\kappa$ = **$0.5270 \pm 0.0514$**.
- Model E (Full SleepAdaptiveNet, 162.3k params): Accuracy = **$66.06\% \pm 4.08\%$**, Macro-F1 = **$0.6596 \pm 0.0425$**, $\kappa$ = **$0.5153 \pm 0.0601$**.
- Paired subject cluster bootstrap ($B=1000$) reveals that all 95% confidence intervals cross zero: $E \text{ vs } A$ [$-1.90\%, +0.50\%$], $E \text{ vs } B$ [$-1.36\%, +0.16\%$], $E \text{ vs } C$ [$-0.15\%, +0.12\%$], $E \text{ vs } D$ [$-1.99\%, +1.02\%$].
- Subject-level win/loss: Model E outperformed Model A on 23 subjects (48.9%), Model A outperformed Model E on 23 subjects (48.9%), with 1 subject tied (2.2%). Across diverse subject folds, conventional matched-capacity models achieve equivalent staging performance, and physiological inter-subject heterogeneity dominates performance variance. All 75 runs are archived in `EXP014_FINAL_REPORT.md`.

## Latest historical recorded results

The latest supplied run artifacts (split hash `a6b569f63126`) report accuracy 0.68659, Macro-F1
0.67791 and κ 0.54722; see EXP-003. The earlier run reported below remains EXP-001. These are
separate saved runs, not independent reproductions.

Recorded run results:

- Complete-graph SleepChannelNet: accuracy 0.6932, macro-F1 0.6851, kappa 0.5529.
- Identity/no-graph SleepChannelNet: accuracy 0.6658, macro-F1 0.6599, kappa 0.5173.
- CNN-GRU early fusion: accuracy 0.6683, macro-F1 0.6462, kappa 0.5069.
- GBM epoch features plus context: accuracy 0.5990, macro-F1 0.5952, kappa 0.4229.
- SleepJEPA pretraining plus fine-tuning: accuracy 0.6411, macro-F1 0.6325, kappa 0.4844.

These results are verified as notebook outputs for the recorded run, not independently reproduced.

## Important constraints

- Preserve existing verified findings and results; under [DEC-021](./DECISIONS.md), exploratory audits, exports, and new protocol formulation are authorized and no longer blocked.
- Do not fabricate metrics or claim that a notebook cell executed when it has not been verified.
- Do not commit the raw dataset recordings or local environment files.

## Next research milestone

Under [DEC-021](./DECISIONS.md), the experimental phase has been formally reopened; the prior restrictions specifying "no further training runs" (from DEC-019 and DEC-020) and "do not change methodology without approval" no longer block audit, review package creation, or new protocol design. While core benchmark evaluations (EXP-012 frozen test benchmark, EXP-013 component attribution, and EXP-014 75-run 5-fold cross-validation) are completed and frozen, new methodological and exploratory investigations are active:
- Investigating the impact of sequence truncation ($T=600$ vs full-night sequences; ~28% labeled epochs dropped).
- Auditing window picking strategies and start offsets.
- Dissecting metadata feature representations (`x_meta`) and ablation in baseline models.
- Auditing normalization mechanics and inter-subject variance predictors (intraclass correlation, sleep fragmentation, motion-wake coupling).
All future training runs remain strictly subject to explicit user authorization under a documented protocol. Zero unauthorized or unapproved training jobs may be launched. All findings and artifacts are continuously synchronized to GitHub (`origin main`).

## Notebook revision note (2026-10-24)

The notebook has an implementation revision adding an SE model, optional HR-motion features, transition regularization/Viterbi decoding, unsupervised head-entropy adaptation, and revised training controls. A validation-only optimizer search and one frozen final test evaluation are now recorded in EXP-004. The previously recorded results above remain historical results from a separate run. Exact SLAMSS-IFS reproduction remains unimplemented/unverified; the project scope is healthy-population longitudinal sleep monitoring.


The supplied latest baseline run and preliminary timestamp QC audit are recorded in EXP-003. The 134-night value is a motion-wake correlation flag, not a confirmed alignment failure.

## Latest revised-source run (2026-10-24)

- Preserved subject split: seed 0, 33/7/7 subjects, split hash `a6b569f63126`.
- Validation-selected training configuration: AdamW, lr 0.001, Plateau scheduler, grouped weight decay 0.01, dropout 0.2; confirmation validation Macro-F1 0.678384.
- Frozen final test, evaluated once: accuracy 0.677715, Macro-F1 0.664803, kappa 0.531064; 7 test subjects, 38 nights, 33,126 known epochs. Subject-bootstrap CIs and detailed reporting are in EXP-004.
- This revised result is below the supplied historical pooled Macro-F1 of 0.67791. No improvement is claimed. Requested ablations and personalization remain unrun.
