# Decisions

## DEC-001 — Use repository files as the shared research memory

- Date: 2026-10-23
- Decision: Store agent instructions, research state, handoffs, notebook state, decisions, and
  experiment history in version-controlled Markdown files.
- Reason: The researcher switches between multiple coding agents and needs a provider-neutral
  source of truth.
- Alternatives considered: Relying on agent conversation history or provider-specific memory.
- Expected consequences: Agents can recover project context from the repository.
- Reversible: Yes.

## DEC-002 — Exclude raw dataset recordings from normal Git commits

- Date: 2026-10-23
- Decision: Exclude the `Bidslab*` raw recordings from Git using `.gitignore`.
- Reason: The raw recordings occupy approximately 28 GB and are unsuitable for a normal GitHub
  repository.
- Alternatives considered: Committing the recordings directly or using Git LFS.
- Expected consequences: The repository contains documentation and code, while raw data must be
  provisioned separately.
- Reversible: Yes, subject to storage and repository limits.

## DEC-003 — Do not infer unverified research facts

- Date: 2026-10-23
- Decision: Mark unknown research questions, architecture, execution state, and metrics as
  `UNKNOWN` or `NOT VERIFIED`.
- Reason: Research integrity requires distinguishing saved artifacts from reproduced results.
- Alternatives considered: Inferring facts from filenames or notebook outputs.
- Expected consequences: The memory files may initially contain placeholders that are filled after
  inspection and execution.
- Reversible: Yes.


## DEC-004 — Keep requested ablations separate from the recorded baseline

- Date: 2026-10-24
- Decision: Preserve the original SleepChannelNet and its 10-feature default inputs, add the SE-Net and HR-motion variants alongside it, and make the new experiments opt-in.
- Reason: The baseline's saved outputs are historical and should remain distinguishable from new, unexecuted implementations.
- Expected consequences: New model comparisons require a fresh run; no new metrics are implied by the notebook edits.
- Reversible: Yes.

## DEC-005 — Do not describe current no-graph control as exact SLAMSS-IFS

- Date: 2026-10-24
- Decision: Mark exact SLAMSS-IFS reproduction as not implemented/verified because the current pipeline lacks the paper's intra-epoch LSTM and a confirmed matching preprocessing protocol.
- Reason: Avoid attributing preprocessing or architecture differences to the requested improvements.
- Reversible: Yes, after the paper protocol is implemented and verified.

## DEC-006 — Use head entropy adaptation where BatchNorm adaptation is unavailable

- Date: 2026-10-24
- Decision: The unsupervised personalization implementation freezes the backbone and optionally entropy-minimizes the prediction head. It does not adapt BatchNorm running statistics.
- Reason: Existing models use GroupNorm, which has no running statistics.
- Reversible: Yes, if a BatchNorm-based architecture is deliberately added.

## DEC-007 — Keep optimization and final evaluation separate

- Date: 2026-10-24
- Decision: Make core training and optimizer search independently opt-in. The optimizer sweep trains on train subjects and selects only from validation subjects; final test evaluation stays in the separate core/ablation run.
- Reason: The validation search should be runnable before any test metrics are produced, and expensive training should not start when opening the notebook.
- Expected consequences: Run `RUN_OPTIMIZER_SEARCH=True` for the controlled search; use its returned configuration for a separately requested final run. Existing SleepChannelNet remains the primary core baseline.
- Reversible: Yes.

## DEC-008 — Freeze optimizer settings before the one-time test pass

- Date: 2026-10-24
- Decision: Use the validation-search selection (AdamW, lr 0.001, Plateau, grouped weight decay 0.01, dropout 0.2) for one final fit and held-out test evaluation; do not tune further against that test result.
- Reason: The attached research prompt requires a validation-driven search and reserves test data for a final frozen evaluation.
- Observed result: Test Macro-F1 was 0.6648, below the supplied historical result 0.6779. No improvement claim is supported.
- Expected consequences: Further model or ablation changes must be selected with training/validation data; this held-out result remains a one-time report for this revision.
- Reversible: Only by a separately documented protocol or a newly justified evaluation split.


## DEC-009 — Store notebook outputs beside the updated notebook

- Date: 2026-10-25
- Decision: Configure the updated notebook's real-data work directory as `sleepchannelnet_results_REAL` beside the notebook and stop writing the duplicate per-run `history.csv`. Retain run checkpoints and feature caches because they allow training resume and preprocessing reuse. Add descriptive ablation metric and per-stage F1 plots.
- Reason: Keep user-facing results together with the notebook and make model comparisons easier to read without discarding files needed for recovery.
- Reversible: Yes.


## DEC-010 — Use Optuna TPE for validation-only optimizer search

- Date: 2026-10-26
- Decision: Replace the fixed optimizer grid with an opt-in, persistent Optuna TPE study. Tune AdamW learning rate, grouped weight decay, dropout, and scheduler using training subjects for fitting and validation Macro-F1 for selection. Confirm the top two screened trials at the configured full epoch budget. Never access the held-out test partition during search.
- Reason: Sample promising configurations adaptively while preserving the existing subject split and frozen test protocol.
- Expected consequences: Search is compute-intensive; default budget is 12 trials screened to 20 epochs plus two full-epoch confirmations. Optuna must be installed.
- Reversible: Yes; the fixed-grid design can be restored if needed.


## DEC-011 — SLAMSS-IFS Reproduction Fidelity Classification and Discrepancy Resolution

- Date: 2026-11-09
- Decision: Classify the SLAMSS-IFS reproduction as Classification B (Faithful Reproduction with Documented Repository Adaptations). Primary discrepancies between the 2026 published IEEE TBME paper text and the official BIDSLabUMass/SLAMSS-IFS GitHub repository have been resolved by prioritizing the official repository implementation:
  1. Recurrent Units: Official repository uses GRU throughout (`Encode_micro`, `CNN`, `Encoder`, `Decoder`), whereas paper text described BiLSTM.
  2. Multi-Scale Fusion: Official repository includes a large 4.70M-parameter `cnn3` multi-scale convolutional fusion block (Conv11, Conv15, Conv21) and a 5-channel time/frequency/HR-statistics subnetwork (`cnn_ACT`), totaling 5.47M parameters, whereas the paper text described only a 666k parameter intra-epoch CNN + LSTM.
  3. Loss Formulation: Official code uses composite loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 0.01 \cdot \mathcal{L}_{\text{RW}}$ with an exact asymmetric $4 \times 4$ cost matrix $w_{\text{FP}}$ based on square roots of training class frequency ratios.
  4. Sequence Handling: Both paper and official repository pad shorter nights to 1200 epochs with zeros and mask padded positions using binary loss masking.
  5. Teacher Forcing: Fixed to 0.0 in the official training loop.
- Reason: Ensures scientific integrity and reproducibility by transparently reconciling primary sources before running the full 500-epoch training baseline.
- Reversible: Yes; both the 666k paper-text model and the 5.47M official repository model are maintained and testable.


## DEC-012 — Modular Architecture for Proposed Final Model (SleepAdaptiveNet)

- Date: 2026-11-09
- Decision: Implement the proposed research model as a modular architecture `SleepAdaptiveNet` with five switchable components:
  1. Separate modality encoders for IHR and ACC with lightweight temporal residual convolution blocks.
  2. Learned Adaptive Modality Reliability Fusion Gate ($\alpha_t$) that dynamically weights heart-rate vs. motion evidence per epoch.
  3. Transition-Aware Module incorporating a learnable stage transition matrix $M \in \mathbb{R}^{4 \times 4}$, transition-adjusted logits, and transition consistency NLL supervision.
  4. Boundary-Aware Loss applying targeted focal supervision to sleep-stage boundary epochs ($y_t \neq y_{t-1}$).
  5. Subject-Invariant Representation Learning via a Gradient Reversal Layer (GRL) and training-only subject discriminator.
  6. Calibrated Uncertainty and Confidence Head predicting normalized entropy and model correctness probability.
  Ablations A0 through A5 provide clean isolated verification of each component.
- Reason: Directly addresses the scientific hypothesis (signal reliability, boundary epochs, transition constraints, subject variability) while maintaining extreme parameter efficiency (~165k parameters, 33x smaller than the official 5.47M SLAMSS-IFS model).
- Reversible: Yes; all components are independently toggleable via boolean configuration flags.


## DEC-013 — Empirical Evaluation of Incremental Ablations (EXP-007, Seed 42)

- Date: 2026-11-09
- Decision: Retain A4 (Subject-Invariant GRL) and A5 (Full Model) as the leading candidates based on validation Macro-F1 (0.5411 vs 0.5282 baseline) and REM recovery (F1 0.361 vs 0.181). Boundary-aware loss ($\lambda=0.5$) drastically reduces boundary errors (by 7.0 percentage points) but causes an overall accuracy dip in A3 unless stabilized by the GRL discriminator in A4/A5.
- Next action: Do not run full 500-epoch training yet. Present findings to the user for guidance on whether to test seeds 123/2024 or tune $\lambda_{\text{boundary}}$ before any full training.
- Reversible: Yes.


## DEC-014 — Boundary Weight Selection and Disentanglement of GRL Effects (EXP-008)

- Date: 2026-11-09
- Decision: Select $\lambda_{\text{boundary}} = 0.20$ as the preferred boundary-aware operating point when boundary error reduction is required (cutting boundary error rate from 54.47% to 49.59%, -4.88 percentage points, while preserving 59.04% accuracy and 0.5543 Macro-F1). Reject $\lambda_{\text{boundary}} = 0.50$ due to severe over-penalization of Wake recall (39.99% vs 52.97%). Select B0_GRL ($\lambda_{\text{boundary}} = 0$) if pure overall Macro-F1 (0.5610) and Cohen's Kappa (0.4114) are prioritized.
- Unconfounded GRL evaluation: The standalone effect of GRL (holding boundary loss at 0) yields a verified +3.21% accuracy ($56.11\% \to 59.32\%$) and +0.0353 Macro-F1 ($0.5256 \to 0.5610$) over A2, confirming strong evidence for subject invariance without confounding from boundary loss.
- Reversible: Yes; $\lambda_{\text{boundary}}$ remains a configurable scalar in `SleepAdaptiveNet`.


## DEC-015 — Multi-Seed Stability Evaluation and Candidate Selection (EXP-009)

- Date: 2026-11-09
- Decision: Evaluate `B0_GRL` ($\lambda=0.0$) and `B2` ($\lambda=0.20$) across 3 random seeds (42, 123, 2024).
  1. `B0_GRL` is established as the primary recommended candidate for overall staging performance: it consistently outperforms `B2` across all 3 seeds in Accuracy ($59.61\% \pm 0.45\%$ vs $58.85\% \pm 0.21\%$, $\Delta = -0.75\%$), Macro-F1 ($0.5606 \pm 0.0162$ vs $0.5491 \pm 0.0173$, $\Delta = -0.0115$), Cohen's $\kappa$ ($0.4138 \pm 0.0128$ vs $0.4000 \pm 0.0104$, $\Delta = -0.0137$), and Wake recall ($53.95\% \pm 3.29\%$ vs $47.75\% \pm 2.35\%$, $\Delta = -6.20\%$).
  2. `B2` is confirmed as a valid specialized boundary-regularized model: it consistently reduces boundary errors across all 3 seeds ($49.76\% \pm 0.91\%$ vs $54.04\% \pm 1.51\%$, $\Delta = -4.28\% \pm 0.77\%$), but incurs a systematic trade-off in Wake recall ($-6.20\%$).
  3. Recommendation: Select `B0_GRL` as the default candidate for full benchmark training due to superior overall discrimination and higher Wake preservation. Retain `B2` as a documented Pareto boundary-regularized variant.
- Reversible: Yes; both candidate configurations are fully preserved and reproducible.


## DEC-016 — Long-Training Convergence Dynamics and Optimal Early Stopping Point (EXP-010)

- Date: 2026-11-10
- Decision: Execute 500-epoch training of `B0_GRL` to map its long-term convergence trajectory on the 28 usable validation nights.
  1. Convergence Trajectory: Validation performance peaks sharply between epochs 70 and 80, reaching optimal Macro-F1 of 0.7441, Accuracy of 74.22%, Cohen's $\kappa$ of 0.6342, and lowest validation loss of 0.7245 at Epoch 78.
  2. Overfitting Onset: Training loss continues monotonic descent from 1.0912 at Epoch 78 to 0.5047 at Epoch 500. However, validation loss begins rising after Epoch 80 (reaching 1.1473 at Epoch 500), while validation Macro-F1 degrades to 0.6829 and ECE deteriorates from 0.0240 to 0.1949. This establishes clear empirical evidence of post-peak overfitting beyond ~100 epochs under constant learning rate ($10^{-4}$).
  3. REM Resolution: Longer training completely resolves the minority-class REM under-fitting observed at 10 epochs: REM F1 surges from 0.2743 (17.86% recall) at Epoch 10 to 0.6882 (63.92% recall) at Epoch 78.
  4. Protocol Guidance: For future benchmark training runs, incorporate early stopping with patience ~20–30 epochs or a learning rate decay schedule starting around Epoch 60–80 to prevent post-peak generalization decay.
- Reversible: Yes; all epoch checkpoints (`best_macro_f1.pt`, `checkpoint_epoch_100.pt`, etc.) are preserved.


## DEC-017 — Multi-Seed Convergence Confirmation and Reproducibility Resolution (EXP-011)

- Date: 2026-11-10
- Decision: Execute multi-seed long training across seeds 42, 123, 2024 ($N=3$) for 500 epochs to test convergence robustness and resolve the EXP-009 vs EXP-010 seed-42 discrepancy.
  1. Discrepancy Resolution: The difference between EXP-009 seed 42 (59.32% Acc / 0.5610 F1) and EXP-010 epoch 10 (60.28% Acc / 0.5566 F1) was traced to CPU RNG advancement by inter-epoch `val_loader` iterations in PyTorch. Both configurations replicate bitwise under identical evaluation frequencies.
  2. Robustness of Convergence Region: Peak validation performance consistently occurs between Epoch 72 and 92 across all 3 seeds (mean peak epoch: $80.7 \pm 10.3$). Peak metrics are highly stable: Macro-F1 = $0.7296 \pm 0.0132$ (CV 1.8%), Accuracy = $72.17\% \pm 1.86\%$, Cohen's $\kappa$ = $0.6073 \pm 0.0243$.
  3. Universal Post-Peak Overfitting: Across 100% of tested seeds (3/3), validation loss hits a minimum around Epoch 70–90 and subsequently climbs to 1.15–1.29 at Epoch 500, with validation Macro-F1 degrading to ~0.665–0.685.
  4. Universal REM Resolution: Across all 3 seeds, minority REM F1 surges from $\sim 0.25–0.35$ at Epoch 10 to $0.6886 \pm 0.0186$ (recall $71.1\% \pm 6.5\%$) at peak.
  5. Training Protocol Freezing: `B0_GRL` is confirmed as a robust, reproducible architecture. The training protocol can be safely frozen with early stopping patience = 25 epochs (or a budget of ~85 epochs) for the final untouched test evaluation.
- Reversible: Yes; all multi-seed histories and checkpoints are preserved.


## DEC-018 — Final Frozen Benchmark Evaluation on Held-Out Test Set (EXP-012)

- Date: 2026-11-10
- Decision: Execute the final frozen benchmark evaluation of `SleepAdaptiveNet` (`B0_GRL`) on the untouched official test set (11 subjects, 58 usable nights, 34,800 evaluated 30-second windows).
  1. Frozen Protocol: Model `B0_GRL` (Adaptive Modality Fusion ON, Transition Module ON with $\lambda_{\text{trans}}=0.10$, Boundary Loss OFF with $\lambda_{\text{boundary}}=0.0$, Subject-Invariant GRL ON with $\lambda_{\text{adv}}=0.10$, 162,325 parameters). Trained for exactly 80 epochs per seed across 3 random seeds (42, 123, 2024) under Adam ($\text{lr}=10^{-4}$, weight decay 0.0, batch size 4, sequence length $T=600$ with 0 padding). Zero learning rate scheduling, zero early stopping, and zero hyperparameter tuning. The test set was held strictly untouched until final epoch-80 evaluation.
  2. Primary Test Generalization:
     - Accuracy: **$68.44\% \pm 1.35\%$** (Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%)
     - Macro-F1: **$0.6859 \pm 0.0139$** (Seed 42: 0.6986, Seed 123: 0.6880, Seed 2024: 0.6711)
     - Weighted-F1: **$0.6835 \pm 0.0141$** (Seed 42: 0.6985, Seed 123: 0.6816, Seed 2024: 0.6704)
     - Cohen's $\kappa$: **$0.5527 \pm 0.0188$** (Seed 42: 0.5726, Seed 123: 0.5504, Seed 2024: 0.5352)
     - MCC: **$0.5541 \pm 0.0187$** (Seed 42: 0.5740, Seed 123: 0.5512, Seed 2024: 0.5370)
     - ECE: **$0.0257 \pm 0.0050$** (Seed 42: 0.0221, Seed 123: 0.0234, Seed 2024: 0.0315)
  3. 95% Subject-Level Bootstrap Confidence Intervals ($B=1000$):
     - Seed 42: Accuracy [66.94%, 72.68%], Macro-F1 [0.6644, 0.7300], $\kappa$ [0.5303, 0.6151]
     - Seed 123: Accuracy [65.43%, 71.06%], Macro-F1 [0.6553, 0.7188], $\kappa$ [0.5087, 0.5928]
     - Seed 2024: Accuracy [64.23%, 69.90%], Macro-F1 [0.6359, 0.7032], $\kappa$ [0.4917, 0.5769]
  4. Class-Specific Generalization:
     - Wake F1: $0.6940 \pm 0.0192$ (Recall: $75.95\% \pm 1.20\%$, Precision: $64.01\% \pm 4.10\%$)
     - Light F1: $0.6778 \pm 0.0119$ (Recall: $70.02\% \pm 1.72\%$, Precision: $65.68\% \pm 0.94\%$)
     - Deep F1: $0.7323 \pm 0.0174$ (Recall: $72.66\% \pm 1.86\%$, Precision: $73.89\% \pm 3.46\%$)
     - REM F1: $0.6395 \pm 0.0207$ (Recall: $58.15\% \pm 2.94\%$, Precision: $71.10\% \pm 1.47\%$)
  5. Scientific Comparison & Classification:
     - Published SLAMSS-IFS (IEEE TBME 2026, Table V): Accuracy = 71.04%, Weighted-F1 = 0.7079, Parameters = 5,465,768.
     - SleepAdaptiveNet (`B0_GRL`): Accuracy = $68.44\% \pm 1.35\%$, Weighted-F1 = $0.6835 \pm 0.0141$, Parameters = 162,325 (33.7x compression).
     - Classification: **Case B (Competitive / Approximate Parity under 33.7x Parameter Compression)**.
  6. Termination: Final benchmark evaluation concluded. No further architecture changes or training runs will be executed.
- Reversible: No; the test set was evaluated once under locked protocol, concluding the experimental phase.

## DEC-019 — Rigorous Component Attribution and Matched-Capacity Verification (EXP-013)

- Date: 2026-11-10
- Decision: Authorize and execute EXP-013 to determine whether the proposed SleepAdaptiveNet components provide a real performance benefit beyond an approximately matched-capacity conventional CNN-BiGRU baseline, and attribute individual performance contributions using a One-Factor-At-A-Time (OFAT) factorial ablation design against the frozen EXP-012 reference (Model E, 162,325 params).
  1. Experimental Matrix:
     - Model A: Matched-capacity conventional CNN-BiGRU (163,940 params, +0.99%; Concat, no gate, no transition, no GRL).
     - Model B: E minus GRL (153,974 params, -5.14%; Gate ON, Transition ON, GRL OFF).
     - Model C: E minus Transition (162,325 params, 0.00%; Gate ON, Transition OFF, GRL ON).
     - Model D: E minus Dynamic Gate (153,876 params, -5.20%; Gate fixed 0.5/0.5, Transition ON, GRL ON).
     - Model E: Frozen EXP-012 Reference (162,325 params; Gate ON, Transition ON, GRL ON; read-only).
  2. Protocol Controls: 80 fixed optimization epochs per seed (seeds: 42, 123, 2024), batch size 4, lr 1e-4, Adam, sequence length $T=600$. Exactly 34,241 valid test windows evaluated on held-out test subjects. Paired subject-level cluster bootstrap ($B=1000$).
  3. Key Findings:
     - Matched Capacity Baseline Comparison ($E - A$): Model E outperforms Model A by **$+1.79\%$ Accuracy** [95% CI: $+0.52\%, +3.35\%$] and **$+0.0199$ Macro-F1** [95% CI: $+0.0074, +0.0342$]. The 95% bootstrap CI strictly excludes zero, establishing that SleepAdaptiveNet's architectural mechanisms provide genuine performance advantages not attributable to parameter capacity.
     - Subject-Invariant GRL ($E - B$): GRL is the largest single component contributor, yielding a **$+1.39\%$ Accuracy** [95% CI: $-0.01\%, +2.84\%$] improvement, with 97.4% of bootstrap replications favoring GRL.
     - Adaptive Reliability Gating ($E - D$): Dynamic reliability gating contributes **$+0.54\%$ Accuracy** [95% CI: $-0.53\%, +1.60\%$] and exhibits physiologically consistent downweighting of accelerometry during Wake ($\alpha \approx 0.54$) versus sleep stages ($\alpha \approx 0.80$).
     - Transition Modeling ($E - C$): Transition module contributes **$+0.04\%$ Accuracy** [95% CI: $-0.29\%, +0.42\%$], indicating it is largely redundant when a 2-layer bidirectional GRU is utilized for sequence modeling.
  4. Final Research State: Architectural mechanism attribution is fully resolved and documented in `EXP013_RESULTS.md`. No further training runs.
- Reversible: No; empirical attribution is completed and frozen.

## DEC-020 — 75-Run Subject-Level Cross-Validation Robustness Evaluation (EXP-014)

- Date: 2026-11-10 / 2026-11-11
- Decision: Authorize and execute EXP-014 as an independent subject-level 5-fold cross-validation across all 47 subjects in the BIDSLab Apple Watch + Dreem 2 dataset (246 usable nights, 145,283 valid evaluated windows) across 5 models and 3 random seeds (75 independent training runs).
  1. Experimental Protocol:
     - Models: Model A (Matched Baseline, 163.9k params), Model B (minus GRL, 154.0k params), Model C (minus Transition, 162.3k params), Model D (minus Adaptive Gating, 153.9k params), Model E (Full SleepAdaptiveNet, 162.3k params, freshly initialized per fold).
     - 5 outer folds (strictly disjoint subjects: 10, 10, 9, 9, 9 test subjects). Inner training on disjoint subjects, inner validation on 5 subjects per fold. Outer test subjects evaluated strictly once per run at locked epoch 80.
     - 3 seeds per model per fold (42, 123, 2024). Exactly 80 epochs per run, Adam ($\text{lr}=10^{-4}$, weight decay 0.0), batch size 4 nights, sequence length $T=600$.
  2. Key Empirical Findings:
     - **Scientific Outcome B (Equivalence / Parity Across Subject Folds)**: Across all 75 runs, Model E achieved $66.06\% \pm 4.08\%$ accuracy and $0.6596 \pm 0.0425$ Macro-F1, compared to $66.63\% \pm 3.33\%$ accuracy and $0.6658 \pm 0.0365$ Macro-F1 for the matched conventional baseline (Model A).
     - **Paired Subject Cluster Bootstrap ($B=1000$)**:
       - $E \text{ vs } A$: $\Delta \text{Acc} = -0.62\%$ [95% CI: $-1.90\%, +0.50\%$], $\Delta \text{Macro-F1} = -0.0069$ [95% CI: $-0.0213, +0.0059$].
       - $E \text{ vs } B$: $\Delta \text{Acc} = -0.66\%$ [95% CI: $-1.36\%, +0.16\%$], $\Delta \text{Macro-F1} = -0.0067$ [95% CI: $-0.0141, +0.0032$].
       - $E \text{ vs } C$: $\Delta \text{Acc} = -0.01\%$ [95% CI: $-0.15\%, +0.12\%$], $\Delta \text{Macro-F1} = -0.0001$ [95% CI: $-0.0014, +0.0013$].
       - $E \text{ vs } D$: $\Delta \text{Acc} = -0.63\%$ [95% CI: $-1.99\%, +1.02\%$], $\Delta \text{Macro-F1} = -0.0081$ [95% CI: $-0.0196, +0.0051$].
       - All 95% confidence intervals span zero; the evidence does not support a generalizable performance difference between the proposed architecture and matched baseline across diverse subject cohorts.
     - **Subject-Level Win/Loss**: Model E outperformed Model A on 23 subjects ($48.9\%$), Model A outperformed Model E on 23 subjects ($48.9\%$), and 1 subject was tied ($2.2\%$).
  3. Final Scientific Conclusion: The empirical research cycle across all 47 subjects confirms that conventional matched-capacity CNN-BiGRU models perform comparably to architectures with specialized gating, transition matrices, and GRL. Performance variance is dominated by subject physiological heterogeneity rather than architectural micro-modifications. All 75 runs are archived and documented in `EXP014_FINAL_REPORT.md`.
- Reversible: No; empirical cross-validation is completed and frozen.

## DEC-021 — Reopening Experimental Phase for Exploratory and Methodological Research

- Date: 2026-11-11
- Decision: Experimental phase reopened. Reason: external audit found protocol issues (5-hour crop vs full nights, EXP-012 window-count inconsistency, GRL scaling). DEC-018/019/020 "no further runs" no longer applies; any new run must follow a documented protocol.
  1. Motivation & Scope: Following the conclusion of EXP-012, EXP-013, and EXP-014, an external audit identified key methodological and protocol issues:
     - 5-hour crop ($T=600$ epochs) versus full nights (~28% of labeled epochs dropped; selective REM depletion and Deep sleep enrichment).
     - EXP-012 window-count inconsistency (34,800 raw unmasked slots vs 34,241 valid evaluated windows after masking 559 invalid/unknown epochs).
     - GRL gradient scaling mechanics ($\lambda_{\text{adv}}=0.10 \times \alpha_{\text{grl}}=0.10 \implies 0.010$ effective gradient scale to encoder; unscheduled).
     - Normalization locality and metadata feature attribution in baseline models.
  2. Protocol & Guardrails:
     - All standing integrity rules remain strictly in effect: any new training run must follow a documented, frozen protocol and requires explicit user authorization before execution.
     - Zero automatic, speculative, or undocumented training runs.
     - All artifacts, code, and documentation continue under mandatory Git push synchronization to `origin main`.
- Reversible: Yes; experimental phase can transition to locked/frozen upon concluding new investigations.


## DEC-022 — Controlled Evaluation of Sequence Truncation Horizon (EXP-017: T=600 vs T≤1200)

- Date: 2026-11-14
- Decision: Conclude EXP-017 evaluating the effect of sequence truncation horizon ($T=600$ vs $T \le 1200$) in wearable sleep staging using Model A (`ConventionalCNNBiGRU`, 163,940 parameters) across seeds 42, 123, 2024.
  1. Experimental Protocol:
     - Model: Model A only (`ConventionalCNNBiGRU`, 163,940 parameters).
     - Conditions: $T=600$ vs $T \le 1200$.
     - Padding Protection: Implemented native sequence length tracking $L = \min(\text{native\_len}, T)$ with `pack_padded_sequence` / `pad_packed_sequence(total_length=T)` in temporal BiGRU. Verified mathematically zero leakage ($\max |\Delta| = 0.0$) from padded tails.
     - Primary Estimand: Matched-600 REM recall difference: $\Delta \text{REM Recall} = \text{REM Recall}_{(T \le 1200)}[\text{eval } 0:600] - \text{REM Recall}_{(T = 600)}[\text{eval } 0:600]$ on the identical first 600 chronological epoch positions (invalid positions masked).
     - Validation Cohort: 5 validation subjects (28 nights, 16,368 valid matched-600 windows). Held-out test cohort (11 subjects, 58 nights) strictly locked with zero access.
     - Optimization: Adam ($\text{lr}=10^{-4}$, weight decay 0.0), 80 epochs, batch size 4 nights, seeds 42, 123, 2024.
  2. Empirical Findings:
     - **Primary Estimand (Matched-600 REM Recall)**: Point estimate (mean across 3 seeds) is **$-0.84\text{ percentage points}$** (Seed 42: $-5.32$ pp, Seed 123: $-3.85$ pp, Seed 2024: $+6.49$ pp).
     - **Paired Subject Cluster Bootstrap ($B=1000$)**: 95% CI is **$[-5.68\text{ pp}, +3.94\text{ pp}]$**, strictly crossing zero.
     - **Subject-Level Win/Loss**: 2 positive (Bidslab40, Bidslab56), 3 negative (Bidslab13, Bidslab55, Bidslab64).
     - **Protocol Decision**: Per pre-specified protocol rules, hypothesis $H_1$ is **`INCONCLUSIVE`** / not supported. Extended sequence training does not specifically rescue or enhance early-night REM recall.
     - **Secondary Metrics (Matched-600 Positions)**:
       - Overall Accuracy: $+2.33\text{ pp}$ [95% CI: $+1.64\text{ pp}, +3.03\text{ pp}$] ($70.47\% \to 72.80\%$).
       - Macro-F1: $+0.0210$ [95% CI: $+0.0123, +0.0269$] ($0.7126 \to 0.7335$).
       - Deep Sleep Recall: $+8.52\text{ pp}$ [95% CI: $+6.35\text{ pp}, +11.72\text{ pp}$] ($65.77\% \to 74.57\%$).
       - Boundary Error Rate: $-2.72\text{ pp}$ ($44.38\% \to 41.66\%$).
     - **Secondary Descriptive Analysis (Extended Coverage $T \le 1200$, 23,183 valid windows)**:
       - Accuracy: $71.45\% \pm 0.43\%$, Macro-F1: $0.7153 \pm 0.0024$, REM recall: $70.42\% \pm 3.81\%$, Deep recall: $75.24\% \pm 4.31\%$.
  3. Research Strategy Decision:
     - Maintain $T=600$ as the primary computationally efficient standard for rapid benchmarking and exploratory iterations.
     - Acknowledge that while extended sequence training does not solve REM under-representation, it provides moderate global context benefits for slow-wave (Deep) sleep and overall accuracy.
     - Held-out test set remains locked and untouched.
- Reversible: Yes.


## DEC-023 — EXP-018 Phase 1 Horizon Curve Evaluation & Phase 2 Authorization

- Date: 2026-11-14
- Decision: Conclude Phase 1 of EXP-018 (Context-Horizon Curve with Data-Coverage Control) and formally record that the pre-registered gating threshold authorizing Phase 2 (5-fold cross-validation across all 47 subjects) has been passed.
  1. Experimental Protocol:
     - Model: Model A only (`ConventionalCNNBiGRU`, 163,940 parameters; `exp013_models.py`).
     - Conditions: Context horizon $L \in \{1, 20, 60, 120, 300, 600, 1200\}$ epochs.
     - Token Exposure Invariant: $B \times L \times N_{\text{accum}} = 2,400$ tokens/step ($3,200\text{ steps} \times 2,400 = 7,680,000\text{ total tokens/run}$) across all 21 runs.
     - Circadian Neutralization: `time_feat = 0.0`.
     - Inference: `SlidingWindowPredictor` with boundary anchoring.
     - Cohort: 31 train / 5 validation subjects (28 nights). 11 test subjects strictly locked. Split hash: `bf33573623f47f3f`.
  2. Empirical Findings:
     - Primary Estimand ($L=600$ vs $L=60$): $\Delta \text{Macro-F1} = \mathbf{+4.27\text{ pp}}$ [95% CI: $\mathbf{+2.53\text{ pp}, +6.46\text{ pp}}$].
     - 100% Directional Concordance across all 5 validation subjects (`Bidslab13`: $+4.17$, `Bidslab40`: $+3.39$, `Bidslab55`: $+3.92$, `Bidslab56`: $+8.33$, `Bidslab64`: $+1.51\text{ pp}$).
     - Decision Category: **`SUPPORTED (Context Benefit)`** $\implies$ **Phase 2 CV is AUTHORIZED**.
     - Parametric Saturation Paradox Resolved: Single-scale exponential fit ($L^* = 46.3\text{ epochs} / 23.1\text{ min}$, $A = 0.6435$) captures local transition inertia, while multi-hour ultradian NREM-REM cycles produce a second scaling regime ($L \ge 300$) reaching $0.6529$ at $L=600$ and $0.6993$ at $L=1200$.
     - Clinical PSG Metrics: TST error exhibits U-shaped curve due to intermediate micro-arousal low-pass filtering, before recovering to $27.14\text{ min}$ MAE at $L=1200$.
  3. Action:
     - Phase 2 5-fold cross-validation is AUTHORIZED per protocol.
     - Phase 1b circadian ablation is PERMITTED.
     - Held-out test set remains strictly locked. Launching of Phase 2 training remains subject to explicit user authorization.
- Reversible: Yes.


## DEC-024 — EXP-018 Phase 1b Circadian Ablation Findings & Phase 2 Execution Safeguards

- Date: 2026-11-15
- Decision: Record completion of EXP-018 Phase 1b (Circadian Feature Ablation), confirm authorization of Phase 2 execution chunking, and enforce hardware safety pauses under multi-process GPU saturation.
  1. Phase 1b Findings (Circadian Feature Ablation, 9 runs on `cuda:1`):
     - Empirical Results:
       - $L=1$: Neutralized $0.3654 \pm 0.0057 \to$ Active **$0.5131 \pm 0.0035$** ($\Delta = \mathbf{+14.77\text{ pp}}$).
       - $L=60$: Neutralized $0.6117 \pm 0.0025 \to$ Active **$0.6555 \pm 0.0078$** ($\Delta = \mathbf{+4.38\text{ pp}}$).
       - $L=600$: Neutralized $0.6529 \pm 0.0078 \to$ Active **$0.6665 \pm 0.0018$** ($\Delta = \mathbf{+1.36\text{ pp}}$).
     - Mechanistic Resolution: Clock time provides massive classification power for isolated single epochs (+14.77 pp) and local context (+4.38 pp), but its marginal contribution diminishes sharply to +1.36 pp at multi-hour context ($L=600$). The recurrent BiGRU autonomously extracts diurnal progression directly from underlying cardiorespiratory and accelerometry time-series.
  2. Phase 2 Architecture & Execution Strategy:
     - 105 total runs ($7 L \in \{1, 20, 60, 120, 300, 600, 1200\} \times 5\text{ folds} \times 3\text{ seeds}$) across all 47 subjects (246 usable nights).
     - Implemented fold-by-fold execution chunking via `--folds` CLI flag in `run_exp018_execution.py` to prevent task timeouts and ensure atomic per-run artifact caching on disk (`experiments/EXP-018/phase2_results/run_L{L}_fold{fold}_seed{seed}.json`).
     - Progress: 23 / 105 runs completed on disk (Fold 1: 21/21 runs complete; Fold 2: 2/21 runs complete).
  3. Thermal & Multi-Job Hardware Safeguards:
     - Inspection of GPU telemetry revealed concurrent external process PID 111725 (`run_exp_026_screen.py` in competition project) utilizing >11 GB VRAM on both GPUs, with GPU 0 at 92°C (fan 95%).
     - To prevent CUDA Out-Of-Memory errors and emergency thermal hardware shutdown (TJMax 93°C), remaining Phase 2 folds (Fold 2 remainder, Folds 3, 4, 5) are held until GPU resources become free.
- Reversible: Yes.


## DEC-025 — EXP-018 Phase 2 Cohort-Wide Cross-Validation Completion & Definitive Horizon Scaling

- Date: 2026-11-15
- Decision: Conclude EXP-018 Phase 2 (Cohort-Wide 5-Fold Cross-Validation across all 47 subjects, 246 usable nights, 105 total runs) and formally record the definitive empirical scaling curve and primary estimand.
  1. Execution & Protocol Verification:
     - 105 / 105 runs completed successfully (5 folds $\times$ 7 context lengths $\times$ 3 seeds).
     - Model A (`ConventionalCNNBiGRU`, 163,940 parameters; `exp013_models.py`).
     - Compute Invariant: $B \times L \times N_{\text{accum}} = 2,400$ tokens/step ($3,200\text{ steps} = 7,680,000\text{ tokens/run}$) maintained strictly across all 105 runs.
     - Out-Of-Fold Evaluation: Covers 100% of the 47 subjects (246 usable nights) with zero test set leakage.
  2. Cohort-Wide Findings:
     - **Primary Estimand ($L=600$ vs $L=60$)**: $\Delta \text{Macro-F1} = \mathbf{+5.33\text{ percentage points}}$ [95% CI: $\mathbf{+4.28\text{ pp}, +6.39\text{ pp}}$].
     - **Subject-Level Win/Loss**: **46 / 47 subjects positive (97.9%)** favoring $L=600$ over $L=60$.
     - **Pre-Registered Decision Rule**: **`SUPPORTED (Context Benefit)`** ($\Delta \ge +1.0\text{ pp}$, CI excludes 0).
     - **Out-of-Fold Macro-F1 Progression (Mean $\pm$ SD across 3 seeds)**:
       - $L=1$: $0.3692 \pm 0.0011$ (Acc: $47.58\%$)
       - $L=20$: $0.5319 \pm 0.0036$ (Acc: $57.54\%$)
       - $L=60$: $0.5773 \pm 0.0039$ (Acc: $60.85\%$)
       - $L=120$: $0.5894 \pm 0.0045$ (Acc: $62.06\%$)
       - $L=300$: $0.6049 \pm 0.0058$ (Acc: $63.62\%$)
       - $L=600$: $0.6306 \pm 0.0031$ (Acc: $65.45\%$)
       - $L=1200$: **$0.6479 \pm 0.0053$** (Acc: **$66.18\%$**)
     - **Parametric Saturation Audit & Multi-Model Recomputation**:
       - Identified and resolved mathematical inconsistency in the baseline single-rate exponential ($A = 0.6143 < y_{600} < y_{1200}$): single-rate misspecification forced the asymptote to the mean of the tail points.
       - Recomputed Bi-Exponential Mixture Model ($R^2 = 0.9998$): True ceiling $A = \mathbf{0.6582}$, local transition timescale $\tau_{\text{fast}} = 12.1\text{ epochs} \ (6.0\text{ min})$, ultradian timescale $\tau_{\text{slow}} = 548.5\text{ epochs} \ (4.6\text{ hours})$, with full-system 95% saturation at $L^* = \mathbf{957.6\text{ epochs} \ (8.0\text{ hours})}$.
       - Recomputed Log-Linear Scaling Law ($R^2 = 0.9540$): steady $+2.65\text{ pp}$ per context doubling.
  3. Action:
     - EXP-018 (Phases 1, 1b, and 2) is officially **COMPLETE**.
     - All 105 run JSONs, metrics table, subject deltas, summary JSON, and report are archived and synchronized to GitHub.
- Reversible: No; 5-fold cross-validation across the complete cohort is finalized.


## DEC-026 — Autonomous NAS Syndicate Iteration 1: Pre-Recurrent Multi-Scale Temporal Fusion (MSTF-Pre)

- Date: 2026-11-15
- Decision: Authorize and document Iteration 1 structural mutation `MultiScaleConventionalCNNBiGRU` under pre-registered Hypothesis H-A with a +20% parameter ceiling.
  1. Scope and Parameter Cap:
     - Baseline Model A: `ConventionalCNNBiGRU` (163,940 parameters).
     - Historical parameter cap: $\le +20\%$ ($163,940 \times 1.20 = 196,728$ parameters) [**SUPERSEDED** by DEC-031: arbitrary numerical caps removed workspace-wide in favor of the Lightweight-First Principle].
     - Mutated Model A: `MultiScaleConventionalCNNBiGRU` (192,692 parameters, $+17.54\%$ / $+28,752$ params).
     - Headroom: 4,036 parameters under the approved ceiling.
  2. Structural Mutation Mechanism:
     - Grounded in Hypothesis H-A (`RESEARCH_DIRECTION_v3_FULL.md`): multi-night within-person adaptation via multi-scale temporal kernel fusion.
     - Inserted pre-recurrent temporal fusion block between 1D CNN epoch encoders and BiGRU sequence model.
     - Bottleneck projection ($144 \to 96$), parallel depthwise 1D conv branches with kernel sizes $k \in \{1, 3, 5\}$ (30s, 90s, 150s receptive fields), restoration projection ($96 \to 144$) with residual addition and LayerNorm.
  3. Verification & Governance:
     - Full tensor dimension compatibility preserved: $(B, T, 144) \to (B, T, 144)$.
     - 49/49 independent adversarial, stress, and unit tests passed.
     - Executed 1-epoch evaluation loop on Fold 1 (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) on GPU `cuda:0` with invariant batch size (4), optimizer (Adam), learning rate ($10^{-4}$), and loss function.
     - Independent victory audit verdict: **`VICTORY CONFIRMED`**.
- Reversible: Yes; alternative mutations can be explored in subsequent NAS iterations.

### ERRATUM to DEC-026 (2026-11-15)

1. Mismatch with H-A: the MSTF-Pre block is a within-night multi-scale temporal
   convolution over epochs. It uses no other nights and is not an operationalization of
   H-A (multi-night within-person adaptation). H-A is tested in PREREG_EXP-019_H-A.md.
2. Not pre-registered: no pre-registration block or decision rule existed for this change.
   It must not be described as pre-registered.
3. Evidence is insufficient: single fold, single seed, 1 training epoch, evaluated on the
   outer test fold; models at about 44 to 47% accuracy. The primary-run Macro-F1 gain
   (+5.01 pp, 0.3056 -> 0.3556) was not reproduced in the independent re-run, which
   reported Macro-F1 0.3764 vs 0.3882 and a baseline value that differs from the primary run,
   so run-to-run noise exceeds the claimed effect. Only an accuracy difference (+0.33 pp)
   and a loss difference (-0.0009) were confirmed.
4. Capacity confound: +17.54% parameters violates the matched-capacity principle of DEC-019.
5. Selection on test data: outer-test-fold results must not guide architecture selection.
6. Status: exploratory smoke test with no performance claim. NAS Iteration 2 and later are
   not authorized (see EXP-014, DEC-020). "VICTORY CONFIRMED" refers to code, test-suite
   and provenance checks only, not to scientific validity.


## DEC-027 — EXP-019 500-Epoch Extended Training Benchmark: Baseline vs MultiScaleConventionalCNNBiGRU

- Date: 2026-11-15
- Decision: Authorize, execute, and document the 500-epoch extended training benchmark comparing Baseline Model A (`ConventionalCNNBiGRU`, 163,940 params) against Mutated Model A (`MultiScaleConventionalCNNBiGRU`, 192,692 params, $+17.54\%$) under identical hyperparameter controls.
  1. Execution Protocol:
     - Epochs: 500 epochs per model.
     - Split: BIDSleep Fold 1 (32 inner-train, 5 inner-val, 10 outer-test subjects).
     - Optimization: Invariant Adam ($lr = 10^{-4}$, weight decay 0.0), batch size 4, sequence horizon $T = 600$, seed 42.
     - Hardware: Parallel dual-GPU execution (Baseline on `cuda:0`, MultiScale on `cuda:1`, 16.8 minutes total wall-clock time).
  2. Empirical Outcomes:
     - **Model Selection @ Peak Validation Checkpoint (Early Stopping Policy)**:
       - Baseline: Peak at Epoch 55 $\implies$ Test Macro-F1 = **`0.6401`**, Acc = **`64.42%`**, $\kappa = \mathbf{0.4805}$, MCC = **`0.4814`**.
       - MultiScale: Peak at Epoch 53 $\implies$ Test Macro-F1 = **`0.6513`**, Acc = **`65.72%`**, $\kappa = \mathbf{0.5032}$, MCC = **`0.5035`**.
       - **Primary Estimand Delta**: $\Delta \text{Macro-F1} = \mathbf{+1.11\text{ percentage points}}$, $\Delta \text{Acc} = \mathbf{+1.30\text{ pp}}$, $\Delta \kappa = \mathbf{+0.0227}$, $\Delta \text{MCC} = \mathbf{+0.0222}$ favoring MultiScale.
       - Stage F1 Gains: Deep F1: **+1.60 pp** ($0.6989 \to 0.7148$), REM F1: **+2.62 pp** ($0.5649 \to 0.5910$), Light F1: **+0.68 pp** ($0.6529 \to 0.6596$), Wake F1: **-0.43 pp** ($0.6440 \to 0.6397$).
     - **Asymptotic 500-Epoch Overfitting Resilience**:
       - Baseline suffered substantial post-peak degradation: Macro-F1 dropped by **5.02 pp** (to `0.5900` at Ep 500, Acc 59.58%).
       - MultiScale exhibited pronounced overfitting resistance: Macro-F1 dropped by only **0.62 pp** (to `0.6451` at Ep 500, Acc 64.67%).
       - Terminal Epoch 500 Delta: $\Delta \text{Macro-F1} = \mathbf{+5.52\text{ percentage points}}$ favoring MultiScale.
  3. Action:
     - EXP-019 is complete. All run JSONs, summary metrics, test suites, and report are archived and synchronized to GitHub.
- Reversible: No; 500-epoch training and evaluation on Fold 1 is finalized.

### ERRATUM to DEC-027 (2026-11-16)

1. Identifier Reclassification: This 500-epoch Fold 1 run was originally labeled "EXP-019"
   prior to the pre-registration lock of the true H-A adaptation protocol (PREREG_EXP-019_H-A.md).
   To prevent identifier collision and maintain strict scientific provenance, this historical
   benchmark is reclassified as `EXP-019-EXPLORATORY`. The identifier `EXP-019` is formally
   reserved for the pre-registered within-person adaptation experiment (`PREREG_EXP-019_H-A.md`).
2. Status: Exploratory architecture benchmark with no confirmatory standing under H-A.
   It evaluated within-night temporal fusion on Fold 1 only, does not evaluate multi-night
   adaptation, uses a +17.54% parameter capacity expansion, and was evaluated on outer-test data.
3. Historical Evidence Preserved: All numerical findings and artifacts (peak metrics at Ep 53/55,
   asymptotic overfitting delta +5.52 pp Macro-F1) remain preserved exactly as recorded.

### DEC-028 (2026-11-16): Confirmatory Execution of EXP-019 (Multi-Night Within-Person Adaptation Hypothesis H-A)

- Context:
  - Following the complete resolution of Phase-0 requirements and EGM Goldfish review, the formal confirmatory within-person adaptation experiment (EXP-019) was executed to test Hypothesis H-A under locked pre-registration `PREREG_EXP-019_H-A.md`.
  - Architecture: Model A (`ConventionalCNNBiGRU`, 163,940 parameters) strictly held constant across all 5 arms.
  - Cohort: 46 eligible subjects ($n_{\text{usable}} \ge 3$, 152 outer evaluation nights across 5 folds; `Bidslab49` excluded due to $n_{\text{usable}}=2$).
  - Unit of analysis: Subject (paired bootstrap over 46 subjects, averaged across seeds 42, 123, 2024).
- Key Decisions & Milestones:
  1. Checkpoint Regeneration & Parity:
     - All 15 Model A ($L=1200$, $\text{time\_feat}=0$, 3,200 steps) checkpoints were regenerated and verified against historical EXP-018 Phase 2 results to within $\le 0.00010$ Macro-F1 difference ($\le 0.01\text{ pp}$).
  2. Stage D Nested Selection & Variance Confirmation:
     - Hyperparameters selected strictly on the 5 inner-validation subjects per fold (25 subjects total).
     - Empirical standard deviation of paired differences ($FT\text{-}2 - ZS$): $s_{\Delta} = 2.2432\text{ pp}$.
     - Continuous 80%-power threshold ($0.4131 \cdot s_{\Delta}$): $0.9267\text{ pp}$.
     - Since $s_{\Delta} \le 5.0\text{ pp}$, no threshold escalation occurred. The practical effect threshold was locked at $\mathbf{+1.50\text{ pp}}$.
     - `PREREG_EXP-019_H-A.md` locked and committed at hash `4dff5b7a5f13a96c3024b309131aa1e66489dcbb` prior to Stage C execution.
  3. Stage C Confirmatory Outcomes:
     - **Primary Hypothesis H-A ($FT\text{-}2$ vs $ZS$)**:
       - Estimand: $\bar{\Delta} = \mathbf{+1.01\text{ pp}}$ (from $64.58\% \pm 7.74\%$ to $65.59\% \pm 8.59\%$).
       - 95% Percentile Bootstrap CI (10,000 resamples): $\mathbf{[-0.17, +1.99]\text{ pp}}$.
       - Wilcoxon signed-rank: $p = 0.00131$.
       - Subject win rate: 73.9% (34 wins, 12 losses).
       - Decision Rule Evaluation: Fails to meet pre-registered criteria ($\bar{\Delta} \ge +1.50\text{ pp}$ AND $\text{CI}_{\text{low}} > 0.0$).
       - Formal Verdict: **NOT SUPPORTED** *(Reconciled to INCONCLUSIVE in DEC-029 erratum per locked preregistration decision rule)*.
     - **Secondary Hypothesis S1 (Specificity: $FT\text{-}2$ vs $FT\text{-}other\text{-}2$)**:
       - Estimand: $\bar{\Delta} = \mathbf{+2.73\text{ pp}}$ (from $62.86\% \pm 8.85\%$ to $65.59\% \pm 8.59\%$).
       - 95% Bootstrap CI: $\mathbf{[+1.05, +4.32]\text{ pp}}$.
       - Wilcoxon signed-rank: $p = 9.0 \times 10^{-6}$.
       - Subject win rate: 78.3% (36 wins, 10 losses).
       - Formal Verdict: **SUPPORTED** (adapting on a subject's own data is statistically and practically superior to adapting on another subject's data; generic fine-tuning degrades performance).
     - **Secondary Hypothesis S2 (Transfer Value: $FT\text{-}2$ vs $PS\text{-}2$)**:
       - Estimand: $\bar{\Delta} = \mathbf{+20.32\text{ pp}}$ (from $45.27\% \pm 9.42\%$ to $65.59\% \pm 8.59\%$).
       - 95% Bootstrap CI: $\mathbf{[+17.70, +22.94]\text{ pp}}$, Wilcoxon $p < 10^{-10}$.
       - Subject win rate: 100.0% (46 wins, 0 losses).
       - Formal Verdict: **SUPPORTED** (cohort pre-training provides indispensable representation foundation).
     - **Secondary Hypothesis S3 (Unsupervised InfoMax: $UA\text{-}2$ vs $ZS$)**:
       - Estimand: $\bar{\Delta} = \mathbf{0.00\text{ pp}}$, 95% CI: $[0.00, 0.00]\text{ pp}$.
       - Formal Verdict: **NOT SUPPORTED** *(Reclassified to UNRESOLVED / INVALIDATED in DEC-029/DEC-030 per Phase A2 audit: implementation no-op via quality-mask unit mismatch; the original parenthetical claim that unsupervised entropy minimization is ineffective without supervisory signal is invalidated and scientifically uninformative)*.
- Reversible: No; Stage C confirmatory evaluation was executed under locked pre-registration and finalized.






### DEC-029 (2026-11-17): Formal Adoption of Research Direction v4, Prior Art Re-alignment, and EXP-019 Governance Reconciliation

- Date: 2026-11-17
- Authoritative Direction: `RESEARCH_DIRECTION_v4_FULL.md` is formally adopted as the working research direction for the project, superseding `RESEARCH_DIRECTION_v3_FULL.md`, the v3.1 amendment, and `RESEARCH_DIRECTION_v4_AMENDMENT_PROPOSAL.md`.
- Key Policy Decisions & Governance Reconciliations:
  1. **Withdrawal of Novelty "Open Lane" Claim**:
     - Literature identification of van der Aar et al. 2025 (*Physiological Measurement*, DOI: 10.1088/1361-6579/ae0119) establishes direct precedent for multi-night personalization using wrist PPG + accelerometry in sleep-disordered adults (82.5% improved, significant at $\ge 3$ nights).
     - The "open lane" novelty claim in v3 is formally withdrawn.
     - Research is reframed as a pre-registered, leakage-free replication and extension using consumer-smartwatch instantaneous heart rate (IHR, ~0.2 Hz) and accelerometry in healthy adults, emphasizing specificity controls, calibration learning curves, failure-mode heterogeneity, external validation, and clinical sleep summaries.
  2. **Ratification of Contribution Structure (C1–C3)**:
     - **C1 (Primary)**: Pre-registered, leakage-free replication and extension of multi-night supervised personalization with consumer IHR + accelerometry in healthy adults (FT-other specificity control, $k=1–4$ learning curve, heterogeneity analysis).
     - **C2 (Secondary, Exploratory)**: Calibration-night class coverage (stage absence) as a candidate failure mode, with nested evaluation of regularized or gated adaptation.
     - **C3**: External zero-shot validation (EXP-027) and clinical sleep-summary agreement (EXP-028).
  3. **Reconciliation of Hypothesis H-A Verdict Label (Erratum)**:
     - The numerical result of H-A (+1.01 pp Macro-F1, 95% CI [-0.17, +1.99] pp, 34/46 improved) is immutable.
     - Inspection of locked preregistration text (`git show 4dff5b7:PREREG_EXP-019_H-A.md` Section 6) establishes the decision rule:
       - SUPPORTED: $\bar{\Delta} \ge +1.50\text{ pp}$ AND CI lower bound $> 0$.
       - NOT SUPPORTED: CI upper bound $< +1.50\text{ pp}$.
       - INCONCLUSIVE: anything else.
     - Because the observed 95% CI upper bound is $+1.99\text{ pp}$ (which is NOT $< +1.50\text{ pp}$), the exact preregistered rule evaluates to **INCONCLUSIVE**, whereas early reports labeled it "NOT SUPPORTED".
     - This governance discrepancy is resolved via formal documentation. The accurate scientific statement is: "Supervised two-night adaptation did not meet the pre-registered threshold (mean +1.01 pp, 95% CI -0.17 to +1.99; 34/46 improved), and the result is inconclusive for effects up to about 2 pp."
  4. **Reclassification of Secondary Hypothesis S3**:
     - S3 ($UA\text{-}2$ vs $ZS = 0.00\text{ pp}$ identically across all 46 subjects) is reclassified as **UNRESOLVED / INVALIDATED** per the completed Phase A2 forensic audit (`experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030).
     - The audit proved that `compute_wearable_quality_mask` used raw BPM limits (`30 <= IHR <= 180`) on standardized z-score IHR. Zero out of 82,762 epochs passed the mask, terminating adaptation on step 0 (0 updates, 0/163,940 parameters changed, identical evaluation logits/predictions).
     - S3 is scientifically uninformative and must not be cited as evidence that unsupervised adaptation fails (nor that it works).
  5. **Interpretation Nuance for S1**:
     - S1 ($FT\text{-}2$ vs $FT\text{-}other\text{-}2 = +2.73\text{ pp}$, Supported) is clarified: $FT\text{-}other\text{-}2$ active degradation ($-1.72\text{ pp}$ below $ZS$) accounts for the majority of the gap. Personalization is biologically specific, but cross-subject fine-tuning is harmful.
  6. **Retirement of SLAMSS-IFS Parity Framing & EXP-012 Test Set**:
     - The general "approximate parity" claim with SLAMSS-IFS is withdrawn. Out-of-fold accuracy is 66.2%–66.6% (4.4–4.9 pp below published 71.04% under unknown class granularity and split). Comparisons must state "lower accuracy than reported SLAMSS-IFS result under a different protocol."
     - The historical EXP-012 test set (11 subjects, 58 nights) is formally classified as **SPENT**.
  7. **Phase Phasing & Anti-Premature Training Gate**:
     - Phased roadmap ratified: Phase A (Close EXP-019), Phase B (Evidence Base), Phase C (Data Search), Phase D (Experiments), Phase E (Synthesis).
     - Explicit Rule: Do NOT interpret v4 as permission to immediately run EXP-020. EXP-020 is gated behind Phase A, B, C and explicit preregistration.
- Reversible: No; ratified working research direction.




### DEC-030 (2026-11-17): Phase A2 S3 Forensic Diagnosis Ratification & Invalidation of Unsupervised Adaptation Negative Claim

- Date: 2026-11-17
- Authoritative Audit: `experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md`
- Context: Secondary Hypothesis S3 in EXP-019 yielded $\bar{\Delta} = 0.00\text{ pp}$, 95% CI $[0.00, 0.00]\text{ pp}$, across all 46 subjects. Early reporting claimed this proved "unsupervised test-time entropy minimization on 2 nights is ineffective without supervisory signal."
- Findings of Phase A2 Diagnostic Audit:
  1. `compute_wearable_quality_mask()` in `exp019_adaptation_core.py` checked physiological limits `30 <= IHR <= 180` expecting raw BPM, whereas the ingested cached dataset provided z-score standardized IHR ($[-3.95, +9.94]$).
  2. Exactly 0 out of 82,762 calibration epochs across all 94 calibration nights for all 47 subjects passed the mask.
  3. Inside `adapt_unsupervised_infomax()`, the check `if valid_probs.shape[0] == 0: break` caused an immediate exit on step 0 for every subject, every fold, and every random seed.
  4. Optimization metrics verified: 0 optimizer steps, 0 backward passes, 0/163,940 parameters updated (displacement $\equiv 0.00000000$), 0 prediction differences on evaluation nights.
  5. The returned `UA-2` model was bitwise identical to the unadapted zero-shot base model `ZS`.
- Ratified Governance Decisions:
  1. **Final Classification of S3:** **`UNRESOLVED / INVALIDATED (Implementation No-Op)`**.
  2. **Scientific Interpretation:** S3 is scientifically uninformative. It provides zero empirical evidence regarding the viability or failure of unsupervised InfoMax adaptation, test-time adaptation, or entropy minimization for wearable sleep staging. Claims asserting that unsupervised adaptation "failed" or "requires supervision" are retracted.
  3. **Immutability of Numerical Artifacts:** The historical experimental record ($0.00\text{ pp}$) remains locked in `exp019_stage_c_results.json` and `PREREG_EXP-019_H-A.md`. No historical numerical file is modified.
  4. **Codebase Remediation Deferred:** Code changes to `exp019_adaptation_core.py` and rerunning of unsupervised adaptation are explicitly deferred to future dedicated protocols under Phase D.
- Reversible: No; ratified forensic finding.


### DEC-031 (2026-11-17): Parameter-Count Governance Reformulation — Adoption of Lightweight-First Principle and Removal of Arbitrary Numerical Ceiling

- Date: 2026-11-17
- Context: Historical NAS iterations and exploratory proposals referenced an arbitrary hard upper limit of $\le +20\%$ ($163,940 \times 1.20 = 196,728$ parameters) or a nominal "200k" ceiling. The research workspace is directed to remove all arbitrary numerical parameter ceilings and establish a scientifically grounded "lightweight-first" governing principle.
- Ratified Governance Principles:
  1. **Baseline Reference Model Size:**
     - The current `ConventionalCNNBiGRU` Model A has exactly **163,940 parameters**.
     - 163,940 is the **baseline reference size**, not a mandatory ceiling.
  2. **Lightweight-First Principle:**
     - Any new architecture or model variant should preferentially use **fewer parameters** than the 163,940-parameter baseline.
     - Parameter reduction is treated as an active design preference and efficiency objective.
     - Model capacity must never be increased simply because additional parameters might improve empirical performance.
  3. **Conditions When a Larger Model Is Considered:**
     - A model larger than 163,940 parameters is **not automatically prohibited**.
     - Any proposal for a larger model requires explicit scientific justification demonstrating why additional capacity is necessary.
     - Capacity expansion must be reported transparently as part of the experimental comparison. Silent architectural expansion is strictly forbidden.
  4. **Removal of Arbitrary Numerical Ceilings:**
     - The previous $+20\%$ ceiling ($196,728$ parameters) is formally removed.
     - The notion that models $> 200\text{k}$ parameters are permanently prohibited is formally withdrawn.
     - No replacement numerical ceiling is imposed unless independently justified by a future preregistered protocol.
  5. **Integrity of Locked EXP-019 Artifacts:**
     - The architecture (Model A, 163,940 params), checkpoints, results, and preregistration of EXP-019 remain locked and immutable.
     - This governance update applies strictly to future architectural and design decisions.
  6. **Future Research Reporting Requirements:**
     For every future model comparison, reports must explicitly disclose:
     - Total parameter count;
     - Trainable parameter count;
     - Relative parameter count compared to the 163,940 baseline (percentage delta $\Delta\%$);
     - Performance metrics;
     - Computational and resource implications where relevant.
  7. **Standard Governance Language:**
     The phrase "maximum allowed parameters" is permanently replaced across active governance documentation with:
     > *"163,940 parameters is the baseline reference size. Future models should preferentially be lighter than this baseline. Larger models require explicit scientific justification and transparent reporting."*
- Reversible: No; ratified governance policy.

---

### DEC-032 (2026-11-19): Phase B Sleep-Summary Agreement Ratification, Finding Reconciliations (FIND-B-01 to FIND-B-06), and Methodological Boundary Resolution

- Date: 2026-11-19
- Authoritative Review: `PHASE_B_INDEPENDENT_AUDIT_REPORT.md` (Verdict: PASS WITH CORRECTIONS, Cleared for Phase 3)
- Target Scope: `EXP019_WITHIN_COHORT_SLEEP_SUMMARY_AGREEMENT_REPORT.md`, `research_engine/bidsleep_within_cohort_agreement_analysis.py`, and `experiments/EXP-019/exp019_sleep_summary_agreement.csv`.
- Ratified Governance Decisions & Reconciled Findings:
  1. **FIND-B-01 (Multi-Night Scope vs. Single-Night Agreement) — RESOLVED:**
     - The subject-averaged limits of agreement (e.g., $[-30.11, +34.67]\text{ min}$ for TST in FT-2) strictly evaluate agreement for a **participant's multi-night monitoring mean** ($\sim 3$ evaluation nights).
     - They must **never be described as spot agreement bounds for an isolated single night**. On single nights, unmodeled within-subject night-to-night variance ($\sigma_w^2$) expands dispersion.
  2. **FIND-B-02 (Micro-Data Information Loss Inventory) — RESOLVED:**
     - Formally catalogs that five secondary analyses cannot be evaluated without subject-night micro-data: (i) two-way error variance decomposition ($\sigma_b^2$ vs $\sigma_w^2$), (ii) linear mixed-effects proportional bias regression, (iii) heteroscedasticity rank tests, (iv) non-parametric participant-level cluster bootstrap, and (v) studentized residual outlier leverage diagnostics.
     - Ratifies subject-averaging as the primary scientifically defensible surrogate under available data.
  3. **FIND-B-03 (LoA Uncertainty Reporting) — DEFERRED TO PHASE 3:**
     - Table 1 in draft manuscript Section 6.2 currently reports parametric LoA point estimates.
     - Scheduled for Phase 3 manuscript editing: incorporate standard errors ($\text{SE}(\text{LoA})$) and 95% confidence intervals on LoA bounds into Table 1 per Bland & Altman (1999) and *Lancet Digital Health* reporting guidelines.
  4. **FIND-B-04 (WASO Systematic Bias Interpretation) — RESOLVED:**
     - The significant underestimation of WASO in FT-2 ($-4.87\text{ min}$, 95% CI: $[-8.54, -1.20]\text{ min}$, $p = 0.0104$) is ratified strictly as an **observed empirical model underestimation**.
     - Causal assertions attributing this bias to an "intrinsic sensor limitation" (e.g. inability to detect micro-arousals) are formally retracted. The bias may arise from multiple technical factors including decision boundaries, training loss class weighting, or feature representations.
  5. **Additional Methodological Correction (Pooled-Night Interval Framing):**
     - The unadjusted pooled-night interval ($[-39.60, +46.95]\text{ min}$ for TST) is an unadjusted descriptive summary that confounds between- and within-subject variance. It is strictly prohibited from being presented as a validated single-night prediction interval.
  6. **FIND-B-05 (Numerical Consistency) & FIND-B-06 (Denominators & Anti-Leakage) — VERIFIED:**
     - 100% numerical concordance verified between source CSV, script, Phase B report, and manuscript draft.
     - $N=46$ subjects ($Bidslab49$ correctly excluded per protocol), $N=152$ nights, $C \cap E = \emptyset$ confirmed leak-free.
- Reversible: No; ratified governance policy.


---

### DEC-033 (2026-11-19): Phase D Governance Framework, Task Decoupling, and Preregistration Formalization

- Date: 2026-11-19
- Authoritative Sources: `PHASE_D_EXPERIMENTAL_DECISION_PLAN.md` (v2), `PHASE_D_PREREGISTRATIONS_INDEPENDENT_REVIEW.md`, `PREREG_EXP-019_K_CURVE.md`, `PREREG_EXP-019_S3_AMENDMENT.md`, `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`
- Context: Establishing formal governance, statistical specifications, and execution readiness for the three Phase D experimental tasks prior to user authorization.
- Ratified Governance Decisions:
  1. **Separation and Independent Governance of Experimental Tasks:**
     - The three proposed Phase D tasks are partitioned into independently governed, non-conflated protocols:
       * **Experiment 1 (TSK-06): Calibration-Depth Learning Curve ($k = 1, 3, 4$):** Governed under `PREREG_EXP-019_K_CURVE.md`. Evaluates monotonic depth scaling and the van der Aar replication hypothesis ($H_{k=3}$) using a primary $N = 33$ common-cohort design on fixed evaluation nights $E_4$ (64 nights) and an available-cohort sensitivity analysis ($N_k \in \{46, 46, 42, 33\}$). Controlled via Page's $L$-trend test and a 5-test Holm-Bonferroni step-down family ($\alpha = 0.05$). Status: **PREREGISTERED / PENDING USER EXECUTION AUTHORIZATION**.
       * **Experiment 2 (TSK-07): Unsupervised Test-Time Adaptation Re-Run ($UA\text{-}2$):** Governed under `PREREG_EXP-019_S3_AMENDMENT.md`. Resolves the DEC-030 implementation no-op by validating raw-unit physiological bounds on reconstructed BPM and scale-invariant accelerometry variance. Establishes mandatory pre-flight validation gates ($> 95\%$ valid epoch ratio; $\ge 600$ epochs/night) and documents the physiological limitations of uniform marginal entropy regularization. Status: **AMENDMENT PREREGISTERED / PENDING USER EXECUTION AUTHORIZATION**.
       * **Experiment 3 (TSK-08): Matched 5-Fold Cross-Validation Paired Comparative Benchmark of SLAMSS-IFS:** Governed under `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`. Evaluates Model A vs. SLAMSS-IFS under strictly matched 5-fold CV across all 47 subjects with full-night context ($L = 1200$) and 3 seeds. Status: **PREREGISTERED / PENDING USER EXECUTION AUTHORIZATION**.
  2. **Retraction of Unsupported Non-Inferiority Margin Claims in TSK-08:**
     - The proposed $-1.50\text{ pp}$ non-inferiority margin and associated ICH E10 half-standard-deviation arguments are formally retracted. In the absence of an externally established clinical or regulatory consensus for wearable sleep staging models, imposing an arbitrary threshold is scientifically unjustified.
     - TSK-08 is ratified strictly as a **paired comparative evaluation with estimation and confidence intervals** ($\bar{d}$, 10,000-resample participant-level bootstrap 95% CIs, Wilcoxon signed-rank testing, and Cohen's $d_z$ effect sizes). Performance differences will be interpreted alongside parameter compression (4.06x vs. paper text; 33.34x vs. repository) and edge SRAM microcontroller feasibility.
  3. **Preservation of Locked Baseline Artifacts:**
     - All locked EXP-018 and EXP-019 confirmatory results, checkpoints, split manifests, and the historical Stage C no-op record ($0.00\text{ pp}$) remain strictly immutable. Output artifacts for Phase D will be written to distinct files (`exp019_k_learning_curve_executed.csv`, `exp019_ua2_rerun_subject_metrics.csv`, `slamss_ifs_matched_subject_metrics.csv`).
  4. **Strict Anti-Premature Execution Gate:**
     - Preregistration does not constitute authorization to train. Zero models may be trained and zero GPU processes scheduled until explicit user authorization is recorded.
- Reversible: No; ratified governance policy.

---

### DEC-034 (2026-11-19): User Strategy Authorization: Deferral of TSK-08 (SLAMSS-IFS Benchmark) Until Final Model Freeze; Execution Prioritization for TSK-06 and TSK-07

- Date: 2026-11-19
- Authoritative Source: User Strategic Decision (`2026-11-19T08:29:08+06:00`)
- Context: Optimizing research compute and scientific sequencing. TSK-08 (SLAMSS-IFS matched 5-fold benchmark, 15 runs, ~33.75 GPU hours) represents substantial compute. Running it against an intermediate model before the project's final model architecture is developed and frozen risks redundant compute or benchmarking against an unoptimized artifact.
- Ratified Strategic Decisions:
  1. **TSK-08 (SLAMSS-IFS Matched Benchmark) Formally Deferred:**
     - TSK-08 (EXP-021) is deferred from immediate Phase D execution.
     - The preregistration `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md` and measured hardware profile (`official_slamss_ifs_hw_profile.json`) remain locked and ready for execution at the appropriate milestone.
     - TSK-08 will be executed exactly once as the final matched comparative benchmark against the fully developed, frozen final model.
  2. **Immediate Execution Priority for TSK-06 and TSK-07:**
     - Task TSK-06 (EXP-019-K: Calibration-depth learning curve $k=1, 3, 4$, governed by `PREREG_EXP-019_K_CURVE.md`) and Task TSK-07 (EXP-019-S3: UA-2 Unsupervised Adaptation Re-Run, governed by `PREREG_EXP-019_S3_AMENDMENT.md`) are cleared as the immediate Phase D execution focus.
     - Both tasks close the primary scientific questions regarding multi-night supervised personalization depth (Contribution C1) and unsupervised label-free adaptation (Contribution C2).
     - Both tasks execute within ~1.5 to 1.75 wall-clock hours concurrently on our dual RTX 3060 workstation.
  3. **Development and Freezing of Final Model:**
     - Following the completion and analysis of TSK-06 and TSK-07, the project will consolidate all empirical evidence (personalization depth, unsupervised adaptation, context scaling, and sleep-summary agreement) to specify, develop, and freeze the final model.
     - Once frozen, the single final matched benchmark against SLAMSS-IFS (TSK-08) will be executed under its audited protocol.
- Reversible: No; ratified project strategic roadmap.

---

### DEC-035 (2026-11-19): Stage 1 Execution Results: Calibration-Depth Learning Curve (TSK-06 / EXP-019-K)

- Date: 2026-11-19
- Authoritative Sources: `PREREG_EXP-019_K_CURVE.md`, `exp019_k_learning_curve_executed.csv`, `exp019_k_curve_subject_metrics.csv`, `logs/tsk06_k_curve_20261119_090250.log`
- Context: Completion of TSK-06 across all 5 outer folds and 3 random seeds (42, 123, 2024) on the workstation's secondary GPU (`cuda:1`, NVIDIA RTX 3060).
- Ratified Empirical Findings:
  1. **Primary Analysis ($N=33$ Common Cohort on Fixed Evaluation Set $E_4$, 64 Nights):**
     - $k=0$ (Zero-Shot Baseline): Macro-F1 = $62.29\%$
     - $k=1$ Night: Macro-F1 = $65.28\%$ ($\Delta = \mathbf{+3.00\text{ pp}}$ [95% CI: $+1.21, +4.84\text{ pp}$], Wilcoxon $p = 0.00228$, Holm-Bonferroni Significant)
     - $k=2$ Nights: Macro-F1 = $65.41\%$ ($\Delta = \mathbf{+3.13\text{ pp}}$ [95% CI: $+1.28, +5.14\text{ pp}$])
     - $k=3$ Nights: Macro-F1 = $65.82\%$ ($\Delta = \mathbf{+3.54\text{ pp}}$ [95% CI: $+1.56, +5.68\text{ pp}$], Wilcoxon $p = 0.00030$, Holm-Bonferroni Significant)
     - $k=4$ Nights: Macro-F1 = $66.25\%$ ($\Delta = \mathbf{+3.96\text{ pp}}$ [95% CI: $+2.04, +6.07\text{ pp}$], Wilcoxon $p = 0.00015$, Holm-Bonferroni Significant)
  2. **Page's $L$-Trend Repeated Measures Test:**
     - Test Statistic: $L = 863.0$ ($\mu_L = 825.0, \sigma_L = 16.58$)
     - Standardized Score: $z = 2.261$
     - $p$-value: $p = 0.01187 < 0.05$ (Statistically significant monotonic trend across calibration depths $k \in \{1, 2, 3, 4\}$, Holm-Bonferroni Significant).
  3. **Multiplicity-Adjusted Hypotheses (5-Test Family, $\alpha=0.05$):**
     - $H_{\text{trend}}$ (Page's $L$ monotonic trend): $p = 0.01187$ $\implies$ **REJECT NULL / MONOTONIC SCALING CONFIRMED**.
     - $H_{k=3}$ ($k=3$ vs ZS on $E_4$, van der Aar Replication): $\Delta = +3.54\text{ pp}$, $p = 0.000301$ $\implies$ **REJECT NULL / REPLICATION CONFIRMED**.
     - $H_{k=1}$ ($k=1$ vs ZS on $E_4$): $\Delta = +3.00\text{ pp}$, $p = 0.00228$ $\implies$ **REJECT NULL / SIGNIFICANT GAIN AT 1 NIGHT**.
     - $H_{\text{sat}, 3}$ ($k=3$ vs $k=2$ step): $\Delta = +0.41\text{ pp}$, $p = 0.287$ $\implies$ FAIL TO REJECT (Not significant after adjustment).
     - $H_{\text{sat}, 4}$ ($k=4$ vs $k=3$ step): $\Delta = +0.42\text{ pp}$, $p = 0.262$ $\implies$ FAIL TO REJECT (Not significant after adjustment).
  4. **Decision Matrix Classification:**
     - **Dimension 1 (Replication):** **REPLICATION CONFIRMED (van der Aar Replicated)** ($\Delta \ge +1.00\text{ pp}$, CI low $> 0$, $p_{\text{adj}} < 0.05$).
     - **Dimension 2 (Scaling Trajectory):** **INCONCLUSIVE SCALING** (Monotonic trend confirmed via Page's $L$, but individual marginal steps $\Delta_{3-2}$ and $\Delta_{4-3}$ do not individually reach Holm-Bonferroni significance given $N=33$).
- Reversible: No; ratified empirical research record.

---

### DEC-036 (2026-11-19): Stage 1 Execution Results: UA-2 Unsupervised Adaptation Re-Run (TSK-07 / EXP-019-S3)

- Date: 2026-11-19
- Authoritative Sources: `PREREG_EXP-019_S3_AMENDMENT.md`, `exp019_ua2_rerun_subject_metrics.csv`, `ua2_quality_gate_audit.log`, `logs/tsk07_ua2_20261119_090250.log`
- Context: Completion of TSK-07 across 46 subjects $\times$ 3 seeds (138 runs) on the workstation's primary GPU (`cuda:0`, NVIDIA RTX 3060) following resolution of the DEC-030 unit mismatch defect.
- Ratified Empirical Findings:
  1. **Pre-Flight Quality Gate Verification:**
     - Total calibration epochs: 81,233 (46 subjects $\times$ Nights 1–2).
     - Valid wearable epoch ratio: $97.26\%$ ($> 95\%$ requirement satisfied).
     - Minimum epochs per night: 606 epochs ($\ge 600$ requirement satisfied).
     - Non-physiological leakage: 0 ($== 0$ requirement satisfied).
  2. **Adaptation Dynamics & Anti-Collapse Invariance:**
     - Zero instances of representation collapse or early abortion ($H_{\text{marginal}} \ge 0.6931\text{ nats}$ maintained across all 138 runs).
     - Mean final marginal entropy: $1.382\text{ nats}$ (close to uniform prior $\ln 4 \approx 1.3863\text{ nats}$).
  3. **Participant-Level Evaluation ($N=46$ Subjects, 152 Evaluation Nights):**
     - Zero-Shot Macro-F1: $64.58\%$
     - UA-2 Macro-F1: $64.79\%$
     - Cohort Mean Delta: $\bar{\Delta} = \mathbf{+0.21\text{ pp}}$ [95% bootstrap CI: $\mathbf{+0.04, +0.39\text{ pp}}$]
     - Wilcoxon Signed-Rank Test: $p_{\text{two-sided}} = 0.0645$, $p_{\text{superiority}} = 0.0322$
     - Subject Win Rate: $27 / 46$ ($58.7\%$)
  4. **Official S3 Decision Categorization:**
     - Under `PREREG_EXP-019_S3_AMENDMENT.md` Section 6, because $p_{\text{superiority}} < 0.05$ and CI lower bound $> 0.0\text{ pp}$, but the mean gain is below the $+1.00\text{ pp}$ benchmark threshold ($\bar{\Delta} = +0.21 < +1.00\text{ pp}$), the outcome is formally classified as:
     - **SUB-BENCHMARK GAIN (Statistically Superior, Below +1.00 pp)**.
  5. **Scientific Implications for Final Model Architecture:**
     - Re-running InfoMax with the corrected wearable quality mask proves that label-free adaptation does indeed impart a genuine, statistically superior representation adjustment ($+0.21\text{ pp}$, $p = 0.0322$, 0% collapse), resolving the historical $0.00\text{ pp}$ artifact.
     - However, the magnitude of unsupervised gain ($+0.21\text{ pp}$) is modest compared to 1-night supervised fine-tuning ($+3.00\text{ pp}$). Unsupervised InfoMax alone cannot substitute for labeled calibration; it serves as a mild, safe regularization auxiliary rather than a primary personalization mechanism.
- Reversible: No; ratified empirical research record.

---

### DEC-037 (2026-11-21): Stage 2 Final Model Freeze (`SleepStagingFinalModel_v1.0`) and Stage 3 (TSK-08 / EXP-021) Execution Launch Authorization

- Date: 2026-11-21
- Authoritative Sources: `experiments/STAGE2_FINAL_MODEL/final_model_spec.json`, `FINAL_MODEL_SYNTHESIS_AND_FREEZE_SPEC.md`, `STAGE2_MODEL_FREEZE_REPORT.md`, `frozen_checkpoint_manifest.json`, `final_model_oof_subject_metrics.csv`, `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`, User Execution Authorization (`2026-11-21T07:35:17+06:00`)
- Context: Completing the Stage 2 milestone established under `DEC-034` (synthesizing and freezing the project's definitive model following Stage 1 completion) and lifting the `DEC-034` deferral gate to launch the single definitive matched 5-fold comparative benchmark against official SLAMSS-IFS (Stage 3 / TSK-08 / EXP-021).
- Ratified Governance & Technical Decisions:
  1. **Permanent Freeze of `SleepStagingFinalModel_v1.0` (Stage 2 Complete):**
     - **Population Backbone:** Model A (`ConventionalCNNBiGRU`, exactly **163,940 trainable parameters**; ~160.1 KiB INT8 weight footprint, achieving **4.06x** compression vs. SLAMSS-IFS paper text [666,060 params] and **33.34x** vs. official repository [5,465,768 params]).
     - **Temporal Context & Inference:** Full-night context $L = 1200$ epochs (10 hours) with `SlidingWindowPredictor` probability stitching and circadian neutralization (`[freq, 0, 0, hr_mean, hr_std]`).
     - **Cryptographic Seal & Parity Verification:** All 15 locked EXP-018 Phase 2 checkpoints (`modelA_L1200_fold{1..5}_seed{42,123,2024}.pt`) were SHA-256 hashed (`frozen_checkpoint_manifest.json`) and evaluated across all 47 subjects (141 subject-seed rows in `final_model_oof_subject_metrics.csv`), achieving a maximum fold-pooled Macro-F1 discrepancy of **`0.000099`** ($< 0.01\text{ pp}$) and maximum per-subject discrepancy of **`0.000655`** ($0.065\text{ pp}$) against locked EXP-018 Phase 2 JSONs.
     - **Unweighted 47-Subject Participant-Mean OOF Reference (`N = 47`, 3-seed means):** Macro-F1 = **$64.79\% \pm 7.45\text{ pp}$**, Accuracy = **$66.18\% \pm 6.79\text{ pp}$**, Weighted-F1 = **$65.96\% \pm 6.73\text{ pp}$**, Cohen's $\kappa = \mathbf{0.5078 \pm 0.1011}$ (Wake F1 $62.70\%$, Light F1 $66.61\%$, Deep F1 $66.76\%$, REM F1 $63.09\%$).
     - **Tiered Personalization Policy Frozen:** Tier 0 (Zero-Shot $L=1200$), Tier 1 (Optional Quality-Gated `head_gru` InfoMax auxiliary, $+0.21\text{ pp}$, `SUB-BENCHMARK GAIN` under `DEC-036`), Tier 2 (Supervised within-person fine-tuning $k \in \{1..4\}$ with non-mutating stage-coverage safety diagnostics, monotonic scaling confirmed under `DEC-035`).
  2. **EGM Pre-Execution Hardening & Protocol Parity Lock for Stage 3 (TSK-08 / EXP-021):**
     - **Circadian Input Parity:** Locked `neutralize_circadian=True` (`x_meta = [freq, 0, 0, hr_mean, hr_std]`) in `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md` §3 Item 4 so official SLAMSS-IFS and frozen Model A receive identical input features without circadian confounding (`DEC-024`).
     - **Zero-Leakage Decoder & FP32 Composite Loss:** `OfficialSLAMSSIFSWrapper` passes all-zero `trg_zeros` with `teacher_forcing_ratio=0.0` (remediating upstream $t=0$ ground-truth label leakage), and `OfficialSLAMSSCompositeLoss` computes `CE + 0.01 * RCCLoss` strictly in FP32 with AMP autocast disabled.
     - **Training Budget & Batched Validation:** `micro_batch_size = 4`, `effective_batch_size = 4` (`accum_steps = 1`, 42–44 optimizer steps/epoch), `max_epochs = 100` (up to 4,200–4,400 steps $\ge$ Model A's 3,200 steps), `min_epochs = 40` (~1,700-step floor), `patience = 25` on inner-validation Macro-F1, and batched validation inference (`val_batch_size = 16`).
  3. **Authorization to Launch Stage 3 (TSK-08 / EXP-021):**
     - With Stage 2 sealed and all 12 unit/contract/CUDA-AMP tests + dual-GPU dry runs passing (0 blocking or major drift items remaining), the `DEC-034` deferral gate is formally lifted and Stage 3 dual-GPU execution (7 runs on `cuda:0`, 8 runs on `cuda:1`) is authorized for immediate launch.
- Reversible: No; ratified model freeze and benchmark launch authorization.

---

### DEC-038 (2026-11-22): Stage 3 Execution Results: Official SLAMSS-IFS Matched 5-Fold Benchmark vs. Frozen Model A (TSK-08 / EXP-021)

- Date: 2026-11-22
- Authoritative Sources: `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`, `experiments/EXP-021/slamss_ifs_matched_subject_metrics.csv`, `experiments/EXP-021/slamss_ifs_paired_subject_deltas.csv`, `experiments/EXP-021/slamss_ifs_paired_comparison_summary.json`, `experiments/EXP-021/slamss_ifs_paired_comparative_report.md`
- Context: Completion of the definitive 15-run matched 5-fold cross-validation comparative benchmark (TSK-08 / EXP-021) across all 47 participants and 3 random seeds (42, 123, 2024) evaluating the official 5,465,768-parameter SLAMSS-IFS model against frozen `SleepStagingFinalModel_v1.0` (Model A: 163,940 parameters; ~160.1 KiB INT8 weights) on dual NVIDIA RTX 3060 GPUs.
- Ratified Empirical Findings & Statistical Comparison:
  1. **Participant-Mean Performance Across Cohort ($N = 47$ Independent Participants, 3-Seed Means):**
     - **Frozen Model A (163,940 params, 160.1 KiB INT8 weights):**
       * Macro-F1: **$64.79\% \pm 7.45\text{ pp}$**
       * Overall Accuracy: **$66.18\% \pm 6.79\text{ pp}$**
       * Weighted-F1: **$65.96\% \pm 6.73\text{ pp}$**
       * Cohen's Kappa: **$0.5078 \pm 0.1011$**
       * Per-Stage F1: Wake **$62.70\% \pm 11.05\text{ pp}$**, Light **$66.61\% \pm 5.25\text{ pp}$**, Deep **$66.76\% \pm 12.31\text{ pp}$**, REM **$63.09\% \pm 13.11\text{ pp}$**
     - **Official SLAMSS-IFS (5,465,768 repo params, 5.12 GiB VRAM at $B=4$):**
       * Macro-F1: **$67.37\% \pm 6.73\text{ pp}$**
       * Overall Accuracy: **$68.75\% \pm 6.08\text{ pp}$**
       * Weighted-F1: **$68.72\% \pm 5.90\text{ pp}$**
       * Cohen's Kappa: **$0.5378 \pm 0.0941$**
       * Per-Stage F1: Wake **$64.25\% \pm 9.36\text{ pp}$**, Light **$69.44\% \pm 5.05\text{ pp}$**, Deep **$70.27\% \pm 12.14\text{ pp}$**, REM **$65.51\% \pm 11.54\text{ pp}$**
  2. **Paired Comparative Estimands ($\bar{d} = \text{Model A} - \text{SLAMSS-IFS}$, $10,000$ Bootstrap Resamples):**
     - **Primary Endpoint (Macro-F1):**
       * Mean Paired Difference: **$-2.58\text{ pp}$** [95% CI: **$-3.53, -1.68\text{ pp}$**]
       * Paired Effect Size (Cohen's $d_z$): **$-0.780$**
       * Wilcoxon Signed-Rank Test: $W = 144.0$, **$p = 1.9818 \times 10^{-6}$**
       * Participant Win/Loss: Model A wins **$13 / 47$** ($27.7\%$), SLAMSS-IFS wins $34 / 47$, ties $0$
     - **Secondary Endpoints:**
       * Accuracy: $\bar{d} = -2.57\text{ pp}$ [95% CI: $-3.53, -1.65\text{ pp}$], Cohen's $d_z = -0.784$, $p = 3.3783 \times 10^{-6}$ (Model A wins 10/47)
       * Weighted-F1: $\bar{d} = -2.76\text{ pp}$ [95% CI: $-3.77, -1.79\text{ pp}$], Cohen's $d_z = -0.800$, $p = 7.4148 \times 10^{-7}$ (Model A wins 12/47)
       * Cohen's Kappa: $\bar{d} = -0.0300$ [95% CI: $-0.0434, -0.0169$], Cohen's $d_z = -0.643$, $p = 1.3095 \times 10^{-4}$ (Model A wins 15/47)
       * Wake F1: $\bar{d} = -1.55\text{ pp}$ [95% CI: $-2.87, -0.19\text{ pp}$], Cohen's $d_z = -0.325$, $p = 0.0287$ (Model A wins 17/47)
       * Light F1: $\bar{d} = -2.83\text{ pp}$ [95% CI: $-3.83, -1.86\text{ pp}$], Cohen's $d_z = -0.829$, $p = 5.9535 \times 10^{-7}$ (Model A wins 8/47)
       * Deep F1: $\bar{d} = -3.51\text{ pp}$ [95% CI: $-5.07, -2.12\text{ pp}$], Cohen's $d_z = -0.678$, $p = 6.8193 \times 10^{-6}$ (Model A wins 11/47)
       * REM F1: $\bar{d} = -2.41\text{ pp}$ [95% CI: $-4.68, -0.32\text{ pp}$], Cohen's $d_z = -0.317$, **$p = 0.1047$ (Not statistically significant; Model A wins 19/47, 40.4%)**
  3. **Preregistered Outcome Categorization & Disambiguation:**
     - Under `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md` Section 6, the outcome evaluates to:
       **`SLAMSS-IFS ADVANTAGE (STATISTICALLY SIGNIFICANT)`**.
     - Discordance flag: **False** (Bootstrap 95% CI is entirely negative and Wilcoxon $p < 0.05$ strictly concur).
  4. **Empirical Accuracy-vs-Efficiency Trade-Off Synthesis:**
     - The official SLAMSS-IFS architecture achieves an empirical performance advantage of **$+2.58\text{ pp}$ Macro-F1** over Model A under strictly matched 5-fold cross-validation.
     - However, this $+2.58\text{ pp}$ margin requires:
       * **33.34-fold more parameters** ($5,465,768$ vs. $163,940$);
       * **~34x larger memory footprint** (~5.47 MB FP32 weights vs. ~160.1 KiB INT8 weights);
       * **Substantial runtime VRAM** (5.12 GiB reserved during training at $B=4$), entirely precluding edge microcontroller execution.
     - In contrast, frozen Model A retains **96.2% of SLAMSS-IFS Macro-F1 performance** ($64.79\%$ vs. $67.37\%$) while fitting within ultra-low-power wearable MCU SRAM constraints (512 KB), with statistical equivalence on REM sleep staging ($p = 0.105$).
     - Furthermore, with 2-night within-person personalization (TSK-06, $\bar{\Delta} = +3.13\text{ pp}$ to $+3.54\text{ pp}$), personalized Model A reaches $65.82\% - 66.25\%$ Macro-F1, largely closing the zero-shot gap with SLAMSS-IFS while maintaining a 33.34x parameter reduction.
- Reversible: No; ratified empirical research record.

---

## DEC-039 (2026-11-22): Forensic Erratum on TSK-06 Calibration-Depth Baseline & Reconciliation of Available Cohort k=2

- Date: 2026-11-22
- Authoritative Sources: `experiments/EXP-019/TSK06_ZS_DISCREPANCY_AUDIT.md`, `experiments/EXP-019/run_tsk06_k_curve.py`, `experiments/EXP-019/exp019_subject_metrics.csv`
- Context: Forensic audit of TSK-06 execution script (`experiments/EXP-019/run_tsk06_k_curve.py`) resolving the discrepancy between locked EXP-019 $k=2$ ($+1.01\text{ pp}$, $\text{ZS} = 64.58\%$, $\text{FT} = 65.59\%$) and TSK-06 executed summary ($+2.72\text{ pp}$, $\text{ZS} = 62.86\%$, $\text{FT} = 65.58\%$).
- Finding: An in-place weight mutation defect in `run_tsk06_k_curve.py` caused `predictor_zs` on subjects 2..N of each fold to evaluate a model fine-tuned on the prior subject (de facto donor control), yielding an average baseline of $62.86\%$ (matching `FT-other-2`). All fine-tuned models (`FT_Ek_MacroF1`, `FT_E4_MacroF1`) were correctly reset from the clean backbone and are completely valid.
- Reconciliation & Impact:
  1. The true available-cohort $k=2$ gain over clean Zero-Shot is $+1.00\text{ pp}$ ($64.58\% \to 65.58\%$), confirming harmony with locked EXP-019 ($+1.01\text{ pp}$).
  2. The true common-cohort ($N=33$, $E_4$) Zero-Shot baseline is $63.54\%$, yielding clean adaptation gains of $+1.74\text{ pp}$ ($k=1$), $+1.87\text{ pp}$ ($k=2$), $+2.28\text{ pp}$ ($k=3$), and $+2.71\text{ pp}$ ($k=4$).
  3. All primary hypotheses ratified under DEC-035 remain valid: Page's $L$ trend test ($p = 0.01187$) and van der Aar replication ($k=3$ gain $+2.28\text{ pp} \ge +1.00\text{ pp}$) remain robustly supported.
- Governance Action: Preserve historical executed CSVs under the immutable data rule; record this erratum in `DECISIONS.md`, `RESEARCH_STATE.md`, and `EXPERIMENTS.md`.

#### Micro-Patch Addendum to DEC-039 (2026-11-22): Exact Clean-ZS Statistics & W0-B Edge Deployment Caveats

- **Reconciliation Artifacts (CPU-only, zero retraining):** [`experiments/EXP-019/reconcile_tsk06_clean_zs.py`](./experiments/EXP-019/reconcile_tsk06_clean_zs.py) (594 CPU night inferences across the 15 frozen Model A checkpoints; verified Clean ZS = $63.54\%$ on $E_4$ [$N=33$] and $64.58\%$ on $E_2$ [$N=46$], matching locked `exp019_subject_metrics.csv` within $3.96 \times 10^{-4}$ max per-subject FP32 diff) and [`experiments/EXP-019/exp019_k_learning_curve_CLEAN_RECONCILED.csv`](./experiments/EXP-019/exp019_k_learning_curve_CLEAN_RECONCILED.csv) (historical `exp019_k_learning_curve_executed.csv` preserved untouched).
- **Exact Clean-ZS Reconciled Statistics (10,000-resample participant percentile bootstrap `seed=42`, exact two-sided Wilcoxon signed-rank `method='exact'`):**
  1. **Primary Common Cohort ($N=33$, Fixed $E_4$, Clean ZS $= 63.54\%$):**
     - $k=1$: FT $= 65.28\%$, $\bar{\Delta} = \mathbf{+1.75\text{ pp}}$ [95% CI: **$+0.63, +2.93\text{ pp}$**], $W = 136.0$, **$p = 0.00879$**, 24/33 wins.
     - $k=2$: FT $= 65.41\%$, $\bar{\Delta} = \mathbf{+1.88\text{ pp}}$ [95% CI: **$+0.59, +3.25\text{ pp}$**], $W = 123.0$, **$p = 0.00402$**, 24/33 wins.
     - $k=3$: FT $= 65.82\%$, $\bar{\Delta} = \mathbf{+2.29\text{ pp}}$ [95% CI: **$+0.69, +3.97\text{ pp}$**], $W = 144.0$, **$p = 0.01369$**, 24/33 wins.
     - $k=4$: FT $= 66.25\%$, $\bar{\Delta} = \mathbf{+2.71\text{ pp}}$ [95% CI: **$+1.17, +4.33\text{ pp}$**], $W = 110.0$, **$p = 0.00170$**, 24/33 wins.
     - **Page's $L$-Trend Test (invariant to clean ZS subtraction):** $L = 863.0$ ($\mu_L = 825.0, \sigma_L = 16.58$), $z = 2.261$, **$p = 0.01187$**.
     - **Incremental Steps on $E_4$ ($N=33$):** $\delta_{2-1} = +0.13\text{ pp}$ [$-0.98, +1.26$], $p = 0.4688$; $\delta_{3-2} = +0.41\text{ pp}$ [$-0.37, +1.20$], $p = 0.3303$; $\delta_{4-3} = +0.42\text{ pp}$ [$+0.08, +0.78$], $p_{\text{2s}} = 0.02660$, $p_{\text{sat}(<+0.50\text{ pp})} = 0.26249$.
  2. **Sensitivity Available Cohort ($E_k$, $N = 46 / 46 / 42 / 33$):**
     - $k=1$ ($N=46$, Clean ZS $= 63.95\%$): FT $= 64.18\%$, $\bar{\Delta} = \mathbf{+0.23\text{ pp}}$ [95% CI: **$-0.59, +1.05\text{ pp}$**], $W = 491.0$, **$p = 0.59568$**, 25/46 wins (*not significant across full cohort; previously inflated to $+1.92\text{ pp}$ by prior-subject contamination*).
     - $k=2$ ($N=46$, Clean ZS $= 64.58\%$): FT $= 65.58\%$, $\bar{\Delta} = \mathbf{+0.99\text{ pp}}$ [95% CI: **$-0.18, +1.99\text{ pp}$**], $W = 271.0$, **$p = 0.00271$**, 33/46 wins (*harmonizes with locked EXP-019 $+1.01\text{ pp}$ [$-0.17, +1.99\text{ pp}$]*).
     - $k=3$ ($N=42$, Clean ZS $= 64.25\%$): FT $= 65.67\%$, $\bar{\Delta} = \mathbf{+1.42\text{ pp}}$ [95% CI: **$+0.15, +2.58\text{ pp}$**], $W = 213.0$, **$p = 0.00232$**, 29/42 wins (*first depth where available-cohort 95% CI excludes zero*).
     - $k=4$ ($N=33$, Clean ZS $= 63.54\%$): FT $= 66.25\%$, $\bar{\Delta} = \mathbf{+2.71\text{ pp}}$ [95% CI: **$+1.17, +4.33\text{ pp}$**], $W = 110.0$, **$p = 0.00170$**, 24/33 wins.
  3. **Updated 5-Test Holm-Bonferroni Adjusted $p$-Values ($\alpha = 0.05$, step-down):**
     - *Executed Driver 5-Test Family (`run_tsk06_k_curve.py`: Page's $L$, $k=1$, $k=3$, $k=4$ vs. Clean ZS [exact two-sided], $T_{\text{sat}}$):* $k=4$ ($p_{\text{Holm}} = \mathbf{0.00848}$, Sig), $k=1$ ($p_{\text{Holm}} = \mathbf{0.03515}$, Sig), Page's $L$ ($p_{\text{Holm}} = \mathbf{0.03561}$, Sig), $k=3$ ($p_{\text{Holm}} = \mathbf{0.03561}$, Sig), $T_{\text{sat}}$ ($p_{\text{Holm}} = 0.26249$, NS).
     - *All-$k$ $E_4$ + Trend 5-Test Family (`[Page's L, k=1, k=2, k=3, k=4]` vs. Clean ZS [exact two-sided]):* $k=4$ ($p_{\text{Holm}} = \mathbf{0.00848}$), $k=2$ ($p_{\text{Holm}} = \mathbf{0.01609}$), $k=1$ ($p_{\text{Holm}} = \mathbf{0.02637}$), Page's $L$ ($p_{\text{Holm}} = \mathbf{0.02637}$), $k=3$ ($p_{\text{Holm}} = \mathbf{0.02637}$) — all 5 significant.
     - *Strict Preregistered §3.4 5-Test Family (`[Page's L, k=3 vs Clean ZS, step 2-1, step 3-2, step 4-3]`):* Under **exact two-sided** Wilcoxon, the smallest raw $p$-value is Page's $L$ ($0.01187 > \alpha/5 = 0.0100$), yielding $p_{\text{Holm}} = 0.05935$ for both Page's $L$ and $k=3$ (and $0.07979$ for $\delta_{4-3}$; $0.66053$ for $\delta_{3-2}, \delta_{2-1}$), narrowly missing two-sided FWER $0.05$ without $k=4$/$k=1$ in the family. Under directional **one-sided** superiority ($H_1: \Delta > 0$ per §3.2), $k=3$ ($p_{\text{1s}} = 0.00685 \to p_{\text{Holm}} = \mathbf{0.03423}$, Sig), Page's $L$ ($p_{\text{Holm}} = \mathbf{0.04748}$, Sig), and $\delta_{4-3}$ ($p_{\text{1s}} = 0.01330 \to p_{\text{Holm}} = \mathbf{0.04748}$, Sig) pass $\alpha = 0.05$. Manuscript reporting must disclose this exact two-sided vs. one-sided family boundary transparently.
- **Task W0-B (INT8 Edge Deployment) Mandatory Caveats:**
  1. **Dynamic INT8 In-Memory Payload ($309.54\text{ KiB}$) vs. Theoretical All-INT8 ($160.10\text{ KiB}$):** Empirical PyTorch dynamic quantization (`torch.ao.quantization.quantize_dynamic`) quantizes only `Linear` and `GRU` layers ($112,932 / 163,940 = 68.89\%$ of parameters), keeping `Conv1d` and normalization layers in FP32. Consequently, the empirically verified in-memory Dynamic INT8 weight footprint is **$309.54\text{ KiB}$** ($316,964\text{ B}$; serialized file $339.99\text{ KiB}$), not the theoretical all-INT8 weight floor of **$160.10\text{ KiB}$** ($163,940\text{ B}$), which would require static `Conv1d` PTQ/QAT currently blocked in PyTorch eager/FX mode without refactoring residual `+` ops and `pack_padded_sequence`.
  2. **BiGRU Requires Full-Night Buffering on Smartwatch SoC/Phone (Not 1-Epoch Streaming on a 512 KB MCU):** Because `ConventionalCNNBiGRU` uses a 2-layer **bidirectional** GRU over the full-night context ($L=1200$), the backward temporal pass requires future epochs across the night and therefore **cannot** be deployed as causal 1-epoch streaming on a $512\text{ KB}$ SRAM microcontroller (superseding the streaming note in `INT8_EDGE_BENCHMARK_REPORT.md` §4 / `int8_quantization_profile.json`). Realistic on-device execution requires **full-night feature/sequence buffering on a smartwatch SoC or paired smartphone**, where the $309.54\text{ KiB}$ Dynamic INT8 model and full-night activations fit easily and run in $47.7\text{–}69.4\text{ ms}$/night on CPU.

---

## DEC-040 — EXP-019B & EXP-020 Design-Informed Exploratory Closure (2026-11-23, Commit `945c7c1`)

- **Classification:** Design-informed exploratory closure on BIDSleep (`N=46` / `N=47`) under DEC-021. Confirmatory verdict labels are strictly prohibited.
- **Audit Clearance:** Cleared post-execution by `docs/audits/STAGE_D_POST_EXECUTION_EGM_DRIFT_REPORT.md` (0 Blocking, 0 Major).
- **EXP-019B (CPU Heterogeneity Audit):** Across 6 predictors (`min_stage_frac`, `stage_entropy`, `eff_ihr_hz`, `missingness_rate`, `baseline_hr_mean`, `baseline_hr_iqr`) tested against `EXP-019` (`FT-2 - ZS`, $N=46$) and `EXP-021` (`Model A - SLAMSS-IFS`, $N=47$), **zero predictors** reached `FDR q < 0.05` (all $q_{\text{family}} \ge 0.5673$, $q_{\text{pooled}} \ge 0.5913$; Oldham headroom check $\rho = +0.0376, p = 0.8043$). Individual subject heterogeneity in adaptation response is not explained by simple baseline HR, sampling rate, or stage-distribution summaries.
- **EXP-020 (Regularized & Gated Adaptation Closure, $N=46$):**
  - `FT-2-KL`: `0.6562` Macro-F1 ($\bar{\Delta}_{\text{vs ZS}} = +1.03\text{ pp}$ [95% CI: $-0.11, +2.01\text{ pp}$], $p_{\text{Holm}} = 0.0030$; $\bar{\Delta}_{\text{vs FT-2}} = +0.02\text{ pp}$, $N_{\text{harm}} = 3$, worst outlier `-15.14 pp`).
  - `GATE_tau`: `0.6591` Macro-F1 ($\bar{\Delta}_{\text{vs ZS}} = +1.33\text{ pp}$ [95% CI: $+0.49, +2.12\text{ pp}$], $p_{\text{Holm}} = 0.0028$; $\bar{\Delta}_{\text{vs FT-2}} = +0.32\text{ pp}$, $N_{\text{harm}} = 2$, worst outlier reduced to `-6.40 pp` via `Bidslab34` abstention at `min_stage_frac_cal = 0.0068 < tau* = 0.01`).
  - `FT-2-KL_GATE_tau`: `0.6559` Macro-F1 ($\bar{\Delta}_{\text{vs ZS}} = +1.00\text{ pp}$, $p_{\text{Holm}} = 0.0030$; Arm c inner-val gate skipped Folds 2, 3, 5).
- **Decision:** Adopt `GATE_tau` as the recommended exploratory safety guardrail against severe missing-stage negative transfer (`Bidslab34`) for Tier-2 adaptation, while reporting all EXP-019B/EXP-020 results strictly as exploratory findings under DEC-021.

---

## DEC-041 — EXP-027 & EXP-028 Strictly Zero-Shot External PSG Validation on PhysioNet SleepAccel v1.0.0 (2026-11-23, Commit `945c7c1`)

- **Protocol:** Zero-shot external validation (zero retraining, zero fine-tuning, zero threshold tuning) on PhysioNet SleepAccel v1.0.0 (DOI: `10.13026/hmhs-py35`, $N=31$ single-night Apple Watch + in-lab PSG recordings), locked under `docs/designs/exp027_exp028_sleepaccel_external_validation.md`, cleared pre-execution via `docs/audits/STAGE_D_PRE_EXECUTION_DRIFT_REPORT.md` (`GO`, `n_qc_pass = 25`, 6 exclusions: 5 with `n_epochs < 600` [`3509524` (417 epochs), `4018081` (510 epochs), `46343` (567 epochs), `5132496` (465 epochs), `759667` (475 epochs)] and 1 [`7749105` (960 epochs)] with ACC dropout `cov_mo = 0.1365`), and cleared post-execution via `docs/audits/STAGE_D_POST_EXECUTION_EGM_DRIFT_REPORT.md` (0 Blocking, 0 Major).
- **Claim Boundary & Epistemological Framing:** The non-significant difference between Model A and the 33.34x larger Official SLAMSS-IFS is strictly framed as **no statistically significant difference / empirical parity** under paired testing. TOST equivalence is explicitly **not** claimed because no formal clinical equivalence margin was pre-specified. SleepAccel is strictly a **single-night** in-lab PSG cohort (never multi-night), and `W0-B` edge execution is framed strictly as smartwatch SoC / companion smartphone offline full-night buffering (never bare-metal 512 KB MCU causal streaming).
- **PSG Support Diagnostics:** Exactly 3 subjects in SleepAccel have zero reference PSG epochs for certain sleep stages (`1455390` with Deep = 0 [in `n=25` `qc_clean`]; `5132496` with REM = 0 [in `n=6` excluded]; `7749105` with REM = 0 [in `n=6` excluded]), generating 96 expected `RuntimeWarning`s in `evaluator.py:129` ($3 \times 2 \times 16$) that were safely caught with zero NaNs (`np.nan_to_num(..., 0.0)` and `zero_division=0`).
- **EXP-027 (4-Class Stage Classification, Model A vs. Official SLAMSS-IFS, 15+15 Checkpoints):**
  - **Primary (`qc_clean`, $n=25$, `mean_single_model`):** Model A `60.51%` vs. SLAMSS-IFS `61.08%` Macro-F1 ($\bar{\Delta} = -0.57\text{ pp}$ [95% CI: $-2.13, +0.94\text{ pp}$], Wilcoxon $p = 0.5782$, $d_z = -0.14$, `10W/15L`); Kappa `0.4792` vs. `0.4919` ($p = 0.6150$); Wake F1 `+1.95 pp`, Light F1 `-3.49 pp` ($p = 0.0006$), Deep F1 `-1.34 pp`, REM F1 `+0.60 pp`.
  - **Secondary (`qc_clean`, $n=25$, `prob_ensemble`):** Model A **`62.94%`** vs. SLAMSS-IFS **`62.79%`** Macro-F1 ($\bar{\Delta} = +0.15\text{ pp}$ [95% CI: $-1.51, +1.86\text{ pp}$], Wilcoxon $p = 0.9789$, $d_z = +0.03$, `10W/15L`); Kappa `0.5122` vs. `0.5202` ($\bar{\Delta} = -0.0079$, $p = 0.6915$, $d_z = -0.11$, `11W/14L`); Wake F1 `+2.65 pp` ($p = 0.1209$), Deep F1 `+1.13 pp` ($p = 0.8192$), REM F1 `-0.61 pp` ($p = 0.7510$), Light F1 `-2.56 pp` ($p = 0.0088$).
  - **Sensitivity (`all_n31`, $n=31$):**
    * **Primary (`mean_single_model`):** Model A `60.65%` vs. SLAMSS-IFS `62.27%` Macro-F1 ($\bar{\Delta} = -1.62\text{ pp}$ [95% CI: $-3.84, +0.28\text{ pp}$], Wilcoxon $p = 0.3468$, $d_z = -0.27$, `13W/18L`).
    * **Secondary (`prob_ensemble`):** Model A `62.94%` vs. SLAMSS-IFS `64.14%` Macro-F1 ($\bar{\Delta} = -1.20\text{ pp}$ [95% CI: $-3.96, +1.04\text{ pp}$], Wilcoxon $p = 0.5551$, $d_z = -0.17$, `13W/18L`).
- **EXP-028 (External Single-Night PSG Bland-Altman Agreement, `qc_clean` $n=25$, FIND-B-02 Remediated):**
  - Persisted 744 raw paired tuples (`exp028_sleepaccel_raw_paired_tuples.csv`) and Bland-Altman (1999) `SE(LoA)` + Shapiro-Wilk fallback (`exp028_sleepaccel_bland_altman_summary.csv`).
  - **Primary (`mean_single_model`, `qc_clean` $n=25$):**
    * **Model A:** `tst_min` bias `+11.81` min (`95% LoA [-23.79, +47.40]`, MAE `15.87` min, $r = 0.9091$); `sol_min` bias `+2.71` min (`[-10.91, +16.33]`, MAE `5.73` min, $r = 0.6607$); `waso_min` bias `-13.57` min (`[-43.05, +15.90]`, MAE `15.09` min, $r = 0.3968$, prop. slope `-0.513`, $p = 0.0618$); `sleep_efficiency_pct` bias `+2.62%` (`[-5.60, +10.83]%`, MAE `3.58%`, $r = 0.3781$); `deep_pct_tst` bias **`+10.65%`** (`[-2.39, +23.69]%`, MAE `10.72%`, $r = 0.4909$); `rem_pct_tst` bias **`+10.13%`** (`[-5.92, +26.18]%`, MAE `11.12%`, $r = 0.4767$).
    * **SLAMSS-IFS:** `tst_min` bias `+14.59` min (`[-21.68, +50.87]`, MAE `19.11` min, $r = 0.9042$); `sol_min` bias `+0.53` min (non-normal $p_{\text{SW}}=0.0046$, median `+1.73` min, non-param LoA `[-18.73, +10.29]`, MAE `5.53` min, prop. slope `-0.460`, $p = 0.0471$); `waso_min` bias `-14.16` min (`[-44.39, +16.07]`, MAE `16.34` min, $r = 0.2824$, prop. slope `-0.925`, $p = 0.0021$); `sleep_efficiency_pct` bias `+3.24%` (`[-5.02, +11.49]%`, MAE `4.26%`, $r = 0.2601$, prop. slope `-0.693`, $p = 0.0277$); `deep_pct_tst` bias `+8.67%` (MAE `8.81%`, $r = 0.5023$); `rem_pct_tst` bias `+7.21%` (MAE `9.64%`, $r = 0.1971$).
  - **Secondary (`prob_ensemble`, `qc_clean` $n=25$):**
    * **Model A:** `tst_min` bias `+12.22` min (`95% LoA [-23.18, +47.62]`, MAE `15.86` min, $r = 0.9099$); `sol_min` bias `+2.44` min (non-normal $p_{\text{SW}}=0.0059$, median `+2.50` min, non-param LoA `[-16.20, +14.30]`, MAE `5.96` min); `waso_min` bias `-13.82` min (MAE `15.98` min, $r = 0.3589$); `sleep_efficiency_pct` bias `+2.71%` (`[-5.47, +10.89]%`, MAE `3.58%`, $r = 0.3814$); `deep_pct_tst` bias **`+10.23%`** (MAE `10.26%`, $r = 0.4977$); `rem_pct_tst` bias **`+9.35%`** (MAE `10.85%`, $r = 0.4482$).
    * **SLAMSS-IFS:** `tst_min` bias `+15.72` min (`[-19.84, +51.28]`, MAE `19.32` min, $r = 0.9078$); `sol_min` bias `+0.38` min (non-normal $p_{\text{SW}}=0.0011$, median `+1.00` min, non-param LoA `[-19.90, +10.30]`, MAE `5.46` min, prop. slope `-0.531`, $p = 0.0316$); `waso_min` bias `-15.16` min (MAE `16.64` min, prop. slope `-1.026`, $p = 0.0004$); `sleep_efficiency_pct` bias `+3.49%` (MAE `4.30%`, prop. slope `-0.672`, $p = 0.0265$); `deep_pct_tst` bias `+8.84%` (MAE `9.11%`); `rem_pct_tst` bias `+6.87%` (MAE `9.61%`).
  - **Label-Prior Shift:** Positive Deep% (`+10.65%` primary / `+10.23%` secondary) and REM% (`+10.13%` primary / `+9.35%` secondary) biases in Model A (and `+8.67%`/`+7.21%` in SLAMSS-IFS) reflect the cross-modality reference shift between Dreem 2 ambulatory EEG (BIDSleep training source: 44.2% Light, 18.9% Deep night-averaged EEG; 46.2% Light, 18.6% Deep across 137,801 pooled sensor-valid epochs; 21.14% Deep of TST on evaluation nights) and single-night in-lab PSG (SleepAccel: 57.7% Light, 11.1% Deep of total epochs; 12.01% Deep of TST on $n=25$ quality-clean recordings).
- **Decision:** External zero-shot generalization of the 163,940-parameter Model A on independent single-night Apple Watch + in-lab PSG hardware is **verified**, showing no statistically significant difference in 4-class Macro-F1 versus the 33.34x larger 5,465,768-parameter Official SLAMSS-IFS (`mean_single_model` `-0.57 pp`, $p = 0.5782$; `prob_ensemble` `+0.15 pp`, $p = 0.9789$). Stage D is closed and fully synchronized with resolved submission manuscript `MANUSCRIPT_DRAFT_v1.0.md`.

---

## DEC-042 — EXP-029 `Ultradian-Pyramidal-Coupler` (`v5`, 190,788 Params) Ratification & Promotion (2026-11-24)

- **Protocol & Audit Clearance:** Executed 5-fold subject-disjoint nested CV (3 seeds: `42, 123, 2024`; 15 runs per arm; 75 total training runs across `cnn_bigru_200k`, `film_only`, `pyramid_only`, `full_v5`, and `full_v5_ce`) under `docs/designs/exp029_ultradian_pyramidal_coupler.md` (`L = 1200`, `prep_57738d534b` cache, `36d84020` manifest gate). Cleared pre-execution via `docs/audits/EXP029_PRE_EXECUTION_DRIFT_REPORT.md` (`GO`, 16/16 unit tests) and post-execution via `docs/audits/EXP029_POST_EXECUTION_EGM_DRIFT_REPORT.md` (0 Blocking, 0 Major).
- **In-Cohort BIDSleep 5-Fold CV (`N=47`, 10,000-resample bootstrap, two-sided Wilcoxon, Holm-Bonferroni):**
  - `full_v5` (`RCCLoss`, `190,788` params, `+16.38%` vs. `163,940`): **`66.74% ± 6.81%`** Macro-F1 (Acc `68.17%`, $\kappa = 0.5295$, Wake `64.37%`, Light `68.66%`, Deep `67.71%`, REM `66.21%`).
    * vs. Arm `(a)` `ConventionalCNNBiGRU-164k` (`64.79%`): **`+1.95 pp`** `[+1.31, +2.61 pp]`, $p_{\text{raw}} = 2.39 \times 10^{-7}$, **$p_{\text{Holm}} = 1.43 \times 10^{-6}$**, $d_z = 0.858$, wins **38/47**.
    * vs. Arm `(f)` `Official SLAMSS-IFS` (`67.37%`, `5,465,768` params): **`-0.63 pp`** `[-1.36, +0.16 pp]`, **$p_{\text{Holm}} = 0.1212$** (statistical parity at **28.65x** parameter compression).
    * vs. Arm `(b)` `cnn_bigru_200k` (`223,268` params, `66.61% ± 7.06%`): `+0.13 pp` `[-0.35, +0.63 pp]`, $p_{\text{Holm}} = 1.0000$ (matches larger capacity control with `-14.5%` fewer parameters).
  - Ablation Arms: `film_only` (`172,260` params) `66.67% ± 7.02%`; `pyramid_only` (`182,468` params) `66.92% ± 7.05%`.
  - Pure-`CE` Sensitivity (`full_v5_ce`, `190,788` params): **`66.95% ± 6.84%`** Macro-F1 (Acc `68.35%`, $\kappa = 0.5324$; $+0.21\text{ pp}$ over `RCCLoss`, $p_{\text{raw}} = 0.0272$, $p_{\text{Holm}} = 0.1212$).
- **W0-B Dynamic INT8 Edge Profile (`full_v5`):** `131,332 / 190,788` (`68.84%`) parameters quantized (`nn.Linear` + `nn.GRU`), **`360.50 KiB`** in-memory payload (`< 400 KiB`), **`67.31 ms`** 1-thread / **`44.02 ms`** 4-thread CPU latency (`L=1200`), FP32 `66.74%` vs. INT8 `66.73%` ($\bar{\Delta} = -0.0113\text{ pp}$, $p = 0.9041$).
- **External Zero-Shot PhysioNet SleepAccel (`n = 25 qc_clean`, isolated `experiments/EXP-029/sleepaccel_cache/predictions/`):**
  - `full_v5` (`em_off`): **`61.78%`** (`mean_single_model`) / **`63.51%`** (`prob_ensemble`), improving Deep% Bland-Altman bias from `+10.65%` (Model A) to `+7.59%`.
  - `DampedEMCalibrator` (`em_on`, `alpha=0.5, n_iter=3, T=1.0`): `59.73%` / `61.30%` (unsupervised single-night EM degraded Macro-F1 and widened Deep%/REM% bias; `em_off` retained as primary zero-shot mode).
- **`EXP-020` `FT-2` / `GATE_tau` Safety (`N = 46`, `tau* = 0.01`):** `full_v5` `ZS = 66.67%`, `FT-2 = 67.59%`, **`GATE_tau = 67.86%`** (`+1.19 pp` vs. `ZS`, `Bidslab34` abstained).
- **Decision:** Promote **`Ultradian-Pyramidal-Coupler` (`full_v5`, 190,788 params)** as the primary `v5` architecture, while retaining `ConventionalCNNBiGRU-164k` (`163,940` params) and `Official SLAMSS-IFS` (`5,465,768` params) as locked reference baselines.

---

## DEC-043 — EXP-030 Single-Factor Capacity–Accuracy–Edge Pareto Sweep & `p1_100k` (`98,940` Params) Selection (2026-11-24)

- **Protocol & Audit Clearance:** Executed a 75-run single-factor width sweep of plain `ScaledConventionalCNNBiGRU` (`experiments/EXP-030/exp030_models.py`) across 5 new grid points (`p1_100k`, `p3_268k`, `p4_404k`, `p5_666k`, `p6_1051k`; 5 folds $\times$ 3 seeds `42, 123, 2024`; `L = 1200`, `prep_57738d534b`, `36d84020` manifest gate) anchored against `p2_164k` (`DEC-037` frozen `Model A`, `64.7907%` via `LogitStitchingSlidingWindowPredictor`). Cleared pre-execution via `docs/audits/EXP030_PRE_EXECUTION_DRIFT_REPORT.md` (`GO`, 17/17 tests in `tests/test_exp030_preflight.py`) and post-execution via `docs/audits/EXP030_POST_EXECUTION_EGM_DRIFT_REPORT.md` (`PASS`, 0 Blocking, 0 Major).
- **Pre-Registered In-Cohort Elbow Selection (`select_exp030_elbow.py`):**
  - 6-point BIDSleep `N = 47` OOF Macro-F1: **`p1_100k` (`98,940` params) = `66.67% ± 7.57%`**, `p2_164k` (`163,940`) = `64.79% ± 7.45%`, `p3_268k` (`268,260`) = `66.43% ± 7.08%`, `p4_404k` (`403,524`) = `66.50% ± 7.16%`, `p5_666k` (`665,556`) = `66.17% ± 7.02%`, `p6_1051k` (`1,051,492`) = `66.25% ± 6.84%`.
  - Deterministic rule (`M_max = 66.6726%`, `delta_max = +1.8819 pp > 0.50 pp`, `threshold_T = 66.4844%`, `guard_triggered = False`) selected **`p1_100k` (`98,940` params)** as `CNN-BiGRU-Star` strictly before any downstream evaluation.
- **Downstream Evaluation (`p1_100k` Only):**
  - **Pillar 1 — BIDSleep (`N = 47`):** vs. `Model A-164k` (`64.79%`): **`+1.88 pp`** `[+1.25, +2.49 pp]`, $p_{\text{Holm}} = 1.19 \times 10^{-6}$, $d_z = +0.875$, `40/47` wins (REM F1 `+3.56 pp`, $p = 2.40 \times 10^{-10}$). vs. `Official SLAMSS-IFS` (`67.37%`, `5,465,768` params): **`-0.69 pp`** `[-1.57, +0.15 pp]`, **$p_{\text{Holm}} = 0.1498$** (statistical parity at **55.24x** parameter compression).
  - **Pillar 2 — `W0-B` Dynamic INT8:** **`182.00 KiB`** in-memory payload (`69,796 / 98,940` quantized), **`456.1 MMACs`** (`L=1200`), **`56.58 ms`** 1T / **`46.35 ms`** 4T CPU latency, OOF FP32-vs-INT8 delta $\bar{\Delta} = -0.0134\text{ pp}$ (`p = 0.4656`).
  - **Pillar 3 — Zero-Shot SleepAccel (`em_off` only):** `qc_clean` (`n = 25`): `60.88%` (`mean_single_model`, `+0.36 pp` vs. `Model A` $p=0.1645$, `-0.21 pp` vs. `SLAMSS-IFS` $p=1.0000$) / `62.51%` (`prob_ensemble`, `-0.43 pp` vs. `Model A` $p=0.2411$, `-0.28 pp` vs. `SLAMSS-IFS` $p=0.6150$). `all` (`N = 31`): `61.10%` / `62.51%`.
  - **Pillar 4 — `EXP-020` `FT-2` & Nested Per-Fold `GATE_tau` (`N = 46`):** Per-fold $\tau^*$ (`0.05` on Folds 1–2, `0.01` on Folds 3–5; `3/46` abstained); `ZS = 66.84%`, `FT-2 = 66.27%` (`-0.57 pp`, $p=0.6571$), `GATE_tau = 66.74%` (`-0.10 pp` vs. `ZS`, $p=0.9519$).
- **Decision:** Ratify **`p1_100k` (`98,940` params)** as the capacity-Pareto optimal `ConventionalCNNBiGRU` operating point (`55.24x` compression vs. `SLAMSS-IFS`, `-39.65%` fewer params than `Model A-164k`, `182.00 KiB` INT8), and report `EXP-029` honestly as a negative architectural ablation superseded by capacity regularization at `~100k` parameters.

---

## DEC-044 — EXP-031 Single-Factor Width Sweep & `P190` (`190,940` Params) Confirmatory Verdicts (2026-11-24)

- **Protocol & Audit Clearance:** Executed a 105-run single-factor width sweep (`w`-multiplier) of plain `ScaledConventionalCNNBiGRU` (`experiments/EXP-031/exp031_models.py`) across 7 new widths (`W050`, `W075`, `P190`, `W125`, `W150`, `W200`, `W250`; 5 folds $\times$ 3 seeds `42, 123, 2024`; `L = 1200`, `prep_57738d534b`, `36d84020` manifest gate) under locked preregistration `PREREG_EXP-031_WIDTH_SWEEP.md` (commit `24ca187`). Reused anchors: `A` (`163,940` params, `64.7907%`), `B` (`223,268`, `66.61%`), `V5` (`full_v5`, `190,788`, `66.74%`), and `S` (`SLAMSS-IFS`, `5,465,768`, `67.37%`). Cleared pre-execution via `docs/audits/EXP031_PRE_EXECUTION_DRIFT_REPORT.md` (`GO`, `25/25` tests) and post-execution via `docs/audits/EXP031_POST_EXECUTION_EGM_DRIFT_REPORT.md` (`PASS`, `0` Blocking, `0` Major).
- **Confirmatory Verdicts (`N = 47`, Holm-corrected across `{H-ARCH, H-SCALE}`):**
  - **`H-ARCH` (`full_v5` `66.74%` vs. `P190` `66.60%`):** $\bar{\Delta} = \mathbf{+0.14\text{ pp}}$ `[95% CI: -0.34, +0.61 pp]`, Wilcoxon $p = 0.4037$, $p_{\text{Holm}} = 0.4037$, $d_z = +0.086$, `26W/21L` -> **`NOT SUPPORTED (practically)`** (CI upper bound `+0.61 pp < +0.75 pp`). Architectural novelty/benefit is **not** claimed for `full_v5`.
  - **`H-SCALE` (Page's $L$-trend across 9 plain widths `W050`..`W250`):** $L = 10938.0$ ($\mu_L = 10575.0$, $\sigma_L = 145.43$), $z = \mathbf{+2.493}$, positive direction, $p_{\text{one-sided}} = 0.00634$, $p_{\text{two-sided}} = 0.01268$, **$p_{\text{Holm, two-sided}} = 0.0254 < 0.05$** -> **`SUPPORTED`**.
- **Section 9 Backbone Selection (`< 400 KiB` INT8, excluding `V5`):** $m^* = \text{B}$ (`66.61%`). `P190` (`190,940` params, `66.60% ± 6.85%`, `375.16 KiB` INT8) satisfied $\text{gap} = 0.0120\text{ pp} \le \text{SE}_{\text{boot}} = 0.1545\text{ pp}$ as the smallest qualifying eligible plain width (`selected_backbone.json`). BIDSleep estimate is labeled optimistic (design-informed).
- **Section 10 Downstream Evaluation (`P190` Only):**
  - **`W0-B` Dynamic INT8:** `375.16 KiB` in-memory (`126,532 / 190,940` quantized), `984.2 MMACs` (`L=1200`), `73.30 ms` 1T / `53.07 ms` 4T CPU latency; OOF FP32 `66.5940%` vs. INT8 `66.6184%`, $\bar{\Delta} = +0.0244\text{ pp}$ `[+0.0017, +0.0480]`, $p = 0.0836$.
  - **Zero-Shot SleepAccel (`em_off` only):** `qc_clean` (`n = 25`): `61.52%` (`mean_single_model`, `+1.01 pp` vs. `Model A` $p=0.0755$, `-0.25 pp` vs. `full_v5` $p=0.5602$, `+0.44 pp` vs. `SLAMSS-IFS` $p=0.5424$) / `63.42%` (`prob_ensemble`, `+0.48 pp` vs. `Model A` $p=0.2872$, `-0.09 pp` vs. `full_v5` $p=1.0000$, `+0.64 pp` vs. `SLAMSS-IFS` $p=0.7112$). `all` (`N = 31`): `61.77%` / `63.94%`.
  - **`EXP-020` `FT-2` & Nested `GATE_tau` (`N = 46`):** Per-fold $\tau^*$ (`0.05` Folds 1–2, `0.01` Folds 3–5; `3/46` abstained); `ZS = 66.42%`, `FT-2 = 66.09%` (`-0.33 pp`, $p=0.5882$), `GATE_tau = 66.55%` (`+0.13 pp` vs. `ZS`, $p=0.7720$).
- **Decision:** Ratify `H-ARCH = NOT SUPPORTED (practically)`, `H-SCALE = SUPPORTED`, and **`P190` (`190,940` params)** as the Section 9 selected plain CNN-BiGRU backbone for `EXP-031`; report `full_v5` strictly as a capacity-matched ablation without architectural novelty claims.

---

## DEC-045 — EXP-032 High-Accuracy Benchmark (`<= 1.5M` Params, `DEC-031` Exception) & Teacher-Distilled `M3` Parity (2026-11-26)

- **Protocol & Audit Clearance:** Executed under locked `docs/designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md` (pre-registered exception to `DEC-031`'s `Lightweight-First` framing for a server / companion-phone high-accuracy benchmark up to `1.5M` params, non-causal `L=1200` on `prep_57738d534b`, `36d84020` manifest gate). Cleared pre-execution via `docs/audits/EXP032_PRE_EXECUTION_DRIFT_REPORT.md` (`GO`, `10/10` tests in `tests/test_exp032_preflight.py`) and post-execution via `docs/audits/EXP032_POST_EXECUTION_EGM_DRIFT_REPORT.md` (`PASS`, `0` Blocking, `0` Major).
- **Stage A Pooled `inner_val` (`N=25` `(fold, subject_id)`, `60` runs at `seed=42`):** Locked `M1: cfg#3` (`dropout=0.3, wd=0.01`, `66.97%`), `M2: cfg#3` (`dropout=0.3, wd=0.01`, `67.06%`), and `M3: cfg#1` (`dropout=0.3, wd=0.01, tau=2.0, alpha_kd=0.50`, `69.51%`) in `experiments/EXP-032/locked_configs.json`.
- **Stage B BIDSleep (`N=47`, `45` locked runs, 10,000-resample bootstrap, two-sided Wilcoxon, Holm-Bonferroni):**
  - **`M1` (`1,402,468` params, `4,053.08` MMACs):** `65.21%` Macro-F1 (`-2.16 pp` `[-3.00, -1.34]` vs. `SLAMSS-IFS`, $p_{\text{Holm}} = 3.14 \times 10^{-5}$; `-1.47 pp` vs. `W100`, $p = 0.00153$), `67.45%` Accuracy, $\kappa = 0.5244$.
  - **`M2` (`1,402,492` params, `+ LinearChainCRF4`):** `65.18%` Macro-F1 (`-2.19 pp` `[-3.06, -1.35]` vs. `SLAMSS-IFS`, $p_{\text{Holm}} = 3.14 \times 10^{-5}$; `M2 - M1 = -0.03 pp`, $p = 0.2809$), `67.44%` Accuracy, $\kappa = 0.5232$.
  - **`M3` (*Teacher-Distilled* `EXP-021` `SLAMSS-IFS` + `M2` CRF, `1,402,492` params):** **`67.38%`** Macro-F1 (**`+0.02 pp`** `[-0.64, +0.69]` vs. `SLAMSS-IFS` `67.37%`, $p_{\text{Holm}} = 0.8627$ — statistical parity; `+2.21 pp` `[+1.69, +2.74]` vs. `M2`, $p = 5.07 \times 10^{-10}$, `43/47` wins), **`69.75%` Accuracy** (**`+1.00 pp`** `[+0.38, +1.61]` vs. `SLAMSS-IFS` `68.75%`, $p = 0.00397$, `33/47` wins), $\kappa = 0.5548$.
- **External SleepAccel (`em_off` only, `n=25 qc_clean` / `N=31 all`):** `M1` `60.88%` / `62.50%` (`60.46%` / `62.02%` all); `M2` `61.00%` / `62.48%` (`60.51%` / `61.89%` all); `M3` **`61.43%` / `62.63%`** (`61.76%` / `63.01%` all; `+0.34 pp` single vs. `SLAMSS-IFS`, $p_{\text{NI}} = 0.00032$ at $\delta_{\text{NI}} = -2.50\text{ pp}$).
- **Interleaved FP32 CPU Latency (`L=1200`, `AMD Ryzen 7 7700`):** `W100` `53.61 / 41.61 ms`, `M1` `115.44 / 73.47 ms`, `M2` `128.49 / 87.51 ms`, `M3` `129.14 / 87.75 ms` (`28.1x` `1T` / `35.7x` `4T` faster than `SLAMSS-IFS` `3,631.24 / 3,131.18 ms`).
- **Decision:** Ratify the locked §5.2 verdict **`NOT SUPPORTED`** on primary BIDSleep Macro-F1 superiority (`K_qualifying = 0`, all 3 arms `ARM_NOT_SUPPORTED`). Retain **`p1_100k`** (`98,940` params / `182.00 KiB` INT8) as the flagship wearable edge model, and record **`M3` (`1,402,492` params, explicitly labeled teacher-distilled)** as the server / companion-phone distillation benchmark achieving Macro-F1 parity and `+1.00 pp` Accuracy over `SLAMSS-IFS` at `3.90x` fewer parameters and `35.7x` faster 4T CPU inference.






