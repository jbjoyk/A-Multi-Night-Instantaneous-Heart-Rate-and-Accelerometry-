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


