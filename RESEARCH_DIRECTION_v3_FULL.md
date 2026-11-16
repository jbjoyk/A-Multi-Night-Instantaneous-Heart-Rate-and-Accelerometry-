# SLEEP Research Direction (v3, full)

**Supersedes** `RESEARCH_DIRECTION.md` (v2).
**Lock rule:** Commit this file to `origin main` and record the commit hash in `EXPERIMENTS.md`. After that commit, sections 4, 6, 7 and 9 may not be edited to match results. Changes go into section 15 as a dated amendment with the reason. If v2 was already committed, record v3 as an amendment to v2 (reason: Q1-readiness review, section 0), and do not rewrite history.

Items marked `TO FILL` or `TO VERIFY` are facts nobody has confirmed yet. Do not remove the marker until someone has checked.

---

## 0. Reality check: is the old contribution enough for Q1?

No. Three results by themselves would not carry a Q1 paper:

1. **"Longer temporal context helps"** (L60 to L600, +5.33 pp Macro-F1, 46/47 subjects improved). Sequence context improving sleep staging is well established. This is a design choice and a sanity check, not a contribution.
2. **A single fusion-mechanism comparison** (interaction module vs concatenation). Incremental architecture work, which reviewers discount heavily.
3. **"We beat a reported number."** Without protocol and metric matching, this is not a valid comparison, and with matching it still tells reviewers little about why.

Further weaknesses reviewers will raise regardless of the method:

- 47 healthy adults, one site, one watch/headband setup. Generalization claims are weak without external validation.
- Labels come from a headband (Dreem 2) with expert review by a single scorer, not full PSG.
- The dataset and the baseline paper come from the same group, so reviewers likely know the data and protocol well.

**The open lane.** The dataset's own usage notes say the authors trained per-night models and encourage future work on approaches that leverage multi-night information within individuals, because the dataset is well suited to that. The dataset was published in 2026 and had only two citing publications when checked (Nov 2026). Multi-night, within-person modeling is therefore the most defensible primary contribution. `TO VERIFY` by a systematic literature search before any novelty claim.

**Realistic positioning:** with the multi-night adaptation contribution, external validation, a leakage-free benchmark, and clinical sleep-summary agreement, a mid-tier Q1 biomedical-engineering venue is plausible, not guaranteed. A top-tier venue is not realistic on one 47-subject dataset. Verify each target journal's current quartile on JCR/Scimago for the year of submission, because quartiles change.

---

## 1. Problem statement (rewritten; replaces the old Bengali draft)

**Why the old draft cannot be used:** it described the dataset (title, 47 participants, 253 nights, Dreem 2, AASM) as work this project carried out. This project did not collect the data. It also closely tracks the dataset documentation, which creates misrepresentation and plagiarism risk. Always state plainly that the data are a public dataset, cited as Song et al. (2026), PhysioNet, doi:10.13026/rees-1092, with the original publication Song et al. (2025/2026), IEEE TBME, doi:10.1109/TBME.2025.3612158.

**Problem statement (English, paper-ready draft):**

Sleep stages are conventionally scored from polysomnography (PSG), which is costly and impractical for repeated, multi-night, at-home monitoring. Consumer smartwatches record photoplethysmography-derived heart rate and accelerometry, making long-term sleep monitoring feasible, but staging from these signals alone remains substantially less accurate than EEG-based scoring, and accuracy degrades for individuals not seen during training. Most wearable sleep-staging models are trained and evaluated one night at a time, or pool nights across subjects, so they do not exploit the fact that the same person generates repeated recordings with person-specific physiology and behavior. Publicly available multi-night wearable recordings with EEG-derived labels have only recently appeared.

Using the public BIDSleep dataset (47 healthy adults, up to seven nights each, Apple Watch instantaneous heart rate and accelerometry, Dreem 2 EEG-derived AASM labels), this work asks whether information from a person's earlier nights can improve staging on that person's later nights, how many nights are needed, and whether this improvement holds under strict subject-level, leakage-free evaluation and on an independent external dataset.

---

## 2. Research gaps (rewritten: three gaps, each closed by this project)

The old list had 20 gaps. A paper that claims 20 gaps closes none of them, and several of them cannot be addressed with this data. Retained gaps below need citations (`TO FILL`: add references from a systematic literature search; none of the claims below are verified until then).

| # | Gap | What this project does about it |
|---|---|---|
| G1 | Within-person multi-night information is unexploited in wearable sleep staging: models are trained per night or pooled across nights, and it is unknown how many labeled or unlabeled nights of a person materially improve staging on later nights. | Pre-registered multi-night adaptation study (H-A). |
| G2 | Subject-independent staging from low-rate IHR (~0.2 Hz) and accelerometry is limited, and evaluation protocols differ across studies (night-level vs subject-level splits, different class granularity), so published numbers are hard to compare. | Leakage-free subject-level benchmark with re-implemented baselines and a protocol-matched comparison (sections 6 to 8). |
| G3 | Wearable staging is reported as stage-level accuracy, rarely as clinically used sleep-summary measures, and robustness to missing or noisy data is under-reported. | Sleep-summary agreement (TST, sleep onset latency, WASO, sleep efficiency, REM%, N3%) with Bland-Altman analysis; controlled-missingness robustness test (H-R, optional). |

**Disposition of the 20 original gaps** (numbered in the order of the original document):

| Original gap | Disposition |
|---|---|
| 1 Single-night data; 10 Personalized learning; 16 Longitudinal behavior | Merged into G1. Keep. |
| 17 Cross-subject generalization | G2 and H-A. Keep. |
| 18 Temporal dependency | Supporting hypothesis H-C. Keep as supporting. |
| 8 Multimodal integration; 19 HR and movement joint effect | Supporting hypothesis H-F. Keep as supporting. |
| 11 Noise/missing data; 20 Clinical-level reliability | G3. Keep. |
| 9 Transition analysis with XAI; 15 Explainable AI | Exploratory analysis only (E3, E4). Claim only what is delivered. |
| 2 Few multi-night datasets; 14 Few open Apple Watch multi-night datasets; 5 EEG synchronization | Closed by the dataset authors. Use as motivation, not as your gap. |
| 3 Black-box commercial algorithms; 4 Limited HR+accelerometry accuracy | Motivation only. |
| 6 Low subject diversity; 7 Lab vs real-world | Limitation of this data (healthy adults, single site). Move to Limitations. |
| 12 Computational cost / real-time deployment | Drop, unless you measure parameters and latency. |
| 13 Sleep disorder detection | Drop. Healthy-only data cannot support it. |

---

## 3. Claimed contributions (maximum three)

1. **C1 (primary):** a pre-registered, leakage-free study of multi-night within-person adaptation for wearable sleep staging, quantifying how many calibration nights are needed and for whom it helps.
2. **C2:** a leakage-free, protocol-documented subject-level benchmark on BIDSleep with re-implemented baselines, plus external validation on an independent Apple Watch dataset.
3. **C3:** clinical sleep-summary agreement and robustness analysis for the final model.

Not claimed: sleep disorder detection, clinical-grade accuracy, real-time deployment, population-level generalization.

---

## 4. Research questions and hypotheses (LOCKED)

### Primary research question (confirmatory)

> Does adapting a subject-independent IHR and accelerometry sleep-staging model with a few of a person's earlier nights improve staging on that person's later nights, compared with the same subject-independent model applied without adaptation?

### Primary hypothesis H-A (confirmatory)

> Adapting a subject-independent model with k = 2 labeled calibration nights of a held-out subject improves Macro-F1 on that subject's later nights compared with the same model without adaptation (zero-shot).

**Protocol:**

- Subject-level folds. The backbone is trained only on training-fold subjects.
- For each held-out subject, order nights chronologically by `recStart`. The first k nights are calibration nights, and the remaining nights are evaluation nights. Calibration and evaluation nights never overlap.
- All arms are evaluated on the **same** evaluation nights.
- Arms:
  - **ZS:** subject-independent model, no subject data.
  - **FT-k:** ZS model fine-tuned on k calibration nights, with epochs, learning rate and layers frozen in advance and never tuned on evaluation nights.
  - **PS-k:** model trained from scratch on the same k nights only (per-subject baseline).
  - **UA-k (secondary):** label-free adaptation (for example, per-subject normalization statistics or test-time adaptation) using k nights' signals, no labels.
- Subjects: every subject has at least 3 nights per the dataset documentation, so k = 2 leaves at least 1 evaluation night. Report how many evaluation nights each subject has.

**Decision rule (see section 9 for why this threshold):** H-A is supported if the paired subject-level mean difference in Macro-F1 (FT-2 minus ZS) is at least **+1.5 pp** and the bootstrap 95% CI (resampling subjects) excludes 0. Otherwise it is **not supported**, with no rescue tuning.

**Secondary tests under H-A:**

- FT-2 vs PS-2: adaptation should beat per-subject training from scratch. Same rule.
- Learning curve for k = 1, 2, 3, 4 (descriptive; subjects with fewer nights drop out at larger k, so report n per k).
- UA-2 vs ZS (label-free variant, same rule).

### Supporting hypothesis H-C (confirmatory replication of the EXP-018 context finding)

> In the final pipeline under subject-level cross-validation, L600 improves Macro-F1 over L60.

Decision rule: paired difference at least +2.0 pp and CI excludes 0. (EXP-018 observed +5.33 pp, which was exploratory.)

### Supporting hypothesis H-F (confirmatory, secondary)

> At L600 and matched parameter count, a cross-modal IHR-accelerometry interaction module improves Macro-F1 over concatenation fusion.

Decision rule: same as H-A. If not supported, report it as a negative result.

### Optional hypothesis H-R (freeze its definition before running)

> Under controlled IHR missingness, a reliability-aware model degrades less in Macro-F1 than the unweighted baseline.

Before this can be frozen: describe real missingness in the data (Phase 0), derive realistic gap patterns from it, and define the reliability measure in writing.

### Exploratory analyses (no confirmatory claim)

- **E1:** per-stage F1 for every comparison. No directional assumption for any stage.
- **E2:** time-of-night. EXP-018 observed explicit time-of-night benefit shrinking with context (L1 +14.77, L60 +4.38, L600 +1.36 pp Macro-F1). Descriptive only.
- **E3:** stage-transition error analysis.
- **E4:** modality ablation (HR only, accelerometry only, both), temporal importance (permutation or saliency). Claim interpretability only for what is actually delivered.
- **E5:** heterogeneity: whether adaptation gain correlates with subjects' zero-shot performance, and per-night-index effects (first-night effects may confound chronological adaptation).

---

## 5. Provenance

This direction was written after EXP-001 to EXP-018 and EXP017 had been run. Everything from those experiments is **exploratory**. Only experiments run after the lock commit, following the pre-registration protocol (section 10), are **confirmatory**. Papers must label results accordingly.

---

## 6. Data and task definition (LOCKED)

- **Data:** BIDSleep, PhysioNet doi:10.13026/rees-1092. Files per night: `hr.csv`, `motion.csv`, `labels.mat`. IHR is ~0.2 Hz, PPG-derived via HealthKit. Epochs are 30 s. Alignment: `k = floor((t - recStart) / 30) + 1`.
- **Target labels:** `expert_label` (primary). Report `dreem_label` results as a sensitivity analysis. State how label 5 (Unknown) is handled (`TO FILL`: recommended: exclude those epochs from training and evaluation, and report how many).
- **Class granularity:** report 5-class (W/N1/N2/N3/REM) as primary, plus 4-class and 3-class (W/NREM/REM) for comparability. Record which granularity the SLAMSS-IFS 71.04% refers to: `TO VERIFY`.
- **Noise ceiling:** labels come from a headband plus single-scorer review. State this as a limitation.
- **Descriptive Phase 0 stats (report in the paper):** nights per subject, per-subject class distributions, IHR missingness patterns, per-subject variability.

---

## 7. Evaluation protocol (LOCKED)

- **Subject-level splits only.** Never split by night within a subject for model evaluation. The dataset documentation recommends subject-level analysis to avoid leakage. The only within-subject night split allowed is the chronological calibration/evaluation split in H-A, and only for held-out subjects.
- **Audit:** confirm EXP-001 to EXP-018 all used subject-level splits. Any that did not are marked invalid in `EXPERIMENTS.md`. Result: `TO FILL`.
- **Fold assignment:** fixed subject-level folds, committed to the repo as a manifest file.
- **Frozen test set (EXP-012):** used only for final evaluation. Keep a test-access log (date, experiment, reason). Number of test evaluations to date: `TO FILL`. Model and hyperparameter selection use validation subjects only. Composition of the frozen set: `TO FILL`.
- **Seeds:** at least 3 per configuration. Average over seeds within each subject first, then treat **subjects** as the unit of analysis. Seeds are not independent samples.
- **Causality:** state whether L600 uses past-only or past-and-future context: `TO FILL`. If non-causal, the paper says results apply to offline staging and makes no real-time claim.
- **Time-of-night feature:** excluded from primary models unless a deployment justification is written. Report both with and without.
- **Metrics:** Macro-F1 (primary), Cohen's kappa, accuracy, weighted F1, MCC, sensitivity/specificity, per-stage F1, confusion matrix. Clinical layer: TST, sleep onset latency, WASO, sleep efficiency, REM%, N3% with Bland-Altman agreement.

---

## 8. Baselines and external validation

**Baselines (all on identical subject-level splits):**

1. Handcrafted-feature gradient-boosting model on IHR and accelerometry features.
2. A standard sequence model (for example, CNN with BiLSTM or TCN) at L600.
3. **SLAMSS-IFS re-implementation** (`TO VERIFY`: confirm what the baseline method is and its protocol from the TBME paper, doi:10.1109/TBME.2025.3612158). Record whether the reported 71.04% is subject-independent and which class granularity it uses.

**Wording rule:** until baseline 3 is re-run on your splits, write "achieved higher accuracy than the reported SLAMSS-IFS result under a different protocol." Superiority claims require a paired subject-level test on identical splits.

**External validation:** the public Apple Watch dataset by Walch et al. (University of Michigan, 31 subjects, one PSG night each, PSG-scored labels; doi:10.13026/hmhs-py35; the dataset description notes exclusions for apnea and REM sleep behavior disorder).

- Zero-shot transfer: train on BIDSleep training subjects, evaluate on Walch subjects.
- Needs label mapping (its labels use 0-5 coding with REM = 5, and it has no multi-night data), alignment of its HR and acceleration formats, and caution: different lab, different label source (PSG vs headband), different sampling.
- Multi-night adaptation (H-A) cannot be tested there because each subject has one night. Report the pipeline's zero-shot performance drop and baseline comparison with CIs. Pre-register it as a secondary, descriptive result with no superiority claim unless the paired CI excludes 0.

---

## 9. Statistics and power

- Unit of analysis: subject. Paired differences, 10,000-resample bootstrap 95% CI, plus Wilcoxon signed-rank as a robustness check. Report how many subjects improved.
- Multiplicity: H-A is the single primary hypothesis for the paper's headline claim. H-C and H-F are supporting; apply Holm correction across H-A, H-C and H-F, and report corrected and uncorrected results.
- **Power (estimate, assumption-dependent):** EXP-018's CI half-width of about 1.06 pp on 47 subjects implies a standard deviation of paired subject-level differences of about 3.6 pp. If a similar spread holds, the standard error is about 0.52 pp and the minimum detectable effect at 80% power is about **1.5 pp**. A +1.0 pp target would have only about 45% power and would often be inconclusive. This is why the threshold above is +1.5 pp (v2 used +1.0 pp, which was underpowered). After Phase 1, re-estimate the spread from the real paired comparisons and revise the threshold before lock if the spread differs.
- Report effect sizes and CIs, not only p-values.

---

## 10. Pre-registration protocol (every confirmatory experiment)

Before running, add a block to `EXPERIMENTS.md` and commit it:

1. Hypothesis ID and one-sentence prediction
2. Single factor changed, everything else held constant
3. Primary metric and decision rule (threshold and CI condition)
4. Splits (manifest hash), seeds, parameter budget
5. What counts as "not supported"

After running: record the outcome against the rule exactly as written, including negative results. Do not edit the block.

---

## 11. Experiment roadmap

IDs are suggestions; keep your own numbering if different.

| Phase | Work | Experiments |
|---|---|---|
| 0 Hygiene | Audit splits; fill test-access log; lock this document; fold manifest; descriptive data statistics; check SLAMSS-IFS protocol | none (analysis only) |
| 1 Baselines | Gradient-boosting baseline; sequence-model baseline at L600; SLAMSS-IFS re-implementation; estimate paired-difference spread | EXP-019 to EXP-021 |
| 2 Context replication | H-C under subject-level CV | EXP-022 |
| 3 Core contribution | H-A: ZS vs FT-k vs PS-k vs UA-k; learning curve over k | EXP-023, EXP-024 |
| 4 Supporting | H-F fusion comparison; optional H-R robustness | EXP-025, EXP-026 |
| 5 External | Zero-shot transfer to Walch dataset | EXP-027 |
| 6 Final | Frozen test evaluation; clinical sleep-summary metrics; exploratory E1 to E5 | EXP-028 |

Rules throughout: no random architecture search, no blind hyperparameter optimization, no fashionable architecture without a scientific reason, no tuning on the test set, no changes to hypotheses or decision rules after seeing results, no experiments chosen only because they raise accuracy.

---

## 12. Outcomes and what each means for the paper

| Outcome | Condition | Framing |
|---|---|---|
| A | H-A supported, and a protocol-matched comparison favors the final model over re-implemented SLAMSS-IFS | Method and benchmark paper with performance and mechanism claims |
| B | H-A supported, baseline not beaten or comparison not possible | Personalization/benchmark paper, no superiority claim |
| C | H-A not supported | Negative-result and benchmark paper: leakage-free multi-night evaluation, quantified limits of per-person adaptation with IHR + accelerometry |

Outcome C is publishable in an appropriate venue. It is not a failed project, but it will likely target a lower tier than Outcome A or B.

---

## 13. Q1 readiness checklist

- [ ] Split audit complete, no leakage; fold manifest in repo
- [ ] Test-access log complete and small
- [ ] SLAMSS-IFS protocol checked and baseline re-implemented on identical splits
- [ ] H-A pre-registered before it was run; outcome reported exactly as specified
- [ ] External validation reported with CIs
- [ ] Clinical sleep-summary agreement reported
- [ ] Per-stage results and confusion matrices reported, including weak stages
- [ ] Code, split manifests and configs released (note the dataset license: Open Data Commons Attribution)
- [ ] Limitations stated: healthy adults, single site, headband plus single-scorer labels, no disorder generalization, offline vs real-time (per causality)
- [ ] Reporting checklist for prediction-model studies followed if the target journal asks for one (for example TRIPOD+AI; `TO VERIFY` current requirement)
- [ ] Systematic literature search completed; novelty claims checked against it
- [ ] Target journal's current quartile and scope verified

**Candidate venues** (verify quartile and fit for the submission year): IEEE Journal of Biomedical and Health Informatics, Computers in Biology and Medicine, Biomedical Signal Processing and Control for engineering-focused framing; Sleep, Journal of Sleep Research or Sleep Medicine for a clinical framing, which needs the clinical metrics in section 7. IEEE TBME is the venue of the baseline paper and a higher bar.

---

## 14. Paper skeleton

Working title: *Leveraging multi-night information for subject-adaptive sleep staging from wearable instantaneous heart rate and accelerometry.*

1. Introduction: problem (section 1), gaps G1 to G3, contributions C1 to C3
2. Related work: wearable sleep staging, protocol heterogeneity, personalization
3. Data and task (section 6), with explicit dataset attribution
4. Methods: backbone, adaptation arms, baselines
5. Evaluation protocol and pre-registration statement (sections 7, 9, 10)
6. Results: Phase 1 baselines, H-C, H-A (primary), H-F, external validation, clinical metrics, exploratory analyses clearly labeled
7. Discussion: why adaptation helps or does not, who benefits, limitations
8. Conclusion

---

## 15. Amendments

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |

---

## 16. One-sentence target for every research and coding agent

> Develop and rigorously evaluate subject-independent and multi-night-adaptive wearable sleep staging from IHR and accelerometry on BIDSleep, testing pre-registered single-factor hypotheses under strict subject-level, leakage-free evaluation, comparing with SLAMSS-IFS only under a documented protocol-matched paired comparison, validating externally, and reporting negative results as faithfully as positive ones.
