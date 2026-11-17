# SLEEP Research Direction (v4, full and consolidated)

**Date:** 2026-11-17
**Supersedes:** `RESEARCH_DIRECTION_v3_FULL.md`, the v3.1 amendment and `RESEARCH_DIRECTION_v4_AMENDMENT_PROPOSAL.md`. Those files stay in the repo as history; this file is the working direction once approved and committed.
**Approval:** Per the KB governance rule, a change of direction needs your explicit approval. Commit this file and record the hash in `EXPERIMENTS.md` to approve it.
**Never edited by this document:** the EXP-019 preregistration (commit `4dff5b7`), its numbers, arms, configurations and stored artifacts.
**Markers:** `TO FILL` and `TO VERIFY` are facts nobody has confirmed. Do not remove a marker until someone has checked the source.

---

## 1. Where the project stands (honest summary)

| Established | Evidence | Status |
|---|---|---|
| Longer temporal context improves subject-independent staging (L60 to L600: Macro-F1 57.73% to 63.06%, +5.33 pp, 95% CI [+4.28, +6.39], 46/47 subjects) | EXP-018 Phase 2 | Counts as confirmatory only if `git log` shows its decision rule was committed before the Phase 2 runs: `TO VERIFY` |
| Full-night context (L1200) is better than the 5-hour crop (Macro-F1 0.6479 vs 0.6306; accuracy 66.18% vs 65.45%) | EXP-018 Phase 2 | Exploratory |
| Architecture mechanisms (gating, transition, GRL) give no detectable gain over matched Model A | EXP-014: 75 runs, 47 subjects; Model E vs A Macro-F1 -0.0069, CI [-0.0213, +0.0059] | Exploratory; architecture search is closed |
| Two-night supervised adaptation gives a small, heterogeneous gain: FT-2 vs ZS +1.01 pp, CI [-0.17, +1.99], 34/46 subjects improved | EXP-019 (locked) | Verdict label open, see section 5 |
| The gain is not explained by extra optimization alone: FT-2 vs FT-other-2 +2.73 pp, CI [+1.05, +4.32] | EXP-019 S1 | Careful wording needed: much of the gap is harm from donor-night fine-tuning (FT-other-2 is 1.72 pp below ZS) |
| Unsupervised adaptation (UA-2) equals ZS to the last decimal in all 46 subjects | EXP-019 S3 | **Unresolved:** looks like a no-op; implementation check pending |
| Failure case: Bidslab34 -16.34 pp, Bidslab65 -6.57, Bidslab10 -6.47 | EXP-019 | Mechanism unproven; exploratory audit links Bidslab34 to a calibration night with essentially no Deep sleep |

**What is not established:** that adaptation works; that gating or regularization fixes failures; any superiority over SLAMSS-IFS (protocol not matched); anything about sleep disorders or other populations.

**New external fact that shapes this direction:** supervised multi-night personalization of wrist PPG plus accelerometry staging has already been published (van der Aar et al., *Physiological Measurement*, 2025, DOI 10.1088/1361-6579/ae0119: n = 59 sleep-disordered adults, up to 7 home nights, 82.5% of subjects improved, significant at 3 or more training nights). It was absent from the KB when v3 called multi-night modeling an "open lane." That claim is withdrawn.

---

## 2. Problem statement

Sleep stages are conventionally scored from polysomnography (PSG), which is costly and impractical for repeated multi-night at-home monitoring. Consumer smartwatches record photoplethysmography-derived heart rate and accelerometry, so long-term monitoring is feasible, but staging from these signals alone is less accurate than EEG-based scoring and degrades for people not seen in training. Personalization from a person's earlier nights has been reported to help wrist-PPG staging in sleep-disordered adults, and similar ideas exist for EEG-based staging. It is not established how much a consumer-device instantaneous heart rate (about 0.2 Hz) and accelerometry model in **healthy adults** benefits from a few labeled earlier nights, who is harmed, and why, when evaluated chronologically, without leakage, and under pre-registered rules.

Data are the public BIDSleep dataset (47 healthy adults, up to seven nights each, Apple Watch IHR and accelerometry, Dreem 2 EEG-derived AASM labels), cited as Song et al. (2026), PhysioNet, DOI 10.13026/rees-1092, with the associated *IEEE TBME* paper, DOI 10.1109/TBME.2025.3612158. This project did not collect the data.

---

## 3. Research gaps (three, each addressed)

| # | Gap | Addressed by |
|---|---|---|
| G1 | Within-person multi-night adaptation has been reported for wrist PPG in sleep-disordered adults; its behavior with consumer IHR in healthy adults under chronological, leakage-free, pre-registered evaluation, including who is harmed and why, is unquantified. | EXP-019 (locked), the k = 1 to 4 curve, EXP-019B, EXP-020 |
| G2 | Subject-independent staging from low-rate IHR and accelerometry is limited, and published numbers are hard to compare because evaluation protocols differ (split unit, crop, class definition). | Leakage-free subject-level benchmark with re-implemented baselines (section 8) |
| G3 | Stage-level accuracy is rarely translated into clinical sleep-summary measures, and robustness to missing or noisy data is under-reported. | EXP-028 (sleep-summary agreement); robustness analysis (optional) |

The 20 gaps in the original Bengali Research Gaps document are dispositioned in `RESEARCH_DIRECTION_v3_FULL.md` section 2 (kept, merged, motivation only, limitation, or dropped).

---

## 4. Contributions and claim boundaries

**Contributions (maximum three):**
- **C1 (primary):** a pre-registered, leakage-free replication and extension of multi-night supervised personalization with consumer IHR and accelerometry in healthy adults. It includes the specificity control (FT-other), the k = 1 to 4 learning curve, and a heterogeneity analysis.
- **C2 (secondary, exploratory):** calibration-night class coverage as a candidate failure mode, with a nested evaluation of regularized or gated adaptation.
- **C3:** external zero-shot validation and clinical sleep-summary agreement.

**May claim (once the work in section 9 is done):**
- Effect size and heterogeneity of two-night adaptation with consumer IHR in healthy adults, under a protocol that avoids the known leakage pitfalls.
- That fine-tuning on a person's own nights did better than fine-tuning on other people's nights, with the harm of the latter stated.
- A relationship between the gain and the number of calibration nights (k = 1 to 4), compared with the earlier report of about 3 nights.

**Must not claim:**
- "First" multi-night personalization, or novelty of supervised fine-tuning.
- That personalization works. The pre-registered H-A was not met.
- That gating or KL regularization is new in sleep staging (not yet systematically searched).
- Superiority over SLAMSS-IFS without a protocol-matched re-run.
- Anything about sleep disorders, clinical-grade accuracy, or real-time use.

---

## 5. Hypotheses, questions and their status

| ID | Question or hypothesis | Status |
|---|---|---|
| **H-A** | Fine-tuning on a held-out subject's first 2 labeled nights improves Macro-F1 on later nights vs the zero-shot model | **Locked, EXP-019.** Result: +1.01 pp, CI [-0.17, +1.99]. See the verdict note below. |
| S1 | FT-2 beats FT-other-2 | Locked. Supported (+2.73 pp, CI [+1.05, +4.32]) |
| S2 | FT-2 beats per-subject-from-scratch (PS-2) | Locked. Supported (+20.32 pp). Weak baseline, expected |
| S3 | Label-free adaptation (UA-2) beats ZS | Locked result 0.00 pp. **Unresolved pending no-op check** |
| K | Gain vs number of calibration nights k = 1 to 4 | Pre-registered descriptive output, not yet reported |
| H-C | L600 beats L60 | EXP-018 Phase 2; confirmatory only if its decision rule was committed before the runs (`TO VERIFY`) |
| X1 to X10, G0 to G3 | Why adaptation helps or harms, from calibration-night and baseline information | Exploratory (`EXP-019B_HETEROGENEITY_ANALYSIS_PLAN.md`) |
| EXP-020 | Regularized or gated adaptation reduces harm without losing the average gain | Exploratory with nested validation (section 9, step D1) |
| RQ-X | Zero-shot transfer to an external PSG cohort | Pre-registered (EXP-027) |
| RQ-Y | Sleep-summary agreement with clinical margins | Pre-registered (EXP-028) |

**Verdict note on H-A.** The locked preregistration text I drafted defines SUPPORTED (Δ at least +1.5 pp and CI lower bound above 0), NOT SUPPORTED (CI upper bound below +1.5 pp) and INCONCLUSIVE (otherwise). With CI [-0.17, +1.99] that rule gives INCONCLUSIVE, while the EXP-019 report says NOT SUPPORTED. Run `git show 4dff5b7:PREREG_EXP-019_H-A.md`, make the report and KB match the locked text, and cite the commit. This changes a label, not a number. Until then the safe sentence is: "Supervised two-night adaptation did not meet the pre-registered threshold (mean +1.01 pp Macro-F1, 95% CI -0.17 to +1.99; 34/46 subjects improved), and the result is inconclusive for effects up to about 2 pp."

**Power note.** The spread of paired subject-level differences is about 3.7 pp (derived from the CI). With 46 subjects, the minimum detectable effect at 80% power is about 1.5 pp. A true +1.0 pp effect would need roughly 107 subjects, so BIDSleep alone could never have confirmed it.

---

## 6. Data and task (locked)

- **Data:** BIDSleep. Usable nights: 246 of 253. IHR about 0.2 Hz (HealthKit), 3-axis accelerometry, 30 s epochs.
- **Task:** 4-class (Wake/Light/Deep/REM); valid labels 0 to 3, others masked. Definition of "Light" in AASM terms: `TO VERIFY`. Whether the published 71.04% (Table V) uses this granularity: `TO VERIFY`.
- **Label source:** `TO FILL` (`expert_label` or `dreem_label`), with the other as a sensitivity analysis.
- **Context:** full night (L = 1200, sliding-window inference). The 5-hour crop (T = 600) dropped about 28% of labeled epochs with REM depleted (DEC-021); report all cropped results as cropped.
- **Model:** Model A (`ConventionalCNNBiGRU`, 163,940 parameters) as the matched baseline. No architecture search.
- **Time-of-night:** excluded from primary models; report with and without.
- **Label noise ceiling:** headband labels with single-scorer review; state as a limitation.

---

## 7. Evaluation protocol (locked)

- **Subject-level splits only.** Never split by night within a subject, except the chronological calibration/evaluation split for held-out subjects in adaptation studies.
- **Folds:** fixed 5-fold subject-disjoint manifests (EXP-014/EXP-018); manifest hash `TO FILL`.
- **The EXP-012 test set (11 subjects, 58 nights) is spent.** It was evaluated at least in EXP-004 and EXP-012 (and `TO VERIFY` EXP-013 and EXP-001 to EXP-003), then used inside later CV folds. Final claims rest on out-of-fold nested CV across all 47 subjects plus external validation. Maintain the test-access log in `EXPERIMENTS.md`.
- **Audit status:** EXP-012, EXP-014 and EXP-018 use subject-disjoint splits; EXP-001 to EXP-004: `TO VERIFY`. Any run that split by night is marked invalid.
- **Seeds:** at least 3. Average within subject first; the unit of analysis is the subject.
- **Statistics:** paired differences, 10,000-resample subject-level percentile bootstrap, Wilcoxon signed-rank as a robustness check, Holm correction across confirmatory secondary tests, effect sizes with CIs.
- **Metrics:** Macro-F1 (primary), kappa, accuracy, weighted F1, MCC, sensitivity/specificity, per-stage F1, confusion matrices, and sleep-summary agreement (TST, SOL, WASO, sleep efficiency, REM%, deep%) with Bland-Altman analysis.
- **Causality of context:** `TO FILL` (past-only or past-and-future). If non-causal, results apply to offline staging only.

---

## 8. Baselines and comparison rules

1. Matched Model A (the baseline for every adaptation arm).
2. A handcrafted-feature gradient-boosting model and a standard sequence model (CNN with BiLSTM or TCN) at full-night context.
3. SLAMSS-IFS: re-run on the same subject-level splits. The official-repository reproduction (DEC-011, Classification B) must be independently re-run before any publication claim. Protocol and class granularity of the published 71.04%: `TO VERIFY`.
4. Reference to van der Aar et al. 2025, with protocol differences stated (sleep-disordered adults, home-night training, PSG testing, raw PPG, all combinations of nights). No numeric superiority claim across protocols.

**Wording until item 3 exists:** "achieved lower/higher accuracy than the reported SLAMSS-IFS result under a different protocol." Out-of-fold accuracy is about 66.2% to 66.6%, roughly 4.4 to 4.9 pp below the published 71.04%, with mismatched protocols. The "parity" framing in DEC-018 rests on one 11-subject split and is withdrawn as a general claim. Fold accuracies range from about 62.5% to 72.3%, so no comparison may be made from one small split.

---

## 9. Work plan

### Phase A: Close out EXP-019 (no change to locked numbers)
- **A1.** Settle the verdict label with `git show` (section 5) and log an erratum.
- **A2.** S3 no-op check: relative parameter change of UA-2 vs ZS, logit differences, and whether adapted weights are used at inference. If UA-2 is a no-op, log an amendment and re-run S3 only, flagged "re-run after implementation fault."
- **A3.** Report the outputs the preregistration required but the report omitted: k = 1 to 4 learning curve with n per k; heterogeneity vs ZS and night-index effects; per-stage F1 and confusion matrices; sleep-summary agreement; selected configurations per fold; Holm-adjusted p-values for S1 to S3; the Stage D spread estimate.
- **A4.** Edit S1 wording: the benefit over FT-other-2 partly reflects harm from donor-night fine-tuning.
- **A5.** Complete the NAS Iteration 1 erratum (single fold, 1 epoch, not H-A, not pre-registered, not reproduced in the independent re-run; NAS Iteration 2 not authorized).

### Phase B: Evidence base
- **B1.** Apply the KB round-2 corrections (wrong Heremans PubMed ID 35877770 to 35508121 in six places; P011 and P009 STRONG grades; P101 stale text and missing author Colin Quirk; `literature_map.md` "Underexplored" labels for gating; verify the remaining resolved authors; Lorenzen's JBHI DOI and PMID).
- **B2.** Systematic search: PubMed, IEEE Xplore, Scopus or Web of Science, arXiv, medRxiv; 2018 to date; forward citations of Song 2026, Phan 2020, Mikkelsen and De Vos 2018 and van der Aar 2025. Query blocks: personalization/adaptation with sleep staging and PPG/heart rate/accelerometry/smartwatch; abstention/gating/negative transfer with sleep staging; "instantaneous heart rate" with accelerometry and sleep; multi-night/longitudinal with sleep staging and wearable. Log query string, database, date, hits, duplicates, exclusion reasons.
- **B3.** Read the full text of the core papers before citing: van der Aar 2025, Phan 2020, Lorenzen 2024, Heremans 2022, Mikkelsen and De Vos 2018, Song 2026, Walch 2019.
- **Exit:** a "closest ten" table with differences stated and a dated novelty statement.

### Phase C: Data
- **C1.** Search for a second multi-night labeled wearable dataset (at least 3 labeled nights per person; wrist PPG or IHR plus accelerometry; EEG or PSG-derived labels; obtainable). Check the data-availability statements of van der Aar 2025 and of papers found in B2. Availability of any candidate: `TO VERIFY`. SleepAccel and MESA have one night per person, so they only support EXP-027.
- **Why:** every BIDSleep subject was used in EXP-019 Stage C, so only new data can give a clean confirmatory adaptation test.

### Phase D: Experiments
- **D1. EXP-020 (exploratory, not locked until a preregistration block is committed).** Arms: ZS; FT-2 (reuse locked EXP-019 predictions); FT-2 plus KL (weight from {0.1, 1, 10}, selected on each fold's 5 inner-validation subjects only); GATE (adapt only if every class has at least tau of calibration epochs, otherwise keep ZS). tau is not tuned (too few inner-validation subjects); report a sensitivity curve over tau in {1%, 2%, 3%, 5%}. Endpoints (descriptive): mean paired difference vs ZS with CI; minimum subject difference; 10th-percentile subject difference; number of subjects at -5 pp or worse; mean among adapted subjects. No SUPPORTED or NOT SUPPORTED labels. Because it was designed after seeing Stage C, it is "design-informed, exploratory."
- **D2. EXP-019B** (heterogeneity analysis plan): gates G0 to G3 first, blind predictor extraction, ten primary tests with Benjamini-Hochberg control, exploratory label.
- **D3. EXP-027:** zero-shot transfer to SleepAccel (31 subjects, one PSG night each, PSG labels). Requires label mapping (REM coded 5, no N4), HR and acceleration alignment, and caution on lab and label-source differences. Descriptive plus paired CIs.
- **D4. EXP-028:** sleep-summary agreement with a priori Bland-Altman margins.
- **D5. If a second dataset is found (C1):** pre-register a replication of the EXP-019 adaptation analysis there with the same decision rule structure. This is the clean confirmatory test.

### Phase E: Analysis and writing
Write results exactly against the pre-specified rules; label exploratory analyses; report negative results.

---

## 10. Decision gates and paper outcomes

| Gate | Condition | Consequence |
|---|---|---|
| After A | S3 was a no-op | Re-run S3 (documented). Otherwise keep as is. |
| After B2 | Close prior art beyond van der Aar 2025 found | Re-scope C1 further. |
| After C1 | Second dataset found and analyzable | Replicate adaptation analysis with a locked preregistration (D5). Strongest Q1 lever. |
| After C1 | No second dataset | Single-dataset paper framed as replication, extension and failure analysis; state the limitation openly. |

**Outcome types (carried from v3):**
- **A:** a confirmatory adaptation result (second dataset) plus a protocol-matched comparison favoring the final model over re-implemented SLAMSS-IFS: method and benchmark paper.
- **B:** a supported adaptation result without a baseline win or comparison: personalization/benchmark paper, no superiority claim.
- **C:** H-A not met and nothing replicated: negative-result and benchmark paper (leakage-free multi-night evaluation, quantified limits of two-night adaptation). This is the current position on BIDSleep alone.

**Venue realism (verify quartiles for the submission year on JCR or Scimago):** with a second dataset and Phases A, B and D complete, a mid-tier Q1 venue in biomedical engineering or sleep is plausible. On BIDSleep alone, mid-tier venues are more realistic and Q1 is uncertain. Nothing is guaranteed. Candidate venues: IEEE JBHI, Computers in Biology and Medicine, Biomedical Signal Processing and Control (engineering framing); *Sleep*, *Journal of Sleep Research* or *Sleep Medicine* (clinical framing, needs the clinical metrics). IEEE TBME is the venue of the dataset paper and a higher bar.

---

## 11. Q1 readiness checklist

- [ ] Verdict label settled and report corrected (A1)
- [ ] S3 no-op check closed (A2)
- [ ] All pre-registered EXP-019 outputs reported, including the k-curve (A3)
- [ ] KB round-2 corrections applied (B1)
- [ ] Systematic search logged; closest-ten table written; novelty statement dated (B2)
- [ ] Core papers read in full (B3)
- [ ] Second-dataset search documented (C1)
- [ ] SLAMSS-IFS re-run on identical splits, or "different protocol" wording used
- [ ] Split audit for EXP-001 to EXP-004; test-access log complete
- [ ] External validation reported with CIs (EXP-027)
- [ ] Sleep-summary agreement reported (EXP-028)
- [ ] Limitations stated: healthy adults, single site, headband labels with single scorer, no disorder generalization, offline vs real-time
- [ ] Code, fold manifests and configs released (dataset license: Open Data Commons Attribution)
- [ ] Reporting checklist followed if the target journal requires one (for example TRIPOD+AI; `TO VERIFY`)

---

## 12. Risks

- Prior art may be larger than found; B2 is not done.
- One dataset: 47 healthy adults, one site, single-scorer headband labels.
- No fresh subjects: any post-EXP-019 method result is contaminated by knowledge of the failure cases, even with nested hyperparameters.
- Small numbers: 46 eligible subjects, a heavy left tail of three subjects, and only 5 inner-validation subjects per fold.
- KB quality: until round-2 corrections are applied, do not cite KB statements in the paper.

---

## 13. Discipline (what is not allowed)

Random architecture search; adding fashionable architectures; tuning on test or outer-fold data; changing hypotheses, thresholds or decision rules after seeing results; selecting experiments because they raise accuracy; describing any experiment as "pre-registered" without a committed preregistration block; modifying the locked EXP-019 result.

---

## 14. Pre-registration protocol (every confirmatory experiment)

Before running, commit a block in `EXPERIMENTS.md` with: hypothesis ID and prediction; the single factor changed; primary metric and decision rule (threshold and CI condition); splits (manifest hash), seeds, parameter budget; and what counts as "not supported." After running, record the outcome exactly against the rule. Do not edit the block.

---

## 15. Paper skeleton (working)

Working title: *Does a few labeled nights of personal data help wearable sleep staging? A pre-registered, leakage-free study with consumer-smartwatch heart rate and accelerometry.*

1. Introduction: problem, gaps G1 to G3, contributions C1 to C3
2. Related work: personalization of sleep staging (EEG and PPG), protocol heterogeneity, adaptation failure
3. Data and task, with explicit dataset attribution
4. Methods: backbone, adaptation arms, controls, baselines
5. Evaluation protocol and pre-registration statement
6. Results: baselines, context, H-A (as pre-specified), S1 to S3, k-curve, heterogeneity (exploratory), EXP-020 (exploratory), external validation, clinical agreement
7. Discussion: why gains are small in healthy adults (a hypothesis, from the prior paper's rationale that personalization matters when altered autonomic modulation is expected), failure modes, limitations
8. Conclusion

---

## 16. One-sentence target for every research and coding agent

> Quantify, under strict subject-level, chronological, leakage-free and pre-registered evaluation, how much two or more labeled earlier nights help consumer-smartwatch IHR and accelerometry sleep staging in healthy adults, who is harmed and why, without architecture search, without changing any locked result, and with exploratory analyses labeled as exploratory and negative results reported as faithfully as positive ones.

---

## 17. Amendment log

| Date | Version | Change | Reason |
|---|---|---|---|
| 2026-11-17 | v4 | Consolidated v3, v3.1 and the v4 proposal. Withdrew "open lane." Reframed C1 as replication and extension. Marked EXP-012 test set spent. Moved to full-night context and 4-class definition. Added Phases A to E and decision gates. | Prior-art discovery (van der Aar 2025); EXP-019 result; KB review |
| | | | |
