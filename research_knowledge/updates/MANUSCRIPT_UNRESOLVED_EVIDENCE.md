# Manuscript Unresolved Evidence & Open Verification Register

**Document Status:** Complete Forensic Companion Deliverable to Manuscript Draft v0.1  
**Governance Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029), Sections 6, 7, 8, 10, 11; DEC-021  
**Author:** Antigravity / EGM Forensic Auditor  
**Date:** 2026-11-18  

---

## 1. Executive Summary

This register explicitly catalogues all empirical assertions, protocol dependencies, and external validations in `MANUSCRIPT_DRAFT_v0.1.md` that remain classified as `TO VERIFY`, `TO FILL`, `BOUNDED DRAFT`, or `DEFERRED`.

In strict adherence to research governance rules:
- No unconfirmed fact may be stated in a publication draft without an explicit marker or qualification.
- No hypothesis may be described as confirmatory without a verifiable pre-registration commit hash.
- All post-hoc designs must carry formal prior-access disclosures.

---

## 2. Inventory of Unresolved Evidence & Action Items

| ID | Topic / Concept | Exact Unresolved Question / Marker | Current Empirical Status | Required Verification / Resolution Action | Risk / Publication Impact |
|:---:|:---|:---|:---|:---|:---|
| **UNRES-01** | **Second Dataset Acquisition (C1)** | Can an independent multi-night wearable dataset ($\ge 3$ nights/subject, wrist PPG/IHR, EEG labels) be acquired? | **DEFERRED (In Progress)**. Candidates identified (van der Aar 2025 data-availability request; SleepAccel/MESA are 1-night only). | Formal request to research lab / institutional data access under DEC-029. Proceed on BIDSleep alone in interim. | If no second dataset is acquired, manuscript is framed as single-dataset replication, extension, and failure analysis (Outcome Type C). |
| **UNRES-02** | **External Zero-Shot Validation (EXP-027)** | Zero-shot generalization of Model A on SleepAccel ($N=31$, 1 PSG night each). | **DEFERRED (Pending Protocol Approval)**. | Pre-register EXP-027 with exact label mapping (REM=5, N4 absent), HR alignment, and lab differences. | Demonstrates whether the baseline model generalizes beyond the BIDSleep acquisition protocol. |
| **UNRES-03** | **SLAMSS-IFS Protocol-Matched Re-Run** | Did the published 71.04% use identical 4-class Wake/Light/Deep/REM granularity, and what does it achieve under 5-fold CV? | **PARTIALLY VERIFIED**. Paper text specifies 666k params; official repo implements 5.47M params. 71.04% was evaluated on single 11-subject test partition (Cohort Beta). | Execute independent 5-fold CV of official repo model (`experiments/EXP-005/slamss_ifs_baseline.py`) under identical manifests before claiming superiority/inferiority. | Manuscript wording strictly adheres to: *"achieved lower/higher accuracy under a different evaluation protocol."* |
| **UNRES-04** | **Label Source Discrepancy** | Exact numeric performance delta between `expert_label` (technologist-reviewed) and `dreem_label` (automated algorithm). | **TO VERIFY**. `expert_label` is used as primary reference across all experiments. | Run sensitivity analysis evaluating Model A against raw `dreem_label` to quantify technologist scoring impact. | Quantifies label noise ceiling and manual review contribution. |
| **UNRES-05** | **Temporal Context Causality** | Is the BiGRU temporal sequence model causal (past-only) or non-causal (bidirectional, past-and-future)? | **CONFIRMED NON-CAUSAL (Offline Only)**. Model A uses a Bidirectional GRU (`bidirectional=True`). | Explicitly state in Methods (Section 4.2) and Limitations that the model operates in an offline, post-hoc staging capacity, not real-time sequential processing. | Eliminates any over-claim regarding real-time closed-loop sleep intervention. |
| **UNRES-06** | **EXP-018 Phase 2 Pre-Registration Status** | Was the decision rule for context length ($L=600$ vs $L=60$) committed before Phase 2 runs? | **TO VERIFY via `git log`**. If committed before runs, counts as confirmatory; otherwise exploratory. | Inspect git commit timestamps for EXP-018 design vs execution runs. | Determines whether context length effect is reported as confirmatory or exploratory. |
| **UNRES-07** | **Clinical Sleep-Summary Agreement (EXP-028)** | Agreement between Model A and Dreem expert labels for clinical parameters (TST, WASO, SE, REM%, Deep%). | **DATA EXTRACTED (exp019_sleep_summary_agreement.csv)**; formal Bland-Altman statistical analysis pending. | Author script computing Bland-Altman bias, 95% limits of agreement, and MAE across all 246 nights. | Essential for clinical journal submission (Gap G3 / Contribution C3). |
| **UNRES-08** | **Reporting Checklist Compliance** | Does the target journal mandate a specific AI reporting checklist (e.g., TRIPOD+AI, CONSORT-AI, or STARD-AI)? | **TO VERIFY per journal**. | Inspect author guidelines for candidate journals (*IEEE JBHI*, *Computers in Biology and Medicine*, *Sleep*) upon journal selection. | Ensures immediate desk-review compliance upon formal submission. |
| **UNRES-09** | **Unsupervised Adaptation (UA-2) Fault Check** | Why did UA-2 yield identical outputs to ZS to the last decimal place in all 46 subjects? | **CONFIRMED IMPLEMENTATION NO-OP**. Zero parameter delta was applied during evaluation. | Do not cite UA-2 as an algorithmic failure of entropy minimization; report as an implementation fault/no-op. | Preserves scientific integrity by distinguishing software bugs from genuine negative results. |
| **UNRES-10** | **Core Literature Paywall Boundaries** | Verification of full-text access for P006 (Phan 2020), P007 (Lorenzen 2024), and P054 (Heremans 2022). | **BOUNDED AT ABSTRACT-ONLY**. Four core papers (`P001`, `P002`, `P056`, `P106`) are verified via full text; three papers are bounded by publisher paywalls (IEEE / IOP). | Bounded until institutional subscription credentials are provided; citation claims in manuscript are restricted strictly to verified abstract/author-summary facts. | Ensures citations do not exceed actual text access depth. |

---

## 3. Action Plan for Final Manuscript Compilation

1. **Pre-Requisite for Results Section (Sections 6–8):**
   - Incorporate Candidate 2 SLAMSS-IFS comparison matrix into baseline benchmark discussion.
   - Present locked EXP-019 results verbatim (+1.0054 pp Macro-F1, 95% CI [-0.17, +1.99] pp, Wilcoxon p=0.00131) under the pre-registered label: **INCONCLUSIVE**.
   - Present Candidate 4 BIDSleep cohort characteristics and calibration stage coverage findings to support the mechanistic explanation of negative transfer in `Bidslab34`.
2. **Pre-Requisite for Launching Further Experiments:**
   - Any execution of EXP-019B (heterogeneity predictors) or EXP-020 (KL regularization and quality gating) requires explicit user approval under DEC-021.
   - When approved, EXP-019B and EXP-020 results must be reported under their committed SHA-256 draft protocols with explicit disclosure of prior data access.
