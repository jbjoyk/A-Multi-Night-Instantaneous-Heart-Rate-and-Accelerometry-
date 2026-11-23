# Manuscript Unresolved Evidence & Open Verification Register

**Document Status:** Stage E Synchronized Companion Register to `MANUSCRIPT_DRAFT_v0.2.md`  
**Governance Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (`DEC-029`), `DEC-035`–`DEC-041`  
**Author:** Antigravity / EGM Forensic Auditor  
**Date:** 2026-11-23  

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
| **UNRES-01** | **Second Multi-Night Dataset (C1)** | Can a second public multi-night healthy-adult smartwatch+EEG dataset ($\ge 3$ nights/subject) be acquired? | **CLOSED FOR v4 / OUT-OF-SCOPE FUTURE TARGET**. Systematic audit across PhysioNet, NSRR, and OpenNeuro confirmed no second public multi-night healthy-adult smartwatch+EEG dataset currently exists. Single-night external PSG validation completed (`EXP-027`/`EXP-028`). | Prospective multi-site clinical-disorder (`DREAMT v2.2.0`) and pediatric/neurodevelopmental multi-night (`Simons Sleep Project DS0000089`) cohorts cited as future external validation targets. | Framed transparently in *Research in context* and Limitations. |
| **UNRES-02** | **External Zero-Shot PSG Validation (EXP-027)** | Zero-shot generalisation of Model A vs. Official SLAMSS-IFS on PhysioNet SleepAccel ($N=31$, single-night PSG). | **RESOLVED (`DEC-041`)**. Executed on 15+15 frozen checkpoints (`qc_clean` $n=25$ primary, `all_n31` $N=31$ sensitivity). `mean_single_model` PRIMARY (`60.51%` vs. `61.08%`, $-0.57\text{ pp}$, $p=0.5782$); `prob_ensemble` SECONDARY (`62.94%` vs. `62.79%`, $+0.15\text{ pp}$, $p=0.9789$). | Closed; integrated into `MANUSCRIPT_DRAFT_v0.2.md` Table 3. | Framed strictly as no statistically significant difference / empirical parity (no TOST equivalence claim). |
| **UNRES-03** | **SLAMSS-IFS Protocol-Matched Re-Run (EXP-021)** | Matched 5-fold CV comparison of Official SLAMSS-IFS (5,465,768 repo params) vs. frozen Model A (163,940 params). | **RESOLVED (`DEC-038`)**. Executed across $N=47$ subjects (3 seeds): Model A `64.79%` vs. SLAMSS-IFS `67.37%` Macro-F1 ($\bar{d} = -2.58\text{ pp}$, $p = 1.98 \times 10^{-6}$, `33.34x` compression). | Closed; integrated into `MANUSCRIPT_DRAFT_v0.2.md` Table 2. | Exact accuracy–efficiency trade-off quantified. |
| **UNRES-04** | **Label Source Discrepancy** | Exact numeric performance delta between `expert_label` (technologist-reviewed) and `dreem_label` (automated algorithm). | **OUT-OF-SCOPE FOR v4**. `expert_label` is used consistently as primary reference across all BIDSleep experiments. | Optional future sensitivity analysis; Dreem 2 surrogate label ceiling disclosed in Limitations. | Label noise ceiling transparently bounded. |
| **UNRES-05** | **Temporal Context Causality** | Is the BiGRU temporal sequence model causal or non-causal (offline)? | **RESOLVED (`DEC-039` Addendum `W0-B`)**. `ConventionalCNNBiGRU` is bidirectional (`L=1200`), requiring smartwatch SoC / companion smartphone offline full-night buffering (not 512 KB MCU 1-epoch streaming). | Closed; explicitly stated in Methods §2.2, Table 2 footnote, and Limitations §4.3. | Eliminates real-time closed-loop or bare-metal streaming overclaims. |
| **UNRES-06** | **EXP-018 Phase 2 Pre-Registration Status** | Was the decision rule for context length ($L=600$ vs $L=60$) committed before Phase 2 runs? | **CLOSED AS EXPLORATORY**. Context-length ablation is treated as exploratory design evidence supporting frozen `L=1200` Model A (`DEC-037`). | No confirmatory claim made for `EXP-018` context ablation in `v0.2`. | Zero confirmatory-claim risk. |
| **UNRES-07** | **External & Internal Sleep-Summary Agreement (EXP-028)** | Bland–Altman (1999) agreement with `SE(LoA)`, raw paired tuples, and Shapiro–Wilk non-parametric LoA fallbacks. | **RESOLVED (`DEC-041`, `FIND-B-02` Remediated)**. Persisted 744 raw paired tuples (`exp028_sleepaccel_raw_paired_tuples.csv`) and full `EXP-028` Bland–Altman summary (`qc_clean` $n=25$, `mean_single_model` PRIMARY + `prob_ensemble` SECONDARY). | Closed; integrated into `MANUSCRIPT_DRAFT_v0.2.md` Table 4 (plus Table 1 for BIDSleep `N=46`). | Exposes Dreem 2-to-PSG label-prior shift (`Deep% +10.65%`, `REM% +10.13%`). |
| **UNRES-08** | **Reporting Checklist Compliance** | Target journal AI reporting checklist and house structure. | **RESOLVED (`v0.2`)**. Formatted to *The Lancet Digital Health* house structure (<=300-word Structured Summary + mandatory *Research in context* panel) with full TRIPOD+AI compliance subsection (§2.5). | Closed; ready for submission-package checklist attachment. | Desk-review compliant for *The Lancet Digital Health* / *npj Digital Medicine*. |
| **UNRES-09** | **Unsupervised Adaptation (UA-2 / TSK-07)** | Resolution of historical `0.00 pp` UA-2 implementation no-op. | **RESOLVED (`DEC-036`)**. Re-executed after `DEC-030` unit-mismatch fix across 46 subjects $\times$ 3 seeds (138 runs, 0% collapse): `+0.21 pp` [95% CI `+0.04, +0.39`], $p_{\text{sup}}=0.0322$ (`SUB-BENCHMARK GAIN`). | Closed; integrated into `MANUSCRIPT_DRAFT_v0.2.md` §3.2. | Historical `DEC-030` defect and fix transparently disclosed. |
| **UNRES-10** | **Core Literature Paywall Boundaries** | Full-text access for paywalled secondary references (Phan 2020, Lorenzen 2024, Heremans 2022). | **BOUNDED (OUT-OF-SCOPE FOR v0.2)**. Core primary references (Song 2026, Walch 2019, van der Aar 2025, Bland & Altman 1999) are verified; `v0.2` restricts citations to verified sources. | No paywalled abstract-only claims relied upon for quantitative comparisons in `v0.2`. | Zero citation overreach. |

---

## 3. Stage E Closure Summary
All Stage 3 (`EXP-021`, `W0-B`), Stage 1 errata (`TSK-06` Clean-ZS, `TSK-07`), and Stage D (`EXP-019B`, `EXP-020`, `EXP-027`, `EXP-028`) items are **CLOSED** and synchronized across `MANUSCRIPT_DRAFT_v0.2.md` and `MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv`. Remaining future targets (`DREAMT v2.2.0`, `Simons Sleep Project DS0000089`) are explicitly scoped as prospective post-v4 external validations.
