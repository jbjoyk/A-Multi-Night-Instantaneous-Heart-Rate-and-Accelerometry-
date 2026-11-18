# Comprehensive Resolution Report: BIDSleep Candidates 2, 4, and 5

**Document Status:** Complete Resolution Report  
**Role:** `egm-resolve`  
**Target Audit Report:** `BIDSLEEP_CANDIDATES_1_TO_5_INDEPENDENT_AUDIT_REPORT.md`  
**Governance Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029) and DEC-021  
**Resolution Date:** 2026-11-18  

---

## 1. Executive Summary of Resolutions

This document formally details the empirical, code, and documentation resolutions applied in response to the independent adversarial audit of Candidates 2, 4, and 5. Every deficiency identified by the independent reviewer has been remediated with verified evidence, corrected code, or reconciled documentation.

### Summary Resolution Scorecard

| Candidate | Audit Finding | Resolution Category | Primary Artifact Modified | Verification Status |
|---|---|:---:|---|:---:|
| **Candidate 2** | Informal "70.5%" accuracy cited for EXP-012 Cohort Beta test baseline. | **Documentation Correction** | `research_knowledge/updates/SLAMSS_IFS_BASELINE_PROTOCOL_AUDIT.md` | **RESOLVED** (Replaced with 3-seed mean: $68.44\% \pm 1.35\%$). |
| **Candidate 2** | Clarify 33.3× vs 4.06× parameter compression targets. | **Documentation Correction** | `research_knowledge/updates/SLAMSS_IFS_BASELINE_PROTOCOL_AUDIT.md` | **RESOLVED** (33.3× repo vs 4.06× paper text explicitly separated). |
| **Candidate 4** | Per-$k$ cohort denominator bug in diagnostic script (`eligible_subs = 47` for all $k$). | **Code & Data Correction** | `research_engine/bidsleep_cohort_diagnostics.py`<br>`bidsleep_calibration_class_coverage_summary.csv` | **RESOLVED** (Fixed to evaluable denominators: $k=1: 47, k=2: 46, k=3: 42, k=4: 33$). |
| **Candidate 4** | Excluded-subject count stated as 5 subjects instead of 6. | **Documentation & Data Correction** | `research_knowledge/updates/BIDSLEEP_COHORT_DATA_INTEGRITY_AUDIT.md` | **RESOLVED** (7 excluded nights across 6 distinct subjects). |
| **Candidate 4** | `Bidslab34` calibration Deep sleep stated as "6 epochs". | **Documentation Correction** | `research_knowledge/updates/BIDSLEEP_COHORT_DATA_INTEGRITY_AUDIT.md` | **RESOLVED** (Corrected to 15 epochs [0.68%] across Nights 2 and 3). |
| **Candidate 4** | Overstated causal language linking class scarcity to negative transfer. | **Documentation Correction** | `research_knowledge/updates/BIDSLEEP_COHORT_DATA_INTEGRITY_AUDIT.md` | **RESOLVED** (De-escalated to descriptive observation and candidate mechanism). |
| **Candidate 5** | Manuscript Section 3.2 cited 5 excluded subjects. | **Documentation Correction** | `research_knowledge/updates/MANUSCRIPT_DRAFT_v0.1.md` | **RESOLVED** (Synchronized to 6 subjects). |
| **Candidate 5** | Manuscript Section 3.3 lacked evaluable cohort denominators across $k=1..4$. | **Documentation Correction** | `research_knowledge/updates/MANUSCRIPT_DRAFT_v0.1.md` | **RESOLVED** (Synchronized to 47, 46, 42, 33). |
| **Candidate 5** | Manuscript Section 4.2 cited only 33.3× compression without distinguishing targets. | **Documentation Correction** | `research_knowledge/updates/MANUSCRIPT_DRAFT_v0.1.md` | **RESOLVED** (Synchronized 33.3× repo vs 4.06× paper text). |
| **Candidate 5** | Manuscript Section 4.4 and Claim Matrix contained uncalibrated causal phrasing. | **Documentation Correction** | `research_knowledge/updates/MANUSCRIPT_DRAFT_v0.1.md`<br>`MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv` | **RESOLVED** (Framed as exploratory candidate mechanism; noted 15 epochs / 0.68%). |
| **Candidate 5** | Core literature paywall boundaries not explicitly tracked. | **Documentation Correction** | `research_knowledge/updates/MANUSCRIPT_UNRESOLVED_EVIDENCE.md` | **RESOLVED** (Added UNRES-10 documenting paywalled papers P006, P007, P054). |

---

## 2. Itemized Before-and-After Values and Evidence

### 2.1 Candidate 2: SLAMSS-IFS Baseline Audit
*File Modified:* `research_knowledge/updates/SLAMSS_IFS_BASELINE_PROTOCOL_AUDIT.md`

1. **EXP-012 Cohort Beta Test Accuracy:**
   - **Before:** Dimension 10 and Section 3 line 49 claimed: *"when our lightweight model (SleepAdaptiveNet, 162k parameters) was evaluated on this exact same 11-subject test partition, it achieved 70.5% accuracy"*.
   - **After:** *"EXP-012 Cohort Beta 3-Seed Mean: Accuracy: $68.44\% \pm 1.35\%$ (Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%; Macro-F1: $0.6859 \pm 0.0139$; Weighted F1: $0.6835 \pm 0.0141$; DEC-018; `exp012_subject_metrics.csv`)."*
   - **Evidence:** Ratified decision `DEC-018` lines 174–182; raw metrics table `experiments/EXP-012/exp012_subject_metrics.csv`; multi-seed test results `exp012_test_results.json`.
2. **Parameter-Count Compression Ratios:**
   - **Before:** Section 3 line 58 stated: *"Model A contains exactly 163,940 parameters (a 33.3x compression relative to the official SLAMSS-IFS codebase)"* without contrasting the published paper text.
   - **After:** Section 3 subsection 3 explicitly distinguishes both targets:
     - *Official Author Repository Codebase (`BIDSLabUMass/SLAMSS-IFS`):* Contains **5,465,768 parameters**, yielding a **33.3-fold parameter compression** ($5,465,768 / 163,940 = 33.34\times$).
     - *Published Paper Text Specification:* Describes an intra-epoch CNN + LSTM model totaling **666,060 parameters**, yielding a **4.06-fold parameter compression** ($666,060 / 163,940 = 4.06\times$).
   - **Evidence:** Source code execution of `experiments/EXP-005/test_cross_implementation.py` confirming 5,465,768 parameters for the official repository, 666,060 parameters for `slamss_ifs_baseline.py`, and 163,940 parameters for `exp013_models.py ConventionalCNNBiGRU`.

---

### 2.2 Candidate 4: BIDSleep Cohort Diagnostics
*Files Modified:* `research_engine/bidsleep_cohort_diagnostics.py`, `research_knowledge/updates/bidsleep_calibration_class_coverage_summary.csv`, `research_knowledge/updates/BIDSLEEP_COHORT_DATA_INTEGRITY_AUDIT.md`

1. **Diagnostic Script Denominator Bug Fix:**
   - **Before:** In `research_engine/bidsleep_cohort_diagnostics.py` line 66, filtering was performed using `usable_inv['usable_night_idx'] <= k`, which erroneously grouped all 47 subjects at $k=3$ and $k=4$ because every subject has at least one night with index $\le k$.
   - **After:** Code updated to filter on `n_usable >= k + 1`:
     ```python
     evaluable_subs = sub_night_counts[sub_night_counts >= k + 1].index
     sub_k = usable_inv[(usable_inv['subject_id'].isin(evaluable_subs)) & (usable_inv['usable_night_idx'] <= k)]
     ```
   - **Verification Results:** The script was executed to regenerate `bidsleep_calibration_class_coverage_summary.csv`. The resulting evaluable cohort sizes are:
     - $k=1$: **47 subjects** (all 47 possess $\ge 2$ usable nights, 198 evaluation nights)
     - $k=2$: **46 subjects** (46 possess $\ge 3$ usable nights, 152 evaluation nights; `Bidslab49` excluded with only 2 usable nights)
     - $k=3$: **42 subjects** (42 possess $\ge 4$ usable nights, 106 evaluation nights)
     - $k=4$: **33 subjects** (33 possess $\ge 5$ usable nights, 64 evaluation nights)
   - **Evidence:** Exactly matches `experiments/EXP-019/exp019_k_learning_curve.csv` and `review_package/night_inventory.csv`.
2. **Excluded Subjects Arithmetic Bug:**
   - **Before:** Section 1 line 23 stated: *"Exact Inventory of Excluded Nights (7 Total across 5 Subjects)"*.
   - **After:** *"Exact Inventory of Excluded Nights (7 Total across 6 Subjects): `Bidslab08` (Night 7), `Bidslab16` (Night 2), `Bidslab34` (Night 1), `Bidslab42` (Nights 1 and 3), `Bidslab49` (Night 1), `Bidslab60` (Night 1)."*
   - **Evidence:** `review_package/night_inventory.csv` rows where `usable == False` confirm exactly 6 distinct subject IDs.
3. **`Bidslab34` Calibration Deep Sleep Count:**
   - **Before:** Section 3 line 68 claimed: *"only 0.7% across its 2 calibration nights ($0.0\%$ on Night 2, $1.3\%$ on Night 3) ... (6 epochs)"*.
   - **After:** *"only 0.68% across its 2 calibration nights (exactly 15 Deep-sleep epochs out of 2,194 total labeled epochs: 0 epochs on Night 2, 15 epochs on Night 3)."*
   - **Evidence:** `review_package/night_inventory.csv` rows for `Bidslab34`: Night 2 (1078 labeled, 0 Deep), Night 3 (1116 labeled, 15 Deep), total labeled $= 2194$, total Deep $= 15$, $15 / 2194 = 0.0068368 \approx 0.68\%$.
4. **De-escalation of Causal Claims:**
   - **Before:** Section 3 framed class absence as the established cause of personalization failure and claimed $k \ge 3$ stabilization proved personalization success.
   - **After:** Re-framed as: *"Descriptive Coverage Stabilization at $k \ge 3$... While improved stage coverage at higher $k$ provides a descriptive physiological rationale consistent with the observation of van der Aar et al. (2025) that personalized staging achieves statistically reliable gains at $k \ge 3$ nights, this observation represents a candidate mechanism rather than an established causal law. Pre-adaptation stage scarcity across the entire cohort does not exhibit a statistically significant monotonic correlation with personalization delta ($q = 0.9711$)."*

---

### 2.3 Candidate 5: Manuscript Draft v0.1
*Files Modified:* `research_knowledge/updates/MANUSCRIPT_DRAFT_v0.1.md`, `research_knowledge/updates/MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv`, `research_knowledge/updates/MANUSCRIPT_UNRESOLVED_EVIDENCE.md`

1. **Contribution C2 Framing (Section 1.4):**
   - **Before:** *"we isolate calibration-night class coverage (specifically the total absence or scarcity of Deep slow-wave sleep) as a primary etiology of severe negative transfer."*
   - **After:** *"we explore calibration-night class coverage (such as severe scarcity of Deep slow-wave sleep observed in outlier cases) as a candidate failure mechanism associated with negative transfer."*
2. **Quality Screening Subject Count (Section 3.2):**
   - **Before:** *"Exactly 7 nights across 5 participants failed this criterion..."*
   - **After:** *"Exactly 7 nights across 6 participants failed this criterion..."*
3. **Evaluable Denominators Across $k=1..4$ (Section 3.3):**
   - **Before:** Section 3.3 mentioned only 46 subjects for EXP-019 without detailing the $k$-specific evaluable cohort progression.
   - **After:** Added complete $k=1..4$ evaluable denominators: $k=1: N=47$ (198 eval nights); $k=2: N=46$ (152 eval nights); $k=3: N=42$ (106 eval nights); $k=4: N=33$ (64 eval nights).
4. **Parameter Compression Ratios (Section 4.2):**
   - **Before:** *"Model A is an audited, lightweight deep neural sequence architecture comprising exactly 163,940 trainable parameters (a 33.3-fold parameter compression relative to the 5.47-million parameter SLAMSS-IFS official codebase)."*
   - **After:** *"Model A is an audited, lightweight deep neural sequence architecture comprising exactly 163,940 trainable parameters. This design achieves a 33.3-fold parameter compression ($5,465,768 / 163,940 = 33.34\times$) relative to the 5.47-million parameter official SLAMSS-IFS author repository codebase, and a 4.06-fold parameter compression ($666,060 / 163,940 = 4.06\times$) relative to the architecture specification in the published paper text."*
5. **Quality-Gated Adaptation & Outlier Framing (Section 4.4):**
   - **Before:** Presented gating as a standard safeguard without empirical context.
   - **After:** Formally contextualized around `Bidslab34`'s 15 Deep-sleep epochs (0.68%) across calibration nights, explicitly noting that because cohort-wide pre-adaptation stage scarcity does not show a statistically significant monotonic correlation with personalization delta ($q = 0.9711$), gating is evaluated strictly as an exploratory, design-informed heuristic.
6. **Spent Test Partition Disclosure & EXP-012 Baseline (Section 5.1):**
   - **Before:** Stated Cohort Beta was spent without citing its exact empirical baseline.
   - **After:** Cited the authoritative 3-seed mean: $68.44\% \pm 1.35\%$ (Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%; Macro-F1: $0.6859 \pm 0.0139$; DEC-018).
7. **Claim Matrix Synchronization:**
   - Synchronized claims CLM-008, CLM-009, CLM-017, and CLM-021 in `MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv`.
8. **Unresolved Evidence Register:**
   - Added `UNRES-10` in `MANUSCRIPT_UNRESOLVED_EVIDENCE.md` formally recording that core papers `P006`, `P007`, and `P054` are bounded at abstract/author-summary depth due to publisher paywalls (IEEE and IOP Publishing).

---

## 3. Governance Invariants & State Confirmation

| Governance Invariant | Prescribed Standard | Actual Workspace State | Status |
|---|---|---|:---:|
| **Candidate 1 Status** | Genuinely complete; do not reopen split audit. | `EXP001_EXP004_SPLIT_LEAKAGE_AUDIT.md` and `EXPERIMENTS.md` remain untouched and preserved as COMPLETE. | **CONFIRMED** |
| **Candidate 3 Status** | Preserve draft protocols unchanged; hashes bitwise verified. | `EXP019B_PREREGISTRATION_DRAFT.md` (hash: `30ca80f1...`) and `EXP020_PREREGISTRATION_DRAFT.md` (hash: `c226cc41...`) remain completely unchanged and bitwise identical. | **CONFIRMED** |
| **Phase D Authorization Gate** | EXP-019B and EXP-020 remain strictly unauthorized. | Neither protocol was locked, scheduled, or executed. Zero GPU/model training processes launched. | **CONFIRMED** |
| **Research Direction & Decisions** | `RESEARCH_DIRECTION_v4_FULL.md` and DEC-029 unchanged. | `git diff RESEARCH_DIRECTION_v4_FULL.md DECISIONS.md` is empty. | **CONFIRMED** |
| **Locked EXP-019 Artifacts** | EXP-019 checkpoints, results, manifests immutable. | `git diff experiments/EXP-019/` is empty. | **CONFIRMED** |
| **External Validation** | External validation remains deferred. | No external runs launched; SleepAccel/MESA remain deferred. | **CONFIRMED** |
| **Git Working Tree** | No Git commits created; no push to origin. | `git log -n 1` remains at HEAD `29c1e63`. Working tree has uncommitted updates awaiting instruction. | **CONFIRMED** |

---

## 4. Next Workflow Action

Per the user request and project governance rules:
1. **Next Mandatory Step:** An independent `/egm-review` is required to verify all Candidate 2, 4, and 5 resolutions before any Git commit is considered.
2. **Phase D Experimental Gate:** EXP-019B and EXP-020 remain strictly unauthorized until the user explicitly issues authorization under DEC-021.
