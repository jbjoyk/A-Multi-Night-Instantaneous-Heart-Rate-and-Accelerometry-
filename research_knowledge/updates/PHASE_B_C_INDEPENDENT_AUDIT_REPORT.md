# Independent Conformance and Evidence Audit: Phase B and Phase C Execution

**Document Audited:** `research_knowledge/updates/2026-11-18_phase_b_and_c_execution.md`  
**Authoritative Reference:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029)  
**Audit Protocol:** EGM Goldfish Review Protocol (`/egm-review`)  
**Audit Date:** 2026-11-18  
**Audit Type:** Strict Independent Conformance and Evidentiary Review  

---

## 1. Executive Summary & Verdict Overview

This audit evaluates the latest execution report against the approved, immutable requirements of `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029). The workspace has reopened exploratory audits and literature intelligence, but remains under strict governance: no alterations to the research direction, no modification of locked `EXP-019` artifacts, and no unauthorized training runs.

### Individual Task Assessment
| Task | Description | Status | Primary Audit Finding |
| :--- | :--- | :---: | :--- |
| **B1** | KB Round-2 Corrections | **COMPLETE** | All 6 round-2 corrections verified against primary sources (NCBI E-utilities, PMC) and applied cleanly across 9 files. Zero occurrences of wrong PMID `35877770` remain. |
| **B2** | Systematic Literature Search | **PARTIALLY COMPLETE** | PubMed (E-utilities) and Europe PMC executed across 4 query blocks; forward citations tracked. **Missing:** Dedicated structured query execution logs for IEEE Xplore, Scopus/Web of Science, and medRxiv. |
| **B3** | Core Full-Text Examination & Exit | **PARTIALLY COMPLETE** | Conceptual assessments and closest-ten table are sound, but: (1) underlying paper cards (`P106`, `P007`, `P054`, `P056`) retain `abstract-only` and `TO FILL` tags; (2) novelty statement contains an unsupported priority claim (`"The first preregistered..."`); (3) C2 is overstated as an established finding rather than a candidate mechanism. |
| **C1** | Second Obtainable Dataset Search | **PARTIALLY COMPLETE** | Correctly established that van der Aar 2025 is restricted/private and that SleepAccel, MESA, and DREAMT are single-night only. **Defect:** Unsubstantiated absolute claim asserting that no public dataset exists "in the world literature" without repository-level census logs. |

---

## 2. Phase B Detailed Conformance Checks

### 2.1 B1 — Knowledge Base Corrections (PASSED)
- **Heremans PMID (`35877770` $\to$ `35508121`):** Verified via NCBI E-utilities. `PMID:35877770` is Laksono et al. (*PLoS One*, 2022, child stunting in Indonesia). `PMID:35508121` is Heremans ERM et al. (*J Neural Eng*, 2022, DOI: `10.1088/1741-2552/ac6ca8`). Replaced in all 6 occurrences (`literature_map.md`, `methods_catalog.md`, `competing_hypotheses.md`, `research_gaps.md`, `novelty_registry.md`). Grep confirms zero remaining instances.
- **`P011` Grade:** Li et al. 2026 is an adjacent survey (0 new subjects, no primary empirical statistical testing). Correctly downgraded from `STRONG` to `MODERATE` in `P011_*.md` and `evidence_matrix.csv`.
- **`P009` Grade:** Zhao et al. 2023 (*ICML*) is a computer vision benchmark study (CIFAR/ImageNet). Correctly downgraded from `STRONG` to `MODERATE` in `P009_*.md` and `evidence_matrix.csv`.
- **`P101` Author List & Text:** Verified via PMC11099244 that Colin Quirk (`Quirk C`) was omitted from the author string. Added `Quirk C`. Stale text claiming `evidence tier (STRONG)` in line 44 was eliminated.
- **`literature_map.md` Gating Labels:** Clarified that gating and selective prediction have extensive precedents in ML and EEG sleep staging; removed any implication that gating is novel to sleep staging per se.
- **Lorenzen JBHI Verification:** Verified via PubMed that `PMID:38833404` and DOI `10.1109/JBHI.2024.3409165` correspond to Lorenzen KP, Heremans ERM, de Vos M, Mikkelsen KB (*IEEE JBHI* 28(10):5804–5815, 2024).

### 2.2 B2 — Systematic Literature Search (DEFICIENT IN DATABASE COVERAGE)
`RESEARCH_DIRECTION_v4_FULL.md` Section 9 B2 states:
> *"Systematic search: PubMed, IEEE Xplore, Scopus or Web of Science, arXiv, medRxiv; 2018 to date; forward citations of Song 2026, Phan 2020, Mikkelsen and De Vos 2018 and van der Aar 2025. Query blocks: personalization/adaptation with sleep staging and PPG/heart rate/accelerometry/smartwatch; abstention/gating/negative transfer with sleep staging; 'instantaneous heart rate' with accelerometry and sleep; multi-night/longitudinal with sleep staging and wearable. Log query string, database, date, hits, duplicates, exclusion reasons."*

- **Executed:** PubMed (E-utilities API) and Europe PMC across Blocks 1–4, yielding 12, 11, 2, and 55 hits respectively. Forward citations of Song 2026 (0 citations), van der Aar 2025 (0 citations), and Phan 2020 (25 citations) were audited.
- **Omission / Gap:** The search log (`search_log.md`) and execution report do NOT contain distinct, structured query logs for:
  1. **IEEE Xplore:** Critical engineering database where biomedical wearable staging papers (including TBME, JBHI, EMBC) reside.
  2. **Scopus or Web of Science:** Mandated citation index for cross-publisher coverage.
  3. **medRxiv:** Mandated preprint server for recent clinical and wearable sleep studies.
- **Finding:** While Google Search and Europe PMC provided broad discovery, skipping distinct logging for IEEE Xplore, Scopus/WoS, and medRxiv violates the literal text of v4 Section 9 B2.

### 2.3 B3 — Core Full-Text Examination & Exit Deliverables (PARTIALLY COMPLETE)

#### A. Artifact-Level Retrieval Depth Discrepancy
The execution report synthesizes findings from all seven papers. However, inspecting the repository's structured paper cards reveals that four papers have not been updated to reflect full-text extraction:
- `research_knowledge/papers/P106_van_der_aar_2025_autonomic_ppg_personalization.md`: Retains `retrieval_depth: abstract-only`, with 11 unpopulated `TO FILL` markers (hypothesis, dataset name, preprocessing, architecture layers, subject split, temporal split, external validation, statistical testing, failure-mode details, leakage risks, reproducibility).
- `research_knowledge/papers/P007_lorenzen2023_wearable_eeg_personalization.md`: Retains `retrieval_depth: abstract-only` with `UNKNOWN` markers.
- `research_knowledge/papers/P054_from_unsupervised_to_semi_supervised_adv.md`: Retains `retrieval_depth: abstract-only` with `UNKNOWN` markers.
- `research_knowledge/papers/P056_personalizing_deep_learning_models_for_a.md`: Retains `retrieval_depth: abstract-only` with `UNKNOWN` markers.
*v4 Section 1 line 7 dictates: "Markers: TO FILL and TO VERIFY are facts nobody has confirmed. Do not remove a marker until someone has checked the source." Leaving these paper cards as `abstract-only` while claiming B3 full-text completion is an internal contradiction.*

#### B. Closest-Ten Comparison Table
The closest-ten comparison table in Section 4 is accurate and methodologically sound. It covers van der Aar 2025, Song 2026, Lorenzen 2024, Phan 2020, Mikkelsen & De Vos 2018, Heremans 2022, Walch 2019, Sridhar 2020, Davidson 2026, and Kotzen 2024. Modality, reference ground truth, cohort size, multi-night depth, evaluation protocol, and key findings are faithfully represented.

#### C. Novelty Statement Audit (CRITICAL DEFECT DETECTED)
The novelty statement in Section 4 contains two critical governance violations:

1. **Unsupported Priority Claim (`"The first preregistered..."`):**  
   Line 131 states:
   > *"Contribution C1 (Primary): The first **preregistered, leakage-free, chronological replication and extension** of multi-night supervised personalization..."*  
   **Violation:** `RESEARCH_DIRECTION_v4_FULL.md` Section 4 explicitly rules:
   > *"Must not claim: 'First' multi-night personalization, or novelty of supervised fine-tuning."*  
   Attaching `"preregistered"` to `"first"` creates an unverified priority claim. Preregistration of personal health, wearable, and sleep studies exists across OSF, AsPredicted, and clinical registries. Claiming global priority cannot be proven and contradicts v4 Section 4, which defines C1 simply as:
   > *"C1 (primary): a pre-registered, leakage-free replication and extension of multi-night supervised personalization with consumer IHR and accelerometry in healthy adults."* (using the indefinite article `"a"`, not `"the first"`).

2. **Overstatement of Candidate Mechanism in C2:**  
   Line 135 states:
   > *"Contribution C2 (Secondary, Exploratory): Empirical identification of calibration-night class coverage (severe label shift) as a primary failure mode..."*  
   **Violation:** Under v4 Section 4, C2 is strictly defined as:
   > *"C2 (secondary, exploratory): calibration-night class coverage as a candidate failure mode, with a nested evaluation of regularized or gated adaptation."*  
   Elevating an exploratory post-hoc audit of a single extreme subject (`Bidslab34`) into an established `"primary failure mode"` oversteps empirical bounds before EXP-019B and EXP-020 have even executed.

3. **Interpretation of $H_A$:**  
   Line 128 correctly reports $H_A$ as **INCONCLUSIVE** ($\bar{\Delta} = +1.01\text{ pp}$, 95% CI: $[-0.17, +1.99]\text{ pp}$). This is fully compliant.

---

## 3. Phase C Detailed Conformance Checks

### 3.1 Candidate Dataset Audit (PASSED)
The audit evaluated candidate datasets against the five v4 criteria ($k \ge 3$ labeled nights per person; wrist PPG/IHR; accelerometry; EEG/PSG reference labels; public/obtainable):
- **Kempenhaeghe Home Cohort (van der Aar et al. 2025):** Confirmed restricted/private. Data were collected in a clinical population ($n=59$) under Kempenhaeghe Center for Sleep Medicine and Philips proprietary governance. The published paper contains no public repository link or data DOI. Access requires institutional permissions and ethical clearance. **Correctly classified as unobtainable.**
- **SleepAccel (Walch et al. 2019, PhysioNet):** Confirmed single-night only ($n=31$, 1 overnight PSG per subject). **Correctly classified as ineligible for multi-night adaptation.**
- **MESA Sleep (NSRR):** Confirmed single-night PSG per subject ($n>2000$). Wrist actigraphy spans multiple nights, but concurrent PSG/EEG reference exists for only one night. **Correctly classified as ineligible.**
- **DREAMT (PhysioNet 2024):** Confirmed single-night clinical PSG per subject ($n=100$, Empatica E4). **Correctly classified as ineligible.**
- **Sleep-EDF / Sleep-EDF Expanded (PhysioNet):** Confirmed EEG-only (no wrist PPG, no accelerometry; $k=2$ nights only). **Correctly classified as ineligible.**
- **ISRUC-Sleep Group 2:** Confirmed clinical PSG only ($n=8$, $k=2$ nights only; no wearable wrist sensors). **Correctly classified as ineligible.**

### 3.2 Global Negative Claim Defect (MAJOR FINDING)
Section 5 of the execution report states:
> *"Audited Outcome: NO obtainable second multi-night dataset currently exists in the world literature meeting all criteria..."*  
**Audit Finding:** The user prompt explicitly warned:
> *"Do not accept an absolute claim that no public dataset exists unless the search coverage justifies it."*  
Asserting that no dataset exists "in the world literature" is an unverified universal negative. The executed searches covered PubMed, Europe PMC, PhysioNet, and NSRR. Repositories such as OpenNeuro, Zenodo, Dryad, Figshare, and institutional archives were not systematically indexed. The accurate, defensible statement must be:
> *"No qualifying second dataset was identified across the surveyed public databases (PhysioNet, NSRR, PubMed, Europe PMC) or peer-reviewed literature to date."*

---

## 4. Categorized Findings & Deficiencies

### 🔴 Critical Issues
1. **Banned Priority Claim in Novelty Statement:** The execution report (Section 4, line 131) claims `"The first preregistered, leakage-free, chronological replication..."`. This violates v4 Section 4 (`"Must not claim: 'First' multi-night personalization..."`) and constitutes an unverified global priority claim.
2. **Knowledge Base Artifact Desynchronization (B3):** Paper cards `P106`, `P007`, `P054`, and `P056` remain flagged as `retrieval_depth: abstract-only` with unresolved `TO FILL` markers, directly contradicting the claim that full-text examination for B3 is complete.

### 🟡 Major Issues
3. **Database Omissions in Search Log (B2):** `search_log.md` logs only PubMed and Europe PMC. Dedicated structured query strings, dates, and hit counts for IEEE Xplore, Scopus/Web of Science, and medRxiv are missing.
4. **Premature Elevation of C2 Mechanism:** Framing calibration-night class coverage as an established `"primary failure mode"` in the novelty statement overstates evidence derived from an exploratory single-subject audit; it must remain framed as a `"candidate failure mode"` per v4 Section 4.
5. **Sweeping Universal Negative in C1:** Stating that no dataset exists "in the world literature" is an overclaim that exceeds the documented repository search bounds.

### 🟢 Minor Issues & Evidence Gaps
6. **PRISMA Tracking:** `search_log.md` summarizes hit counts at the batch level without logging explicit deduplication counts or a PRISMA-style exclusion flowchart.
7. **van der Aar 2025 Paywall Constraints:** Exact hyperparameter layer freezing and loss function details for van der Aar 2025 remain behind the IOP Science paywall and are marked `TO FILL` in `P106`.

---

## 5. Exact Minimum Corrective Actions

To bring Phase B and Phase C execution into full conformance with `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029), the following minimum corrections must be applied:

1. **Correct Novelty Statement in `research_knowledge/updates/2026-11-18_phase_b_and_c_execution.md`:**
   - In Section 4 (Contribution C1), replace `"The first preregistered, leakage-free..."` with `"A preregistered, leakage-free..."` exactly matching v4 Section 4.
   - In Section 4 (Contribution C2), replace `"Empirical identification of calibration-night class coverage (severe label shift) as a primary failure mode..."` with `"Empirical evaluation of calibration-night class coverage (severe label shift) as a candidate failure mode..."`.
2. **Synchronize Paper Cards (`P106`, `P007`, `P054`, `P056`):**
   - In `P106`, resolve the `TO FILL` markers using the extracted data (clinical cohort $n=59$, wrist raw PPG + accel, home wearable EEG fine-tuning vs in-lab PSG test, significant at $\ge 3$ nights, 17.5% non-responders) and update `retrieval_depth` to `full-text (open preprint / author accepted summary)`.
   - In `P007`, `P054`, and `P056`, update `retrieval_depth` to `full-text` and populate architecture and sample details.
3. **Document IEEE Xplore, arXiv, and medRxiv Queries in `search_log.md`:**
   - Append dedicated query lines for IEEE Xplore, arXiv, and medRxiv reflecting the exact search strings, hits, and screening outcomes.
4. **Moderate Dataset Negative Claim in Section 5:**
   - Replace `"NO obtainable second multi-night dataset currently exists in the world literature..."` with `"No qualifying second dataset was identified across the surveyed public repositories (PhysioNet, NSRR, OpenNeuro) and literature to date..."`.

---

## 6. Exit Condition Determinations

- **Phase B Exit Condition:** **NOT SATISFIED**  
  *Justification:* The dated novelty statement contains a forbidden priority claim (`"The first"`), overclaims C2 as an established finding, and leaves core Knowledge Base cards in `abstract-only` status.
- **Phase C Exit Condition:** **CONDITIONALLY SATISFIED PENDING TEXT CORRECTION**  
  *Justification:* The substantive finding (that no eligible second dataset exists among investigated cohorts, confirming the Decision Gate for a single-dataset paper) is empirically verified; only the sweeping universal phrasing requires moderation.
