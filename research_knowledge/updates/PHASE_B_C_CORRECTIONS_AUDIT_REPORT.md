# Independent Conformance Audit Report: Phase B and Phase C Corrections

**Document Audited:** `research_knowledge/updates/2026-11-18_phase_b_and_c_execution.md`  
**Authoritative Reference:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029)  
**Prior Audit Reference:** `research_knowledge/updates/PHASE_B_C_INDEPENDENT_AUDIT_REPORT.md`  
**Audit Protocol:** EGM Goldfish Review Protocol (`/egm-review`)  
**Audit Date:** 2026-11-18  
**Audit Type:** Strict Independent Verification of Phase B/C Corrections  

---

## 1. Executive Summary & Task Verdicts

This audit verifies whether the targeted corrections applied to Phase B and Phase C work resolve the critical and major defects identified in `PHASE_B_C_INDEPENDENT_AUDIT_REPORT.md` while remaining strictly within the governance boundaries of `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029).

### Task Status Summary

| Task | Scope | Final Verdict | Exit Condition Status | Primary Finding |
| :--- | :--- | :---: | :---: | :--- |
| **B1** | KB Round-2 Corrections | **COMPLETE** | **SATISFIED** | All 6 round-2 corrections verified against primary records (NCBI E-utilities, PMC11099244) and applied cleanly across 9 files. Erroneous PMID `35877770` is completely eliminated. |
| **B2** | Systematic Literature Search | **PARTIALLY COMPLETE** | **NOT SATISFIED** | Structured query logs and hit counts recorded for PubMed (80 hits), Europe PMC, arXiv (23 hits), and medRxiv/bioRxiv (38 hits). Direct API searches on IEEE Xplore and Scopus/WoS were not executed due to institutional credential boundaries. |
| **B3 & Exit** | Core Full-Text Examination & Exit Deliverables | **PARTIALLY COMPLETE** | **NOT SATISFIED** | Exactly 4 of the 7 required core papers are verified full text (`P001`, `P002`, `P056`, `P106`). The remaining 3 (`P006`, `P007`, `P054`) are abstract-verified under publisher paywalls (IEEE and IOP). The dated novelty statement is compliant (priority claims removed; C2 candidate mechanism preserved; H-A inconclusive). |
| **C1** | Second Dataset Search & Decision Gate | **BOUNDED COMPLETE** | **SATISFIED** | Census across public repositories (PhysioNet, NSRR, OpenNeuro) and literature confirms no qualifying second multi-night dataset exists. Claim is properly bounded; Decision Gate for a single-dataset paper is enforced. |

---

## 2. Verification of Novelty Corrections

| Check Item | Requirement in v4 / Audit | Observed Status in Execution Report | Audit Finding |
| :--- | :--- | :--- | :--- |
| **Priority Claims** | No unauthorized "first" claims permitted (v4 Section 4) | The execution report line 154 replaced `"The first preregistered..."` with `"A pre-registered, leakage-free replication and extension..."`. A global grep confirms that the only occurrence of `"first"` is the negative claim: `"We do not claim to be the 'first' to personalize wearable or photoplethysmography-based sleep staging."` | **PASS (Resolved)** |
| **C2 Mechanism** | C2 must be framed as a candidate mechanism, not an established primary failure mode | Section 4 line 158 defines C2 as: *"Empirical evaluation of calibration-night class coverage (severe label shift) as a candidate failure mode for classifier-head collapse under few-night adaptation, with a nested evaluation of regularized or gated adaptation."* | **PASS (Resolved)** |
| **H-A Verdict** | $H_A$ must be reported as INCONCLUSIVE | Section 4 line 151 explicitly records: *"The locked confirmatory hypothesis (H-A, EXP-019) was inconclusive (+1.0054 pp Macro-F1, 95% CI [-0.17, +1.99] pp, Wilcoxon p=0.00131). Because the 95% confidence interval includes zero, these results do not establish a reliable personalization gain."* | **PASS (Compliant)** |
| **C1/C2/C3 Hierarchy** | Preserve primary (C1), secondary exploratory (C2), and external/clinical (C3) hierarchy | Section 4 preserves C1 (pre-registered replication, specificity control, $k$-learning curve, heterogeneity), C2 (calibration class coverage candidate failure mode), and C3 (SleepAccel zero-shot, clinical summary agreement). | **PASS (Preserved)** |

---

## 3. Bibliographic and Cohort Consistency Audit

A forensic line-by-line inspection was conducted across the paper cards, `evidence_matrix.csv`, and `research_knowledge/updates/2026-11-18_phase_b_and_c_execution.md` to resolve cohort and citation discrepancies against primary records:

### 3.1 P006: Phan et al. (2020)
- **Primary Source Evidence:** PubMed Abstract (PMID: `32392550`, DOI: `10.1088/1361-6579/ab921e`): *"Experimental results on the Sleep-EDF Expanded database consisting of 75 subjects show that sleep staging personalization with single-night data is possible with help of the proposed KL-divergence regularization."*
- **Audit Finding:** The execution report previously stated in Section 3 and the closest-ten table that Phan 2020 evaluated `"20 healthy subjects from Sleep-EDF"`. This mixed up the 20-subject legacy Sleep-EDF release with Sleep-EDF Expanded.
- **Resolution Applied:**
  - Section 3 line 97 corrected to: `75 healthy subjects from Sleep-EDF Expanded (150 nights, 2 nights per subject); 5-class sleep staging (pre-trained on MASS).`
  - Closest-ten table row 4 corrected to: `$n=75$ healthy adults; 2 nights/subject (150 nights, Sleep-EDF Expanded)`.
  - Paper card `P006_phan2020_kl_personalization.md` and `evidence_matrix.csv` confirmed synchronized to 75 subjects / 150 nights.

### 3.2 P056: Mikkelsen and De Vos (2018)
- **Primary Source Evidence:** arXiv:1801.02645 Section II (Data): *"This data set contains 2 nights of sleep EEG from 19 subjects (38 nights in total), together with a single night from an additional subject."*
- **Audit Finding:** The closest-ten table in the execution report previously listed `$n=75$ healthy adults`.
- **Resolution Applied:** Corrected across Section 3, Section 4 closest-ten table row 5, paper card `P056_personalizing_deep_learning_models_for_a.md`, and `evidence_matrix.csv` to: `$n=19$ adults with 2 nights (38 nights total) + 1 single-night subject (39 nights total)`.

### 3.3 Publication Identifiers for P106, P007, P054
- **`P106` (van der Aar et al. 2025):**
  - Verified from publisher PDF: *Physiological Measurement* 46 (2025) `095012` (Open Access CC BY 4.0).
  - DOI: `10.1088/1361-6579/ae0119`. PMID: `40882681`.
  - Discrepancy noted and corrected: The execution report previously cited article ID `095001`. Corrected to `095012`.
- **`P007` (Lorenzen et al. 2024):**
  - Verified from PubMed / IEEE: *IEEE Journal of Biomedical and Health Informatics* 28(10):5804–5815, October 2024.
  - DOI: `10.1109/JBHI.2024.3409165`. PMID: `38833404`. TechRxiv preprint: `10.36227/techrxiv.23938569`.
- **`P054` (Heremans et al. 2022):**
  - Verified from CrossRef: *Journal of Neural Engineering* 19(3):`036044`, June 2022.
  - DOI: `10.1088/1741-2552/ac6ca8`. PMID: `35508121`. Clinical trial registration: `S64190`.
  - Discrepancy noted and corrected: Execution report previously cited `036015`; paper card cited `036021`. Corrected to official article ID `036044` across execution report, paper card, and `evidence_matrix.csv`.

---

## 4. Full-Text Retrieval Status Audit

`RESEARCH_DIRECTION_v4_FULL.md` Section 9 (task B3) requires:
> *"Core full text: van der Aar 2025, Phan 2020, Lorenzen 2024, Heremans 2022, Mikkelsen and De Vos 2018, Song 2026, Walch 2019. Read each in full. Update its Knowledge Base card with paper_id, research question, hypothesis, dataset, number of subjects/nights, sensors, labels, task definition, preprocessing, architecture, adaptation method, calibration data, evaluation protocol, subject split, temporal split, external validation, metrics with uncertainty, statistical testing, failure cases, limitations, leakage risks, reproducibility."*

### Paper-by-Paper Verification

| Paper ID | Authors & Year | Venue & Identifier | Actual Text Access Achieved | Retrieval Depth Classification | Card Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **`P001`** | Song et al. (2026) | *IEEE TBME* (PMC12931632) | Complete open PMC full text (HTML/XML) | `full-text` | All fields verified |
| **`P002`** | Walch et al. (2019) | *Sleep* (PMC6930135) | Complete open PMC full text (HTML/XML) | `full-text` | All fields verified |
| **`P106`** | van der Aar et al. (2025) | *Physiol. Meas.* (`10.1088/1361-6579/ae0119`) | Complete publisher PDF (65,375 bytes, CC BY 4.0) | `full-text` | All 11 `TO FILL` markers resolved |
| **`P056`** | Mikkelsen & De Vos (2018) | *arXiv:1801.02645* | Complete arXiv PDF (33,594 bytes) | `full-text` | All fields resolved |
| **`P006`** | Phan et al. (2020) | *Physiol. Meas.* (`10.1088/1361-6579/ab921e`) | PubMed abstract + Lirias metadata (IOP paywall) | `abstract-only` | Core facts verified; full text paywalled |
| **`P007`** | Lorenzen et al. (2024) | *IEEE JBHI* (`10.1109/JBHI.2024.3409165`) | PubMed abstract + TechRxiv excerpts (IEEE paywall) | `abstract-only` | Core facts verified; full text paywalled |
| **`P054`** | Heremans et al. (2022) | *J. Neural Eng.* (`10.1088/1741-2552/ac6ca8`) | PubMed abstract + CrossRef metadata (IOP paywall) | `abstract-only` | Core facts verified; full text paywalled |

### Audit Finding
- Exactly **4 papers** (`P001`, `P002`, `P056`, `P106`) have verified full text.
- Exactly **3 papers** (`P006`, `P007`, `P054`) are bounded at abstract/author-summary extraction due to publisher paywalls (IEEE and IOP Publishing).
- `evidence_matrix.csv` had previously misclassified `P006` as `full-text`. This error was detected and corrected to `abstract-only`.
- **Conclusion:** Task B3 is **PARTIALLY COMPLETE**. It cannot be marked COMPLETE under v4 Section 9 until full text for `P006`, `P007`, and `P054` is genuinely retrieved and verified.

---

## 5. Database Coverage Audit

`RESEARCH_DIRECTION_v4_FULL.md` Section 9 (task B2) mandates searches across PubMed, IEEE Xplore, Scopus or Web of Science, arXiv, and medRxiv.

### Systematic Verification of Search Logs

| Database / Source | Execution Classification | Evidence in Repository | Verified Query Logs & Metrics |
| :--- | :---: | :--- | :--- |
| **PubMed** | **Search Performed** | `search_log.md` lines 10–13 | NCBI E-utilities API executed for 4 query blocks (Block 1: 12 hits; Block 2: 11 hits; Block 3: 2 hits; Block 4: 55 hits). Total 80 hits screened. |
| **Europe PMC** | **Search Performed** | `search_log.md` line 14 | REST API executed across Blocks 1–4. Confirmed zero additional multi-night public datasets. |
| **arXiv** | **Search Performed** | `search_log.md` line 15 | `export.arxiv.org/api/query` executed for 4 query blocks (Block 1: 6 hits; Block 2: 14 hits; Block 3: 0 hits; Block 4: 3 hits). Total 23 hits screened. |
| **medRxiv / bioRxiv** | **Search Performed** | `search_log.md` line 16 | Europe PMC Preprints API (`SRC:PPR`) executed for 4 query blocks (Block 1: 9 hits; Block 2: 6 hits; Block 3: 0 hits; Block 4: 23 hits). Total 38 hits screened. |
| **IEEE Xplore** | **Search Attempted (Blocked via REST API) / Performed via Indexing** | `search_log.md` line 17 | Direct REST API blocked due to requirement for institutional `X-ApiKey`. Literature discovery performed via indexed journal records (TBME, JBHI) in PubMed and TechRxiv. |
| **Scopus or Web of Science** | **Search Not Performed (Direct) / Performed via Forward Citation Sweeps** | `search_log.md` lines 18–19 | Direct database interface was not queried due to lack of institutional subscription credentials. Forward citations were tracked via CrossRef API and PubMed Central. |

### Audit Finding
- Automated searches were performed for PubMed, Europe PMC, arXiv, and medRxiv/bioRxiv.
- IEEE Xplore and Scopus/Web of Science were not directly queried via institutional subscription search engines.
- **Conclusion:** Task B2 is **PARTIALLY COMPLETE**. Marking B2 as COMPLETE would misrepresent database coverage.

---

## 6. Dataset Conclusion & Decision Gate Audit

### 6.1 Bounding of Dataset Claims
- **Audit Check:** Section 5 of the execution report was evaluated to confirm that sweeping global negative claims were removed.
- **Observed Text:**
  > *"Audited Outcome: No qualifying second dataset was identified across the public repositories and literature sources surveyed in this investigation meeting all criteria ($k \ge 3$ labeled nights per subject, wrist PPG/IHR, accelerometry, and EEG/PSG reference labels)."*
- **Audit Finding:** The claim is properly bounded to the surveyed repositories (PhysioNet, NSRR, OpenNeuro) and literature sources.

### 6.2 Decision Gate Enforcement
- **Criteria Verified:** $\ge 3$ labeled nights per person; wrist PPG/IHR; accelerometry; EEG/PSG reference labels; public/obtainable.
- **Evaluated Datasets:**
  - Kempenhaeghe Home Cohort (van der Aar 2025): Restricted/private clinical governance; unobtainable.
  - SleepAccel (Walch 2019): Single night per subject ($n=31$); ineligible for multi-night adaptation.
  - MESA Sleep (NSRR): Single concurrent PSG night ($n>2000$); ineligible for multi-night adaptation.
  - DREAMT (PhysioNet 2024): Single night per subject ($n=100$); ineligible for multi-night adaptation.
  - Sleep-EDF / Sleep-EDF Expanded: EEG only; $k=2$; ineligible.
  - ISRUC-Sleep Group 2: Full clinical PSG only; $k=2$; ineligible.
- **Enforced Outcome:** Single-dataset paper framed as replication, extension, and failure analysis under v4 Section 10. The limitation is stated openly.

---

## 7. Governance Compliance

The following governance invariants were checked via Git status and diff inspections:
1. **`RESEARCH_DIRECTION_v4_FULL.md`**: Unmodified (`git diff --name-only` confirms absence).
2. **DEC-029**: Unmodified (`DECISIONS.md` has no changes).
3. **Locked `EXP-019` Artifacts**: No changes to `experiments/EXP-019/exp019_*`, checkpoints, or preregistration files.
4. **Execution Freeze**: Zero training runs or experimental scripts were launched for `EXP-020` or `EXP-019B`.
5. **Decision Thresholds**: $H_A$ decision threshold ($+1.50\text{ pp}$) and qualitative verdict (`INCONCLUSIVE`) are unchanged.
6. **Git Commit Status**: No commits were made during this audit turn.

---

## 8. Exit Condition Determinations

### Phase B Exit Condition: NOT SATISFIED
- **Justification:** `RESEARCH_DIRECTION_v4_FULL.md` Section 9 mandates full-text examination for all seven core papers and systematic coverage of IEEE Xplore and Scopus/WoS. Because publisher paywalls restrict `P006`, `P007`, and `P054` to abstract/author-summary depth, and direct institutional queries for IEEE Xplore and Scopus/WoS were not executed, Phase B remains **PARTIALLY COMPLETE**. The dated novelty statement, closest-ten table, and round-2 KB corrections are fully compliant.

### Phase C Exit Condition: SATISFIED (BOUNDED COMPLETE)
- **Justification:** The search across public repositories and literature established that no qualifying second multi-night dataset is available. The claim is properly bounded, and the single-dataset paper path defined by the Decision Gate in v4 Section 10 is formally enforced.
