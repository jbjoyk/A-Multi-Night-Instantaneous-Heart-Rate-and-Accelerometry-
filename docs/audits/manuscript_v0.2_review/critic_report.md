# Stage 7 Critic Review Report (`egm-design`, `egm-structural`, `egm-style`)

**Target Document:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md) (Lines 1–252)  
**Cross-Referenced Authorities:** [MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv), [RESEARCH_DIRECTION_v4_FULL.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_DIRECTION_v4_FULL.md), [DECISIONS.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/DECISIONS.md) (`DEC-032` to `DEC-041`)  
**Reviewer Persona:** Senior Editor & TRIPOD+AI Biostatistician (*The Lancet Digital Health*)

---

## Executive Audit Summary

While the empirical calculations, denominator tracking (`N=47`, `N=46`, `N=33` on `E_4`, `n=25` `qc_clean`, `N=31` `all_n31`), and edge-deployment caveats (`309.54 KiB` Dynamic INT8, offline full-night buffering on smartwatch SoC/phone) are mathematically concordant with `DEC-032`–`DEC-041`, **the manuscript in its current v0.2 form would suffer an immediate editorial desk-rejection at *The Lancet Digital Health***. It reads like an internal machine-learning engineering and git-audit log rather than a clinical digital health paper, violates several *Lancet* structural mandates, and commits subtle biostatistical claim-boundary overreaches in the Structured Summary.

Below is the exhaustive breakdown categorized by `[egm-design]`, `[egm-structural]`, and `[egm-style]`.

---

## 1. `[egm-design]` — Biostatistical Design, Claim Boundaries & TRIPOD+AI Rigor

### 🔴 BLOCKER D-1: Exploratory `EXP-020` (`GATE_tau`) and `TSK-06` Override the Null/Inconclusive Primary Confirmatory Outcome (`H-A`) in the Structured Summary
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L20-L23) (Lines 20–23) & [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L38-L40) (Lines 38–40)
* **Finding:** Under TRIPOD+AI and *Lancet* statistical reporting rules, when a pre-registered primary confirmatory hypothesis (`H-A`, two-night supervised fine-tuning `FT-2`, `EXP-019`) evaluates to **inconclusive** (`+1.01 pp`, 95% CI `-0.17 to +1.99`, failing the locked `+1.50 pp` and CI `> 0` criteria), the Structured Summary **Interpretation** must lead with or explicitly reflect that primary confirmatory outcome. Instead, Line 20 buries the word `INCONCLUSIVE` inside parentheses without stating the pre-registered decision threshold (`+1.50 pp`), immediately pivots to post-hoc exploratory `GATE_tau` (`EXP-020`, designed *after* seeing `Bidslab34` fail), and Line 23 (**Interpretation**) states as an established prescriptive rule: *"provided k >= 3 calibration nights, stage-coverage gating, and label-prior shift adjustments are enforced."*
* **Recommendation:**
  1. In **Findings** (Line 20), state explicitly that the primary confirmatory two-night personalisation hypothesis did not meet its pre-registered `+1.50 pp` threshold and was inconclusive (`+1.01 pp`, 95% CI `-0.17 to +1.99`), and explicitly label stage-coverage gating (`+1.33 pp`) as *post-hoc exploratory*.
  2. In **Interpretation** (Line 23), qualify that while two-night unconstrained personalisation is inconclusive, calibration-depth scaling (`k >= 3` nights) and *exploratory* stage-coverage abstention offer candidate safeguards requiring prospective confirmation.

### 🔴 BLOCKER D-2: Unqualified Equivalence Claim ("match") in Structured Summary Interpretation Violates `DEC-041` Non-TOST Boundary
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L23) (Line 23) & [MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv#L41) (`CLM-040`)
* **Finding:** `DEC-041` (Line 829) mandates: *"The non-significant difference between Model A and the 33.34x larger Official SLAMSS-IFS is strictly framed as no statistically significant difference / empirical parity under paired testing. TOST equivalence is explicitly not claimed because no formal clinical equivalence margin was pre-specified."* In a sample of `n=25` QC-clean subjects (`EXP-027`), the 95% CI on the primary Macro-F1 difference (`-0.57 pp`) spans `-2.13 to +0.94 pp` (`p=0.5782`), and in `N=31` it spans `-3.84 to +0.28 pp`. Claiming in the Structured Summary Interpretation (Line 23) that *"Lightweight smartwatch models match 33-fold larger server architectures on external clinical PSG"* asserts statistical equivalence from an absence of significant difference in a small (`n=25`) external sample.
* **Recommendation:** Replace *"match 33-fold larger server architectures"* in Line 23 with *"showed no statistically significant difference in 4-class Macro-F1 compared with a 33-fold larger server architecture on external single-night clinical PSG"*. Update `CLM-040` in the matrix accordingly.

### 🟡 MAJOR D-3: Missing Clinical / Physiological Framing of External WASO Underestimation and Sleep Efficiency Correlation Drop (`EXP-028`)
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L23) (Line 23), [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L39) (Line 39), [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L200-L224) (Lines 200–224), [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L232) (Line 232)
* **Finding:** Line 39 claims consumer IHR and accelerometry safely track Total Sleep Time and Sleep Efficiency. While multi-night subject-averaged Sleep Efficiency on BIDSleep has `r=0.8775` (Table 1, Line 141), single-night external PSG Sleep Efficiency in Table 4 (Line 208) has a weak correlation of `r=0.3781` (with WASO biased by `-13.57 min`, 95% LoA `-43.05 to +15.90 min`, `r=0.3968`, and TST biased by `+11.81 min`, 95% LoA `-23.79 to +47.40 min`). Claiming that Sleep Efficiency is reliably tracked on external single-night PSG when `r=0.3781` and WASO is underestimated by `13.57 min` will draw immediate scrutiny from clinical sleep reviewers.
* **Recommendation:** Distinguish clearly in "Research in context" (Line 39) and Discussion (Line 232) between *multi-night habitual averages* (where TST `r=0.9492` and SE `r=0.8775` are strongly tracked) and *single-night external PSG* (where TST correlation remains high at `r=0.9091` with low SE MAE of `3.58%`, but single-night SE rank correlation drops to `r=0.3781` due to empirical WASO underestimation of `-13.57 min`). Also verify that `DEC-032` (`FIND-B-01`) is explicitly stated in Section 3.3 prose: the BIDSleep Bland–Altman LoAs apply *only* to multi-night participant means (~3.3 nights), never to single-night spot predictions.

### 🟡 MAJOR D-4: Missing Explicit Clarification Why Personalized Model A (`E_4`, `N=33`) Is Not Directly Compared Against Zero-Shot SLAMSS-IFS (`N=47`)
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L164-L171) (Lines 164–171) & [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L230-L235) (Lines 230–235)
* **Finding:** The draft rightly avoids the cohort-conflation error present in `DEC-038` Item 4 (which casually juxtaposed `N=33` `E_4` personalized Model A `65.82%–66.25%` against `N=47` all-night zero-shot SLAMSS-IFS `67.37%`). However, a reader looking at `66.25%` (`k=4`, Section 3.4) and `67.37%` (SLAMSS-IFS, Section 3.1) may attempt that direct comparison themselves.
* **Recommendation:** Add one explicit sentence in Section 3.4 or Section 4.2 noting that `k=1..4` adaptation metrics on `E_4` (`N=33`, 64 nights, clean `ZS=63.54%`) are evaluated on a subset of later chronological nights and cannot be directly compared to full-cohort zero-shot SLAMSS-IFS (`N=47`, 246 nights, `ZS=64.79%` for Model A).

### 🟢 NIT D-5: Denominator Integrity & Claim-Boundary Verification Passed Across All Five Core Checks
* **Affected Scope:** Global audit across Sections 2 & 3
* **Verification Summary:**
  - `N=47` (246 usable nights of 253 total; 7 nights excluded across 6 subjects for `<60%` coverage) is consistently used for `EXP-018`/`EXP-021`.
  - `N=46` (152 evaluation nights; `Bidslab49` excluded a priori with `n_usable=2`, `n_eval=0`) is consistently used for `EXP-019`, `TSK-07`, `EXP-020`, and Table 1 (`df=45`).
  - `N=33` (`E_4`, 64 evaluation nights, clean `ZS=63.54%`) vs. `E_k` (`N=46/46/42/33`) are cleanly separated in Section 3.4.
  - `n=25` (`qc_clean`, 6 pre-flight exclusions: 5 with `<600` epochs, 1 with `cov_mo=0.136`) and `N=31` (`all_n31` sensitivity) are consistently separated in Section 3.6 and Tables 3–4.
  - Bare-metal 512 KB MCU causal streaming is explicitly disclaimed in Lines 80, 122, and 241.
  - Healthy-adult scoping and non-generalization to clinical sleep disorders are explicitly enforced in Lines 39, 59, and 238–240.

---

## 2. `[egm-structural]` — *The Lancet Digital Health* Structural & Editorial Compliance

### 🔴 BLOCKER S-1: "Research in context" Panel Violates Mandatory *Lancet Digital Health* Systematic Search Reporting Requirements
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L32-L34) (Lines 32–34)
* **Finding:** *The Lancet Digital Health* strictly requires the **`Evidence before this study`** subsection of the *Research in context* panel to report: (1) the bibliographic databases searched (e.g., PubMed, IEEE Xplore, Scopus/Web of Science, arXiv, medRxiv, plus data repositories PhysioNet, NSRR, OpenNeuro), (2) the exact date range of the search, (3) the search terms used, and (4) the inclusion/exclusion criteria or language restrictions. Currently, Line 33 only mentions repository audits (`PhysioNet, NSRR, and OpenNeuro`) and cites three papers narratively, omitting the literature search strategy required by both *The Lancet* and [RESEARCH_DIRECTION_v4_FULL.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_DIRECTION_v4_FULL.md#L138) (Phase B2).
* **Recommendation:** Rewrite the opening of **`Evidence before this study`** to explicitly state the databases searched (PubMed, IEEE Xplore, arXiv, medRxiv, and public sleep repositories PhysioNet, NSRR, and OpenNeuro), search dates (e.g., Jan 1, 2018, to Nov 15, 2026), exact search terms (`("sleep staging" OR "sleep stage") AND ("photoplethysmography" OR "instantaneous heart rate" OR "smartwatch" OR "wearable") AND ("personalization" OR "fine-tuning" OR "multi-night" OR "adaptation")`), and inclusion criteria before summarizing Walch et al. (2019), Song et al. (2026), and van der Aar et al. (2025).

### 🔴 BLOCKER S-2: Table Numbering Out of Sequence in Main Text (`Table 2` Appears Before `Table 1`)
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L103-L107) (Section 3.1, Lines 103 & 107: `Table 2`) vs. [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L133-L135) (Section 3.3, Lines 133 & 135: `Table 1`)
* **Finding:** In Section 3.1 (Line 103), the first table cited and rendered is **Table 2** (*Matched 5-Fold Cross-Validation and Edge Deployment Profile*). In Section 3.3 (Line 133), the second table cited and rendered is **Table 1** (*Subject-Averaged Multi-Night Sleep-Summary Bland–Altman Agreement*). All biomedical journals mandate strict sequential numbering of tables in order of first citation.
* **Recommendation:** Renumber **Table 2** (Section 3.1) $\to$ **Table 1**, and renumber **Table 1** (Section 3.3) $\to$ **Table 2**.

### 🔴 BLOCKER S-3: Incomplete, Unnumbered, and Non-Vancouver Reference List
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L246-L251) (Lines 246–251) and in-text citations throughout Lines 33–243
* **Finding:**
  1. The reference list contains only 5 stub items and violates *Lancet* Vancouver style (up to 6 authors listed before *et al.*, full paper title, abbreviated journal title, year, volume, pages/article ID).
  2. **Ref 1 (Line 247):** `"Song et al. SLAMSS-IFS and BIDSleep multi-night dataset. IEEE Trans Biomed Eng 2026..."` uses a fabricated placeholder title and omits author initials.
  3. **Ref 3 (Line 249):** `"van der Aar JF, et al. Supervised multi-night personalization..."` omits co-authors, volume, and article ID (`ae0119`).
  4. **Ref 5 (Line 251):** `"Quirk C, et al. Data leakage in deep learning sleep staging. 2024."` is missing its journal/preprint venue, volume/ID, and DOI, and isn't even cited in the main text!
  5. Essential references mentioned in the text or required by the methodology—such as **TRIPOD+AI** (Collins GS, Moons KGM, Dhiman P, et al. *BMJ* 2024; 385:e078378), the **AASM Scoring Manual** (Berry RB et al.), **Page's L test** (Page EB, *J Am Stat Assoc* 1963), and **Oldham's method** (Oldham PD, *J Chronic Dis* 1962)—are completely missing from the References list, and the main text uses author-year citations instead of numbered Vancouver citations.
* **Recommendation:** Convert in-text citations to numbered Vancouver format `[1]–[10]` (or superscripts) and provide full, verified Vancouver bibliographic entries for all cited works, including TRIPOD+AI, AASM, Song et al., Walch et al., van der Aar et al., Bland & Altman, Quirk et al., Page, and Oldham.

### 🟡 MAJOR S-4: Structured Summary Word Count Verification and Formatting Violations
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L11-L27) (Lines 11–27)
* **Finding:**
  - **Word Count Audit:** The body of the Structured Summary (Background: 40 words + Methods: 81 words + Findings: 120 words + Interpretation: 29 words + Funding: 1 word = **271 words** total; 276 words including the 5 subheadings) is strictly **$\le 300$ words**, satisfying the *Lancet Digital Health* word ceiling. You have **24–29 words of headroom** to fix Findings/Interpretation per **Blocker D-1** and **Blocker D-2**.
  - **Formatting Violation:** The Structured Summary contains **29 inline code backtick blocks** (e.g., `` `N=47` ``, `` `163,940` ``, `` `FT-2` ``, `` `GATE_tau` ``, `` `mean_single_model` ``, `` `INCONCLUSIVE` ``). *The Lancet Digital Health* strictly prohibits code backticks, programmatic variable names (`mean_single_model`, `E_4`, `GATE_tau`), and all-caps software status strings (`INCONCLUSIVE`) in the Structured Summary.
* **Recommendation:** Strip all backticks from the Structured Summary; replace programmatic identifiers (`mean_single_model` $\to$ single-model evaluation; `E_4` $\to$ common four-night evaluation cohort; `GATE_tau` $\to$ pre-adaptation stage-coverage gating; `INCONCLUSIVE` $\to$ inconclusive under pre-registered criteria) while keeping the total word count $\le 300$ words.

### 🟡 MAJOR S-5: Missing Mandatory *Lancet Digital Health* End-Matter Declarations (Data Sharing, Code Availability, Ethics, Author Contributions, Declaration of Interests)
* **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L243-L252) (Lines 243–252)
* **Finding:** *The Lancet Digital Health* and TRIPOD+AI mandate dedicated end-matter sections (and an institutional ethics / secondary public data reuse statement in Methods).
* **Recommendation:** Add a concise Ethics statement in Section 2.1 and add a `Declarations (Ethics, Data & Code Sharing, Competing Interests)` section immediately following Section 4.3.

---

## 3. `[egm-style]` — Clinical Prose vs. Raw ML Engineering / Git Audit Report Style

### 🔴 BLOCKER ST-1: Pervasive Internal Task IDs, Git Commit Hashes, Decision Codes, Python Variable Names, and Repo File Paths in Main Prose
* **Affected Scope:** Throughout the entire manuscript (Lines 5–242).
* **Finding:** The manuscript reads like an internal engineering handoff memo. Readers and editors of *The Lancet Digital Health* have no idea what `DEC-030`, `DEC-039`, `TSK-06`, `TSK-07`, `W0-B`, `EXP-021`, `Cohort Beta`, `mean_single_model`, `qc_clean`, or `experiments/EXP-019/exp019_stage_d_selected_configs.json` mean. Worse, inline backticks are wrapped around nearly every number in the paper.
* **Recommendation:**
  1. **Remove all internal task and decision codes** (`DEC-*`, `TSK-*`, `W0-B`, `EXP-*`, `FIND-B-*`) and local file paths from the narrative prose and section headings.
  2. **Remove all inline code backticks around numbers, percentages, p-values, and clinical acronyms.**
  3. **Translate code identifiers into clinical/scientific terminology.**

### 🟡 MAJOR ST-2: Tables 1–4 Use Raw CSV Column Headers and Condensed Multi-Metric Slash Rows
* **Affected Scope:** Tables 1–4 (Lines 107–223).
* **Finding:** *The Lancet Digital Health* production editors reject tables that mix single-variable rows with slash-compressed dual-variable rows or display raw snake_case variable names.
* **Recommendation:** Replace all snake_case variable names in Tables 1 and 4 with proper clinical labels and unpack the slash-compressed rows in Table 3 (Line 194) and Table 4 (Lines 219–223).

### 🟡 MAJOR ST-3: Bullet-Point Outline Prose in Introduction, Methods, Results, and Discussion
* **Affected Scope:** Section 1.3 (Lines 55–57), Section 2.1 (Lines 68–73), Section 2.2 (Lines 78–80), Section 2.3 (Lines 84–90), Section 3.2 (Lines 128–130), Section 3.4 (Lines 165–169), Section 3.5 (Lines 175–176), Section 4.3 (Lines 239–242)
* **Recommendation:** Convert the bulleted lists in Sections 1.3, 2.1–2.3, 3.2, 3.4, 3.5, and 4.3 into cohesive, publication-grade paragraphs.

### 🟢 NIT ST-4: Framing of Historical Audit Errors (`DEC-030` No-Op and `DEC-039` In-Place Mutation) in Clinical Prose
* **Affected Scope:** Lines 36, 88–89, 130
* **Recommendation:** In *Research in context* (Line 36), frame advance (3) around the *scientific finding*—that quality-gated unsupervised entropy adaptation (`UA-2`) yields a statistically significant but sub-benchmark gain (`+0.21 pp`), proving that label-free adaptation cannot substitute for supervised multi-night calibration—and keep the transparent disclosure of the pre-registration amendment (`S3` unit-mask remediation) in Methods Section 2.3 and Results Section 3.2.
