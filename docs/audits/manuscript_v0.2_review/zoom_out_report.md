# Stage 5 Initial Zoom Out & Inventive Catalyst Audit Report (`egm-zoom-out` + `egm-inventive-catalyst`)

**Target Manuscript:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md) (lines 1–252)  
**Review Persona:** Hostile Senior Editor & TRIPOD+AI Biostatistician (*The Lancet Digital Health*) + Skeptical Architect (`egm-zoom-out`) + Inventive Catalyst (`egm-inventive-catalyst`)  
**Authoritative Anchors Audited:** [MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv), [RESEARCH_DIRECTION_v4_FULL.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_DIRECTION_v4_FULL.md), [DECISIONS.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/DECISIONS.md) (`DEC-032` to `DEC-041`), [exp027_summary.json](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-027/exp027_summary.json), [exp028_sleepaccel_bland_altman_summary.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv)

---

## 1. Executive Verification: Word Count & Claim-Boundary Compliance

### 1.1 Structured Summary Exact Word Count ([MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L11-L27))
- **Background** (line 14): **46 words**
- **Methods** (line 17): **84 words**
- **Findings** (line 20): **123 words**
- **Interpretation** (line 23): **28 words**
- **Funding** (line 26): **1 word**
- **Total Body Word Count:** **282 words** (or **287 words** including the 5 subheadings).
- **Verdict:** **PASSES** the strict $\le 300$-word *Lancet Digital Health* ceiling.

### 1.2 High-Level Claim-Boundary Verification
- **No TOST equivalence claims without a pre-specified margin:** Verified compliant in main text (lines 59, 103, 198 explicitly disclaim TOST equivalence and frame non-significant paired tests as empirical parity / no statistically significant difference; note line 23 "match" requires precision adjustment).
- **No bare-metal 512 KB MCU causal-streaming claims:** Verified compliant (lines 80, 122, 230, 241 explicitly state that full-night $L=1200$ bidirectional GRU requires smartwatch SoC or companion smartphone offline full-night buffering, not 1-epoch causal streaming on a 512 KB microcontroller).
- **No unfair cross-cohort comparison of personalized Model A ($E_4$, $N=33$) vs. zero-shot SLAMSS-IFS ($N=47$):** Verified compliant (unlike the historical note in `DEC-038` item 4, the manuscript strictly compares Model A vs. SLAMSS-IFS under matched 5-fold zero-shot `N=47` on BIDSleep and zero-shot `n=25`/`N=31` on SleepAccel).
- **No overclaiming of clinical sleep-disorder generalization:** Verified compliant (lines 39, 59, 238–239 strictly scope all findings to healthy adults and identify PhysioNet DREAMT v2.2.0 and Simons Sleep Project `DS0000089` as prospective validation targets).

---

## 2. Detailed Audit Findings (`egm-zoom-out` & *Lancet Digital Health* Editorial Review)

### 🔴 FIND-ZO-01 [Blocker] — Structured Summary & Research in Context Overstate Exploratory `GATE_tau` and $k \ge 3$ Scaling Ahead of Locked Confirmatory `H-A` (`INCONCLUSIVE`), and Imply Personalization Applies to External Zero-Shot PSG
- **Severity:** 🔴 Blocker
- **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L20-L23) (Structured Summary `Findings` & `Interpretation`) and [lines 35–40](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L35-L40) (`Research in context`).
- **Editorial & Biostatistical Rationale:**
  1. **Conflation of Zero-Shot External PSG with Multi-Night Calibration (`Interpretation`, line 23):** Line 23 states: *"Lightweight smartwatch models match 33-fold larger server architectures on external clinical PSG and reliably track sleep duration, provided `k >= 3` calibration nights, stage-coverage gating, and label-prior shift adjustments are enforced."* A clinical reader of *The Lancet Digital Health* will misread "provided" as meaning that external PSG parity required $k \ge 3$ calibration nights and stage-coverage gating. In reality, external PSG validation (`EXP-027`/`EXP-028`) was **strictly zero-shot ($k=0$) on single-night recordings**, where multi-night calibration was impossible!
  2. **Exploratory `EXP-020` / `TSK-06` Overshadowing Confirmatory `EXP-019` `H-A` (`INCONCLUSIVE`):** In `Research in context -> Added value of this study` (line 36), advance `(1)` highlights monotonic scaling across $k=1\text{–}4$ nights ($p=0.01187$) and surgical `GATE_tau` gating (`-16.34 pp` reduced to `-6.40 pp`), **without even mentioning that the primary pre-registered confirmatory hypothesis (`H-A`, two-night fine-tuning `FT-2`, `+1.01 pp`, 95% CI `-0.17 to +1.99`) was `INCONCLUSIVE`** or that `GATE_tau` is a post-hoc, design-informed exploratory rule (`DEC-040`).
- **Concrete Remediation:**
  - In **Structured Summary -> Findings** (line 20), explicitly prefix `GATE_tau` with *"exploratory"* (`exploratory stage-coverage gating [GATE_tau]`).
  - In **Structured Summary -> Interpretation** (line 23), decouple the zero-shot external PSG claim from the within-person calibration claim.
  - In **Research in context -> Added value of this study** (line 36) and **Implications** (line 39), explicitly state that locked confirmatory two-night personalisation (`H-A`) was inconclusive (`+1.01 pp`, 95% CI `-0.17 to +1.99`), whereas multi-night scaling reaches significance at $k \ge 3$ nights and post-hoc exploratory stage-coverage gating (`GATE_tau`) mitigates extreme-tail negative transfer pending prospective replication.

---

### 🔴 FIND-ZO-02 [Blocker] — Altitude Drop & Repository Log Leakage: Raw Code Identifiers, File Paths, and Internal Task/Decision IDs Violate *Lancet Digital Health* Clinical Readability
- **Severity:** 🔴 Blocker
- **Affected Scope:** Throughout [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md) (lines 6–242).
- **Simplicity & Editorial Rationale:**
  *The Lancet Digital Health* immediately desk-rejects manuscripts that read like internal software engineering logs or ML repository readmes. Clinicians and biostatisticians should read **"Total Sleep Time (TST, min)"**—never `` `tst_min` ``; **"motion coverage (13.6%)"**—never `` `cov_mo=0.136` ``; **"single-model mean"** and **"15-model probability ensemble"**—never `` `mean_single_model` `` or `` `prob_ensemble` ``; **"calibration-depth analysis"**—never `` `TSK-06` ``.

---

### 🟡 FIND-ZO-03 [Major] — Table 4 Structural Compression Omits Mandatory Bland–Altman Uncertainty (`SE(LoA)`) and `95% CI on LoA Bounds`, and Conflates Metrics into Unreadable Slash-Separated Rows
- **Severity:** 🟡 Major
- **Affected Scope:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L181-L224) (Table 3 line 194 and Table 4 lines 202–224).
- **Concrete Remediation:** Expand slash-packed rows in Table 3 and Table 4 into clean individual rows per metric and include full CIs and non-parametric LoA bounds.

---

### 🟡 FIND-ZO-04 [Major] — Unexplained Reference Prior Discrepancy Between Prose (`11.1%` Deep) and Table 4 / `EXP-028` CSV (`12.01%` Deep\% of TST)
- **Severity:** 🟡 Major
- **Affected Scope:** Lines 36, 68, 142–143, 200, 232.
- **Concrete Remediation:** Explicitly distinguish pooled total-epoch stage proportions (including Wake: BIDSleep `46.2%` Light, `18.6%` Deep across all 246 nights vs. SleepAccel `57.7%` Light, `11.1%` Deep) from subject-averaged percentages of Total Sleep Time (excluding Wake: BIDSleep `21.14%` Deep% of TST on $N=46$ evaluation nights; SleepAccel `12.01%` Deep% of TST on $n=25$ `qc_clean`).

---

### 🟡 FIND-ZO-05 [Major] — TRIPOD+AI & *Lancet* Structural Omissions: Missing Clinical Table Ordering (`Table 2` Appears Before `Table 1`), Missing Data/Code Availability Statement, and Incomplete Reference Citations
- **Severity:** 🟡 Major
- **Affected Scope:** Lines 103–135, 244–252.

---

### 🟢 FIND-ZO-06 [Nit] — Minor Precision & Statistical Reporting Refinements (`LaTeX` Escape Typo, `k=2` Available-Cohort Delta Explanation, and `EXP-019` `S1` Bootstrap CI Discrepancy)
- **Severity:** 🟢 Nit
- **Affected Scope:** Lines 125, 128, 171, 235.

---

## 3. Inventive Catalyst (`egm-inventive-catalyst`): TRIZ Zero-Compromise Resolution of Audit Provenance vs. *Lancet* Clinical Prose

### 3.1 The Physical & Editorial Contradiction
- **Requirement A (Internal EGM Audit & Reproducibility Governance):** Every claim, cohort subset, protocol amendment, and software guardrail must be deterministically traceable to its exact git commit (`4dff5b7`), decision record (`DEC-030`–`DEC-041`), experiment/task identifier (`EXP-018`–`EXP-028`, `TSK-06`, `TSK-07`, `W0-B`), manifest path, and variable name (`cov_mo`, `min_stage_frac_cal`, `mean_single_model`).
- **Requirement B (*The Lancet Digital Health* Clinical Readability):** The main manuscript text and clinical tables must be immediately intelligible to sleep physicians, clinical trialists, and biostatisticians—completely free of software variable names, repository directory paths, and internal task codes.

### 3.2 TRIZ Inventive Principles Applied
1. **Principle 1 (Segmentation) & Principle 3 (Local Quality):** Separate *scientific nomenclature* (used in main prose and clinical tables) from *repository provenance identifiers* (consolidated into a dedicated **Section 2.5 Protocol Governance & Provenance Mapping Table** or **Data Sharing & Reproducibility Note**).
2. **Principle 24 (Intermediary / Canonical Translation Layer):** Replace every raw code token in main text and tables with its canonical clinical/biostatistical term, while binding each canonical term *once* in Section 2.5 / Data Sharing to its immutable repository artifact.
