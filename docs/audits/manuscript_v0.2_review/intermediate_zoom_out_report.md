# Stage 8 Intermediate Zoom Out Audit Report (`egm-zoom-out`)

**Target Manuscript:** [MANUSCRIPT_DRAFT_v0.2.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md) (lines 1–252)  
**Cross-Evaluated Reports:** Stage 5 ([zoom_out_report.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/zoom_out_report.md)), Stage 6 ([comprehension_report.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/comprehension_report.md)), Stage 7 ([critic_report.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/critic_report.md)), Stage 9 ([implementation_readiness_report.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/implementation_readiness_report.md))  
**Stage 8 Verdict:** **APPROVED FOR UNIFIED REMEDIATION — ZERO WORD-BUDGET VIOLATIONS AFTER RECONCILING 4 CROSS-STAGE CONFLICTS**

---

## 1. Word-Budget & Architectural Bloat Impact Audit (*The Lancet Digital Health* Limits)

In *The Lancet Digital Health*, the word ceilings are:
- **Structured Summary:** strictly $\le 300$ words.
- **Main Text (Introduction + Methods + Results + Discussion):** $\le 3,500\text{–}4,500$ words (**excluding** the Structured Summary, *Research in context* panel, Tables 1–4, end-matter Declarations, and References).

### 1.1 Exact Word-Count Accounting of `MANUSCRIPT_DRAFT_v0.2.md` vs. Post-Remediation Target

| Manuscript Section | Current v0.2 Word Count | Net Remediation Delta | Projected v0.3 Word Count | *Lancet Digital Health* Ceiling & Status |
| :--- | :---: | :---: | :---: | :--- |
| **Structured Summary** ([lines 11–27](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L11-L27)) | 271 body words (276 w/ headings) | $+14$ words (re-anchoring `H-A` threshold, non-TOST phrasing, zero-shot decoupling) | **285 body words** (290 w/ headings) | **$\le 300$ words — PASS** ($10\text{–}15$ words margin) |
| **Research in context** ([lines 30–40](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L30-L40)) | 465 words | $+75$ words (systematic search dates/DBs/Boolean terms + `H-A` inconclusive framing) | **~540 words** | Excluded from main-text word count — **PASS** |
| **Main Text Prose** (Sections 1–4, excluding Tables 1–4) | **~2,490 words** | $+235$ words (acronym expansions, denominator clarifications, `DEC-039` & `DEC-030` 1-sentence scientific disclosures, prior-shift disambiguation, minus stripped repo paths/codes) | **~2,725 words** | **$\le 3,500\text{–}4,500$ words — PASS** ($>775$ words below lower ceiling!) |
| **Tables 1–4** + **Section 2.5 Provenance Mapping** | 4 tables (62 data rows) | $+8$ unpacked rows in Tables 3 & 4 + compact Section 2.5 Provenance Table | 4 clinical tables (70 data rows) + 1 compact governance table | Excluded from main-text word count — **PASS** |
| **Declarations & References** | 85 words (5 stub refs) | $+240$ words (End-matter Declarations + 10 full Vancouver refs) | **~325 words** | Excluded from main-text word count — **PASS** |

---

## 2. Cross-Stage Conflict Analysis & Unified Zero-Bloat Resolutions

1. **Conflict 1 (Removing Internal Audit/Bug Jargon vs. Mandatory Forensic Disclosures for `DEC-039` and `DEC-030`/`TSK-07`):**
   - Strip all `DEC-*`, `TSK-*`, `EXP-*`, `W0-B`, and `Cohort Alpha/Beta` code strings from the narrative sentences, state the scientific methodology and transparent pre-registration software quality amendments (`UA-2` input-scaling unit mismatch remediation and `k=1..4` clean frozen-checkpoint zero-shot baseline reconciliation) in clean clinical English, and consolidate all internal repository IDs (`EXP-018`–`EXP-028`, `TSK-06`, `TSK-07`, `W0-B`, `DEC-018`–`DEC-041`, commit `4dff5b7`) into a single compact **Section 2.5 Protocol Governance & Provenance Mapping Table**.
2. **Conflict 2 (Structured Summary Corrections Under $\le 300$-Word Ceiling):**
   - Use the pre-counted **286-word** Structured Summary body (**291 words** including the 5 subheadings) that strips all 29 backtick blocks, states the pre-registered `+1.50 pp` superiority threshold for `FT-2` (`+1.01 pp`, inconclusive), explicitly labels stage-coverage gating (`+1.33 pp`) as post-hoc exploratory, decouples zero-shot single-night external PSG parity from multi-night personalisation ($k \ge 3$), and replaces `"match"` with `"showed no statistically significant difference in 4-class Macro-F1 versus"`.
3. **Conflict 3 (Three-Way Reference Stage Proportion Discrepancy Across `44.2%/18.9%`, `46.2%/18.6%`, and `21.14%/12.01%`):**
   - Explicitly distinguish total-recording epoch proportions (BIDSleep `44.2%` Light, `18.9%` Deep across night-averaged Dreem 2 EEG [`46.2%` Light, `18.6%` Deep across 137,801 sensor-valid pooled epochs] vs. SleepAccel `57.7%` Light, `11.1%` Deep) from clinical `% of Total Sleep Time` (excluding Wake: BIDSleep `21.14%` Deep and `29.67%` REM of TST on $N=46$ evaluation nights vs. SleepAccel `12.01%` Deep and `25.21%` REM of TST on $n=25$ quality-clean recordings).
4. **Conflict 4 (Section 3.2 `S1` & `S2` Confidence Intervals — Parametric $t$-Interval vs. 10,000-Resample Percentile Bootstrap CI):**
   - Report the locked 10,000-resample percentile bootstrap 95% CI first (`[+1.05, +4.32] pp` for `S1`, `[+17.70, +22.94] pp` for `S2`) followed immediately in parentheses by the parametric $t$-interval (`[+1.01, +4.44]` and `[+17.60, +23.03]`) and both two-sided/one-sided Wilcoxon $p$-values.
