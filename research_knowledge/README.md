# Research Knowledge & Literature Intelligence System

## 1. Governance & Locked Research Direction
- **Locked Working Title:** *“Leveraging multi-night information for subject-adaptive sleep staging from wearable instantaneous heart rate and accelerometry.”*
- **Locked Scope:** Multi-night, subject-adaptive 4-class sleep staging (Wake, Light, Deep, REM) from wearable instantaneous heart rate (IHR) and 3-axis accelerometry (30-second epochs, Dreem 2 EEG-derived labels on BIDSleep / PhysioNet), using strict subject-level, leakage-free evaluation, external validation, and clinically meaningful evaluation.
- **Governance Rule:** This Knowledge Base is an intelligence layer supporting the locked research direction. It does **not** alter the research direction, modify EXP-019 preregistration/results, or select future hypotheses autonomously. Any change of research direction requires explicit human approval. When contradicting evidence is found (e.g., `P106`), it is explicitly flagged as contradicting a **novelty claim** vs a **locked empirical result**.

---

## 2. Three Epistemic Layers (Strictly Separated)

### Layer 1: `LOCKED` (Immutable Ground Truth of Our Project)
- **Research Direction:** `RESEARCH_DIRECTION_v4_FULL.md` (supersedes `RESEARCH_DIRECTION_v3_FULL.md` and `RESEARCH_DIRECTION_v3.1_AMENDMENT.md`).
- **EXP-019 Preregistration:** `PREREG_EXP-019_H-A.md` (Git commit `4dff5b7a5f13a96c3024b309131aa1e66489dcbb`, manifest hash `36d840207e93...`), Model A (`ConventionalCNNBiGRU`, 163,940 parameters) frozen across all 5 arms (`ZS`, `FT-2`, `FT-other-2`, `PS-2`, `UA-2`), 4-class staging, `L = 1200` context, 46 eligible outer test subjects / 152 evaluation nights (`Bidslab49` excluded).
- **EXP-019 Confirmatory Results & Reconciled Verdict Status:**
  - **H-A (Primary: `FT-2` vs `ZS`):** $\bar{\Delta} = +1.01\text{ pp}$, 95% bootstrap CI $[-0.17, +1.99]\text{ pp}$, Wilcoxon $p = 0.00131$, 34/46 subjects improved. *Governance Reconciled Status:* Under the locked decision rule in `PREREG_EXP-019_H-A.md` at `4dff5b7` (SUPPORTED: $\bar{\Delta} \ge +1.50$ & CI_low > 0; NOT SUPPORTED: CI_high < +1.50; INCONCLUSIVE: anything else), since CI_high = +1.99 > +1.50, the exact preregistered rule evaluates to **INCONCLUSIVE**, while the EXP-019 report labeled it **NOT SUPPORTED**. This is a documented governance discrepancy to reconcile via erratum. Numerical results remain immutable.
  - **S1 (Specificity: `FT-2` vs `FT-other-2`):** $\bar{\Delta} = +2.73\text{ pp}$, 95% CI $[+1.05, +4.32]\text{ pp}$, $p = 9.0 \times 10^{-6}$. **Verdict: SUPPORTED.** (Interpreted with nuance: FT-other-2 is actively harmful at -1.72 pp below ZS, so much of the gap represents donor-night fine-tuning degradation).
  - **S2 (Transfer vs Scratch: `FT-2` vs `PS-2`):** $\bar{\Delta} = +20.32\text{ pp}$, 95% CI $[+17.70, +22.94]\text{ pp}$, $p < 1.0 \times 10^{-10}$. **Verdict: SUPPORTED.** (Weak baseline).
  - **S3 (Unsupervised InfoMax TTA: `UA-2` vs `ZS`):** $\bar{\Delta} = +0.00\text{ pp}$ across all 46 subjects identically. **Verdict: UNRESOLVED.** (Suspected implementation no-op; pending Phase A2 parameter change audit before interpreting as scientific evidence against unsupervised adaptation).

### Layer 2: `EXPLORATORY` (Hypothesis-Generating Only; Never Confirmatory)
- **EXP-019 Heterogeneity (`HETEROGENEITY_AUDIT_REPORT.md` / `audit_heterogeneity_mechanisms.py`):**
  - 34/46 subjects (73.9%) improved under `FT-2` (up to $+6.82\text{ pp}$), while severe degradations occurred in a minority (`Bidslab34`: $-16.34\text{ pp}$; `Bidslab65`: $-6.57\text{ pp}$; `Bidslab10`: $-6.47\text{ pp}$).
  - **Bidslab34 Forensic Observation:** Calibration nights 2–3 had $0.00\%$ and $1.34\%$ Deep sleep (with $41.8\%$ REM, $31.5\%$ Wake), whereas evaluation nights 4–5 rebounded to $17.65\%$ and $18.91\%$ Deep sleep ($D_{\text{JS}}(P_C \parallel P_E) = 0.3319$).
  - **Diagnostic Cohort Correlations:** Post-hoc $D_{\text{JS}}(P_C \parallel P_E)$ correlated with $\Delta$ (Spearman $\rho = -0.2933, p = 0.0479$). Pre-adaptation continuous rank correlations were non-significant after BH-FDR ($q > 0.05$).
  - **RTM Refutation:** Oldham's correction ($\text{Corr}(\frac{ZS+FT}{2}, \Delta)$) yielded $\rho = +0.0376, p = 0.8043$, refuting a linear ceiling/baseline-headroom effect.
  - **Parameter Drift Null:** Stratified within `scope=all` ($n=37$), parameter displacement showed $\rho = -0.0062, p = 0.9711$.
  - **Exploratory Gating Simulation:** $\min_c P_C(y=c) \ge 3.0\%$ AND $D_{\text{JS}}(P_C \parallel P_{\text{pop}}) < 0.35$ passed 44/46 subjects ($+1.39\text{ pp}$ mean) and rejected 2/46 ($-7.54\text{ pp}$ mean) via post hoc tuning on the same 46 subjects.

### Layer 3: `FUTURE` (Evidence-Grounded Questions & Candidate Hypotheses)
- Maintained in `research_gaps.md`, `competing_hypotheses.md`, and `novelty_registry.md`. Requires explicit human + EGM reasoning, adversarial review, and preregistration before any experiment.

---

## 3. Existing Infrastructure Reused vs. Created
- **Reused (External to this VM):** NotebookLM notebook `Sleep_research` (`notebooks/b1fb0d70-179b-4147-847d-b4585edaacd9`, 9 sources including Song et al. 2026 PDF, `Datasets Informations.docx`, `Problem Statement.docx`, `Research Gaps.docx`, and `EGM Workflow and Agent Sequence`).
- **Local Git Repo Note:** The user's local git repository (`/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/`) is not mounted in this cloud sandbox; therefore, `research_knowledge/` is built cleanly here for direct drop-in integration without modifying any locked repo files.

---

## 4. Evidence Hierarchy, Retrieval Depth & Relevance Scoring
- **Retrieval Depth (`retrieval_depth`):** `full-text` (full paper body read and extracted) | `abstract-only` (title/abstract/metadata screened only; unretrieved fields marked `UNKNOWN` or `TO FILL`).
- **Evidence Strength:** `STRONG` (strictly requires `full-text` retrieval + verified `subject_split` + verified `statistical_testing`; records lacking any of these are capped at `MODERATE`) | `MODERATE` (reasonably rigorous or `STRONG` candidate pending full-text verification) | `WEAK` (small/limited dataset, unclear protocol, weak stats) | `SPECULATIVE` (proposed mechanism without direct empirical test).
- **Current-Direction Relevance:** `HIGH` (directly informs multi-night subject-adaptive wearable sleep staging) | `MEDIUM` (transferable method, adjacent leakage methodology, or theory) | `LOW` (weakly connected) | `OUT-OF-SCOPE` (does not materially help current direction).

---

## 5. Repository Structure
```
research_knowledge/
├── README.md                    # Governance, epistemic layers, and system architecture
├── search_strategy.md           # Multi-database, forward-citation, PRISMA & screening rules
├── literature_map.md            # Research landscape across 16 sub-areas
├── evidence_matrix.csv          # Machine-readable extraction matrix (P001–P106)
├── research_gaps.md             # Classified gaps & evidence-grounded future questions
├── competing_hypotheses.md      # Mechanisms for heterogeneous adaptation (Observed/Proposed/Tested/Confirmed)
├── methods_catalog.md           # Transferable & sleep-specific methods catalog
├── datasets_catalog.md          # Wearable & reference sleep datasets catalog
├── novelty_registry.md          # Precedent & novelty checks for candidate directions
├── search_log.md                # Chronological log of all search queries & yields
├── papers/                      # Individual 33-field paper extraction records (P001–P106)
└── updates/                     # Append-only update & conclusion-change logs
```
