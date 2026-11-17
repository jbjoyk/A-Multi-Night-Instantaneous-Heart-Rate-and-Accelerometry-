# Literature Search Strategy

**Last Updated:** 2026-11-17 (KB v2 Errata Applied)  
**Governance:** All searches serve the locked research direction (*multi-night, subject-adaptive sleep staging from wearable IHR and accelerometry*). Adjacent fields are mined strictly for transferable methods and failure-mode theory, not to replace the research topic.

---

## 1. Search Scope & Priority Tiers

### Tier 1: Core Domain (Target `HIGH` Relevance)
- Wearable sleep staging (PPG, IHR, accelerometry, consumer smartwatches)
- Multi-night & longitudinal sleep staging
- Personalized / subject-adaptive / few-shot sleep staging
- External validation & cross-dataset generalization in wearable sleep staging
- Clinical sleep-summary agreement (TST, SOL, WASO, SE, Bland-Altman)

### Tier 2: Failure Modes, Criticism & Negative Results (Mandatory Contradiction Search)
- Test-time adaptation (TTA) failure modes, label/prior shift, batch dependency, model collapse
- Negative transfer in physiological/EEG/wearable adaptation
- Data leakage in physiological ML (epoch-level / subject-dependent splits vs subject-independent splits)
- Normalization artifacts across nights/subjects

### Tier 3: Transferable Methods (Target `MEDIUM` Relevance)
- Selective prediction, abstention, confidence-aware adaptation
- Risk monitoring & distribution-shift detection (label shift, covariate shift)
- Regularized fine-tuning (KL-divergence regularization, Fisher/EWC, PEFT)
- Multi-source domain adaptation & domain generalization for physiological time series

---

## 2. Recency & Foundational Balance
- **Priority Window:** 2023–2026 (capture modern TTA, negative transfer, foundation models, and recent wearable benchmarks).
- **Foundational Anchors:** Include pre-2023 landmark papers where they define the dataset, baseline, or core personalization method (e.g., Walch et al. 2019 *Sleep*, Sridhar et al. 2020 *npj Digital Medicine*, Phan et al. 2020 *Physiological Measurement* 41(6):064004, Banluesombatkul et al. 2020/2021 *MetaSleepLearner*).

---

## 3. Multi-Database, Forward-Citation & Contradiction-First Protocol
1. **Multi-Database Search:** Query across **PubMed, IEEE Xplore, arXiv, Semantic Scholar, and Google Scholar** (never rely on a single search engine).
2. **Mandatory Forward-Citation Tracking:** Run forward-citation (`Cited by`) sweeps on key anchor papers (e.g., `P001` Song et al. 2026, `P006` Phan et al. 2020, `P056` Radar/PPG personalization, `P106` van der Aar et al. 2025).
3. **Closest-Prior-Art Rule:** Before claiming novelty for any direction or method, run a dedicated query matching the exact modality + task + multi-night personalization setup (e.g., `wearable PPG accelerometry multi-night sleep staging personalization`).
4. **Contradiction-First Queries:** For every core concept, explicitly query for supporting evidence, negative results, competing alternatives, and methodological criticism.
5. **Governance Flag on Contradicting Evidence:** Whenever contradicting evidence is discovered (such as `P106`), explicitly flag whether it contradicts a **novelty claim** (e.g., "first multi-night wearable personalization") versus a **locked empirical result** (which remains immutable in Layer 1).

---

## 4. Two-Stage PRISMA-Style Screening & Update Pipeline
1. **Search:** Execute multi-database queries + forward-citation sweeps + contradiction queries; log exact query strings and dates in `search_log.md`.
2. **Deduplicate:** Check against `evidence_matrix.csv` by DOI/PMID/arXiv ID/title.
3. **Two-Stage Screen:**
   - *Stage 1 (Title/Abstract):* Assign preliminary Relevance (`HIGH`/`MEDIUM`/`LOW`/`OUT-OF-SCOPE`); reject `OUT-OF-SCOPE`.
   - *Stage 2 (Full-Text Retrieval):* Retrieve full text for `HIGH`-relevance records whenever accessible; record `retrieval_depth: full-text | abstract-only` explicitly.
4. **Extract:** Populate 33-field schema in `papers/PXXX_*.md` and `evidence_matrix.csv`. Use `UNKNOWN` or `TO FILL` for unverified fields; separate `Paper reports:` from `We infer:`.
5. **Classify:** Assign Evidence Strength (`STRONG` strictly requires `full-text` + verified `subject_split` + verified `statistical_testing`; otherwise cap at `MODERATE`) and Stance (`Supporting`/`Contradicting`/`Neutral`).
6. **Update Synthesis:** Append new findings to `competing_hypotheses.md`, `methods_catalog.md`, `datasets_catalog.md`, `research_gaps.md`, and `literature_map.md`.
7. **Log Update:** Create a dated entry in `updates/YYYY-MM-DD_*.md` preserving historical conclusions and stating the reason for every change.
