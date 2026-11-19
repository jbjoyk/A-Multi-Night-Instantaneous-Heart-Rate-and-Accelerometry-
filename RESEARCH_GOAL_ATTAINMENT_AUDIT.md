# Independent Research Goal Attainment Audit: BIDSleep Personalization Project
## Critical Assessment of Scientific Progress, Evidence Gaps, and Publication Viability

**Audit Date:** 2026-11-19  
**Auditor Persona:** Independent Adversarial Research Auditor & Senior Methodology Reviewer (EGM Protocol)  
**Governing Documents & Baselines Audited:**
- Authoritative Research Direction: [`RESEARCH_DIRECTION_v4_FULL.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_DIRECTION_v4_FULL.md) (Approved under DEC-029)
- Active Governance & State Records: [`RESEARCH_STATE.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_STATE.md), [`HANDOFF.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/HANDOFF.md), [`DECISIONS.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/DECISIONS.md) (DEC-001 through DEC-032)
- Locked Experimental Artifacts: `experiments/EXP-019/` (all 7 confirmatory artifacts, SHA-256 verified)
- Prior Independent Audits: `PHASE_A_INDEPENDENT_AUDIT_REPORT.md`, `PHASE_B_INDEPENDENT_AUDIT_REPORT.md`, `PHASE_3_INDEPENDENT_REVIEW_REPORT.md`
- Working Manuscript & Claim Matrix: [`MANUSCRIPT_DRAFT_v0.1.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.1.md), [`MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv)

---

## 1. Executive Summary & Central Verdict

### **Central Question: How far has the project actually progressed toward its original scientific and publication goals?**

The project has achieved **high computational maturity, strict governance discipline, and a complete manuscript draft (Sections 1–8)**. It has rigorously exposed historical split leakage, terminated unprincipled architecture searching (NAS), audited dataset signal quality, established anti-leakage manifests, and performed a methodologically sound sleep-summary agreement analysis.

However, when audited against its **substantive scientific and publication objectives** as defined in `RESEARCH_DIRECTION_v4_FULL.md`, the project exhibits a **substantial divergence between manuscript completeness and scientific goal attainment**. 

The completed manuscript documents what is fundamentally an **Outcome C** paper:
- The primary confirmatory personalization hypothesis (H-A) is **INCONCLUSIVE** ($+1.01\text{ pp}$ Macro-F1, $95\%\text{ CI: } [-0.17, +1.99]\text{ pp}$) because an $N=46$ cohort is structurally underpowered to detect a $1.0\text{ pp}$ effect.
- The $k = 1, 3, 4$ calibration-depth learning curve was **never executed** (explicitly marked `UNEXECUTED_IN_EXP019`).
- The unsupervised adaptation arm (S3, UA-2) suffered an implementation defect and was a **degenerate execution no-op**.
- The proposed algorithmic failure mitigations (FT-2-KL and $\text{GATE}_\tau$, EXP-020) were **never executed**.
- External validation on an independent PSG dataset (SleepAccel, EXP-027) was **never executed (deferred)**.
- The primary published benchmark (SLAMSS-IFS, 71.04% acc) was **never re-run under a matched 5-fold cross-validation protocol**.

### **FINAL SCIENTIFIC VERDICT: NOT SCIENTIFICALLY READY FOR TOP-TIER Q1 SUBMISSION**

```
================================================================================
RESEARCH GOAL ATTAINMENT AUDIT: MASTER CLASSIFICATION
================================================================================
1. Objective C1 (Multi-Night Personalization):    PARTIALLY MET (k=2 done; k=1,3,4 & S3 missing)
2. Objective C2 (Coverage & Gated Adaptation):   NOT MET (EXP-020 unexecuted; mechanism unproven)
3. Objective C3 (External Validation & Agreement): PARTIALLY MET (Agreement done; External deferred)
4. Algorithmic Research Contribution:           NOT MET (Model A is standard; methods unexecuted)
5. Matched Baseline Evaluation:                  PARTIALLY MET (Model A 5-fold done; SLAMSS-IFS unrun)
6. Evidence Base for Credible Q1 Publication:    NOT MET (Current evidence supports Q2/mid-tier only)
================================================================================
RECOMMENDED NEXT PHASE: PHASE D (FOCUSED TARGETED EXPERIMENTS: k-CURVE & MATCHED BASELINE)
DO NOT PROCEED DIRECTLY TO JOURNAL SUBMISSION.
================================================================================
```

---

## 2. Objective-by-Objective Goal Attainment Audit

### 2.1 Contribution C1: Multi-Night Supervised Personalization
- **Authoritative Mandate (`RESEARCH_DIRECTION_v4_FULL.md` lines 52, 41):** A pre-registered, leakage-free replication and extension of multi-night supervised personalization with consumer IHR and accelerometry in healthy adults, including specificity controls (FT-other), the $k = 1 \text{ to } 4$ learning curve, and heterogeneity analysis.
- **Goal Attainment Classification: PARTIALLY MET**
- **Supporting Experimental Evidence:**
  1. *Hypothesis H-A ($k=2$):* Executed under locked commit `4dff5b7`. Achieved $\bar{\Delta} = +1.0054\text{ pp}$ Macro-F1 ($64.58\% \to 65.59\%$), $95\%\text{ CI: } [-0.17, +1.99]\text{ pp}$, Wilcoxon $p = 0.00131$, win rate $73.9\%$ ($34/46$). Because the lower confidence bound spans zero ($-0.17 \le 0.0$), it strictly evaluates to **INCONCLUSIVE** under pre-registered decision rules.
  2. *Specificity Control (S1):* Executed. $\Delta_{\text{S1}} = +2.73\text{ pp}$ [95% CI: $+1.01, +4.44\text{ pp}$], $p = 0.00249$. Supported. Verified that personal fine-tuning outperforms donor fine-tuning.
  3. *From-Scratch Baseline (S2):* Executed. $\Delta_{\text{S2}} = +20.32\text{ pp}$, $p < 10^{-10}$. Supported. Verified that population pre-training is indispensable.
- **Unresolved Scientific Questions & Failures:**
  1. *Missing $k$-Learning Curve ($k=1, 3, 4$):* In the locked artifact `exp019_k_learning_curve.csv`, lines 2, 4, and 5 for $k \in \{1, 3, 4\}$ are explicitly populated with `UNEXECUTED_IN_EXP019`. The empirical trajectory comparing personal gains across 1, 2, 3, and 4 nights does not exist. This is a critical gap: the published literature (van der Aar et al., 2025) reported that personalization only reached statistical significance at $k \ge 3$ nights. By failing to execute $k=3$ and $k=4$, the project cannot answer whether the inconclusive result at $k=2$ is resolved with additional calibration nights.
  2. *Degenerate Unsupervised Control (S3):* Unsupervised adaptation ($UA\text{-}2$) suffered a silent implementation defect (`compute_wearable_quality_mask` applied raw BPM range $[30, 180]$ to z-scored features $[-3.95, +9.94]$, filtering $100\%$ of epochs and resulting in 0 parameter updates). S3 is an execution no-op; whether label-free adaptation works on consumer vitals remains completely unknown.
  3. *Negative Transfer Decomposition (S1):* Forensic decomposition shows that $63\%$ of the $+2.73\text{ pp}$ S1 advantage arises from donor-night degradation (FT-other-2 is $-1.72\text{ pp}$ below zero-shot ZS), rather than substantial positive transfer ($+1.01\text{ pp}$).
- **Remaining Requirements to Fully Meet C1:**
  - Execute $k=1$, $k=3$, and $k=4$ adaptation runs across evaluable cohorts ($N=47, 42, 33$).
  - Re-run S3 (UA-2) after removing the broken quality mask.

---

### 2.2 Contribution C2: Calibration-Night Class Coverage and Adaptation Failure
- **Authoritative Mandate (`RESEARCH_DIRECTION_v4_FULL.md` lines 53, 81, 147):** Calibration-night class coverage as a candidate failure mode, with nested evaluation of regularized (FT-2-KL) or gated ($\text{GATE}_\tau$) adaptation to eliminate negative transfer without losing average gains.
- **Goal Attainment Classification: NOT MET**
- **Supporting Experimental Evidence:**
  1. *Exploratory Observation:* Forensic audit of EXP-019 identified `Bidslab34` as experiencing catastrophic negative transfer ($-16.34\text{ pp}$ vs ZS; $-20.79\text{ pp}$ vs FT-other), associated with a calibration night containing only 15 Deep sleep epochs ($0.68\%$).
  2. *Preregistration Draft:* A design-informed protocol was authored (`EXP020_PREREGISTRATION_DRAFT.md`, SHA-256: `c226cc41...`).
- **Unresolved Scientific Questions & Failures:**
  1. *Zero Execution:* Under DEC-029 guardrails, **EXP-020 was NEVER executed**. No model checkpoints were trained; no KL penalty ($\lambda_{\text{KL}}$) was tested; no gating threshold ($\tau \in \{1\%, 2\%, 3\%, 5\%\}$) was evaluated.
  2. *Mechanism Statistically Unsupported:* In Phase A exploratory testing, cohort-wide correlation between calibration-night Deep sleep prevalence and adaptation gain was statistically non-significant ($q = 0.9711$, FDR-corrected). Deep sleep scarcity was an idiosyncratic anomaly of a single subject (`Bidslab34`), not a statistically validated general mechanism for personalization failure.
  3. *Unverified Solution:* Whether regularized adaptation or gating actually prevents negative transfer in wearable sleep staging remains entirely unproven.
- **Remaining Requirements to Fully Meet C2:**
  - Authorize and execute EXP-019B to formally test multiple candidate failure mechanisms.
  - Authorize and execute EXP-020 to determine whether KL regularization or quality gating produces a statistically verifiable benefit.

---

### 2.3 Contribution C3: External Validation and Clinical Sleep-Summary Agreement
- **Authoritative Mandate (`RESEARCH_DIRECTION_v4_FULL.md` lines 54, 82, 83, 149):** External zero-shot validation (SleepAccel cohort, EXP-027) and clinical sleep-summary agreement (TST, SOL, WASO, SE, REM%, Deep%, EXP-028).
- **Goal Attainment Classification: PARTIALLY MET**
- **Supporting Experimental Evidence:**
  1. *Clinical Sleep-Summary Agreement (EXP-028 Internal):* **MET.** Evaluated across $N=46$ subjects ($152$ nights) using Bland–Altman agreement with verified $\text{SE}(\text{LoA})$ and 95% CIs on bounds. Established that macro duration (TST bias $+2.28\text{ min}$, $r = 0.9492$) is well-tracked, while internal architecture (Deep% $r = 0.4868$, REM% $r = 0.4478$) exhibits poor agreement, and WASO is empirically underestimated ($-4.87\text{ min}$, $p = 0.0104$).
  2. *Second Multi-Night Dataset Census (C1):* **MET.** Systematic search confirmed that no qualifying second public multi-night wearable dataset with sleep stage labels exists.
- **Unresolved Scientific Questions & Failures:**
  1. *External Validation Dropped (EXP-027):* **NOT MET / DEFERRED.** Zero-shot transfer to the independent SleepAccel dataset (31 subjects, PSG labels) was formally deferred under DEC-029 and dropped from the active manuscript (Section 8.1 Item 6).
  2. *Generalizability Unestablished:* The entire empirical evidence base rests on a single geographic site, a single healthy young demographic profile, a single watch model (Apple Watch Series 6), and a single dry-sensor headband reference (Dreem 2). Zero external generalizability has been demonstrated.
  3. *Micro-Data Information Loss:* Because the underlying artifact is an aggregate summary table (48 rows), single-night prediction intervals and two-way error variance decomposition ($\sigma_b^2$ vs $\sigma_w^2$) could not be estimated.
- **Remaining Requirements to Fully Meet C3:**
  - Execute EXP-027 (SleepAccel zero-shot transfer) to establish whether Model A functions across different sensors and clinical PSG scoring.

---

### 2.4 Algorithmic Research Contribution
- **Authoritative Mandate (`RESEARCH_DIRECTION_v4_FULL.md` lines 17, 97, 130):** Evaluate whether novel sequence architecture mechanisms or lightweight designs provide competitive advantages.
- **Goal Attainment Classification: NOT MET**
- **Supporting Experimental Evidence:**
  1. *NAS Termination:* Neural Architecture Search (NAS Iteration 1) was invalidated as a single-fold smoke test on unadapted data (A5 erratum) and permanently terminated.
  2. *Architectural Invariance:* In EXP-014 (75 runs across 47 subjects), complex mechanisms (gating, transition heads, gradient reversal layers) showed zero detectable benefit over baseline Model A (Macro-F1 difference $-0.0069$, $95\%\text{ CI: } [-0.0213, +0.0059]$).
  3. *Model A Nature:* Model A (`ConventionalCNNBiGRU`, 163,940 parameters) is a standard, off-the-shelf combination of 1D convolutions and a 2-layer BiGRU. It contains zero novel mathematical formulations, zero novel loss functions, and zero novel architectural primitives.
  4. *Compression vs Innovation:* The manuscript frames Model A as a "$33.3\times$ parameter compression" relative to the 5.47M parameter official SLAMSS-IFS author repository. While practical for embedded systems, channel reduction in a standard CNN-BiGRU is an engineering optimization, not a substantive algorithmic discovery.
- **Unresolved Scientific Questions & Failures:**
  - The project does not offer an algorithmic contribution. Attempting to sell Model A as an algorithmic advance will fail peer review at any top-tier machine learning or biomedical informatics venue.

---

### 2.5 Matched Baseline Evaluation
- **Authoritative Mandate (`RESEARCH_DIRECTION_v4_FULL.md` lines 118–125):** Re-run SLAMSS-IFS (Song et al., 2026) on the identical 5-fold subject-disjoint splits and full-night context ($L=1200$), resolving the published 71.04% accuracy claim under an audited protocol.
- **Goal Attainment Classification: PARTIALLY MET**
- **Supporting Experimental Evidence:**
  1. *Model A Benchmark:* Model A was benchmarked under strict 5-fold CV across all 47 subjects (Macro-F1 64.58%–65.59%, accuracy 66.63%–67.12%).
  2. *Handcrafted Baseline:* Handcrafted feature gradient boosting (LightGBM) was evaluated in EXP-018 (Macro-F1 ~0.59).
- **Unresolved Scientific Questions & Failures:**
  1. *SLAMSS-IFS Unmatched:* The official SLAMSS-IFS codebase was **never re-run under the 5-fold cross-validation protocol**.
  2. *Disparate Protocol Defense:* The published SLAMSS-IFS result (71.04% acc, 0.7079 weighted F1) was evaluated on a single 11-subject holdout (`Cohort Beta`). While the manuscript correctly disclaims superiority and notes protocol differences, it fails to resolve whether SLAMSS-IFS actually outperforms Model A when evaluated under identical, leakage-free 5-fold CV.
  3. *Reviewer Vulnerability:* A reviewer from *IEEE TBME* (where SLAMSS-IFS was published) will immediately demand: *"Why did you not run the public SLAMSS-IFS code on your 5 folds to prove whether your lightweight model matches it?"*
- **Remaining Requirements to Fully Meet Baseline Evaluation:**
  - Execute a matched 5-fold cross-validation re-run of SLAMSS-IFS on identical splits and full-night context.

---

### 2.6 Evidence Needed to Support a Credible Q1 Publication
- **Authoritative Mandate (`RESEARCH_DIRECTION_v4_FULL.md` Sections 10 and 11):** Meet Q1 readiness checklist criteria and attain Outcome A or B.
- **Goal Attainment Classification: NOT MET (for Top-Tier Q1; PARTIALLY MET for Mid-Tier Q2 / Specialized Engineering)**
- **Critical Evaluation:**
  `RESEARCH_DIRECTION_v4_FULL.md` defined three paper outcome types:
  - **Outcome A:** Confirmatory adaptation on a second dataset + matched baseline win $\implies$ Method and benchmark paper (Strong Q1).
  - **Outcome B:** Supported adaptation without baseline win $\implies$ Personalization paper.
  - **Outcome C:** H-A not met and nothing replicated $\implies$ Negative-result and benchmark paper on BIDSleep alone.

The project currently sits squarely in **Outcome C**:
```
Outcome C Reality:
- Confirmatory hypothesis H-A is INCONCLUSIVE (+1.01 pp, CI [-0.17, +1.99]).
- Second dataset: None exists.
- External validation (EXP-027): Deferred / unexecuted.
- Proposed failure mitigations (EXP-020): Unexecuted.
- Learning curve (k=1, 3, 4): Unexecuted.
- Unsupervised adaptation (S3): Execution no-op.
- Matched baseline re-run (SLAMSS-IFS): Unexecuted.
```

#### Journal Tier Analysis:
1. **Top-Tier Q1 Venues (*IEEE Transactions on Biomedical Engineering*, *The Lancet Digital Health*, *Sleep*):**
   - **Expectation:** Conclusive positive findings, large cohorts, cross-site external validation, or groundbreaking algorithmic advances.
   - **Probability of Acceptance in Current State:** **Extremely Low (< 10%)**. A paper that reports an inconclusive 1-pp gain on 46 subjects, with zero external validation, an unexecuted learning curve, an unexecuted mitigation technique, and no matched baseline run will be seen as an incomplete project.
2. **Mid-Tier Q1 / High Q2 Venues (*Computers in Biology and Medicine*, *Biomedical Signal Processing and Control*, *IEEE JBHI*):**
   - **Expectation:** Solid methodological rigor, thorough benchmarking, and honest error analysis.
   - **Probability of Acceptance in Current State:** **Moderate (~40–50%)**, provided the paper is framed strictly as a *methodological critique* of data leakage, protocol heterogeneity, and the power realities of wearable personalization.
   - **Major Risk:** Reviewers will still object to the unexecuted $k$-curve ($k=3, 4$) and the lack of a matched SLAMSS-IFS baseline.

---

## 3. Dissecting the "Manuscript Completeness" Illusion

A central failure mode in computational research is **conflating a completed manuscript draft with achieved scientific goals**:
1. *Manuscript Draft v0.1 has 8 sections written:* This demonstrates documentation capability, not scientific closure.
2. *Section 6.2 has a beautiful Table 1:* The table is mathematically verified, but it reports on an aggregate summary of a single experiment where H-A was inconclusive, $k$-depths 1, 3, 4 are absent, and S3 is a no-op.
3. *Section 4.4 describes FT-2-KL and $\text{GATE}_\tau$:* These equations describe proposals that were never run. In a scientific audit, unexecuted methods have zero evidential weight.
4. *Claim Matrix has 40 claims:* 38 are verified, but they verify what was *observed* (including negative results and execution failures), not that the original scientific goals of the research were fulfilled.

A well-written paper about an incomplete experiment is still an incomplete experiment.

---

## 4. The Most Important Remaining Scientific Work

Before any journal submission is contemplated, the following concrete, high-leverage experimental tasks represent the **minimum scientific work required to elevate this study into a defensible Q1 contribution**:

```
Priority 1: Execute the k = 1, 3, 4 Learning Curve (EXP-019 Completion)
   - Cost: Low (inference and fine-tuning on existing splits; ~2–3 hours GPU time).
   - Value: CRITICAL. Answers whether personalization reaches significance at k=3 or 4 nights,
     directly testing the core claim of van der Aar et al. (2025).

Priority 2: Re-run S3 (UA-2) with Corrected Feature Mask
   - Cost: Low (fix the BPM threshold bug on z-scored features; run entropy minimization).
   - Value: HIGH. Replaces an embarrassing "execution no-op" disclosure with actual empirical
     evidence on whether unsupervised test-time adaptation functions on wearable vitals.

Priority 3: Matched 5-Fold Cross-Validation Re-Run of SLAMSS-IFS
   - Cost: Moderate (run official repository code on identical 5-fold manifests).
   - Value: HIGH. Protects the manuscript against immediate rejection from IEEE TBME reviewers
     by providing a true apples-to-apples comparison.

Priority 4: External Zero-Shot Transfer on SleepAccel (EXP-027)
   - Cost: Moderate (data harmonization, zero-shot inference across 31 PSG nights).
   - Value: VERY HIGH. Fulfills Contribution C3, breaking the single-dataset limitation.
```

---

## 5. Strategic Recommendation: What Phase to Run Next

> [!CAUTION]
> **DO NOT AUTOMATICALLY ADVANCE TO SUBMISSION PREPARATION.**
> Advancing to journal submission now locks the paper into an incomplete Outcome C state, ensuring an uphill battle with reviewers and a high probability of rejection at top venues.

### **RECOMMENDED NEXT ACTION: TRANSITION TO PHASE D (TARGETED EXPERIMENTS)**

The project should formally execute **Phase D (Targeted Experimental Completion)** under strict, pre-registered protocols to close the critical empirical gaps:
1. **Step D1 (Immediate):** Complete the pre-registered $k$-curve ($k = 1, 3, 4$) and re-run S3 (UA-2) with the corrected mask.
2. **Step D2 (Baseline Closure):** Execute the matched 5-fold CV run of SLAMSS-IFS.
3. **Step D3 (External Validation):** Execute EXP-027 zero-shot transfer on SleepAccel.
4. **Step D4 (Manuscript Synthesis):** Only after Steps D1–D3 are completed should the manuscript be finalized and submitted.

This transforms the manuscript from a defensive, inconclusive negative-result report into an authoritative, exhaustive, and unassailable benchmark of wearable sleep staging personalization.

---

*(Audit complete. No code executed; no locked files modified; research direction preserved.)*
