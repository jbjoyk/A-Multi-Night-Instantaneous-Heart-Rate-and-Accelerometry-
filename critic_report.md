# Stage 7 Critic Review Report: `docs/designs/exp019_adaptation.md` (FeatureRange: Lines 1–238)

* **Target Document:** [`docs/designs/exp019_adaptation.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/docs/designs/exp019_adaptation.md#L1-L238)
* **Audited Range:** FeatureRange (Lines 1 to 238)
* **Reviewers:**
  - Hostile Design Critic (`egm-design`)
  - Document Architect (`egm-structural`)
  - Technical Copy Editor (`egm-style`)
* **Review Date:** 2026-11-16
* **Overall Assessment:** **ACTION REQUIRED (CRITICAL FLAWS & EDITORIAL REVISIONS IDENTIFIED)**

---

## 1. Executive Summary

Stage 7 evaluated the high-level specification (`FeatureRange`, lines 1–238) across architecture, methodology, statistical validity, structural formatting, and technical tone. 

While the protocol exhibits strong separation between inner validation and outer testing, the **Hostile Design Critic** identified three critical blocking vulnerabilities that threaten experimental validity:
1. An operational validity paradox in Arm 5 ($UA\text{-}2$) regarding wearable-only masking vs. EEG label leakage.
2. Unconstrained Shannon entropy mode collapse in $UA\text{-}2$.
3. A statistical power formulation error regarding the compound decision rule for $N=46$.

The **Document Architect** confirmed high structural adherence with minor Markdown linting defects (missing blank lines between headings and lists). The **Technical Copy Editor** identified confirmatory bias in goal statements and uncalibrated claims that must be reframed neutrally.

---

## 2. Synthesized Review Findings by Severity

### 🔴 Blockers (System Architecture & Statistical Rigor)

#### Finding 1: The $UA\text{-}2$ Masking Paradox (Wearable-Only Adaptation vs. EEG Leakage)
* **Reviewer:** `egm-design`
* **Affected Scope:** Line 143 (`Arm 5: UA-2`)
* **Vulnerability:** Line 143 specifies Shannon entropy minimization on unlabeled calibration inputs. In BIDSleep, recordings contain invalid epochs (off-wrist, sensor detachment, motion artifact). The valid mask in the dataset is derived from Dreem EEG headband signal quality (`labels.mat`). If $UA\text{-}2$ uses this mask, it utilizes EEG ground-truth metadata, destroying the claim of "label-free wearable adaptation." If it does not use the mask, it minimizes entropy on off-wrist noise, forcing confident stage predictions on corrupted data.
* **Mitigation:** Formulate an explicit wearable-only valid-epoch heuristic for $UA\text{-}2$ (e.g., physiological IHR range $30 \le \text{HR} \le 180$ bpm combined with non-zero accelerometry variance), or explicitly restrict calibration to the sleep interval bounded by bedtime/wake-up.

#### Finding 2: Unconstrained Shannon Entropy Minimization Mode Collapse
* **Reviewer:** `egm-design`, `egm-inventive-catalyst`
* **Affected Scope:** Line 143, Line 173
* **Vulnerability:** Minimizing Shannon entropy ($\mathcal{L}_{\text{ent}} = -\sum p_c \log p_c$) on an imbalanced 4-class problem (Light sleep $\approx 50\%$) has a degenerate global minimum: predicting 100% Light sleep yields zero entropy but drops Macro-F1 to $\sim 0.15$.
* **Mitigation:** Augment $UA\text{-}2$ with a marginal distribution diversity penalty (InfoMax objective: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ent}} - \lambda \mathcal{H}(\bar{p})$) and add an automated collapse safeguard abort check if predicted stage entropy collapses below $0.5 \cdot \mathcal{H}_{\text{prior}}$.

#### Finding 3: Compound Decision Rule Statistical Power Formulation
* **Reviewer:** `egm-design`
* **Affected Scope:** Line 28, Line 58
* **Vulnerability:** The document asserts 80% power at $\Delta = 1.5\text{ pp}$ using $\text{MDE} = 0.4087 \cdot s_{\Delta}$. However, because the decision rule requires *both* $\hat{\Delta} \ge 1.5\text{ pp}$ *and* CI lower bound $> 0$, if the true population effect is exactly $1.5\text{ pp}$, the sampling distribution is centered at $1.5\text{ pp}$, giving $P(\hat{\Delta} \ge 1.5\text{ pp}) = 0.50$ (50% power). Furthermore, the step-function threshold escalation at $s_{\Delta} > 5.0\text{ pp}$ creates an abrupt discontinuity.
* **Mitigation:** Clarify that to achieve 80% power for $\hat{\Delta} \ge 1.5\text{ pp}$, the true detectable effect is $\delta \ge 1.95\text{ pp}$. Replace the step-function with a continuous smooth rule: $\text{Threshold} = \max(1.5, 0.4131 \cdot s_{\Delta})$, where $0.4131 = 2.8016 / \sqrt{46}$ for the audited cohort of $N=46$.

---

### 🟡 Majors (Methodology, Optimization & Codebase Parity)

#### Finding 4: Step-Count & Batching Representation for $PS\text{-}2$ and $FT\text{-}2$
* **Reviewer:** `egm-design`
* **Affected Scope:** Lines 171–175
* **Vulnerability:** With batch size fixed to 2 nights per step, 5–15 epochs for $FT\text{-}2$ equates to only 5–15 gradient steps, which is insufficient for Adam momentum de-biasing at low LR. For $PS\text{-}2$, 50–150 gradient steps from scratch on 163.9k parameters cannot train a functional CNN-BiGRU, turning $PS\text{-}2$ into a crippled strawman.
* **Mitigation:** Feed calibration data via sliding windows ($L=1200$, stride 300) so that 2 nights yield multiple mini-batches per epoch, or provide $PS\text{-}2$ a realistic optimization horizon (e.g. 500–1000 update steps with learning rate warmup).

#### Finding 5: Sampling Protocol in Specificity Control ($FT\text{-}other\text{-}2$)
* **Reviewer:** `egm-design`, `egm-inventive-catalyst`
* **Affected Scope:** Line 139
* **Vulnerability:** Sampling 2 random nights across different subjects creates an artificial chimeric subject, and nights drawn from inner-training subjects represent data the backbone already saw during pre-training.
* **Mitigation:** Enforce that $FT\text{-}other\text{-}2$ samples calibration nights $N_1, N_2$ from a *single* other training subject, assert that drawn nights contain all 4 stages with $\ge 300$ valid epochs, and document pre-training exposure as an unavoidable design boundary.

#### Finding 6: Acceptance Gate for Regenerated Checkpoints
* **Reviewer:** `egm-design`
* **Affected Scope:** Line 57
* **Vulnerability:** All 15 Model A backbones are missing from disk. Newly trained backbones could silently drift from the historical EXP-018 Phase 2 out-of-fold Macro-F1 (64.79%).
* **Mitigation:** Add an explicit verification assertion in `regenerate_modelA_checkpoints.py` confirming that regenerated backbones replicate the out-of-fold EXP-018 Phase 2 Macro-F1 within $\pm 0.5\text{ pp}$.

#### Finding 7: Clerical Compute Budget Arithmetic
* **Reviewer:** `egm-design`
* **Affected Scope:** Line 218
* **Vulnerability:** Line 218 lists Stage C as `5 folds * 46 outer test subjects * 3 seeds * 5 arms`. In 5-fold CV, outer test subjects total 46 across all folds, not 46 per fold.
* **Mitigation:** Correct line 218 to `46 outer test subjects * 3 seeds * 5 arms = 690 runs total`.

#### Finding 8: Confirmation Bias in Research Goals
* **Reviewer:** `egm-style`
* **Affected Scope:** Lines 28–30
* **Vulnerability:** Goals are phrased as conclusions to be confirmed ("Confirm that performance gains are idiographic by demonstrating that...") rather than neutral empirical hypotheses.
* **Mitigation:** Rephrase to objective testing language: "Test whether performance gains are idiographic by evaluating whether..."

#### Finding 9: Missing Blank Lines Before Bullet Lists
* **Reviewer:** `egm-structural`
* **Affected Scope:** Lines 64, 69, 181, 186, 191, 196, 201, 210, 215, 221, 226, 231
* **Vulnerability:** Violates markdownlint rules MD022/MD032 across 12 heading sites where blank lines are omitted before lists.
* **Mitigation:** Insert an empty newline between each `###` heading and its following bullet list.

---

### 🟢 Nits (Clarity, Precision & Formatting)

* **Finding 10 (egm-style):** Replace hyperbolic verbs and idioms ("accuracy swings widely", "pool nights indiscriminately", "definitively resolve", "without ever touching", "Monday and Tuesday to predict Wednesday") with dry descriptive prose.
* **Finding 11 (egm-style):** Neutralize repetitive use of decorative intensifiers (`strict`, `strictly`, `permanently`).
* **Finding 12 (egm-design):** Formalize the Stage D tie-breaking hierarchy: `higher Macro-F1 -> lower LR -> fewer epochs/steps -> fewer trainable parameters`.
* **Finding 13 (egm-design):** Explicitly document `use_rcc_loss=True` to match backbone pre-training.
* **Finding 14 (egm-structural):** Add language specifier ` ```text ` to the ASCII architecture diagram at line 81 (MD040).
* **Finding 15 (egm-structural):** Collapse consecutive blank lines before horizontal dividers (MD012).
