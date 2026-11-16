# Stage 6 Comprehension Test Report: `docs/designs/exp019_adaptation.md`

**Auditor:** Comprehension Goldfish (Naive Reader)  
**Target Document:** [`docs/designs/exp019_adaptation.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/docs/designs/exp019_adaptation.md)  
**Evaluated Ranges:** Feature Specification (lines 1–238), Detailed Implementation Plan (lines 239–367)

---

## High-Level Feature Specification Paraphrase
*(Based strictly on lines 1–238)*

**What this proposes:**  
This specification formalizes experiment EXP-019 to evaluate whether a pre-trained deep learning sleep-staging model can be effectively personalized to an individual using two earlier recorded nights of their wearable data (instantaneous heart rate and accelerometry) to improve sleep stage classification on subsequent nights. To rigorously prove genuine personalization without data snooping or temporal leakage, the design compares two-night fine-tuning against four control baselines (zero-shot, fine-tuning on another person's nights, training from scratch, and unsupervised entropy minimization) across 46 eligible subjects (152 evaluation nights) using chronologically ordered nights ($C \to E$). Hyperparameters are tuned strictly on a separate inner-validation set of 5 subjects per fold (Stage D) to confirm statistical power and lock decision thresholds before a single confirmatory test run (Stage C) on outer test folds.

**How the existing system works, per this document:**  
Currently, consumer wearable sleep models are trained across pooled cross-sectional cohorts and evaluated out-of-fold, resulting in high between-subject variance (accuracy varying from 62.5% to 72.3% across folds) due to individual differences in baseline autonomic tone and heart rate dynamics. Existing models either evaluate nights in isolation or pool multi-night recordings without an established protocol for within-person longitudinal calibration. The current baseline model in the repository is a frozen 1D CNN + BiGRU architecture (Model A, 163,940 parameters) operating on 10-hour context windows ($L=1200$ epochs) targeting 4 sleep stages (Wake, Light, Deep, REM), but its saved weight checkpoints from prior runs (EXP-018) are missing from disk because saving was omitted during previous script runs.

---

## Detailed Implementation Plan Paraphrase
*(Based strictly on lines 239–367)*

The implementation follows a strict 6-step sequential workflow designed to isolate model development from final evaluation:
1. **Regenerate Backbones:** Because EXP-018 failed to save model weights to disk, a dedicated retraining script (`regenerate_modelA_checkpoints.py`) is created to retrain Model A across 5 folds $\times$ 3 seeds and save 15 `.pt` checkpoint files to `experiments/EXP-018/checkpoints/`.
2. **Implement & Execute Stage D Inner-Validation:** A nested hyperparameter search script (`run_exp019_stage_d.py`) searches a pre-declared grid (12 fine-tuning configs, 4 scratch configs, 4 unsupervised configs) exclusively on the 5 inner-validation subjects per fold. It selects the winning hyperparameter setup per fold, measures the empirical standard deviation ($s_{\Delta}$) of paired differences, verifies whether the effect threshold needs escalation beyond $+1.5\text{ pp}$, and exports the selected parameters to JSON.
3. **Formal Pre-Registration Lock:** The preregistration file `PREREG_EXP-019_H-A.md` is updated with audited values (dataset manifest hash, 46 subjects / 152 evaluation nights, confirmed thresholds, and selected configs) and committed to Git (`origin main`) before any test evaluation begins.
4. **Implement & Execute Stage C Confirmatory Pass:** The evaluation script (`run_exp019_stage_c.py`) executes a single locked run across outer test subjects for all 5 arms, computes paired differences and 10,000-sample bootstrap 95% confidence intervals, and outputs final metrics.
5. **Documentation & Verification:** Workspace tracking logs (`EXPERIMENTS.md`, `DECISIONS.md`, `HANDOFF.md`) are updated with final outcomes, unit tests (`test_exp019_invariants.py`) verify temporal disjointness and state isolation, and all results are pushed immediately to GitHub.

---

## Comprehension Findings

### 1. Major Clarity & Self-Containment Gaps

* **Finding 1.1: Missing Target File in Plan Inventory**
  * **Severity:** Major
  * **Affected Scope:** Lines 268–331 vs. Line 343 (`tests/test_exp019_invariants.py`)
  * **Discrepancy Description & Rationale:** Section "Target Files to Create / Modify" (lines 268–331) lists 7 files to create or modify, but completely omits `tests/test_exp019_invariants.py`. However, Section "Verification Plan" (lines 343–352) relies centrally on this file to verify temporal disjointness, cohort sizing, state neutralization, and target masking via `pytest tests/test_exp019_invariants.py -v`. An engineer following the implementation plan step-by-step will find that the test suite does not exist.
  * **Suggested Clarification:** Add `tests/test_exp019_invariants.py` under "Target Files to Create / Modify" (File #8), specifying its test functions, fixtures, and assertions.

* **Finding 1.2: Confusing Dual Usage of the Term "Epoch"**
  * **Severity:** Major
  * **Affected Scope:** Line 98 ("$L=1200$ epochs $\approx 10$ hours"), Line 171 ("epochs $\in \{5, 15\}$"), Line 183 ("$\approx 1,800$ epochs"), Line 281 ("epochs")
  * **Discrepancy Description & Rationale:** The document uses the identical word "epoch" for two fundamentally different concepts without disambiguation:
    1. A physiological time-series segment (the standard 30-second sleep scoring epoch, where $1200 \times 30\text{ s} = 10\text{ hours}$).
    2. A machine learning optimization pass over the training dataset (e.g., fine-tuning for 5 or 15 gradient epochs).  
    A reader fresh to wearable biosignals or ML will be confused when reading that calibration nights have 1,800 epochs but training runs for 15 epochs.
  * **Suggested Clarification:** Explicitly differentiate these terms upon first use: use "30-second sleep scoring epochs" (or "time steps / windows") when describing time-series sequence length, and "training epochs" (or "optimization passes") when describing model training duration.

* **Finding 1.3: Ambiguity Between Raw Labels and Mapped Target Epoch Masking**
  * **Severity:** Major
  * **Affected Scope:** Line 56 (`{0: 0, 1: 1, 2: 1, 3: 2, 4: 3}`) vs. Line 100 (`(target >= 0) & (target < 4)`)
  * **Discrepancy Description & Rationale:** In Line 56, the raw labels in the dataset are indexed 0 to 4 (where raw label 4 represents REM, and 5 represents Unknown/masked). In Line 100, the valid target mask is specified as `(target >= 0) & (target < 4)`. A naive reader may assume that raw label 4 (REM) is being inadvertently excluded by the `< 4` condition because the text does not explicitly state that masking is evaluated *after* mapping `{0,1,2,3,4} \to \{0,1,2,3\}`.
  * **Suggested Clarification:** Explicitly clarify in Line 100: "After remapping the 5 raw labels into 4 target classes ($0\dots 3$), the valid target mask applies to `(target >= 0) & (target < 4)`. Raw label 5 (Unknown) is mapped to -1/invalid and masked out."

---

### 2. Undefined Acronyms & Unlinked Terms (Nits)

* **Finding 2.1: Undefined Clinical Acronyms (PSG, AASM)**
  * **Severity:** Nit
  * **Affected Scope:** Line 197 ("AASM"), Line 198 ("PSG"), Line 201 ("External PSG")
  * **Discrepancy Description & Rationale:** "AASM" and "PSG" appear in Section "Alternatives Considered" without definition. For non-clinical ML engineers, PSG and AASM are domain-specific jargon.
  * **Suggested Clarification:** Define inline on first mention: "American Academy of Sleep Medicine (AASM)" and "Polysomnography (PSG)".

* **Finding 2.2: Undefined Statistical / Trial Design Acronyms (MDE, s_delta)**
  * **Severity:** Nit
  * **Affected Scope:** Line 58 ("$\text{MDE}=0.4087 \cdot s_{\Delta}$")
  * **Discrepancy Description & Rationale:** "MDE" is introduced in the decision rule formula without spelling out "Minimum Detectable Effect", and $s_{\Delta}$ is only partially explained in later text as the paired standard deviation.
  * **Suggested Clarification:** Define "Minimum Detectable Effect (MDE)" and annotate $s_{\Delta}$ explicitly as the standard deviation of within-person differences ($FT\text{-}2 - ZS$).

* **Finding 2.3: Unreferenced Literature Baseline (SLAMSS-IFS)**
  * **Severity:** Nit
  * **Affected Scope:** Line 42 ("SLAMSS-IFS 71.04%")
  * **Discrepancy Description & Rationale:** "SLAMSS-IFS" is cited as a baseline benchmark figure, but lacks a bibliographic citation, author, or link.
  * **Suggested Clarification:** Add a brief citation or footnote identifying the paper/model (Song et al., 2026, IEEE TBME).

* **Finding 2.4: Unexplained Metadata Field in Diagram (`recStart`)**
  * **Severity:** Nit
  * **Affected Scope:** Line 108 (`Held-Out Subject Nights (Ordered by recStart)`)
  * **Discrepancy Description & Rationale:** The Mermaid diagram in line 108 specifies ordering by `recStart`, which is an unintroduced BIDSleep metadata dictionary field.
  * **Suggested Clarification:** Add a brief inline note in Section 2 or diagram caption clarifying that `recStart` refers to the recording start datetime timestamp from the BIDSleep night metadata.

* **Finding 2.5: Script Output Discrepancy (`README.md`)**
  * **Severity:** Nit
  * **Affected Scope:** Line 307 (`Outputs: experiments/EXP-019/exp019_stage_c_results.json and experiments/EXP-019/README.md`)
  * **Discrepancy Description & Rationale:** Line 307 states that the Python script `run_exp019_stage_c.py` outputs `experiments/EXP-019/README.md`.
  * **Suggested Clarification:** Clarify that `run_exp019_stage_c.py` exports `exp019_stage_c_results.json`, while `experiments/EXP-019/README.md` is authored by the researcher to document the findings.
