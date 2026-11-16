# Stage 9 Implementation Readiness Report: `docs/designs/exp019_adaptation.md` (Lines 255–395)

* **Target Document:** [`docs/designs/exp019_adaptation.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/docs/designs/exp019_adaptation.md#L255-L395)
* **Audited Scope:** Detailed Implementation Plan & Verification Plan (Lines 255 to 395)
* **Reviewers:**
  - Pragmatic Tech Lead (`egm-implementation-readiness`)
  - Codebase Technical Quality Auditor (`egm-codebase-details`)
  - Technical Copy Editor (`egm-style`)
* **Review Date:** 2026-11-16
* **Readiness Verdict:** **CONDITIONAL PASS (RESOLVABLE TECHNICAL GAPS IDENTIFIED)**

---

## 1. Executive Summary

Stage 9 audited the Detailed Implementation Plan and Verification Plan against codebase reality, zero-decision actionability, API completeness, and technical tone. 

The reviewers confirmed that the architectural separation between Stage D (inner-validation search) and Stage C (outer confirmatory testing) is topologically sound. However, the audit identified **two codebase blockers** and **five major specification gaps** where the implementation plan must be aligned with actual codebase dynamics before execution:
1. `run_exp018_execution.py` does not contain a `train_model` function or serialize checkpoints. Checkpoint regeneration must directly import the core training building blocks (`ConventionalCNNBiGRU`, `create_infinite_batch_stream`, etc.), run the locked 3,200 steps, and call `torch.save`.
2. Calibration nights have a median length of 868 epochs ($N < 1150$ for $>98\%$ of nights); slicing into $L=1200$ windows with stride 300 is physically impossible. Each night is extracted in full, right-padded to $L=1200$, and batched as a $(B=2, L=1200)$ tensor.
3. A shared adaptation module `experiments/EXP-019/exp019_adaptation_core.py` must be added to house shared PyTorch adaptation functions.
4. Explicit JSON schemas must be documented for `exp019_stage_d_selected_configs.json` and `exp019_stage_c_results.json`.
5. Invalid/padded targets must be set to `-100` (`ignore_index` for PyTorch `CrossEntropyLoss`), not `-1`.

---

## 2. Synthesized Review Findings by Severity

### 🔴 Blockers (Codebase Parity & Physical Tensor Reality)

#### Finding 1: Checkpoint Regeneration Import & Serialization Parity
* **Reviewers:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** Line 265, Lines 295–300
* **Discrepancy:** The plan specifies importing `train_model` from `run_exp018_execution.py`. In reality, the function is `train_single_run`, and it only returns a dictionary of metrics without saving `.pt` checkpoints or returning the model.
* **Remediation:** Specify that `experiments/EXP-018/regenerate_modelA_checkpoints.py` directly coordinates training by importing the core components:
  ```python
  from exp013_models import ConventionalCNNBiGRU
  from exp018_dataset import COMPUTE_SCHEDULE_LOOKUP, RandomCropSleepDataset, create_infinite_batch_stream
  from exp018_inference import SlidingWindowPredictor
  from run_exp018_execution import evaluate_records
  ```
  It will run the locked 3,200-step training loop on `inner_train_subjects`, assert metric parity against historical per-fold artifacts (`run_L1200_fold{fold}_seed{seed}.json`) within $\pm 0.5\text{ pp}$, and explicitly call `torch.save(model.state_dict(), checkpoint_path)`.

#### Finding 2: Calibration Night Tensor Batching vs. Night Length Reality
* **Reviewers:** `egm-codebase-details`
* **Affected Scope:** Line 179, Line 268, Line 304
* **Discrepancy:** The plan stated that calibration nights are sliced into sliding windows of length $L=1200$ with stride 300 to provide multiple mini-batch updates per pass. In `night_inventory.csv`, the median night length is 868 epochs, and over 98% of nights have $N < 1150$. Slicing an 868-epoch night with window 1200 is impossible.
* **Remediation:** Clarify the physical tensor batching invariant:
  Each calibration night is extracted in full and right-padded to $L=1200$ (with valid length passed to `pack_padded_sequence` and padded targets masked to `-100`). The $k=2$ calibration nights of a subject form a mini-batch of shape $(B=2, L=1200, \dots)$. One training epoch over calibration nights consists of exactly one gradient step over this 2-night batch. For 5 (or 15) epochs, exactly 5 (or 15) optimizer steps are executed.

---

### 🟡 Majors (Actionability, Schemas & Testing)

#### Finding 3: Missing Shared Adaptation Module (`exp019_adaptation_core.py`)
* **Reviewers:** `egm-implementation-readiness`
* **Affected Scope:** Lines 287–357
* **Discrepancy:** Stage D, Stage C, and invariant tests require identical PyTorch adaptation functions (fine-tuning loop, InfoMax unsupervised loop, scratch training, wearable-only quality mask heuristic, and state neutralization). The plan lacked a shared core module, risking code duplication between scripts.
* **Remediation:** Add `experiments/EXP-019/exp019_adaptation_core.py` to the plan, encapsulating:
  - `adapt_supervised_ft`
  - `adapt_unsupervised_infomax`
  - `train_per_subject_scratch`
  - `compute_wearable_quality_mask`
  - `reset_model_weights`

#### Finding 4: Explicit JSON Schemas for Inter-Script Contracts
* **Reviewers:** `egm-implementation-readiness`
* **Affected Scope:** Lines 305, 312, 330, 337
* **Discrepancy:** Undefined schemas for `exp019_stage_d_selected_configs.json` and `exp019_stage_c_results.json` risk deserialization failures between Stage D and Stage C.
* **Remediation:** Document the exact JSON schemas for both files in the implementation plan.

#### Finding 5: Target Masking Value (`-100` vs `-1`)
* **Reviewers:** `egm-codebase-details`
* **Affected Scope:** Lines 344, 375
* **Discrepancy:** Padded positions and unknown stages (label 5) must be set to `-100` (`ignore_index` for PyTorch `CrossEntropyLoss`). Passing `-1` triggers an out-of-bounds `IndexError` in PyTorch.
* **Remediation:** Explicitly specify `-100` as the masked target value for invalid/padded positions.

#### Finding 6: Missing `cache_dir` in Script Signatures
* **Reviewers:** `egm-codebase-details`
* **Affected Scope:** Lines 308–313, Lines 332–338
* **Discrepancy:** Functions must accept `cache_dir: str` (`/home/rj2826/.../cache/prep_57738d534b`) and `manifest_dir: str` to load preprocessed feature arrays.
* **Remediation:** Update function signatures to include `manifest_dir` and `cache_dir`.

#### Finding 7: Ambiguous Float Rounding in Stage D Tie-Breaking
* **Reviewers:** `egm-implementation-readiness`
* **Affected Scope:** Line 178, Line 304
* **Discrepancy:** 16-decimal float differences prevent tie-breaker rules from ever triggering.
* **Remediation:** Specify that inner-validation Macro-F1 is rounded to 4 decimal places (`round(macro_f1, 4)`, i.e. $0.01\text{ pp}$) before applying the tie-breaker hierarchy.

#### Finding 8: Stage C Dry-Run Test Isolation
* **Reviewers:** `egm-implementation-readiness`
* **Affected Scope:** Line 383
* **Discrepancy:** Running Stage C `--dry-run` on outer test subjects risks test leakage prior to preregistration lock.
* **Remediation:** Mandate that `--dry-run` operates strictly on synthetic tensors or inner-validation subjects, never outer test data.

---

### 🟢 Nits (Codebase Polish & Style)

* **Finding 9 (egm-codebase-details):** Formally define $\mathcal{H}_{\text{prior}} = \ln(4) \approx 1.3863\text{ nats}$ for the $UA\text{-}2$ abort threshold ($0.5 \cdot 1.3863 = 0.6931\text{ nats}$).
* **Finding 10 (egm-style):** Provide concrete bash commands for the dry-run execution step.
* **Finding 11 (egm-style):** Replace hyperbolic terms ("engine" $\to$ "module/script", "guarantee 100% protocol fidelity" $\to$ "ensure protocol parity").
* **Finding 12 (egm-codebase-details):** Add `experiments/EXP-018/regenerate_modelA_checkpoints.py` to the Git staging command.
* **Finding 13 (egm-implementation-readiness):** Insert `Step 2b: Unit & Invariant Suite Execution` into the phase sequence diagram.
