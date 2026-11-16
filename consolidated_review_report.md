# Consolidated Review Report: `design_draft.md` (EXP-019 Adaptation Specification)

* **Target Document:** [`design_draft.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/design_draft.md)
* **Author:** jbjoyk
* **Audit Execution Mode:** Consolidated Report / Uninterrupted Full Flow (Stages 5 through 9)
* **Date:** 2026-11-16
* **Final Consolidated Verdict:** **APPROVED (100% IMPLEMENTATION READINESS VERIFIED)**

---

## 1. Executive Summary

This consolidated report summarizes the multi-stage adversarial review and verification of [`design_draft.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/design_draft.md), which specifies the protocol, architecture, and implementation plan for experiment **EXP-019** (testing hypothesis **H-A**: within-person multi-night sleep staging adaptation on the BIDSleep dataset).

All five specialized review stages were executed in sequence:
1. **Stage 5 (Initial Zoom Out):** Verified high-altitude architecture, root problem focus, and YAGNI compliance.
2. **Stage 6 (Comprehension Test):** Paraphrased feature specs and implementation plans with zero institutional knowledge, identifying clarity gaps and terminology ambiguities.
3. **Stage 7 (Critic Review):** Conducted hostile architectural critique, structural layout validation, and style analysis, uncovering critical masking and mode collapse risks in Arm 5 ($UA\text{-}2$).
4. **Stage 8 (Intermediate Zoom Out):** Re-audited post-remediation architecture to ensure no complexity bloat or weeds were introduced.
5. **Stage 9 (Implementation Readiness Test):** Audited implementation details against codebase ground truth, eliminating non-existent symbol imports, aligning batching with actual night lengths, adding shared adaptation libraries, and formalizing inter-script JSON contracts.

Every identified finding has been resolved directly within `design_draft.md` and cross-verified against repository ground truth. The specification is cleared for implementation.

---

## 2. Cross-Stage Traceability Matrix

The matrix below maps all findings across the five review stages to their final resolution status in `design_draft.md`:

| ID | Originating Stage & Reviewer | Issue / Finding Description | Initial Severity | Final Status | Applied Resolution in `design_draft.md` |
|---|---|---|---|---|---|
| **TR-01** | Stage 7 (`egm-design`) | **$UA\text{-}2$ Masking Paradox**: Unsupervised test-time adaptation cannot use EEG headband signal quality masks without leaking labels. | 🔴 Blocker | **RESOLVED** | Formulated an explicit wearable-only quality heuristic ($30 \le \text{HR} \le 180\text{ bpm}$ and non-zero accelerometry variance), preserving the true label-free wearable setting. (Lines 144–146) |
| **TR-02** | Stage 7 (`egm-design`) | **$UA\text{-}2$ Mode Collapse**: Unconstrained Shannon entropy minimization collapses to the dominant Light-sleep class. | 🔴 Blocker | **RESOLVED** | Added InfoMax diversity regularization ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ent}} - \lambda \mathcal{H}(\bar{p})$ with $\lambda=1.0$) and an automated abort check if entropy drops below $0.5 \cdot \mathcal{H}_{\text{prior}} = 0.6931\text{ nats}$. (Lines 143–147) |
| **TR-03** | Stage 7 (`egm-design`) | **Compound Decision Power Fallacy**: Compound rule ($\hat{\Delta} \ge 1.5$ AND CI $> 0$) has $50\%$ power at $\delta=1.5\text{ pp}$; discontinuous threshold escalation. | 🔴 Blocker | **RESOLVED** | Clarified power: detecting $\hat{\Delta} \ge 1.5\text{ pp}$ with $80\%$ power requires true effect $\delta \ge 1.95\text{ pp}$. Replaced step-function with continuous smooth rule: $\text{Threshold} = \max(1.5, 0.4131 \cdot s_{\Delta})$ for $N=46$. (Lines 28, 58) |
| **TR-04** | Stage 9 (`egm-readiness`, `egm-codebase`) | **Non-Existent Symbol & Serialization in EXP-018**: `train_model` does not exist in `run_exp018_execution.py`, and `train_single_run` does not save `.pt` files. | 🔴 Blocker | **RESOLVED** | Replaced with direct coordination of `ConventionalCNNBiGRU`, `create_infinite_batch_stream`, etc., to run the 3,200-step training loop, assert metric parity ($\pm 0.5\text{ pp}$), and serialize `.pt` checkpoints. (Lines 294–302) |
| **TR-05** | Stage 9 (`egm-codebase`) | **Physical Tensor Batching Invariant**: Median night length is 868 epochs ($< 1150$ for $>98\%$ of nights); stride-300 slicing is physically impossible. | 🔴 Blocker | **RESOLVED** | Clarified tensor reality: full nights right-padded to $L=1200$ (padded targets $=-100$) stacked into a $(B=2, L=1200)$ batch. One epoch equals one gradient step over this batch. (Lines 179–181) |
| **TR-06** | Stage 5 (`egm-zoom-out`) | **Filesystem Namespace Collision**: Existing exploratory files in `experiments/EXP-019/` collide with new confirmatory scripts. | 🟡 Major | **RESOLVED** | Added Step 0 pre-execution cleanup migrating historical files to `experiments/EXP-019-EXPLORATORY/`. (Lines 261–263, 289–293) |
| **TR-07** | Stage 6 (`egm-comprehension`) | **Missing Target File in Plan Inventory**: `tests/test_exp019_invariants.py` was relied upon in verification but omitted from the file catalog. | 🟡 Major | **RESOLVED** | Added Target File #7 (`tests/test_exp019_invariants.py`) specifying its automated invariant test assertions. (Lines 342–350) |
| **TR-08** | Stage 6 (`egm-comprehension`) | **Dual Usage of "Epoch"**: Ambiguity between 30-second physiological scoring windows and optimization gradient epochs. | 🟡 Major | **RESOLVED** | Disambiguated terminology: explicitly uses "30-second sleep scoring epochs" for time-series windows and "training epochs / steps" for model passes. (Lines 28, 98, 171, 180) |
| **TR-09** | Stage 9 (`egm-readiness`) | **Missing Shared Adaptation Library**: Duplication risk of adaptation loops between Stage D and Stage C. | 🟡 Major | **RESOLVED** | Created Target File #3 (`experiments/EXP-019/exp019_adaptation_core.py`) encapsulating all shared adaptation algorithms and functions. (Lines 304–326) |
| **TR-10** | Stage 9 (`egm-readiness`) | **Missing JSON Contracts**: Undefined schemas for `selected_configs.json` and `stage_c_results.json` risking runtime deserialization bugs. | 🟡 Major | **RESOLVED** | Documented complete, explicit JSON contracts for both files in the specification. (Lines 314–325, 338–353) |
| **TR-11** | Stage 9 (`egm-codebase`) | **Target Masking Discrepancy**: Padded positions and unknown stages were labeled `-1` instead of `-100` (`ignore_index`). | 🟡 Major | **RESOLVED** | Explicitly specified `-100` as the masked target value for invalid/padded positions. (Lines 56, 100, 180) |
| **TR-12** | Stage 9 (`egm-readiness`) | **Float Precision in Stage D Tie-Breaking**: 16-decimal float noise prevents tie-breaker execution. | 🟡 Major | **RESOLVED** | Formalized 4-decimal-place rounding (`round(macro_f1, 4)`) prior to applying the tie-breaker hierarchy. (Lines 68, 178) |
| **TR-13** | Stage 9 (`egm-readiness`) | **Stage C Dry-Run Test Isolation**: Running Stage C dry-run on outer test subjects risks test leakage. | 🟡 Major | **RESOLVED** | Mandated that `--dry-run` operates strictly on mock/synthetic or inner-validation data, asserting outer test records are never accessed. (Lines 336, 381–386) |
| **TR-14** | Stage 7 (`egm-style`) | **Confirmation Bias in Research Goals**: Phrasing goals as conclusions to be confirmed rather than empirical hypotheses. | 🟡 Major | **RESOLVED** | Rephrased Goals items 2 and 3 from "Confirm that..." to neutral testing language: "Test whether...". (Lines 29–30) |
| **TR-15** | Stage 7 (`egm-structural`) | **Markdown Linting Violations**: Missing blank lines before lists across 12 heading sites (MD022/MD032). | 🟡 Major | **RESOLVED** | Inserted blank newlines between all `###` headings and adjoining bullet lists. (Lines 64, 69, 181, 186, 191, etc.) |
| **TR-16** | Stage 7 & 9 (`egm-style`) | **Colloquialisms & Promotional Words**: Use of informal idioms ("accuracy swings widely", "engine"). | 🟢 Nit | **RESOLVED** | Replaced with dry, descriptive technical prose. (Throughout document) |

---

## 3. High-Level Summary by Review Stage

### Stage 5: Initial Zoom Out Review
* **Verdict:** CONDITIONAL PASS $\to$ APPROVED
* **Key Observations:** Verified that the document addresses the primary root problem (between-subject autonomic variance in wearable sleep staging) through within-person multi-night adaptation. Strongly praised the architectural freeze (Model A, 163,940 parameters; Amendment A6) and the empirical negative control ($FT\text{-}other\text{-}2$). Identified the Step 0 directory cleanup requirement and EXP-018 training routine reuse.

### Stage 6: Comprehension Test
* **Verdict:** PASS WITH MINOR ACTION ITEMS $\to$ APPROVED
* **Key Observations:** The Naive Reader successfully paraphrased the high-level research problem and the 6-step implementation sequence without misinterpretation. Caught the omission of `tests/test_exp019_invariants.py` from the file inventory and disambiguated the dual usage of "epoch".

### Stage 7: Critic Review
* **Verdict:** ACTION REQUIRED $\to$ APPROVED
* **Key Observations:** Hostile review identified three major blocking vulnerabilities: the $UA\text{-}2$ masking paradox, Shannon entropy mode collapse, and compound decision power under $N=46$. Structural review validated the heading hierarchy, tables, and Mermaid syntax, while style review neutralized hypothesis confirmation bias. All 16 findings were remediated.

### Stage 8: Intermediate Zoom Out Review
* **Verdict:** UNCONDITIONAL PASS (APPROVED FOR STAGE 9)
* **Key Observations:** Confirmed that the Stage 7 mitigations (InfoMax diversity regularization, physical wearable quality boundaries, smooth MDE threshold formula, and directory isolation) were minimal, robust, and free of architectural bloat or YAGNI violations.

### Stage 9: Implementation Readiness Test
* **Verdict:** CONDITIONAL PASS $\to$ APPROVED
* **Key Observations:** Pragmatic Tech Lead and Codebase Auditor verified code alignment against the active repository. Aligned training loop coordination, physical $(B=2, L=1200)$ tensor batching, shared adaptation library (`exp019_adaptation_core.py`), explicit JSON schemas, and test isolation.

---

## 4. Final Implementation Clearance

The design document [`design_draft.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/design_draft.md) (and its mirror [`docs/designs/exp019_adaptation.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/docs/designs/exp019_adaptation.md)) is **100% verified, actionable, unambiguous, and approved for implementation**.

### Immediate Execution Sequence:
1. **Step 0**: Execute workspace cleanup:
   ```bash
   mkdir -p experiments/EXP-019-EXPLORATORY
   mv experiments/EXP-019/* experiments/EXP-019-EXPLORATORY/ 2>/dev/null || true
   ```
2. **Step 1**: Author and run `experiments/EXP-018/regenerate_modelA_checkpoints.py` to regenerate the 15 Model A checkpoints with verified metrics.
3. **Step 2**: Author `experiments/EXP-019/exp019_adaptation_core.py` and `experiments/EXP-019/run_exp019_stage_d.py`.
4. **Step 2b**: Author and run `tests/test_exp019_invariants.py` (`pytest -v`).
5. **Step 3**: Execute Stage D nested selection across the 25 inner-validation subjects.
6. **Step 4**: Commit locked preregistration in `PREREG_EXP-019_H-A.md` to `origin main`.
7. **Step 5**: Execute Stage C confirmatory evaluation on the 46 outer test subjects across all 5 arms.
