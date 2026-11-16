# Goldfish Review Resolution Report: Stage 9 Implementation Readiness Remediation

* **Target Document:** [`docs/designs/exp019_adaptation.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/docs/designs/exp019_adaptation.md)
* **Resolver:** Primary Orchestrator (Elephant context)
* **Date:** 2026-11-16
* **Scope:** Findings from `implementation_readiness_report.md` (Stage 9 Implementation Readiness Test)

---

## Findings Disposition & Resolution Table

| # | Finding Description | Severity | Disposition | Resolution / Applied Modification |
|---|---|---|---|---|
| 1 | **Non-Existent `train_model` Symbol & Serialization in EXP-018** | **Blocker** | **PROPOSE FIX** | Clarified that `regenerate_modelA_checkpoints.py` directly coordinates training by importing core building blocks (`ConventionalCNNBiGRU`, `create_infinite_batch_stream`, `RandomCropSleepDataset`, `SlidingWindowPredictor`, `evaluate_records`), executes the 3,200-step training loop, asserts metric parity against historical per-fold artifacts ($\pm 0.5\text{ pp}$), and explicitly serializes `.pt` checkpoints. |
| 2 | **Calibration Night Tensor Batching vs. Night Length Reality** | **Blocker** | **PROPOSE FIX** | Replaced invalid stride-300 slicing statement with physical tensor reality: median night length is 868 epochs ($< 1150$ for $>98\%$ of nights). Each calibration night is extracted in full, right-padded to $L=1200$ with native length passed to `pack_padded_sequence` and padded targets set to `-100`. The $k=2$ nights form a $(B=2, L=1200)$ batch; 1 epoch = 1 optimizer step. |
| 3 | **Missing Shared Adaptation Library Module** | **Major** | **PROPOSE FIX** | Added `experiments/EXP-019/exp019_adaptation_core.py` to the plan inventory, encapsulating shared functions for fine-tuning, unsupervised InfoMax, scratch training, wearable-only quality masking, and state neutralization. |
| 4 | **Missing JSON Schemas for Inter-Script Contracts** | **Major** | **PROPOSE FIX** | Explicitly documented the complete JSON schemas for `exp019_stage_d_selected_configs.json` and `exp019_stage_c_results.json`. |
| 5 | **Target Masking Discrepancy (`-100` vs `-1`)** | **Major** | **PROPOSE FIX** | Explicitly specified `-100` as the masked target value for invalid/padded positions (`ignore_index` for PyTorch `CrossEntropyLoss`). |
| 6 | **Missing `cache_dir` and `manifest_dir` in Signatures** | **Major** | **PROPOSE FIX** | Updated script signatures to accept `manifest_dir: str`, `cache_dir: str`, `checkpoint_dir: str`, and `device: str = "cuda:0"`. |
| 7 | **Ambiguous Float Rounding in Stage D Tie-Breaking** | **Major** | **PROPOSE FIX** | Specified that inner-validation Macro-F1 is rounded to 4 decimal places (`round(macro_f1, 4)`) prior to applying the tie-breaker hierarchy. |
| 8 | **Outer-Test Isolation in Stage C Dry-Run** | **Major** | **PROPOSE FIX** | Mandated that `--dry-run` operates strictly on synthetic tensors or inner-validation subjects, asserting outer test records are never touched. |
| 9 | **Prior Entropy Definition for $UA\text{-}2$ Safeguard** | **Nit** | **PROPOSE FIX** | Formally defined $\mathcal{H}_{\text{prior}} = \ln(4) \approx 1.3863\text{ nats}$, with the abort threshold set to $0.5 \cdot 1.3863 = 0.6931\text{ nats}$. |
| 10 | **Smoke Test Bash Commands** | **Nit** | **PROPOSE FIX** | Provided concrete copy-pasteable bash code blocks for the dry-run execution step. |
| 11 | **Style Polish (De-jargoning)** | **Nit** | **PROPOSE FIX** | Replaced "engine" with "module/pipeline" and "guarantee 100% protocol fidelity" with "ensure exact protocol parity". |
| 12 | **Git Staging List Completeness** | **Nit** | **PROPOSE FIX** | Added `experiments/EXP-018/regenerate_modelA_checkpoints.py` to the Git staging command. |
| 13 | **Phase Sequence Representation** | **Nit** | **PROPOSE FIX** | Inserted `Step 2b: Unit & Invariant Suite Execution` into the Mermaid phase diagram. |
