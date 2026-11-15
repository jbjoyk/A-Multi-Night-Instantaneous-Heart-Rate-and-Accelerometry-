# Original User Request

## Initial Request — 2026-11-15T05:18:37Z

Act as an autonomous Neural Architecture Search (NAS) syndicate to continuously iterate architectural mutations until definitively beating the published SLAMSS-IFS paper model on its primary metrics and resolving hypotheses H-F and H-A on the BIDSleep dataset.

Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels
Integrity mode: development

## Requirements

### R1. Continuous Structural Mutation and Architecture Search Loop
Execute a continuous neural architecture search loop starting at Iteration 2 targeting Hypothesis H-F (cross-modal IHR-accelerometry interaction module) and Hypothesis H-A (adaptation/architecture with statistically significant gain across all 47 subjects). For each iteration:
1. Propose one new structural mutation targeting H-F and H-A.
2. Implement the tensor-safe PyTorch `nn.Module` with verified dimension alignment.
3. Execute fast proxy evaluation on Fold 1 for 50 epochs (single seed 42) on `cuda:0`; if Proxy Accuracy < 70.0% or Proxy Weighted-F1 < 0.69, revert the mutation, log the failure, and proceed to the next iteration.
4. If proxy passes, execute full 5-fold cross-validation across all 47 subjects (80 epochs, 3 seeds, outer test evaluation); if aggregated 5-fold Accuracy > 71.04% and Weighted-F1 > 0.7079, proceed to the victory condition, otherwise revert the mutation, log the failure, and continue the loop.
The loop runs continuously without an iteration cap until the victory condition is met or a terminal error occurs.

### R2. Architectural Budget and Parameter Count
Candidate architectures should remain as close to the 162k–196k parameter range as possible to maintain a >90% compression ratio against the 5.47M parameter paper model. Exceeding 196k parameters is permitted as a soft constraint if the architecture meets or exceeds the target performance metrics. Structural modifications must focus on architecture; training batch size, optimizer, and base learning rate must remain consistent across evaluations.

### R3. Evaluation and Hardware Infrastructure
Execute evaluations locally using the host Python virtualenv `./.venv/bin/python`. Run proxy evaluations on single GPU `cuda:0`. For 5-fold cross-validation, parallelize fold evaluations across both available GPUs (`cuda:0` and `cuda:1`). All evaluations must use the pre-computed feature cache and evaluation manifests.

### R4. Victory Condition Reporting and Synchronization
Upon achieving aggregated 5-fold Accuracy > 71.04% and Weighted-F1 > 0.7079, format a final PDF-ready markdown report declaring the SLAMSS-IFS paper benchmark beaten, commit all code and JSON results to Git, and output: "VICTORY CONFIRMED: SLAMSS-IFS BENCHMARK BEATEN. CONTINUOUS LOOP TERMINATED."

## Verification Resources
- Baseline Model A: `ConventionalCNNBiGRU` (163,940 params in `exp013_models.py`)
- Iteration 1 Model: `MultiScaleConventionalCNNBiGRU` (192,692 params in `experiments/EXP-014/nas_iteration1_model.py`)
- Published Paper Target: 71.04% Accuracy, 0.7079 Weighted-F1 (Table V in published SLAMSS-IFS paper)
- Dataset & Manifests: BIDSleep dataset, 5-fold manifests (`experiments/EXP-014/EXP014_MANIFESTS/fold_{1..5}.json`), feature cache `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b`
- Evaluation Utilities: `evaluator.py` (`SleepEvaluator`), `experiments/EXP-019/run_exp019_500epochs.py`, `experiments/EXP-014/evaluate_nas_iteration1.py`

## Acceptance Criteria

### Proxy Evaluation Filtering
- [ ] Proposed mutation implements a tensor-safe `nn.Module` targeting hypothesis H-F or H-A.
- [ ] Fast proxy evaluation completes on Fold 1 for 50 epochs (seed 42) on `cuda:0` via `./.venv/bin/python`.
- [ ] Candidate mutations with Proxy Accuracy < 70.0% or Proxy Weighted-F1 < 0.69 are reverted and logged.

### Definitive Cohort Evaluation
- [ ] Candidate mutations passing proxy filtering execute 5-fold cross-validation across all 47 subjects (80 epochs, 3 seeds) utilizing `cuda:0` and `cuda:1`.
- [ ] Candidate mutations failing the cohort threshold (Accuracy <= 71.04% or Weighted-F1 <= 0.7079) are reverted and logged, returning to propose a new mutation.

### Victory Condition
- [ ] Aggregated 5-fold cross-validation achieves Accuracy > 71.04% and Weighted-F1 > 0.7079.
- [ ] Final PDF-ready markdown report is generated detailing the winning architecture, ablation findings for H-F and H-A, and comparison to the SLAMSS-IFS benchmark.
- [ ] All code, configs, and JSON metrics are committed to Git, and the terminal message "VICTORY CONFIRMED: SLAMSS-IFS BENCHMARK BEATEN. CONTINUOUS LOOP TERMINATED." is output.
