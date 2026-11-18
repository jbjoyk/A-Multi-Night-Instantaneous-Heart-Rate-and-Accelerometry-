# Handoff Report: Benchmark Targets, Hardware Verification & 5-Fold Cross-Validation Parallelization

**Agent**: `teamwork_explorer_r2_3` (Benchmark & Hardware Explorer)  
**Recipient**: Parent Orchestrator (`5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91`)  
**Timestamp**: 2026-11-15T05:29:00Z  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r2_3`

---

## 1. Observation

1. **Hardware & GPU Environment**:
   - `nvidia-smi` command output:
     - Driver Version `595.99.02`, CUDA Driver `13.2`.
     - GPU 0: `NVIDIA GeForce RTX 3060`, Bus ID `00000000:01:00.0`, Display `On`, Memory: `1074MiB / 12288MiB` used (~11,214 MiB free). Active processes: `/usr/bin/gnome-shell` (203 MiB), Xwayland, and desktop helpers.
     - GPU 1: `NVIDIA GeForce RTX 3060`, Bus ID `00000000:06:00.0`, Display `Off`, Memory: `13MiB / 12288MiB` used (~12,275 MiB free).
   - PyTorch CUDA properties via `./.venv/bin/python`:
     ```text
     PyTorch Version: 2.14.1+cu130
     CUDA Available: True
     Device Count: 2
     Device 0: NVIDIA GeForce RTX 3060, Capability: 8.6, Total Memory: 11902.8 MB, MultiProcessor Count: 28
     Device 1: NVIDIA GeForce RTX 3060, Capability: 8.6, Total Memory: 11911.8 MB, MultiProcessor Count: 28
     ```
   - Python packages via `./.venv/bin/python`: Python `3.14.7`, NumPy `2.5.3`, SciPy `1.18.1`, Scikit-Learn `1.9.1`.

2. **Published Paper Benchmark Target**:
   - `ORIGINAL_REQUEST.md:32`: `"- Published Paper Target: 71.04% Accuracy, 0.7079 Weighted-F1 (Table V in published SLAMSS-IFS paper)"`.
   - `DECISIONS.md:193`: `"Published SLAMSS-IFS (IEEE TBME 2026, Table V): Accuracy = 71.04%, Weighted-F1 = 0.7079, Parameters = 5,465,768."`
   - `experiments/EXP-012/exp012_report.html:1010-1025`: Table V reports Overall Test Accuracy = `71.04%`, Weighted-F1 = `0.7079`, parameters = `5,465,768` (official repository model) / `666,060` (paper text), and Wake Recall = `69.17%`.

3. **Historical Baseline Performance & Deficit**:
   - In EXP-014 (`experiments/EXP-014/EXP014_SUMMARY.csv:2`, `DECISIONS.md:227`):
     - Model A (`ConventionalCNNBiGRU`, 163,940 params): Mean Accuracy = **`66.63% ± 3.33%`**, Mean Weighted-F1 = **`0.6654 ± 0.0339`**, Mean Macro-F1 = `0.6658 ± 0.0365`, Cohen's $\kappa = 0.5251 \pm 0.0505$.
     - Deficit to close: **+4.41 percentage points in Accuracy** and **+0.0425 in Weighted-F1**.
   - In EXP-019 (`experiments/EXP-019/exp019_results/exp019_summary.json:12-26`):
     - MultiScale Model A (`MultiScaleConventionalCNNBiGRU`, 192,692 params, +17.54% capacity) on Fold 1 outer test at best validation checkpoint: Accuracy = `65.72%` (+1.30 pp over baseline `64.42%`), Macro-F1 = `0.6513` (+1.11 pp over baseline `0.6401`). Post-peak overfitting drop was only 0.62 pp vs 5.02 pp for baseline.

4. **Cross-Validation Partitioning & Cohort Scope**:
   - `experiments/EXP-014/EXP014_MANIFESTS/fold_{1..5}.json`:
     - Disjoint outer-test partitions: Fold 1 (10 subjects, 54 nights, 32,098 windows), Fold 2 (10 subjects, 53 nights, 31,291 windows), Fold 3 (9 subjects, 45 nights, 26,599 windows), Fold 4 (9 subjects, 47 nights, 27,630 windows), Fold 5 (9 subjects, 47 nights, 27,665 windows).
     - Sum: Exactly 47 subjects, 246 usable nights, 145,283 valid 30-second windows.
   - Preprocessed feature cache at `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b` contains exactly 246 `.npz` files (one per usable night).

5. **Empirical Runtimes**:
   - EXP-014 (`experiments/EXP-014/EXP014_FINAL_REPORT.md:54`): 75 runs of 80 epochs took 5,745 seconds on 1 GPU $\implies$ 76.60 seconds per run ($\mathbf{0.958\text{ s/epoch}}$).
   - EXP-019 (`experiments/EXP-019/exp019_results/baseline_500epochs_results.json:8`): Baseline 500 epochs with validation took 954.53 seconds on `cuda:0` ($\mathbf{1.909\text{ s/epoch}}$). MultiScale took 1007.52 seconds on `cuda:1` ($\mathbf{2.015\text{ s/epoch}}$).
   - EXP-014 NAS Iteration 1 (`experiments/EXP-014/evaluate_nas_iteration1.py`): 1 epoch on Fold 1 inner-train (42 batches of 4 nights) took ~1.2s; test evaluation (14 batches) took ~1.5s.

6. **Victory Conditions & Mandatory Outputs**:
   - `ORIGINAL_REQUEST.md:26-28, 47-50`:
     - Aggregated 5-fold CV Accuracy > 71.04% AND Weighted-F1 > 0.7079.
     - Final PDF-ready markdown report.
     - Git commit and push all code and results to `origin main`.
     - Terminal message: `"VICTORY CONFIRMED: SLAMSS-IFS BENCHMARK BEATEN. CONTINUOUS LOOP TERMINATED."`

---

## 2. Logic Chain

1. **Hardware Suitability**: From Observation 1, two RTX 3060 12GB GPUs (`cuda:0` and `cuda:1`) with Compute Capability 8.6 are online with ~11.2 GB and ~12.3 GB of free VRAM. Since candidate sleep staging models with batch size 4 and $T=600$ consume <1.8 GB VRAM per training process, each GPU has ample headroom to execute an independent training run without memory pressure or resource starvation.
2. **Benchmark Deficit Analysis**: From Observation 2 and Observation 3, the published gold standard requires > 71.04% Accuracy and > 0.7079 Weighted-F1 across 47 subjects in 5-fold CV. Baseline Model A achieved 66.63% Accuracy and 0.6654 Weighted-F1. Therefore, architectural mutations cannot rely on minor capacity increases alone (as proved in EXP-014 where 5-fold CV equivalence persisted across ~160k models). Mutations must introduce substantive cross-modal synergy (Hypothesis H-F) and subject-level adaptive invariance (Hypothesis H-A) to close the +4.41 pp Accuracy and +0.0425 Weighted-F1 gap.
3. **Dual-GPU Workload Parallelization**: From Observation 4, 5-fold cross-validation with 3 random seeds requires $5 \times 3 = 15$ runs. Running sequentially on 1 GPU requires ~26.5 minutes. Assigning the 15 runs across `cuda:0` and `cuda:1` via a dynamic task queue splits the load (~7 runs on GPU 0, ~8 runs on GPU 1), reducing wall-clock runtime to **~14.4 to 16.0 minutes**.
4. **Collision and Race Condition Prevention**:
   - If processes share a PyTorch CUDA context or use `fork`, deadlocks occur. Therefore, independent subprocesses must be launched using `sys.executable` with explicit `--device cuda:0` and `--device cuda:1` flags.
   - If processes write to a shared results file simultaneously, corruption occurs. Therefore, each run must write to a unique file: `results_fold_{fold}_seed_{seed}.tmp.json` followed by an atomic `os.replace` to `results_fold_{fold}_seed_{seed}.json`.
   - The feature cache `/home/rj2826/ai-research/sleep/.../prep_57738d534b` is read-only, allowing safe concurrent multi-process access without file locking.
5. **Proxy Gating Efficiency**: From Observation 5, running 50 epochs on Fold 1 (seed 42) on `cuda:0` takes only ~60 to 75 seconds. Enforcing the proxy gate ($\text{Accuracy} \ge 70.0\%$, $\text{Weighted-F1} \ge 0.6900$) filters non-viable candidates in ~1.1 minutes, saving ~15 minutes of full CV compute on negative mutations.
6. **Victory Fulfillment**: From Observation 6, achieving aggregated Accuracy > 71.04% and Weighted-F1 > 0.7079 satisfies the R4 requirements, triggering the generation of the PDF-ready report, synchronization via `git push origin main`, and emission of the termination token.

---

## 3. Caveats

1. **GPU 0 Desktop Overhead**: GPU 0 is currently attached to the Xwayland/GNOME display compositor (~1,074 MiB VRAM used). While this leaves >11 GB free, minor display rendering tasks may cause GPU 0 to execute ~2–4% slower than headless GPU 1. The dynamic work queue mitigates this disparity.
2. **Proxy vs. Full CV Generalization Correlation**: Fold 1 in EXP-014 had lower baseline accuracy (63.31%) than Fold 3 (71.79%) and Fold 5 (68.51%). A proxy threshold of 70.0% Accuracy on Fold 1 is stringent and ensures only high-potential candidate architectures proceed to full 5-fold CV.
3. **Circadian Clock Feature Alignment**: All 5-fold CV runs must maintain invariant feature formatting (IHR, accelerometry, 5 metadata/circadian features, batch size 4, Adam lr=1e-4, seq_len=600) to ensure strictly comparable attribution.

---

## 4. Conclusion

1. **Hardware Configuration**: Dual NVIDIA RTX 3060 12GB GPUs (`cuda:0` and `cuda:1`) are fully verified and operational under `./.venv/bin/python` (PyTorch 2.14.1+cu130, Python 3.14.7).
2. **Benchmark Target**: The definitive target is Table V of Song et al. (IEEE TBME 2026): **Accuracy > 71.04%** and **Weighted-F1 > 0.7079** on 5-fold CV across all 47 subjects (145,283 valid windows), surpassing the 5.47M parameter published model while maintaining a >90% compression ratio (~162k–196k parameter budget).
3. **Execution Plan**:
   - Fast Proxy Evaluation: Fold 1, 50 epochs, seed 42 on `cuda:0` (~70s). Filter threshold: Accuracy $\ge$ 70.0% AND Weighted-F1 $\ge$ 0.69.
   - Full 5-Fold Cross-Validation: 15 runs (5 folds $\times$ 3 seeds, 80 epochs) parallelized via dynamic queue across `cuda:0` and `cuda:1` (~15 min wall-clock time).
   - Victory Sequence: Upon exceeding targets, generate PDF-ready Markdown report, commit and push to Git (`origin main`), and output the exact termination phrase.

---

## 5. Verification Method

To independently verify all findings in this report, execute the following commands in the workspace root:

1. **Verify GPU Telemetry & PyTorch Devices**:
   ```bash
   ./.venv/bin/python -c "
   import torch
   print('Device count:', torch.cuda.device_count())
   for i in range(torch.cuda.device_count()):
       p = torch.cuda.get_device_properties(i)
       print(f'cuda:{i}: {p.name}, {p.total_memory / (1024**2):.1f} MB, CC {p.major}.{p.minor}')
   "
   ```
   *Expected Output*: 2 devices detected (`NVIDIA GeForce RTX 3060`, ~11,900 MB each, CC 8.6).

2. **Verify 5-Fold Disjoint Subject Partitions (47 Subjects)**:
   ```bash
   ./.venv/bin/python -c "
   import json
   subs = []
   for f in range(1, 6):
       with open(f'experiments/EXP-014/EXP014_MANIFESTS/fold_{f}.json') as fp:
           subs.extend(json.load(fp)['outer_test_subjects'])
   assert len(subs) == 47 and len(set(subs)) == 47
   print('5-fold manifests verified: 47 unique disjoint subjects.')
   "
   ```

3. **Verify Preprocessed Feature Cache**:
   ```bash
   ls /home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b/*.npz | wc -l
   ```
   *Expected Output*: `246` (246 usable nights).

4. **Verify Table V Published Numbers & Baseline Deficit**:
   ```bash
   grep -n "71.04%" DECISIONS.md
   head -n 2 experiments/EXP-014/EXP014_SUMMARY.csv
   ```
   *Expected Output*: SLAMSS-IFS Table V reports 71.04% Accuracy and 0.7079 Weighted-F1; EXP-014 Model A baseline reports 66.63% Accuracy and 0.6654 Weighted-F1.

### Invalidation Conditions:
- If fewer than 2 CUDA devices are detected by PyTorch, dual-GPU parallelization is invalidated.
- If outer test subjects overlap across fold manifests, 5-fold cross-validation isolation is violated.
- If feature cache files are missing or unreadable, evaluation cannot proceed.
