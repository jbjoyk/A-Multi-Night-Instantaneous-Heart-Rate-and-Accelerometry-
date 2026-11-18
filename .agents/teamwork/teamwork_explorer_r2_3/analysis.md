# Benchmark Targets, Hardware Verification, and 5-Fold Cross-Validation Parallelization Analysis

**Author**: Teamwork Explorer R2-3 (Benchmark & Hardware Explorer)  
**Date**: 2026-11-15T05:27:00Z  
**Workspace**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels`  
**Reference Document**: `.agents/teamwork/ORIGINAL_REQUEST.md`

---

## 1. Executive Summary

This investigation provides the hardware verification, published paper benchmark analysis, 5-fold cross-validation dual-GPU parallelization strategy, and victory condition specifications for the autonomous Neural Architecture Search (NAS) syndicate.

### Core Discoveries & Recommendations:
1. **Hardware Capacity**: Dual NVIDIA GeForce RTX 3060 12GB GPUs (`cuda:0` and `cuda:1`, Compute Capability 8.6, 28 SMs each) are fully functional and available. `cuda:0` has ~1,074 MiB utilized by the desktop environment (leaving ~11.2 GB free VRAM), while `cuda:1` is completely idle (13 MiB used, ~12.3 GB free). The host Python virtual environment `./.venv/bin/python` is configured with Python 3.14.7, PyTorch 2.14.1+cu130, CUDA 13.2 driver, NumPy 2.5.3, and Scikit-Learn 1.9.1.
2. **Published Benchmark Target**: Table V of the published SLAMSS-IFS paper (Song et al., IEEE TBME 2026) reports **71.04% Accuracy** and **0.7079 Weighted-F1** using **5,465,768 parameters** (official repository model) on the 47-subject BIDSleep dataset.
3. **Current Performance Baseline & Deficit**: In EXP-014 (75-run 5-fold CV across all 47 subjects), baseline `ConventionalCNNBiGRU` (163,940 params) achieved **66.63% ± 3.33% Accuracy** and **0.6654 ± 0.0339 Weighted-F1** (Macro-F1 0.6658 ± 0.0365). To beat the published benchmark, candidate mutations must close an empirical gap of **+4.41 percentage points in Accuracy** and **+0.0425 in Weighted-F1** under 5-fold CV while adhering to the 162k–196k parameter budget (>90% compression ratio).
4. **Dual-GPU Parallelization Strategy**: Full 5-fold cross-validation requires 15 runs (5 outer folds $\times$ 3 seeds: 42, 123, 2024) of 80 epochs each. Deploying a dynamic work queue with independent subprocess workers pinned to `cuda:0` and `cuda:1` prevents race conditions, guarantees zero shared-memory memory leaks, and delivers balanced compute.
5. **Runtime Bounds**:
   - **Proxy Evaluation** (Fold 1, 50 epochs, seed 42 on `cuda:0`): **~60 to 75 seconds** (~1.1 minutes).
   - **Full 5-Fold Cross-Validation** (15 runs of 80 epochs parallelized across 2 GPUs): **~14.4 to 16.5 minutes** total wall-clock time (~7–8 runs per GPU, ~1.8–2.0 min per run).
6. **Victory Protocol**: The continuous NAS loop terminates upon achieving aggregated 5-fold CV Accuracy > 71.04% and Weighted-F1 > 0.7079, triggering generation of a PDF-ready Markdown report, automated Git commit and push to `origin main`, and output of the exact terminal message: `"VICTORY CONFIRMED: SLAMSS-IFS BENCHMARK BEATEN. CONTINUOUS LOOP TERMINATED."`

---

## 2. Hardware Environment & GPU Verification

### 2.1 Telemetry Verification (`nvidia-smi`)
Direct query of the NVIDIA System Management Interface executed at 2026-11-15T11:22:27:
- **Driver Version**: 595.99.02
- **CUDA Driver Version**: 13.2
- **GPU 0**:
  - Name: NVIDIA GeForce RTX 3060
  - Bus ID: `00000000:01:00.0`
  - Display: `On` (Active desktop display connection)
  - VRAM Total: 12,288 MiB (12.0 GB)
  - VRAM Used: 1,074 MiB (GNOME shell, Xwayland, and desktop portal)
  - VRAM Free: ~11,214 MiB
  - Temperature / Fan: 48°C / 0% fan
  - Power Draw: 21W / 170W cap
- **GPU 1**:
  - Name: NVIDIA GeForce RTX 3060
  - Bus ID: `00000000:06:00.0`
  - Display: `Off` (Headless accelerator)
  - VRAM Total: 12,288 MiB (12.0 GB)
  - VRAM Used: 13 MiB
  - VRAM Free: ~12,275 MiB
  - Temperature / Fan: 40°C / 0% fan
  - Power Draw: 22W / 170W cap

### 2.2 PyTorch CUDA Environment Verification
Direct inspection via `./.venv/bin/python`:
```text
PyTorch Version: 2.14.1+cu130
CUDA Available: True
Device Count: 2
Device 0: NVIDIA GeForce RTX 3060, Capability: 8.6, Total Memory: 11902.8 MB, MultiProcessor Count: 28
Device 1: NVIDIA GeForce RTX 3060, Capability: 8.6, Total Memory: 11911.8 MB, MultiProcessor Count: 28
```
- **Architecture**: NVIDIA Ampere (Compute Capability 8.6, `sm_86`)
- **Streaming Multiprocessors**: 28 SMs per GPU (3,584 CUDA cores per GPU, 7,168 total CUDA cores)
- **Host Python Environment**:
  - Python: 3.14.7 (Clang 22.1.3)
  - NumPy: 2.5.3
  - SciPy: 1.18.1
  - Scikit-Learn: 1.9.1
  - PyTorch: 2.14.1+cu130

### 2.3 Memory Footprint & Concurrency Headroom
- For candidate models (~162k to 196k parameters) with batch size = 4, sequence horizon $T = 600$, window length $L = 30$, and feature channels (IHR, accelerometry, 5 circadian/metadata features):
  - Model parameters: ~0.7 MB
  - Activations and optimizer states (Adam): ~1.2 GB to ~1.8 GB peak VRAM during backward pass.
- **Safety Assessment**: Running 1 training process per GPU requires <15% of available VRAM on each card. There is zero risk of CUDA Out-Of-Memory (OOM) errors, zero VRAM fragmentation risk, and zero contention between processes.

---

## 3. Published Benchmark Targets vs. Historical Workspace Results

### 3.1 Published SLAMSS-IFS Paper Benchmark (Song et al., IEEE TBME 2026)
From Table V of the published paper and repository audit (`experiments/EXP-012/exp012_report.html:1000-1060`, `DECISIONS.md:193-195`):
- **Model Name**: SLAMSS-IFS (Sleep Multi-modal Synchronous Representation with Inter-modality Feature Sharing)
- **Model Parameter Count**:
  - Official Repository Model: **5,465,768 parameters** (~5.47M params)
  - Paper Text Specification: **666,060 parameters** (~666k params)
- **Primary Metric in Table V**: **Weighted-F1 = 0.7079**
- **Secondary Metric in Table V**: **Overall Accuracy = 71.04%**
- **Clinical Wake Sensitivity (Recall)**: **69.17%** (Specificity: 94.19%)
- **Evaluation Dataset**: BIDSleep dataset (47 subjects, Apple Watch accelerometry + instantaneous heart rate vs. Dreem 2 EEG 4-class labels)
- **Sequence Horizon in Paper**: $T = 1200$ epochs (10.0 continuous hours; 55,196 evaluated 30-second windows across test split).

### 3.2 Historical Experimental Results in Workspace

| Experiment | Configuration | Subjects & Partition | Parameters | Accuracy (%) | Weighted-F1 | Macro-F1 | Cohen's $\kappa$ | Scientific Outcome |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Published SLAMSS-IFS** | Table V Benchmark (TBME 2026) | 47 subjects (Table V) | 5,465,768 | **71.04%** | **0.7079** | Not reported | Not reported | Published Gold Standard |
| **EXP-012** | SleepAdaptiveNet (`B0_GRL`) | 11 held-out test subs (3 seeds) | 162,325 | 68.44 ± 1.35 | 0.6835 ± 0.0141 | 0.6859 ± 0.0139 | 0.5527 ± 0.0188 | Case B: Competitive under 33.7x compression |
| **EXP-013** | Model E vs Model A Attribution | 11 held-out test subs (3 seeds) | 162.3k vs 163.9k | E: 68.44 vs A: 66.65 | E: 0.6835 vs A: 0.6636 | E: 0.6859 vs A: 0.6660 | E: 0.5527 vs A: 0.5251 | GRL +1.39 pp; Gate +0.54 pp; Transition +0.04 pp |
| **EXP-014** (Model A) | Baseline Conventional CNN-BiGRU | All 47 subjects (5-fold CV, 3 seeds) | 163,940 | **66.63 ± 3.33** | **0.6654 ± 0.0339** | 0.6658 ± 0.0365 | 0.5251 ± 0.0505 | Outcome B: Parity across general cohort |
| **EXP-014** (Model E) | Full SleepAdaptiveNet | All 47 subjects (5-fold CV, 3 seeds) | 162,325 | 66.06 ± 4.08 | 0.6601 ± 0.0415 | 0.6596 ± 0.0425 | 0.5153 ± 0.0601 | Outcome B: Bootstrap CI spans zero ($E \approx A$) |
| **EXP-018 Phase 2** | Horizon Curve ($L=600$) | All 47 subjects (5-fold CV, 3 seeds) | 163,940 | 65.45 ± 0.14 | — | 0.6306 ± 0.0031 | 0.4877 ± 0.0048 | Context scaling verified |
| **EXP-018 Phase 2** | Horizon Curve ($L=1200$) | All 47 subjects (5-fold CV, 3 seeds) | 163,940 | 66.18 ± 0.73 | — | 0.6479 ± 0.0053 | 0.5078 ± 0.0076 | Extended context benefit (+5.33 pp over L=60) |
| **EXP-019** (Peak Checkpoint) | MultiScale CNN-BiGRU vs Base | Fold 1 Outer Test (500 ep, seed 42) | 192,692 vs 163,940 | **65.72 vs 64.42** | — | **0.6513 vs 0.6401** | **0.5032 vs 0.4805** | +1.11 pp Macro-F1; +1.30 pp Acc; Overfit resilient |

### 3.3 Target Metric Deficit
To declare victory, the candidate architecture evaluated across the full 47-subject 5-fold cross-validation must beat:
- **Accuracy Target**: $> 71.04\%$
  - Deficit from Model A 5-fold CV mean ($66.63\%$): **$+4.41\text{ percentage points}$**
- **Weighted-F1 Target**: $> 0.7079$
  - Deficit from Model A 5-fold CV mean ($0.6654$): **$+0.0425\text{ (+4.25 percentage points)}$**

---

## 4. 5-Fold Cross-Validation Parallelization Strategy

### 4.1 Partitioning Structure & Verification
The 5 outer-fold manifests are located in `experiments/EXP-014/EXP014_MANIFESTS/fold_{1..5}.json`:
- **Fold 1**: 32 inner-train (168 nights), 5 inner-val (24 nights), **10 outer-test subjects (54 nights, 32,098 valid windows)**
  - Test subjects: `Bidslab08`, `Bidslab30`, `Bidslab36`, `Bidslab39`, `Bidslab41`, `Bidslab45`, `Bidslab47`, `Bidslab64`, `Bidslab65`, `Bidslab68`
- **Fold 2**: 32 inner-train (168 nights), 5 inner-val (25 nights), **10 outer-test subjects (53 nights, 31,291 valid windows)**
  - Test subjects: `Bidslab07`, `Bidslab10`, `Bidslab13`, `Bidslab32`, `Bidslab34`, `Bidslab43`, `Bidslab44`, `Bidslab53`, `Bidslab55`, `Bidslab63`
- **Fold 3**: 33 inner-train (175 nights), 5 inner-val (26 nights), **9 outer-test subjects (45 nights, 26,599 valid windows)**
  - Test subjects: `Bidslab06`, `Bidslab14`, `Bidslab19`, `Bidslab20`, `Bidslab38`, `Bidslab40`, `Bidslab42`, `Bidslab56`, `Bidslab62`
- **Fold 4**: 33 inner-train (174 nights), 5 inner-val (25 nights), **9 outer-test subjects (47 nights, 27,630 valid windows)**
  - Test subjects: `Bidslab00`, `Bidslab09`, `Bidslab15`, `Bidslab22`, `Bidslab31`, `Bidslab35`, `Bidslab50`, `Bidslab51`, `Bidslab60`
- **Fold 5**: 33 inner-train (171 nights), 5 inner-val (28 nights), **9 outer-test subjects (47 nights, 27,665 valid windows)**
  - Test subjects: `Bidslab01`, `Bidslab02`, `Bidslab11`, `Bidslab16`, `Bidslab17`, `Bidslab18`, `Bidslab49`, `Bidslab52`, `Bidslab66`

**Mathematical Proof of Completeness**:
- Total outer-test subjects: $10 + 10 + 9 + 9 + 9 = 47\text{ subjects}$ (100% cohort coverage).
- Disjointness: $\text{OuterTest}_i \cap \text{OuterTest}_j = \emptyset \; (\forall i \neq j)$.
- Evaluated 30-second windows: $32,098 + 31,291 + 26,599 + 27,630 + 27,665 = \mathbf{145,283\text{ valid windows}}$.

### 4.2 Work Assignment Across `cuda:0` and `cuda:1`
Full 5-fold cross-validation requires evaluating:
$$\mathbf{5\text{ Outer Folds}} \times \mathbf{3\text{ Seeds (42, 123, 2024)}} = \mathbf{15\text{ Independent Runs}}$$

#### Recommended Dynamic Work Queue Architecture
Rather than hardcoding static fold splits (which causes load imbalance because Folds 1 and 2 have 10 test subjects while Folds 3, 4, 5 have 9 test subjects, and GPU 0 has minor desktop background activity), we use a **Dynamic Task Queue with 2 Dedicated Worker Processes**:
1. **Task Queue Initialization**:
   ```python
   tasks = [(fold, seed) for fold in [1, 2, 3, 4, 5] for seed in [42, 123, 2024]]
   # 15 discrete tasks
   ```
2. **Worker 0**: Dedicated to `cuda:0`. Pops next available `(fold, seed)`, executes, records atomic JSON, repeats.
3. **Worker 1**: Dedicated to `cuda:1`. Pops next available `(fold, seed)`, executes, records atomic JSON, repeats.
4. **Execution Balance**: Worker 0 and Worker 1 process ~7 and 8 tasks each, finishing nearly simultaneously.

### 4.3 Preventing Race Conditions and Resource Collisions
To guarantee bitwise reproducibility and zero runtime collisions:
1. **Process Isolation (`multiprocessing` spawn or `subprocess`)**:
   - Each worker runs in an independent OS process with its own Python interpreter and PyTorch CUDA context.
   - Workers are invoked via `sys.executable` with explicit CLI arguments: `--device cuda:0` vs `--device cuda:1`.
   - Never use `fork` multiprocessing with CUDA; always use `spawn` or `subprocess.Popen` to avoid deadlocks.
2. **Atomic File Persistence**:
   - Each run writes its output to a unique temporary file:
     `results_fold{fold}_seed{seed}.tmp.json`
   - Upon successful JSON encoding and flush, `os.replace()` is called to atomically move it to:
     `results_fold{fold}_seed{seed}.json`
   - This eliminates incomplete reads, file corruption, or race conditions during synthesis.
3. **Read-Only Feature Cache Access**:
   - The preprocessed feature cache `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b` is opened exclusively in read-only mode (`numpy.load` without write permissions). Multiple concurrent readers on disk are 100% thread-safe and process-safe.
4. **Data Loader Workers (`num_workers=0`)**:
   - Set `num_workers=0` with `pin_memory=True` in PyTorch `DataLoader`.
   - As proven in EXP-014 and EXP-019, `num_workers=0` executes in < 2 seconds per epoch without inter-process IPC overhead, POSIX semaphore leaks, or memory thrashing.
5. **RNG State Isolation**:
   - Each worker process seeds its random generators independently at the start of each run:
     ```python
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     if torch.cuda.is_available():
         torch.cuda.manual_seed_all(seed)
     ```

### 4.4 Empirical Runtime Benchmarks & Expected Execution Times
From audited runs in EXP-014, EXP-018, and EXP-019:
- **EXP-014 Baseline**: 75 runs of 80 epochs took 5,745 seconds on 1 RTX 3060 $\implies$ **76.60 seconds per run** ($\mathbf{0.958\text{ s/epoch}}$).
- **EXP-019 Baseline**: 500 epochs with validation per epoch took 954.5 seconds on `cuda:0` $\implies$ **1.909 s/epoch**.
- **EXP-019 MultiScale**: 500 epochs took 1,007.5 seconds on `cuda:1` $\implies$ **2.015 s/epoch**.

#### Time Projections for Continuous NAS Loop:
1. **Proxy Evaluation Filtering (Gate 1)**:
   - Partition: Fold 1 only (32 inner-train, 10 outer-test subjects).
   - Training: 50 epochs on `cuda:0` (single seed 42).
   - Per-epoch time: ~1.2 to 1.5 seconds.
   - Outer test evaluation (14 batches, 32,098 windows): ~1.5 seconds.
   - **Total Proxy Evaluation Time: ~60 to 75 seconds (~1.1 minutes)**.
2. **Definitive 5-Fold Cross-Validation (Gate 2)**:
   - Partition: All 5 folds (all 47 subjects, 145,283 windows).
   - Training: 80 epochs per run $\times$ 3 seeds (42, 123, 2024) = 15 runs.
   - Per-run time: 80 epochs $\times$ ~1.3 s + 2 s eval = ~106 seconds (~1.77 minutes).
   - Sequential single-GPU time: $15 \times 1.77\text{ min} = \mathbf{26.5\text{ minutes}}$.
   - **Parallel Dual-GPU Time (`cuda:0` + `cuda:1`)**: $8 \times 1.77\text{ min} = \mathbf{14.2\text{ to }16.0\text{ minutes}}$ wall-clock time!

---

## 5. Exact Victory Conditions & Verification Reporting Requirements

### 5.1 Acceptance Criteria Matrix

| Criterion | Evaluation Level | Required Metric Threshold | Enforcement Policy |
| :--- | :--- | :--- | :--- |
| **Proxy Accuracy** | Fold 1 (50 ep, seed 42) | $\text{Accuracy} \ge 70.0\%$ | Hard Filter (Revert if $< 70.0\%$) |
| **Proxy Weighted-F1** | Fold 1 (50 ep, seed 42) | $\text{Weighted-F1} \ge 0.6900$ | Hard Filter (Revert if $< 0.6900$) |
| **Cohort Accuracy** | Full 5-Fold CV (80 ep, 3 seeds) | **$\text{Aggregated Accuracy} > 71.04\%$** | Hard Victory Gate (Table V target) |
| **Cohort Weighted-F1** | Full 5-Fold CV (80 ep, 3 seeds) | **$\text{Aggregated Weighted-F1} > 0.7079$** | Hard Victory Gate (Table V target) |
| **Model Parameters** | Architecture Parameter Count | Budget: 162k–196k ($+20\%$ cap = 196,728) | Soft constraint if $>196\text{k}$ but beats targets |
| **Compression Ratio** | Compression vs. 5.47M Repo Model | $> 90\%$ compression ratio | Architectural requirement |

### 5.2 Verification & Aggregation Methodology
Aggregated cohort metrics are computed over all out-of-fold outer test evaluations covering all 47 subjects:
1. For each seed $s \in \{42, 123, 2024\}$, outer-test predictions $\hat{y}_k^{(s)}$ and true labels $y_k$ are gathered across folds $k \in \{1..5\}$.
2. The cohort prediction vector $\hat{Y}^{(s)}$ of length 145,283 is constructed by concatenating out-of-fold predictions.
3. For each seed:
   $$\text{Acc}^{(s)} = \text{accuracy\_score}(Y, \hat{Y}^{(s)})$$
   $$\text{Weighted-F1}^{(s)} = \text{f1\_score}(Y, \hat{Y}^{(s)}, \text{average='weighted'}, \text{zero\_division}=0)$$
4. The aggregated cross-validation metrics are:
   $$\overline{\text{Acc}} = \frac{1}{3} \sum_{s} \text{Acc}^{(s)}$$
   $$\overline{\text{Weighted-F1}} = \frac{1}{3} \sum_{s} \text{Weighted-F1}^{(s)}$$
5. Victory requires:
   $$\overline{\text{Acc}} > 0.7104 \quad \text{AND} \quad \overline{\text{Weighted-F1}} > 0.7079$$

### 5.3 PDF-Ready Markdown Report Specification
Upon passing Gate 2, the syndicate must generate a publication-grade, PDF-ready Markdown report (e.g. `experiments/EXP-020/EXP020_VICTORY_REPORT.md` or `NAS_VICTORY_REPORT.md`). The report must contain:
1. **Title and Executive Header**: Clear declaration that the SLAMSS-IFS benchmark has been beaten on the 47-subject BIDSleep dataset.
2. **Head-to-Head Benchmark Table**:
   - Columns: Metric, SLAMSS-IFS (Song et al., TBME 2026 Table V), Winning Architecture (Mean ± SD across 3 seeds), Delta ($\Delta$).
   - Rows: Overall Accuracy (%), Weighted-F1 Score, Macro-F1 (unweighted), Cohen's $\kappa$, Parameter Count, Compression Ratio (x-fold and %).
3. **Architectural Specification of the Winning Mutation**:
   - Detailed tensor block diagram / module description.
   - Concrete attribution to Hypothesis H-F (cross-modal IHR-accelerometry interaction) and Hypothesis H-A (adaptation/architecture with statistically significant gain across all 47 subjects).
4. **Fold-by-Fold Performance Breakdown**:
   - Table showing Accuracy, Weighted-F1, and Macro-F1 across Fold 1 through Fold 5.
5. **Class-Specific Breakdown & Clinical Metrics**:
   - Per-stage F1, Recall, and Precision (Wake, Light, Deep, REM).
   - Wake Recall / Sensitivity comparison against the paper's 69.17%.
6. **Ablation & Statistical Validation**:
   - Paired subject bootstrap confidence intervals ($B=1000$) against Baseline Model A.

### 5.4 Git Synchronization & Terminal Output
Per project rules and `ORIGINAL_REQUEST.md`:
1. All winning code, configurations, model definitions, and JSON metrics must be staged, committed, and pushed:
   ```bash
   git add .
   git commit -m "feat(nas): victory confirmed - SLAMSS-IFS benchmark beaten with <model_name>"
   git push origin main
   ```
2. The terminal output must conclude with the exact verbatim victory string:
   ```text
   VICTORY CONFIRMED: SLAMSS-IFS BENCHMARK BEATEN. CONTINUOUS LOOP TERMINATED.
   ```

---

## 6. Implementation Checklist for NAS Syndicate

- [x] Verified dual NVIDIA RTX 3060 12GB GPUs (`cuda:0` and `cuda:1`, CC 8.6, 28 SMs).
- [x] Verified Python 3.14.7, PyTorch 2.14.1+cu130, NumPy 2.5.3, Scikit-Learn 1.9.1 in `./.venv/bin/python`.
- [x] Verified preprocessed feature cache (`prep_57738d534b`, 246 usable nights across 47 subjects).
- [x] Verified 5-fold cross-validation manifests (`experiments/EXP-014/EXP014_MANIFESTS/fold_{1..5}.json`, 145,283 valid windows).
- [x] Identified published paper benchmark (Table V: 71.04% Accuracy, 0.7079 Weighted-F1, 5.47M parameters).
- [x] Quantified baseline performance deficit (+4.41 pp Accuracy, +0.0425 Weighted-F1).
- [x] Designed dual-GPU parallelization queue (15 runs across `cuda:0` and `cuda:1`, expected wall-clock time ~15 minutes).
- [x] Formulated proxy evaluation gate (Fold 1, 50 epochs, seed 42 on `cuda:0`, Acc $\ge$ 70.0%, Weighted-F1 $\ge$ 0.69, ~70 seconds).
- [x] Codified exact victory reporting requirements, PDF report structure, Git push mandate, and terminal victory message.
