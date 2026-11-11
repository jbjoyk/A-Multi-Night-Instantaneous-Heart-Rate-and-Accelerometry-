# Claims Audit for EXP-012

This document provides a line-by-line audit of major technical, architectural, and performance claims made in the previous EXP-012 benchmark report. Each claim is formally audited and categorized according to verifiable source evidence.

---

## 1. Classification Definitions

- **SUPPORTED**: Directly verified by code execution, saved tensor/metric artifacts, and passing mathematical assertions.
- **NOT SUPPORTED**: Asserted in previous prose without verifiable evidence, contradicted by saved data, or derived from unsupported subjective extrapolation.
- **NOT COMPUTED**: The metric or data point cannot be determined from the saved artifacts without reconstructing or simulating unrecorded parameters.
- **REQUIRES SOURCE VERIFICATION**: Plausible based on literature or code, but requires explicit external benchmarking or hardware verification.

---

## 2. Technical Claims Audit

### Claim 1: "34,800 evaluated 30-second windows on the test set"
- **Original Statement**: "34,800 evaluated 30-second windows across 58 complete nights"
- **Classification**: **NOT SUPPORTED / DISCREPANCY RESOLVED**
- **Audit Findings**:
  - Code inspection of the dataset demonstrates that 58 nights padded or truncated to $T=600$ epochs yield exactly $58 \times 600 = 34,800$ total sequence slots.
  - However, within these 34,800 slots, exactly **559 windows** contain the invalid/padding label `-100` (arising from signal artifacts or recording termination prior to 5 hours).
  - The evaluation script correctly filtered out these 559 slots via `valid = (targets >= 0) & (targets < 4) & mask`.
  - The actual number of real, unpadded test windows evaluated is **34,241**.
  - The claim of 34,800 represents the padded matrix allocation, not the evaluated window support.

### Claim 2: "REM support is 8,871 windows"
- **Original Statement**: "REM (3): Support 8,871"
- **Classification**: **NOT SUPPORTED / NUMERICAL DISCREPANCY**
- **Audit Findings**:
  - The ground-truth target vector for the test set contains: Wake = 3,819; Light = 13,816; Deep = 8,294; REM = **8,312**.
  - In the previous report, the table displayed 8,871 by subtracting Wake, Light, and Deep from 34,800 ($34,800 - 3,819 - 13,816 - 8,294 = 8,871$).
  - The 559 invalid/padded slots were erroneously absorbed into the REM support in the text table.
  - The verified ground-truth REM support is **8,312**.

### Claim 3: "Alpha gate values: $\alpha_{\text{Wake}} \approx 0.60 - 0.68$, $\alpha_{\text{Deep}} \approx 0.76 - 0.89$"
- **Original Statement**: "The adaptive gate consistently assigns lowest alpha to Wake ($\alpha \approx 0.60 - 0.68$) and highest alpha to Deep sleep ($\alpha \approx 0.76 - 0.89$) across all seeds."
- **Classification**: **NOT COMPUTED (ON TEST SET)**
- **Audit Findings**:
  - During the test evaluation loop (`evaluate_test_set` in `run_exp012_final_benchmark.py`), the model output dictionary `out["alpha"]` was not collected into the test accumulation lists.
  - Per-window alpha gate values on the test set do not exist in `exp012_test_results.pkl`.
  - While alpha dynamics were verified on the validation set during EXP-010 and EXP-011, asserting specific test-set numerical alpha values without saved test gate activations is mathematically unsupported.
  - Reconstructing probabilities or gate values post-hoc is strictly prohibited. Therefore, test alpha gate values are marked: **NOT COMPUTED**.

### Claim 4: "Physiologically plausible sleep architectures"
- **Original Statement**: "Successfully regularized temporal state sequences, maintaining physiologically plausible sleep architectures."
- **Classification**: **NOT SUPPORTED**
- **Audit Findings**:
  - No sequence-level transition frequency matrices, hypnogram bouts distribution analysis, or comparison against polysomnographic normative sleep architecture tables were executed on the test set predictions.
  - Transition regularization ($\lambda_{\text{trans}}=0.10$) penalizes rapid improbable transitions in loss space, but claiming clinical physiological plausibility without empirical sequence duration metrics is an unsubstantiated qualitative claim.

### Claim 5: "Exceptional calibration (ECE 0.0257)"
- **Original Statement**: "Exceptional probability calibration", "near-perfect probability calibration"
- **Classification**: **DOWNGRADED / SUBJECTIVE LANGUAGE REMOVED**
- **Audit Findings**:
  - The numerical Expected Calibration Error (ECE) is verified:
    - Seed 42: **0.0221**
    - Seed 123: **0.0234**
    - Seed 2024: **0.0315**
    - Mean $\pm$ SD: **$0.0257 \pm 0.0050$**
  - ECE was computed using 10 uniform equal-width bins on $[0, 1]$ over the maximum predicted softmax probabilities on valid windows only.
  - Subjective descriptors ("exceptional", "near-perfect") are removed. The report will state purely: `ECE = 0.0257 ± 0.0050 (10 uniform bins, valid windows)`.

### Claim 6: "Edge feasibility: High"
- **Original Statement**: "Edge Feasibility: High (suitable for on-device wrist staging)"
- **Classification**: **NOT SUPPORTED**
- **Audit Findings**:
  - No deployment, profiling, or execution on embedded edge hardware (e.g., ARM Cortex-M, Apple Watch S-series SIP, Snapdragon Wear) was conducted.
  - While a 162k parameter network is smaller than a 5.47M parameter network, edge feasibility requires evaluating memory footprint, SRAM buffer constraints, fixed-point quantization compatibility, and battery consumption on the target hardware.
  - In the absence of embedded hardware benchmarks, this claim is unverified.

### Claim 7: "Peak GPU VRAM: 265.1 MB vs SLAMSS-IFS ~1.4 GB (5.3x lower memory)"
- **Original Statement**: "Peak VRAM: 265.1 MB vs ~1.4 GB (5.3x lower memory)"
- **Classification**: **NOT SUPPORTED / UNPAIRED ESTIMATE**
- **Audit Findings**:
  - While SleepAdaptiveNet peak VRAM was directly profiled on an NVIDIA RTX 3060 at 265.1 MB during training, the SLAMSS-IFS VRAM value (~1.4 GB) was an estimate derived from an earlier exploratory run with different sequence lengths and batch configurations, not measured side-by-side on the test set under matched pipeline conditions.
  - Paired VRAM benchmarking was not executed.

### Claim 8: "Training time per epoch: 0.91 s vs ~5.8 s (6.4x faster throughput)"
- **Original Statement**: "Training Time per Epoch: ~5.8 s vs 0.91 s (6.4x faster throughput)"
- **Classification**: **NOT SUPPORTED / UNPAIRED ESTIMATE**
- **Audit Findings**:
  - The runtime for SleepAdaptiveNet across 80 epochs was recorded at 73 seconds per seed (~0.91 s/epoch).
  - However, SLAMSS-IFS epoch runtime was not benchmarked side-by-side using the identical data loader, sequence length ($T=600$), worker thread count, and precision settings.
  - Throughput claims must be measured under identical hardware and software harnesses.

### Claim 9: "Overall Test Accuracy: 68.44% ± 1.35%, Macro-F1: 0.6859 ± 0.0139"
- **Original Statement**: "Accuracy: 68.44% ± 1.35%, Macro-F1: 0.6859 ± 0.0139, Weighted-F1: 0.6835 ± 0.0141, Cohen's κ: 0.5527 ± 0.0188"
- **Classification**: **SUPPORTED**
- **Audit Findings**:
  - Recomputed from raw predictions and targets in `exp012_test_results.pkl`:
    - Seed 42: Accuracy = 69.8578%, Macro-F1 = 0.698641, Weighted-F1 = 0.698469, $\kappa$ = 0.572631, MCC = 0.574041.
    - Seed 123: Accuracy = 68.2749%, Macro-F1 = 0.687952, Weighted-F1 = 0.681595, $\kappa$ = 0.550428, MCC = 0.551188.
    - Seed 2024: Accuracy = 67.1826%, Macro-F1 = 0.671103, Weighted-F1 = 0.670444, $\kappa$ = 0.535181, MCC = 0.537047.
  - Multi-seed mean $\pm$ standard deviation:
    - Accuracy: **$68.44\% \pm 1.35\%$**
    - Macro-F1: **$0.6859 \pm 0.0139$**
    - Weighted-F1: **$0.6835 \pm 0.0141$**
    - Cohen's $\kappa$: **$0.5527 \pm 0.0188$**
    - MCC: **$0.5541 \pm 0.0187$**
  - All values match the recomputed arrays to floating-point rounding precision.

### Claim 10: "Parameter Count: 162,325 vs SLAMSS-IFS 5,465,768 (33.7x reduction)"
- **Original Statement**: "162,325 vs 5,465,768 parameters; 33.7x reduction (97.03% compression)"
- **Classification**: **SUPPORTED**
- **Audit Findings**:
  - `SleepAdaptiveNet` (`B0_GRL`) parameter count was counted layer-by-layer: exactly **162,325** trainable parameters.
  - The official `BIDSLabUMass/SLAMSS-IFS` architecture (`model/SLAMSS_IFS_Model.py`) was audited and instantiated: exactly **5,465,768** parameters.
  - Ratio: $5,465,768 / 162,325 = 33.67 \approx 33.7\text{x}$ parameter reduction ($97.03\%$ reduction).
