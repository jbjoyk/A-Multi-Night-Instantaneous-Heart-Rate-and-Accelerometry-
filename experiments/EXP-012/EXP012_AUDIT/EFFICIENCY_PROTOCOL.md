# Protocol for Standardized Hardware Efficiency Benchmarking

> [!IMPORTANT]
> **PROTOCOL SPECIFICATION ONLY — DO NOT EXECUTE YET.**
> This document specifies the rigorous methodology for benchmarking CPU-measured computational and memory efficiency. No efficiency tests will be executed until explicitly authorized.

---

## 1. Benchmarking Objectives

To establish verifiable, reproducible computational efficiency metrics comparing `SleepAdaptiveNet` and official `SLAMSS-IFS` without relying on ungrounded assertions or estimated values.

---

## 2. Standardized Measurement Environment (Controlled Conditions)

All future efficiency benchmarks must record and hold constant:
- **Processor**: Exact CPU Model (from `/proc/cpuinfo`, e.g., AMD Ryzen or Intel Xeon/Core).
- **CPU Thread Count**: Locked to a single thread (`torch.set_num_threads(1)`) to model low-power embedded single-core processors, followed by multi-thread profiling.
- **Operating System & Kernel**: Linux kernel version, glibc version.
- **Python & PyTorch Runtime**: Python 3.14.7, PyTorch 2.14.1 (CPU backend).
- **Batch Size**: Fixed at **Batch Size = 1** (simulating real-time single-night or single-epoch wearable inference).
- **Precision**: Full Precision (FP32), followed by dynamic post-training quantization (INT8).
- **Warmup**: Exactly 50 warmup iterations before recording latency measurements.
- **Repeats**: Minimum 500 timed iterations per measurement.

---

## 3. Metrics to be Measured

For both `SleepAdaptiveNet` and official `SLAMSS-IFS` under identical conditions:

1. **Inference Latency per 30-Second Window**:
   - Single-step forward pass latency (mean $\pm$ std ms, 95th percentile, 99th percentile).
2. **Inference Latency per Full Night (600 Windows)**:
   - Full 5-hour hypnogram sequence forward pass latency (ms).
3. **Peak Resident Set Size (RSS) Memory**:
   - Monitored via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` and `tracemalloc` during inference.
4. **Model Checkpoint Size on Disk**:
   - Serialized `.pt` state dictionary file size (KB / MB).
5. **INT8 Quantization Impact (Optional)**:
   - Quantized model file size.
   - Inference speedup factor on CPU.
   - Classification accuracy delta ($\Delta \text{Acc}$) after dynamic quantization on the validation partition.

---

## 4. Comparability Rules & Missing Implementation Fallback

- If official `SLAMSS-IFS` cannot be compiled or executed under the identical CPU test harness due to environment or missing official weights:
  > The comparison value must be explicitly recorded as: `NOT COMPUTED`.
- Subjective terminology such as *"edge-ready"*, *"deployment-ready"*, or *"real-time capable"* is strictly prohibited.
- All conclusions must be framed objectively as:
  > **"CPU-measured inference latency of X ms and peak RSS of Y MB under single-thread FP32 execution."**
