#!/usr/bin/env bash
# ==============================================================================
# EXP-021 (STAGE 3 / TSK-08): DUAL-GPU ORCHESTRATION SCRIPT
# Hardware: 2 x NVIDIA GeForce RTX 3060 (12 GB VRAM each; independent devices)
# Architecture: Official BIDSLabUMass/SLAMSS-IFS (5,465,768 parameters, AMP, B=4)
# Measured Peak VRAM: 5.12 GiB reserved at B=4, L=1200 (>6.0 GiB headroom per card)
#
# Training Config (EGM-Resolved Parity & Throughput):
#   - micro_batch_size=4, effective_batch_size=4 (42-44 optimizer steps/epoch)
#   - max_epochs=100 (up to 4,200-4,400 steps >= Model A's 3,200 steps), min_epochs=40, patience=25
#   - Batched validation inference (val_batch_size=16 -> ~2s/epoch vs ~30s unbatched)
#   - Thread pinning: OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 (prevents Ryzen 7 7700 contention)
#
# Workload Split per PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md Section 6:
#   GPU 0 (cuda:0): 7 runs -> Folds 1-2 (all 3 seeds) + Fold 3 (seed 42)
#   GPU 1 (cuda:1): 8 runs -> Fold 3 (seeds 123, 2024) + Folds 4-5 (all 3 seeds)
# Expected Wall-Clock Runtime (early stop ~65-85 epochs): ~21-28 hours total
#
# Recovery Note:
#   Individual runs are resume-safe (skipping completed run_results/*.json).
#   To re-run a single pair: .venv/bin/python experiments/EXP-021/run_tsk08_slamss_matched.py --pairs F:S --force
#
# Usage:
#   ./experiments/EXP-021/run_stage3_dual_gpu.sh                # Auto-runs Stage 2 if CSV missing, then Stage 3
#   ./experiments/EXP-021/run_stage3_dual_gpu.sh --run-stage2   # Force Stage 2 freeze first, then Stage 3
#   ./experiments/EXP-021/run_stage3_dual_gpu.sh --dry-run      # Verify Stage 2 & Stage 3 without training
# ==============================================================================

set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
STAGE2_SCRIPT="${REPO_ROOT}/experiments/STAGE2_FINAL_MODEL/freeze_final_model.py"
STAGE2_OOF_CSV="${REPO_ROOT}/experiments/STAGE2_FINAL_MODEL/final_model_oof_subject_metrics.csv"
TSK08_DRIVER="${SCRIPT_DIR}/run_tsk08_slamss_matched.py"
TSK08_ANALYZE="${SCRIPT_DIR}/analyze_tsk08_paired_benchmark.py"

DRY_RUN_ARGS=()
IS_DRY_RUN=0
RUN_STAGE2=0

for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN_ARGS=(--dry-run)
            IS_DRY_RUN=1
            ;;
        --run-stage2)
            RUN_STAGE2=1
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            echo "Usage: $0 [--run-stage2] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

PID_GPU0=""
PID_GPU1=""

cleanup_workers() {
    echo ""
    echo "Signal received — terminating active Stage 3 GPU workers..." >&2
    if [[ -n "${PID_GPU0}" ]]; then
        kill "${PID_GPU0}" 2>/dev/null || true
    fi
    if [[ -n "${PID_GPU1}" ]]; then
        kill "${PID_GPU1}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit 130
}

trap cleanup_workers INT TERM

print_log_tail() {
    local label="$1"
    local log_file="$2"
    echo "--- Last 30 lines of ${label} (${log_file}) ---" >&2
    if [[ -f "${log_file}" ]]; then
        tail -n 30 "${log_file}" >&2
    else
        echo "(log file not found)" >&2
    fi
    echo "--------------------------------------------------------------------------------" >&2
}

echo "================================================================================"
echo "EXP-021 STAGE 3 DUAL-GPU BENCHMARK ORCHESTRATOR (TSK-08: SLAMSS-IFS)"
echo "Repository Root:   ${REPO_ROOT}"
echo "Python Executable: ${VENV_PYTHON}"
echo "CPU Thread Pin:    OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "Mode:              $( [[ ${IS_DRY_RUN} -eq 1 ]] && echo 'DRY-RUN VERIFICATION' || echo 'FULL 15-RUN BENCHMARK' )"
echo "Timestamp (UTC):   $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "================================================================================"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Error: Virtual environment python not found at ${VENV_PYTHON}" >&2
    exit 1
fi

echo "Checking Dual-GPU Workstation Hardware..."
"${VENV_PYTHON}" -c "
import torch
count = torch.cuda.device_count()
print(f'Detected CUDA GPUs: {count}')
for i in range(count):
    props = torch.cuda.get_device_properties(i)
    print(f'  [GPU {i}] {props.name} (VRAM: {props.total_memory / (1024**3):.1f} GB)')
assert count >= 2, f'Dual-GPU orchestrator requires 2 CUDA GPUs, detected {count}.'
"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +"%Y%m%d_%H%M%S")"

# Stage 2 Final Model Freeze Prerequisite (runs if --run-stage2, --dry-run, or CSV missing)
if [[ ${RUN_STAGE2} -eq 1 || ${IS_DRY_RUN} -eq 1 || ! -f "${STAGE2_OOF_CSV}" ]]; then
    STAGE2_LOG="${LOG_DIR}/stage2_freeze_${TS}.log"
    echo ""
    echo "Running Stage 2 Final Model Freeze Verification on cuda:0..."
    echo "  Log: ${STAGE2_LOG}"
    EXIT_STAGE2=0
    "${VENV_PYTHON}" -u "${STAGE2_SCRIPT}" \
        --device "cuda:0" \
        ${DRY_RUN_ARGS[@]+"${DRY_RUN_ARGS[@]}"} > "${STAGE2_LOG}" 2>&1 || EXIT_STAGE2=$?
    if [[ ${EXIT_STAGE2} -ne 0 ]]; then
        echo "Stage 2 freeze verification FAILED with exit code ${EXIT_STAGE2}" >&2
        print_log_tail "Stage 2 Freeze" "${STAGE2_LOG}"
        exit "${EXIT_STAGE2}"
    fi
    echo ">>> Stage 2 freeze verification completed with exit code 0 <<<"
fi

GPU0_LOG="${LOG_DIR}/tsk08_gpu0_7runs_${TS}.log"
GPU1_LOG="${LOG_DIR}/tsk08_gpu1_8runs_${TS}.log"
ANALYZE_LOG="${LOG_DIR}/tsk08_paired_analysis_${TS}.log"

GPU0_PAIRS=(1:42 1:123 1:2024 2:42 2:123 2:2024 3:42)
GPU1_PAIRS=(3:123 3:2024 4:42 4:123 4:2024 5:42 5:123 5:2024)

echo ""
echo "Launching GPU 0 Worker (cuda:0, 7 runs: ${GPU0_PAIRS[*]})..."
echo "  Log: ${GPU0_LOG}"
"${VENV_PYTHON}" -u "${TSK08_DRIVER}" \
    --device "cuda:0" \
    --pairs "${GPU0_PAIRS[@]}" \
    --micro-batch-size 4 \
    --effective-batch-size 4 \
    --val-batch-size 16 \
    --max-epochs 100 \
    --min-epochs 40 \
    --patience 25 \
    --context-len 1200 \
    ${DRY_RUN_ARGS[@]+"${DRY_RUN_ARGS[@]}"} > "${GPU0_LOG}" 2>&1 &
PID_GPU0=$!
echo "  GPU 0 Worker PID: ${PID_GPU0}"

echo ""
echo "Launching GPU 1 Worker (cuda:1, 8 runs: ${GPU1_PAIRS[*]})..."
echo "  Log: ${GPU1_LOG}"
"${VENV_PYTHON}" -u "${TSK08_DRIVER}" \
    --device "cuda:1" \
    --pairs "${GPU1_PAIRS[@]}" \
    --micro-batch-size 4 \
    --effective-batch-size 4 \
    --val-batch-size 16 \
    --max-epochs 100 \
    --min-epochs 40 \
    --patience 25 \
    --context-len 1200 \
    ${DRY_RUN_ARGS[@]+"${DRY_RUN_ARGS[@]}"} > "${GPU1_LOG}" 2>&1 &
PID_GPU1=$!
echo "  GPU 1 Worker PID: ${PID_GPU1}"

echo ""
echo "Both Stage 3 workers running concurrently across dual RTX 3060 GPUs."
echo "Waiting for process completion..."

EXIT_GPU0=0
EXIT_GPU1=0

wait "${PID_GPU0}" || EXIT_GPU0=$?
PID_GPU0=""
echo ">>> GPU 0 Worker exited with code: ${EXIT_GPU0} <<<"

wait "${PID_GPU1}" || EXIT_GPU1=$?
PID_GPU1=""
echo ">>> GPU 1 Worker exited with code: ${EXIT_GPU1} <<<"

if [[ ${EXIT_GPU0} -ne 0 || ${EXIT_GPU1} -ne 0 ]]; then
    echo "================================================================================" >&2
    echo "STAGE 3 EXECUTION FAILED: GPU0=${EXIT_GPU0}, GPU1=${EXIT_GPU1}" >&2
    echo "================================================================================" >&2
    [[ ${EXIT_GPU0} -ne 0 ]] && print_log_tail "GPU 0 Worker" "${GPU0_LOG}"
    [[ ${EXIT_GPU1} -ne 0 ]] && print_log_tail "GPU 1 Worker" "${GPU1_LOG}"
    exit 1
fi

if [[ ${IS_DRY_RUN} -eq 1 ]]; then
    echo "================================================================================"
    echo "STAGE 3 DRY-RUN VERIFICATION SUCCEEDED (Zero production artifacts modified)"
    echo "================================================================================"
    exit 0
fi

echo ""
echo "All 15 SLAMSS-IFS fold-seed runs completed. Running paired comparative inference..."
echo "  Log: ${ANALYZE_LOG}"
EXIT_ANALYZE=0
"${VENV_PYTHON}" -u "${TSK08_ANALYZE}" > "${ANALYZE_LOG}" 2>&1 || EXIT_ANALYZE=$?
if [[ ${EXIT_ANALYZE} -ne 0 ]]; then
    echo "Paired comparative analysis FAILED with exit code ${EXIT_ANALYZE}" >&2
    print_log_tail "Paired Analysis" "${ANALYZE_LOG}"
    exit "${EXIT_ANALYZE}"
fi

echo "================================================================================"
echo "STAGE 3 (TSK-08 / EXP-021) COMPLETE: ALL 15 RUNS & PAIRED REPORT GENERATED"
echo "================================================================================"
exit 0
