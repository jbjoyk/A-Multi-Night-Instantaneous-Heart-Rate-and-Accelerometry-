#!/usr/bin/env bash
# ==============================================================================
# EXP-019 STAGE 1: DUAL-GPU ORCHESTRATION SCRIPT
# Hardware: 2 x NVIDIA GeForce RTX 3060 (12 GB VRAM each)
# Workload:
#   GPU 0 (cuda:0): TSK-07 (UA-2 Unsupervised Adaptation Re-Run, 138 runs, ~28 min)
#   GPU 1 (cuda:1): TSK-06 (Calibration-Depth Learning Curve k in {1,2,3,4}, 363 runs, ~1.5-2.5 h)
# Governance:
#   - Pre-registration: PREREG_EXP-019_K_CURVE.md & PREREG_EXP-019_S3_AMENDMENT.md
#   - TSK-08 (SLAMSS-IFS matched benchmark) remains strictly deferred under DEC-034.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
    echo "================================================================================"
    echo "STAGE 1 DRY RUN MODE ACTIVATED (Verification Only - No Weight Updates)"
    echo "================================================================================"
fi

echo "================================================================================"
echo "EXP-019 STAGE 1 CONCURRENT EXECUTION ORCHESTRATOR"
echo "Repository Root: ${REPO_ROOT}"
echo "Python Executable: ${VENV_PYTHON}"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "================================================================================"

# Verify Python Environment
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Error: Virtual environment python not found at ${VENV_PYTHON}" >&2
    exit 1
fi

# Verify GPU Hardware
echo "Checking GPU Availability..."
"${VENV_PYTHON}" -c "
import torch
count = torch.cuda.device_count()
print(f'Detected CUDA GPUs: {count}')
for i in range(count):
    print(f'  [GPU {i}] {torch.cuda.get_device_name(i)} (VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB)')
assert count >= 1, 'At least 1 CUDA GPU is required.'
"

LOG_DIR="${REPO_ROOT}/experiments/EXP-019/logs"
mkdir -p "${LOG_DIR}"

TSK07_LOG="${LOG_DIR}/tsk07_ua2_$(date +"%Y%m%d_%H%M%S").log"
TSK06_LOG="${LOG_DIR}/tsk06_k_curve_$(date +"%Y%m%d_%H%M%S").log"

echo ""
echo "Launching TSK-07 (UA-2 Rerun) on cuda:0..."
echo "  Log: ${TSK07_LOG}"
"${VENV_PYTHON}" -u "${SCRIPT_DIR}/run_tsk07_ua2_rerun.py" \
    --device "cuda:0" \
    ${DRY_RUN_FLAG} > "${TSK07_LOG}" 2>&1 &
PID_TSK07=$!
echo "  TSK-07 PID: ${PID_TSK07}"

echo ""
echo "Launching TSK-06 (k-Learning Curve) on cuda:1..."
echo "  Log: ${TSK06_LOG}"
"${VENV_PYTHON}" -u "${SCRIPT_DIR}/run_tsk06_k_curve.py" \
    --device "cuda:1" \
    ${DRY_RUN_FLAG} > "${TSK06_LOG}" 2>&1 &
PID_TSK06=$!
echo "  TSK-06 PID: ${PID_TSK06}"

echo ""
echo "Both Stage 1 experiments running concurrently across workstation GPUs."
echo "Waiting for process completion..."

EXIT_CODE_TSK07=0
EXIT_CODE_TSK06=0

wait ${PID_TSK07} || EXIT_CODE_TSK07=$?
echo ">>> TSK-07 exited with code: ${EXIT_CODE_TSK07} <<<"

wait ${PID_TSK06} || EXIT_CODE_TSK06=$?
echo ">>> TSK-06 exited with code: ${EXIT_CODE_TSK06} <<<"

echo "================================================================================"
if [[ ${EXIT_CODE_TSK07} -eq 0 && ${EXIT_CODE_TSK06} -eq 0 ]]; then
    echo "STAGE 1 EXECUTION COMPLETE: ALL EXPERIMENTS SUCCEEDED"
    echo "================================================================================"
    exit 0
else
    echo "STAGE 1 EXECUTION FAILED: Non-zero exit code detected"
    echo "TSK-07 Exit Code: ${EXIT_CODE_TSK07}"
    echo "TSK-06 Exit Code: ${EXIT_CODE_TSK06}"
    echo "================================================================================"
    exit 1
fi
