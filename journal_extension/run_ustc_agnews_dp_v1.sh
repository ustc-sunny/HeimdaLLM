#!/usr/bin/env bash
# Shared-host entry point: all generator and MPI ranks use physical GPU 1.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/home/zzkevin/heimdallm-journal-20260928}"
DP_EPSILON="${HEIMDALLM_DP_EPSILON:-8}"
export HEIMDALLM_WORK_ROOT="$WORK_ROOT" PYTHONNOUSERSITE=1
export HEIMDALLM_DP_RUN_ID="${HEIMDALLM_DP_RUN_ID:-ustc_agnews_dp_eps${DP_EPSILON}_20260928}"
unset CUDA_VISIBLE_DEVICES
exec bash "${SCRIPT_DIR}/run_matpool_agnews_dp_v1.sh" \
    --h5-python "${WORK_ROOT}/envs/kdd/bin/python" \
    --fed-python "${WORK_ROOT}/envs/kdd/bin/python" \
    --generator-python "${WORK_ROOT}/envs/ton/bin/python" \
    --gpu-ids 1 --generator-device cuda:1 \
    --staging-root "${WORK_ROOT}/private_staging" "$@"
