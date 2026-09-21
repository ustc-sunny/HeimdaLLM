#!/usr/bin/env bash
# Single-A40 deployment check: five paired Non-DP arms, held-out train dev set.
set -Eeuo pipefail
WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/root/heimdallm-work}"
CONDA_ROOT="${HEIMDALLM_CONDA_ROOT:-/root/miniconda3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
exec bash "${SCRIPT_DIR}/run_sst2_nondp_feasibility.sh" \
  --stage smoke --condition paired --evaluation dev \
  --run-id matpool_a40_sst2_deployment_smoke_20260921 \
  --real-data "${WORK_ROOT}/data/sst2_deployment/sst_2_data.h5" \
  --real-partition "${WORK_ROOT}/data/sst2_deployment/sst_2_partition.h5" \
  --source-partition-method deployment_balanced100 \
  --task-model "${WORK_ROOT}/models/distilbert-base-uncased" \
  --generator-model "${WORK_ROOT}/models/distilgpt2" \
  --h5-python "${CONDA_ROOT}/envs/heimdallm-kdd/bin/python" \
  --fed-python "${CONDA_ROOT}/envs/heimdallm-kdd/bin/python" \
  --generator-python "${CONDA_ROOT}/envs/heimdallm-ton/bin/python" \
  --results-root "${WORK_ROOT}/results" \
  --gpu-ids 0 --generator-device cuda:0 --mpi-workers 1 \
  --allow-gpu-sharing "$@"
