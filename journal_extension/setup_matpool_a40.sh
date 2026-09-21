#!/usr/bin/env bash
# Recreate the tested single-A40 environment for the AG News journal extension.
set -Eeuo pipefail

WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/root/heimdallm-work}"
CONDA_ROOT="${HEIMDALLM_CONDA_ROOT:-/root/miniconda3}"
BASE_ENV="${HEIMDALLM_BASE_ENV:-myconda}"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_URL="https://drive.google.com/uc?id=10S3Zg9HFmBuDkOusycefkugOCu27s0JT"
DATA_ARCHIVE="${WORK_ROOT}/data/fednlp_data.tar"
DATA_ROOT="${WORK_ROOT}/data/fednlp_data"
TASK_MODEL_DIR="${WORK_ROOT}/models/distilbert-base-uncased"
GENERATOR_MODEL_DIR="${WORK_ROOT}/models/distilgpt2"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    printf 'Conda activation script not found: %s\n' "${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
mkdir -p "${WORK_ROOT}/data" "${WORK_ROOT}/models" "${WORK_ROOT}/results" "${WORK_ROOT}/logs"

if ! conda env list | awk '{print $1}' | grep -Fxq heimdallm-kdd; then
    conda create -y -n heimdallm-kdd --clone "${BASE_ENV}"
fi
if ! conda env list | awk '{print $1}' | grep -Fxq heimdallm-ton; then
    conda create -y -n heimdallm-ton --clone "${BASE_ENV}"
fi

conda run -n heimdallm-kdd pip install -r "${PROJECT_ROOT}/journal_extension/requirements-kdd-matpool.txt"
conda run -n heimdallm-ton pip install -r "${PROJECT_ROOT}/journal_extension/requirements-ton-matpool.txt"

if [[ ! -f "${DATA_ROOT}/data_files/agnews_data.h5" || \
      ! -f "${DATA_ROOT}/partition_files/agnews_partition.h5" ]]; then
    conda run -n heimdallm-kdd gdown "${DATA_URL}" -O "${DATA_ARCHIVE}"
    tar -xzf "${DATA_ARCHIVE}" -C "${WORK_ROOT}/data"
fi

if [[ ! -f "${TASK_MODEL_DIR}/config.json" ]]; then
    TASK_MODEL_DIR="${TASK_MODEL_DIR}" conda run -n heimdallm-kdd python -c \
      'import os; from transformers import AutoModelForSequenceClassification, AutoTokenizer; p=os.environ["TASK_MODEL_DIR"]; AutoTokenizer.from_pretrained("distilbert-base-uncased").save_pretrained(p); AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4).save_pretrained(p)'
fi

if [[ ! -f "${GENERATOR_MODEL_DIR}/config.json" ]]; then
    GENERATOR_MODEL_DIR="${GENERATOR_MODEL_DIR}" conda run -n heimdallm-ton python -c \
      'import os; from transformers import AutoModelForCausalLM, AutoTokenizer; p=os.environ["GENERATOR_MODEL_DIR"]; AutoTokenizer.from_pretrained("distilgpt2").save_pretrained(p); AutoModelForCausalLM.from_pretrained("distilgpt2").save_pretrained(p)'
fi

export HEIMDALLM_WORK_ROOT="${WORK_ROOT}"
export HEIMDALLM_CONDA_ROOT="${CONDA_ROOT}"
bash "${PROJECT_ROOT}/journal_extension/run_matpool_agnews_nondp_v3.sh" --preflight-only

printf 'Setup and preflight complete. Project: %s\n' "${PROJECT_ROOT}"
