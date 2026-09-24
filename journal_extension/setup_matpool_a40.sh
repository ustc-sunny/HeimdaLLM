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
DATA_FALLBACK_ARCHIVE="${HEIMDALLM_DATA_FALLBACK_ARCHIVE:-/mnt/heimdallm-agnews-20260921/heimdallm-agnews.tar.gz}"
TASK_MODEL_DIR="${WORK_ROOT}/models/distilbert-base-uncased"
GENERATOR_MODEL_DIR="${WORK_ROOT}/models/distilgpt2"
MODELSCOPE_VERSION="1.18.0"

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
    if conda run -n heimdallm-kdd gdown "${DATA_URL}" -O "${DATA_ARCHIVE}"; then
        tar -xzf "${DATA_ARCHIVE}" -C "${WORK_ROOT}/data"
    elif [[ -f "${DATA_FALLBACK_ARCHIVE}" ]]; then
        printf 'Google Drive unavailable; restoring AG News from %s\n' "${DATA_FALLBACK_ARCHIVE}"
        tar -xzf "${DATA_FALLBACK_ARCHIVE}" -C "${WORK_ROOT}/data" \
            fednlp_data/data_files/agnews_data.h5 \
            fednlp_data/partition_files/agnews_partition.h5
    else
        printf 'Unable to download data and fallback archive is absent: %s\n' \
            "${DATA_FALLBACK_ARCHIVE}" >&2
        exit 1
    fi
fi

ensure_modelscope() {
    if ! conda run -n "${BASE_ENV}" python -c 'import modelscope' >/dev/null 2>&1; then
        conda run -n "${BASE_ENV}" pip install "modelscope==${MODELSCOPE_VERSION}"
    fi
}

modelscope_file() {
    local model_id="$1"
    local filename="$2"
    local destination="$3"
    conda run -n "${BASE_ENV}" modelscope download \
        --model "${model_id}" "${filename}" --local_dir "${destination}"
}

task_model_complete() {
    [[ -f "${TASK_MODEL_DIR}/config.json" ]] && \
    [[ -f "${TASK_MODEL_DIR}/vocab.txt" || -f "${TASK_MODEL_DIR}/tokenizer.json" ]] && \
    [[ -f "${TASK_MODEL_DIR}/pytorch_model.bin" || -f "${TASK_MODEL_DIR}/model.safetensors" ]]
}

generator_model_complete() {
    [[ -f "${GENERATOR_MODEL_DIR}/config.json" ]] && \
    [[ -f "${GENERATOR_MODEL_DIR}/merges.txt" ]] && \
    [[ -f "${GENERATOR_MODEL_DIR}/vocab.json" ]] && \
    [[ -f "${GENERATOR_MODEL_DIR}/pytorch_model.bin" || -f "${GENERATOR_MODEL_DIR}/model.safetensors" ]]
}

if ! task_model_complete; then
    if ! TASK_MODEL_DIR="${TASK_MODEL_DIR}" conda run -n heimdallm-kdd python -c \
      'import os; from transformers import AutoModelForSequenceClassification, AutoTokenizer; p=os.environ["TASK_MODEL_DIR"]; AutoTokenizer.from_pretrained("distilbert-base-uncased").save_pretrained(p); AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4).save_pretrained(p)'; then
        printf 'Hugging Face unavailable; downloading DistilBERT from ModelScope.\n'
        ensure_modelscope
        mkdir -p "${TASK_MODEL_DIR}"
        for filename in config.json tokenizer_config.json tokenizer.json vocab.txt pytorch_model.bin; do
            modelscope_file distilbert/distilbert-base-uncased "${filename}" "${TASK_MODEL_DIR}"
        done
    fi
fi

if ! generator_model_complete; then
    if ! GENERATOR_MODEL_DIR="${GENERATOR_MODEL_DIR}" conda run -n heimdallm-ton python -c \
      'import os; from transformers import AutoModelForCausalLM, AutoTokenizer; p=os.environ["GENERATOR_MODEL_DIR"]; AutoTokenizer.from_pretrained("distilgpt2").save_pretrained(p); AutoModelForCausalLM.from_pretrained("distilgpt2").save_pretrained(p)'; then
        printf 'Hugging Face unavailable; downloading DistilGPT2 from ModelScope.\n'
        ensure_modelscope
        mkdir -p "${GENERATOR_MODEL_DIR}"
        for filename in config.json generation_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt pytorch_model.bin; do
            modelscope_file distilbert/distilgpt2 "${filename}" "${GENERATOR_MODEL_DIR}"
        done
    fi
fi

task_model_complete || { printf 'DistilBERT download is incomplete.\n' >&2; exit 1; }
generator_model_complete || { printf 'DistilGPT2 download is incomplete.\n' >&2; exit 1; }

export HEIMDALLM_WORK_ROOT="${WORK_ROOT}"
export HEIMDALLM_CONDA_ROOT="${CONDA_ROOT}"
bash "${PROJECT_ROOT}/journal_extension/run_matpool_agnews_nondp_v3.sh" --preflight-only

printf 'Setup and preflight complete. Project: %s\n' "${PROJECT_ROOT}"
