#!/usr/bin/env bash
# Isolated setup for the shared USTC A40 host; preserves existing conda envs.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/home/zzkevin/heimdallm-journal-20260928}"
BASE_PYTHON="${HEIMDALLM_BASE_PYTHON:-/home/zzkevin/miniconda3/envs/fwdllm3.9/bin/python}"
PIP_INDEX="${HEIMDALLM_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
DATA_ROOT="${HEIMDALLM_SOURCE_DATA_ROOT:-/home/zzkevin/HeimdaLLM/xiexiu-final/fednlp_data}"
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${WORK_ROOT}/envs" "${WORK_ROOT}/models" "${WORK_ROOT}/data" \
    "${WORK_ROOT}/logs" "${WORK_ROOT}/results" "${WORK_ROOT}/backups"
for name in kdd ton; do
    env_dir="${WORK_ROOT}/envs/${name}"
    if [[ ! -x "${env_dir}/bin/python" ]]; then
        "$BASE_PYTHON" -m venv "$env_dir"
    fi
    "${env_dir}/bin/python" -m pip install --index-url "$PIP_INDEX" \
        'pip<26' 'setuptools<70' wheel
    "${env_dir}/bin/python" -m pip install --index-url "$PIP_INDEX" \
        --extra-index-url https://download.pytorch.org/whl/cu116 \
        'torch==1.13.1+cu116' 'numpy==1.24.4' 'scipy==1.10.1' \
        'h5py==3.8.0' 'pandas==1.5.3' 'scikit-learn==1.2.2' \
        psutil tqdm regex
done
"${WORK_ROOT}/envs/kdd/bin/python" -m pip install --index-url "$PIP_INDEX" \
    -r "${SCRIPT_DIR}/requirements-kdd-matpool.txt"
"${WORK_ROOT}/envs/ton/bin/python" -m pip install --index-url "$PIP_INDEX" \
    -r "${SCRIPT_DIR}/requirements-ton-matpool.txt" 'modelscope==1.18.0'

for directory in data_files partition_files; do
    mkdir -p "${WORK_ROOT}/data/fednlp_data/${directory}"
    source_file="${DATA_ROOT}/${directory}/agnews_${directory%_files}.h5"
    target_file="${WORK_ROOT}/data/fednlp_data/${directory}/agnews_${directory%_files}.h5"
    [[ -f "$source_file" ]] || { echo "Missing AG News file: $source_file" >&2; exit 1; }
    if [[ ! -e "$target_file" ]]; then
        ln -s "$source_file" "$target_file"
    fi
done
TASK_MODEL_SOURCE="${HEIMDALLM_TASK_MODEL_SOURCE:-/home/zzkevin/models/distilbert-base-uncased}"
if [[ ! -e "${WORK_ROOT}/models/distilbert-base-uncased" ]]; then
    ln -s "$TASK_MODEL_SOURCE" "${WORK_ROOT}/models/distilbert-base-uncased"
fi
GENERATOR_DIR="${WORK_ROOT}/models/distilgpt2"
mkdir -p "$GENERATOR_DIR"
for filename in config.json generation_config.json tokenizer_config.json tokenizer.json \
    vocab.json merges.txt pytorch_model.bin; do
    if [[ ! -s "${GENERATOR_DIR}/${filename}" ]]; then
        "${WORK_ROOT}/envs/ton/bin/modelscope" download \
            --model distilbert/distilgpt2 "$filename" --local_dir "$GENERATOR_DIR"
    fi
done
"${WORK_ROOT}/envs/kdd/bin/python" -m pip freeze > "${WORK_ROOT}/logs/kdd-freeze.txt"
"${WORK_ROOT}/envs/ton/bin/python" -m pip freeze > "${WORK_ROOT}/logs/ton-freeze.txt"
export HEIMDALLM_WORK_ROOT="$WORK_ROOT"
bash "${SCRIPT_DIR}/run_ustc_agnews_dp_v1.sh" --preflight-only
echo 'USTC setup and preflight complete.'
