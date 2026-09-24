#!/usr/bin/env bash
# AG News record-level DP synthetic-guidance experiment.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/root/heimdallm-work}"
CONDA_ROOT="${HEIMDALLM_CONDA_ROOT:-/root/miniconda3}"
DP_EPSILON="${HEIMDALLM_DP_EPSILON:-8}"
DP_DELTA="${HEIMDALLM_DP_DELTA:-1e-5}"
RUN_ID="${HEIMDALLM_DP_RUN_ID:-matpool_agnews_dp_eps${DP_EPSILON}_v1}"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

exec bash "${SCRIPT_DIR}/run_sst2_nondp_feasibility.sh" \
  --dataset agnews --stage smoke --condition client-syn --evaluation dev \
  --generator-mode dp --dp-target-epsilon "$DP_EPSILON" \
  --dp-delta "$DP_DELTA" --dp-max-grad-norm 1.0 \
  --run-id "$RUN_ID" --seeds 57,58,59 \
  --real-data "${WORK_ROOT}/data/fednlp_data/data_files/agnews_data.h5" \
  --real-partition "${WORK_ROOT}/data/fednlp_data/partition_files/agnews_partition.h5" \
  --source-partition-method uniform_client_1000 \
  --client-ids 1,21 --real-matched-client-ids 800,801,802,803,804,805,806,807 \
  --dev-client-ids 900,901,902,903,904,905,906,907,908 --dev-per-label 128 \
  --sample-limit-per-client 120 --generator-epochs 5 --target-per-label 32 \
  --real-matched-limit 16 --rounds 50 --eval-every 1 --timeout-minutes 240 \
  --generator-max-length 192 --generator-max-new-tokens 80 \
  --label-names-json '{"1":"World","2":"Sports","3":"Business","4":"Science and Technology"}' \
  --prompt-template 'Category: {label}\nNews article:\n' \
  --task-model "${WORK_ROOT}/models/distilbert-base-uncased" \
  --generator-model "${WORK_ROOT}/models/distilgpt2" \
  --h5-python "${CONDA_ROOT}/envs/heimdallm-kdd/bin/python" \
  --fed-python "${CONDA_ROOT}/envs/heimdallm-kdd/bin/python" \
  --generator-python "${CONDA_ROOT}/envs/heimdallm-ton/bin/python" \
  --results-root "${WORK_ROOT}/results" \
  --gpu-ids 0 --generator-device cuda:0 --mpi-workers 1 --allow-gpu-sharing "$@"
