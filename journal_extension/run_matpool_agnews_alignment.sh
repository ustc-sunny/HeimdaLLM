#!/usr/bin/env bash
# Repeat the fixed-checkpoint diagnostic at three task-model initializations.
# All three use the same seed-57 synthetic set; these are NOT independent
# end-to-end generation/training repetitions.
set -Eeuo pipefail
WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/root/heimdallm-work}"
CONDA_ROOT="${HEIMDALLM_CONDA_ROOT:-/root/miniconda3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${WORK_ROOT}/results/matpool_agnews_nondp_feasibility_v1"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for seed in 1 2 3; do
  output="${RUN_ROOT}/alignment_seed_${seed}.json"
  [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 1; }
  "${CONDA_ROOT}/envs/heimdallm-kdd/bin/python" \
    "${SCRIPT_DIR}/gradient_guidance_alignment.py" \
    --data-file "${WORK_ROOT}/data/fednlp_data/data_files/agnews_data.h5" \
    --partition-file "${WORK_ROOT}/data/fednlp_data/partition_files/agnews_partition.h5" \
    --partition-method uniform_client_1000 --client-ids 1,21 \
    --real-control-client-ids 800,801,802,803,804,805,806,807 \
    --dev-client-ids 900,901,902,903,904,905,906,907,908 \
    --synthetic "${RUN_ROOT}/seed_57/synthetic/generated/synthetic.jsonl" \
    --model-path "${WORK_ROOT}/models/distilbert-base-uncased" \
    --samples-per-label 32 --dev-samples-per-label 64 \
    --batch-size 16 --zo-dev-batch-size 128 --zo-random-directions 16 \
    --seed "$seed" --device cuda:0 --output "$output" \
    > "${RUN_ROOT}/alignment_seed_${seed}.log" 2>&1
done
