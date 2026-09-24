#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_TC_DIR="${PROJECT_ROOT}/experiments/distributed/transformer_exps/run_tc_exps"

DATASET="sst_2"
TARGET_PER_LABEL=""
STAGE="smoke"
PHASE="all"
CONDITION="paired"
EVALUATION_MODE="dev"
LOCKED_CONFIG=0
SEEDS=""
RUN_ID=""
RESULTS_ROOT="/home/zzkevin/HeimdaLLM+/results/non_dp_feasibility"
STAGING_ROOT="${TMPDIR:-/tmp}/heimdallm_nondp_private"
REAL_DATA="/home/zzkevin/HeimdaLLM/xiexiu-final/fednlp_data/data_files/${DATASET}_data.h5"
REAL_PARTITION="/home/zzkevin/HeimdaLLM/xiexiu-final/fednlp_data/partition_files/${DATASET}_partition.h5"
SOURCE_PARTITION_METHOD="uniform"
DEV_CLIENT_IDS="90,91,92,93,94,95,96,97,98"
DEV_PER_LABEL=256
DEV_SEED=57
REAL_MATCHED_CLIENT_IDS="80,81,82,83,84,85,86,87"
REAL_MATCHED_LIMIT=""
SHUFFLE_MIN_LABEL_AGREEMENT=""
SHUFFLE_MAX_LABEL_AGREEMENT=0.60
TASK_MODEL="/home/zzkevin/models/distilbert-base-uncased"
GENERATOR_MODEL="/home/zzkevin/models/distilgpt2"
H5_PYTHON="/home/zzkevin/miniconda3/envs/fwdllm/bin/python"
GENERATOR_PYTHON="/home/zzkevin/miniconda3/envs/fedllm/bin/python"
FED_PYTHON="/home/zzkevin/miniconda3/envs/fwdllm/bin/python"
MPI_LAUNCHER="mpirun"
CLIENT_IDS=""
ROUNDS=""
EVAL_EVERY=""
CLIENTS_PER_ROUND=""
MPI_WORKERS=1
SAMPLE_LIMIT_PER_CLIENT=""
GENERATOR_EPOCHS=""
GENERATOR_MAX_LENGTH=64
GENERATOR_MAX_NEW_TOKENS=48
LABEL_NAMES_JSON="{}"
GENERATOR_MODE="non_dp"
DP_TARGET_EPSILON=""
DP_NOISE_MULTIPLIER=""
DP_DELTA="1e-5"
DP_MAX_GRAD_NORM="1.0"
INCLUDE_PUBLIC_SYNTHETIC=0
ADAPTER_INIT_SEED=57
PROMPT_TEMPLATE=$'Sentiment: {label}\nMovie review:\n'
TIMEOUT_MINUTES=""
GPU_IDS="auto"
GENERATOR_DEVICE="auto"
MIN_FREE_MIB_PER_PROCESS=4096
MIN_GENERATOR_FREE_MIB=3000
ALLOW_GPU_SHARING=0
MAX_VAR_RETRIES=0
RESUME=0
PREFLIGHT_ONLY=0
PRINT_PLAN=0

usage() {
    cat <<'EOF'
Run paired SST-2 true-NonDP synthetic-guidance feasibility experiments.

Usage:
  bash journal_extension/run_sst2_nondp_feasibility.sh [options]

Main options:
  --dataset sst_2|agnews       Task identifier (default sst_2)
  --stage smoke|pilot          smoke: 2 clients/5 rounds/seed 57;
                               pilot: 8 clients/300 rounds/seeds 1,2,3
  --phase all|prepare|train    Generate/pack and train, or run one phase only
  --condition NAME             paired (default), or one arm: no-cloud,
                               client-syn, public-syn, same-source-real-matched,
                               real-matched, shuffled-label
  --include-public-synthetic   Prepare and run a pretrained-only public-syn arm
  --evaluation dev|final-test Default dev uses reserved client train IDs 90-98
  --locked-config             Required to access official SST-2 final test
  --run-id ID                  Stable result directory name (timestamp by default)
  --resume                     Reuse completed artifacts under an existing run-id
  --seeds CSV                  Paired seeds, for example 1,2,3
  --results-root PATH          Persistent output root
  --staging-root PATH          Private train-text staging root (mode 0700)

Data and model options:
  --real-data PATH
  --real-partition PATH
  --source-partition-method NAME
  --dev-client-ids CSV
  --dev-per-label N
  --dev-seed N
  --real-matched-client-ids CSV
  --real-matched-limit N       Per-client stratified limit (smoke 2, pilot 16)
  --task-model PATH
  --generator-model PATH
  --client-ids CSV             Source uniform clients used both for generation
                               and the fixed real pilot partition

Training options:
  --rounds N
  --eval-every N               N=rounds evaluates round 0 and the final round
  --clients-per-round N        Logical clients sampled each round
  --mpi-workers N              Physical worker processes; total MPI ranks=N+2
  --timeout-minutes N
  --max-var-retries N          Adaptive extra directions after a high variance
                               estimate; default 0 fixes the query budget

Runtime options:
  --h5-python PATH             Python with h5py (legacy fwdllm env)
  --generator-python PATH      Python with torch/transformers/peft
  --fed-python PATH            Python with HeimdaLLM FedFwd dependencies
  --mpi-launcher PATH
  --gpu-ids auto|CSV           Distinct logical CUDA ids (CUDA_VISIBLE_DEVICES
                               must be unset); auto picks GPUs with most free RAM
  --generator-device auto|cpu|cuda:N
  --min-free-mib-per-process N
  --min-generator-free-mib N
  --allow-gpu-sharing          Required if MPI ranks exceed selected GPUs
  --sample-limit-per-client N  Label-stratified generator-training subset
  --generator-epochs N
  --generator-max-length N
  --generator-max-new-tokens N
  --label-names-json JSON      Public raw-label to category-name mapping
  --generator-mode non_dp|dp  Client-local generator training mechanism
  --dp-target-epsilon FLOAT   Calibrate DP noise to this per-client epsilon
  --dp-noise-multiplier FLOAT Use an explicit DP noise multiplier instead
  --dp-delta FLOAT            Per-client delta (default 1e-5)
  --dp-max-grad-norm FLOAT    Per-example global clipping norm (default 1.0)
  --target-per-label N         Total generated quota per class across clients
  --adapter-init-seed N        Shared fresh LoRA initialization seed (default 57)
  --prompt-template TEXT       Must contain {label}; SST-2 template is the default
  --preflight-only             Validate paths/dependencies/GPU capacity, then exit
  --print-plan                 Print resolved settings without touching files
  -h, --help

The No-Cloud arm reuses the exact same synthetic H5 for artifact provenance but
sets alpha=1.0 and skips cloud BP; it still crosses the same synchronization
barrier.  The Client-Syn arm sets alpha=0.5.  Rank 1 and workers always use real
SST-2; only rank 0 receives the explicit synthetic H5 paths.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 ]] || die "option $1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) need_value "$@"; DATASET="$2"; shift 2 ;;
        --target-per-label) need_value "$@"; TARGET_PER_LABEL="$2"; shift 2 ;;
        --stage) need_value "$@"; STAGE="$2"; shift 2 ;;
        --phase) need_value "$@"; PHASE="$2"; shift 2 ;;
        --condition) need_value "$@"; CONDITION="$2"; shift 2 ;;
        --evaluation) need_value "$@"; EVALUATION_MODE="$2"; shift 2 ;;
        --locked-config) LOCKED_CONFIG=1; shift ;;
        --run-id) need_value "$@"; RUN_ID="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --seeds) need_value "$@"; SEEDS="$2"; shift 2 ;;
        --results-root) need_value "$@"; RESULTS_ROOT="$2"; shift 2 ;;
        --staging-root) need_value "$@"; STAGING_ROOT="$2"; shift 2 ;;
        --real-data) need_value "$@"; REAL_DATA="$2"; shift 2 ;;
        --real-partition) need_value "$@"; REAL_PARTITION="$2"; shift 2 ;;
        --source-partition-method) need_value "$@"; SOURCE_PARTITION_METHOD="$2"; shift 2 ;;
        --dev-client-ids) need_value "$@"; DEV_CLIENT_IDS="$2"; shift 2 ;;
        --dev-per-label) need_value "$@"; DEV_PER_LABEL="$2"; shift 2 ;;
        --dev-seed) need_value "$@"; DEV_SEED="$2"; shift 2 ;;
        --real-matched-client-ids) need_value "$@"; REAL_MATCHED_CLIENT_IDS="$2"; shift 2 ;;
        --real-matched-limit) need_value "$@"; REAL_MATCHED_LIMIT="$2"; shift 2 ;;
        --task-model) need_value "$@"; TASK_MODEL="$2"; shift 2 ;;
        --generator-model) need_value "$@"; GENERATOR_MODEL="$2"; shift 2 ;;
        --client-ids) need_value "$@"; CLIENT_IDS="$2"; shift 2 ;;
        --rounds) need_value "$@"; ROUNDS="$2"; shift 2 ;;
        --eval-every) need_value "$@"; EVAL_EVERY="$2"; shift 2 ;;
        --clients-per-round) need_value "$@"; CLIENTS_PER_ROUND="$2"; shift 2 ;;
        --mpi-workers) need_value "$@"; MPI_WORKERS="$2"; shift 2 ;;
        --timeout-minutes) need_value "$@"; TIMEOUT_MINUTES="$2"; shift 2 ;;
        --max-var-retries) need_value "$@"; MAX_VAR_RETRIES="$2"; shift 2 ;;
        --h5-python) need_value "$@"; H5_PYTHON="$2"; shift 2 ;;
        --generator-python) need_value "$@"; GENERATOR_PYTHON="$2"; shift 2 ;;
        --fed-python) need_value "$@"; FED_PYTHON="$2"; shift 2 ;;
        --mpi-launcher) need_value "$@"; MPI_LAUNCHER="$2"; shift 2 ;;
        --gpu-ids) need_value "$@"; GPU_IDS="$2"; shift 2 ;;
        --generator-device) need_value "$@"; GENERATOR_DEVICE="$2"; shift 2 ;;
        --min-free-mib-per-process) need_value "$@"; MIN_FREE_MIB_PER_PROCESS="$2"; shift 2 ;;
        --min-generator-free-mib) need_value "$@"; MIN_GENERATOR_FREE_MIB="$2"; shift 2 ;;
        --allow-gpu-sharing) ALLOW_GPU_SHARING=1; shift ;;
        --sample-limit-per-client) need_value "$@"; SAMPLE_LIMIT_PER_CLIENT="$2"; shift 2 ;;
        --generator-max-length) need_value "$@"; GENERATOR_MAX_LENGTH="$2"; shift 2 ;;
        --generator-max-new-tokens) need_value "$@"; GENERATOR_MAX_NEW_TOKENS="$2"; shift 2 ;;
        --label-names-json) need_value "$@"; LABEL_NAMES_JSON="$2"; shift 2 ;;
        --generator-mode) need_value "$@"; GENERATOR_MODE="$2"; shift 2 ;;
        --dp-target-epsilon) need_value "$@"; DP_TARGET_EPSILON="$2"; shift 2 ;;
        --dp-noise-multiplier) need_value "$@"; DP_NOISE_MULTIPLIER="$2"; shift 2 ;;
        --dp-delta) need_value "$@"; DP_DELTA="$2"; shift 2 ;;
        --dp-max-grad-norm) need_value "$@"; DP_MAX_GRAD_NORM="$2"; shift 2 ;;
        --include-public-synthetic) INCLUDE_PUBLIC_SYNTHETIC=1; shift ;;
        --generator-epochs) need_value "$@"; GENERATOR_EPOCHS="$2"; shift 2 ;;
        --adapter-init-seed) need_value "$@"; ADAPTER_INIT_SEED="$2"; shift 2 ;;
        --prompt-template) need_value "$@"; PROMPT_TEMPLATE="$2"; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --print-plan) PRINT_PLAN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$DATASET" in sst_2|agnews) ;; *) die "unsupported dataset: $DATASET" ;; esac

case "$STAGE" in
    smoke)
        [[ -n "$SEEDS" ]] || SEEDS="57"
        [[ -n "$CLIENT_IDS" ]] || CLIENT_IDS="1,21"
        [[ -n "$ROUNDS" ]] || ROUNDS=5
        [[ -n "$EVAL_EVERY" ]] || EVAL_EVERY="$ROUNDS"
        [[ -n "$SAMPLE_LIMIT_PER_CLIENT" ]] || SAMPLE_LIMIT_PER_CLIENT=16
        [[ -n "$GENERATOR_EPOCHS" ]] || GENERATOR_EPOCHS=1
        [[ -n "$TIMEOUT_MINUTES" ]] || TIMEOUT_MINUTES=60
        GENERATOR_QUOTA_FLAG="--samples-per-label"
        GENERATOR_QUOTA_VALUE=4
        [[ -n "$REAL_MATCHED_LIMIT" ]] || REAL_MATCHED_LIMIT=2
        [[ -n "$SHUFFLE_MIN_LABEL_AGREEMENT" ]] || SHUFFLE_MIN_LABEL_AGREEMENT=0.0
        ;;
    pilot)
        [[ -n "$SEEDS" ]] || SEEDS="1,2,3"
        [[ -n "$CLIENT_IDS" ]] || CLIENT_IDS="1,21,22,27,28,36,52,65"
        [[ -n "$ROUNDS" ]] || ROUNDS=300
        [[ -n "$EVAL_EVERY" ]] || EVAL_EVERY="$ROUNDS"
        [[ -n "$GENERATOR_EPOCHS" ]] || GENERATOR_EPOCHS=5
        [[ -n "$TIMEOUT_MINUTES" ]] || TIMEOUT_MINUTES=720
        GENERATOR_QUOTA_FLAG="--target-per-label"
        GENERATOR_QUOTA_VALUE=64
        [[ -n "$REAL_MATCHED_LIMIT" ]] || REAL_MATCHED_LIMIT=16
        [[ -n "$SHUFFLE_MIN_LABEL_AGREEMENT" ]] || SHUFFLE_MIN_LABEL_AGREEMENT=0.40
        ;;
    *) die "--stage must be smoke or pilot" ;;
esac

if [[ -n "$TARGET_PER_LABEL" ]]; then
    [[ "$TARGET_PER_LABEL" =~ ^[1-9][0-9]*$ ]] || die "target-per-label must be positive"
    GENERATOR_QUOTA_FLAG="--target-per-label"
    GENERATOR_QUOTA_VALUE="$TARGET_PER_LABEL"
fi
if [[ "$DATASET" == "agnews" ]]; then
    SHUFFLE_MIN_LABEL_AGREEMENT=0.15
    SHUFFLE_MAX_LABEL_AGREEMENT=0.35
fi
case "$PHASE" in all|prepare|train) ;; *) die "--phase must be all, prepare, or train" ;; esac
case "$GENERATOR_MODE" in non_dp|dp) ;; *) die "--generator-mode must be non_dp or dp" ;; esac
if [[ "$GENERATOR_MODE" == "dp" ]]; then
    [[ -n "$SAMPLE_LIMIT_PER_CLIENT" ]] || die "DP mode requires --sample-limit-per-client"
    [[ "$LABEL_NAMES_JSON" != "{}" ]] || die "DP mode requires a fixed public --label-names-json table"
    if [[ -n "$DP_TARGET_EPSILON" && -n "$DP_NOISE_MULTIPLIER" ]]; then
        die "choose only one of --dp-target-epsilon and --dp-noise-multiplier"
    fi
    if [[ -z "$DP_TARGET_EPSILON" && -z "$DP_NOISE_MULTIPLIER" ]]; then
        die "DP mode requires --dp-target-epsilon or --dp-noise-multiplier"
    fi
fi
if [[ "$GENERATOR_MODE" == "dp" ]]; then
    GENERATOR_SCRIPT="${SCRIPT_DIR}/dp_client_synthetic.py"
else
    GENERATOR_SCRIPT="${SCRIPT_DIR}/non_dp_client_synthetic.py"
fi
case "$CONDITION" in paired|no-cloud|client-syn|public-syn|same-source-real-matched|real-matched|shuffled-label) ;; *) die "invalid --condition" ;; esac
if [[ "$CONDITION" == "public-syn" ]]; then
    INCLUDE_PUBLIC_SYNTHETIC=1
fi
if [[ "$GENERATOR_MODE" == "dp" && "$INCLUDE_PUBLIC_SYNTHETIC" -eq 1 ]]; then
    die "DP mode currently supports the client-syn/no-cloud arms only"
fi
case "$EVALUATION_MODE" in dev|final-test) ;; *) die "--evaluation must be dev or final-test" ;; esac
if [[ "$EVALUATION_MODE" == "final-test" && "$LOCKED_CONFIG" -ne 1 ]]; then
    die "--evaluation final-test requires --locked-config after configuration selection"
fi

for numeric_value in "$ROUNDS" "$EVAL_EVERY" "$CLIENTS_PER_ROUND" "$MPI_WORKERS" \
    "$GENERATOR_EPOCHS" "$GENERATOR_MAX_LENGTH" "$GENERATOR_MAX_NEW_TOKENS" "$TIMEOUT_MINUTES" "$MIN_FREE_MIB_PER_PROCESS" \
    "$MIN_GENERATOR_FREE_MIB" "$ADAPTER_INIT_SEED" "$DEV_PER_LABEL" "$DEV_SEED" \
    "$REAL_MATCHED_LIMIT"; do
    if [[ -n "$numeric_value" && ! "$numeric_value" =~ ^[1-9][0-9]*$ ]]; then
        die "positive integer expected, got: $numeric_value"
    fi
done
[[ "$MAX_VAR_RETRIES" =~ ^[0-9]+$ ]] || die "non-negative integer expected for --max-var-retries"
if [[ "$MAX_VAR_RETRIES" -eq 0 ]]; then
    FIXED_ZO_QUERY_BUDGET=true
else
    FIXED_ZO_QUERY_BUDGET=false
fi

IFS=',' read -r -a SOURCE_CLIENT_ARRAY <<< "$CLIENT_IDS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
IFS=',' read -r -a DEV_CLIENT_ARRAY <<< "$DEV_CLIENT_IDS"
IFS=',' read -r -a REAL_MATCHED_CLIENT_ARRAY <<< "$REAL_MATCHED_CLIENT_IDS"
LOGICAL_CLIENT_COUNT=${#SOURCE_CLIENT_ARRAY[@]}
[[ "$LOGICAL_CLIENT_COUNT" -gt 0 ]] || die "no source clients selected"
[[ -n "$CLIENTS_PER_ROUND" ]] || CLIENTS_PER_ROUND="$LOGICAL_CLIENT_COUNT"
[[ "$CLIENTS_PER_ROUND" -le "$LOGICAL_CLIENT_COUNT" ]] || die "clients-per-round exceeds fixed pilot clients"
[[ "$MPI_WORKERS" -le "$CLIENTS_PER_ROUND" ]] || die "mpi-workers exceeds clients-per-round and would create empty worker assignments"
[[ "$MPI_WORKERS" -eq 1 ]] || die "this reliable pilot supports exactly one MPI worker; it simulates all selected logical clients"
[[ "$EVAL_EVERY" -le "$ROUNDS" ]] || die "eval-every cannot exceed rounds"
for seed in "${SEED_ARRAY[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed: $seed"
done
for client_id in "${SOURCE_CLIENT_ARRAY[@]}"; do
    [[ "$client_id" =~ ^[0-9]+$ ]] || die "invalid client id: $client_id"
done
for client_id in "${DEV_CLIENT_ARRAY[@]}" "${REAL_MATCHED_CLIENT_ARRAY[@]}"; do
    [[ "$client_id" =~ ^[0-9]+$ ]] || die "invalid reserved client id: $client_id"
done
for source_id in "${SOURCE_CLIENT_ARRAY[@]}"; do
    for reserved_id in "${DEV_CLIENT_ARRAY[@]}" "${REAL_MATCHED_CLIENT_ARRAY[@]}"; do
        [[ "$source_id" != "$reserved_id" ]] || die "FL/generator client $source_id overlaps a reserved client"
    done
done
for dev_id in "${DEV_CLIENT_ARRAY[@]}"; do
    for control_id in "${REAL_MATCHED_CLIENT_ARRAY[@]}"; do
        [[ "$dev_id" != "$control_id" ]] || die "dev client $dev_id overlaps a Real-Matched client"
    done
done

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="${DATASET}_nondp_${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "run-id may contain only letters, digits, dot, underscore, and hyphen"

RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
SHARED_DIR="${RUN_DIR}/shared"
PILOT_PARTITION="${SHARED_DIR}/${DATASET}_${STAGE}_${EVALUATION_MODE}_fixed_clients_partition.h5"
PILOT_METHOD="pilot_uniform_${LOGICAL_CLIENT_COUNT}"
GPU_MAPPING_FILE="${SHARED_DIR}/gpu_mapping.yaml"
MPI_HOST_FILE="${SHARED_DIR}/mpi_host_file"
PROCESS_COUNT=$((MPI_WORKERS + 2))
PRIVATE_STAGING_RUN="${STAGING_ROOT}/${RUN_ID}"
RUN_SPEC_FILE="${SHARED_DIR}/run_spec.json"

print_resolved_plan() {
    cat <<EOF
run_id=${RUN_ID}
dataset=${DATASET}
stage=${STAGE}
phase=${PHASE}
condition=${CONDITION}
evaluation_mode=${EVALUATION_MODE}
locked_config=${LOCKED_CONFIG}
seeds=${SEEDS}
source_client_ids=${CLIENT_IDS}
dev_client_ids=${DEV_CLIENT_IDS}
dev_per_label=${DEV_PER_LABEL}
real_matched_client_ids=${REAL_MATCHED_CLIENT_IDS}
real_matched_limit_per_client=${REAL_MATCHED_LIMIT}
shuffled_label_agreement_interval=[${SHUFFLE_MIN_LABEL_AGREEMENT},${SHUFFLE_MAX_LABEL_AGREEMENT}]
include_public_synthetic=${INCLUDE_PUBLIC_SYNTHETIC}
logical_client_count=${LOGICAL_CLIENT_COUNT}
clients_per_round=${CLIENTS_PER_ROUND}
mpi_workers=${MPI_WORKERS}
mpi_processes=${PROCESS_COUNT} (cloud + server + ${MPI_WORKERS} worker)
rounds=${ROUNDS}
eval_every=${EVAL_EVERY}
pre_update_evaluation=true
max_var_retries=${MAX_VAR_RETRIES}
fixed_zo_query_budget=${FIXED_ZO_QUERY_BUDGET}
real_eval_data=${REAL_DATA}
fixed_real_partition=${PILOT_PARTITION}:${PILOT_METHOD}
generator_mode=${GENERATOR_MODE}
generator_quota=${GENERATOR_QUOTA_FLAG} ${GENERATOR_QUOTA_VALUE}
dp_target_epsilon=${DP_TARGET_EPSILON:-none}
dp_noise_multiplier=${DP_NOISE_MULTIPLIER:-calibrated}
dp_delta=${DP_DELTA}
dp_max_grad_norm=${DP_MAX_GRAD_NORM}
adapter_init_seed=${ADAPTER_INIT_SEED}
results=${RUN_DIR}
private_staging=${PRIVATE_STAGING_RUN}
EOF
}

if [[ "$PRINT_PLAN" -eq 1 ]]; then
    print_resolved_plan
    exit 0
fi

require_file() {
    [[ -f "$1" ]] || die "required file not found: $1"
}

require_executable() {
    if [[ "$1" == */* ]]; then
        [[ -x "$1" ]] || die "executable not found: $1"
    else
        command -v "$1" >/dev/null 2>&1 || die "command not found: $1"
    fi
}

preflight_dependencies() {
    require_file "$REAL_DATA"
    require_file "$REAL_PARTITION"
    [[ -d "$TASK_MODEL" ]] || die "task model directory not found: $TASK_MODEL"
    if [[ "$PHASE" != "train" ]]; then
        [[ -d "$GENERATOR_MODEL" ]] || die "generator model directory not found: $GENERATOR_MODEL"
    fi
    require_file "${SCRIPT_DIR}/prepare_sst2_pilot_partition.py"
    require_file "${SCRIPT_DIR}/summarize_fed_pilot.py"
    require_file "${SCRIPT_DIR}/manage_pilot_run_spec.py"
    require_file "${SCRIPT_DIR}/shuffle_synthetic_labels.py"
    require_file "${SCRIPT_DIR}/validate_guidance_controls.py"
    require_file "${SCRIPT_DIR}/validate_public_generator_control.py"
    require_file "${SCRIPT_DIR}/validate_nondp_smoke.py"
    require_file "${SCRIPT_DIR}/export_client_train_jsonl.py"
    require_file "${SCRIPT_DIR}/build_matched_real_control.py"
    require_file "${SCRIPT_DIR}/non_dp_client_synthetic.py"
    if [[ "$GENERATOR_MODE" == "dp" ]]; then
        require_file "${SCRIPT_DIR}/dp_client_synthetic.py"
        require_file "${SCRIPT_DIR}/validate_dp_release.py"
    fi
    require_file "${SCRIPT_DIR}/pack_synthetic_h5.py"
    require_file "${RUN_TC_DIR}/fedavg_main_tc.py"
    require_executable "$H5_PYTHON"
    "$H5_PYTHON" -c 'import h5py, numpy' >/dev/null
    if [[ "$PHASE" != "train" ]]; then
        require_executable "$GENERATOR_PYTHON"
        if [[ "$GENERATOR_MODE" == "dp" ]]; then
            "$GENERATOR_PYTHON" -c 'import opacus, torch, transformers, peft' >/dev/null
        else
            "$GENERATOR_PYTHON" -c 'import torch, transformers, peft' >/dev/null
        fi
    fi
    if [[ "$PHASE" != "prepare" ]]; then
        require_executable "$FED_PYTHON"
        require_executable "$MPI_LAUNCHER"
        require_executable timeout
        "$FED_PYTHON" -c 'import functorch, h5py, mpi4py, torch, transformers; import transformers.adapters' >/dev/null
    fi
}

write_or_check_run_spec() {
    local sample_limit_value="${SAMPLE_LIMIT_PER_CLIENT:-all}"
    local downstream_arms="no_cloud:1.0,client_syn:0.5,same_source_real_matched:0.5,real_matched:0.5,shuffled_label:0.5"
    if [[ "$INCLUDE_PUBLIC_SYNTHETIC" -eq 1 ]]; then
        downstream_arms="no_cloud:1.0,client_syn:0.5,public_syn:0.5,same_source_real_matched:0.5,real_matched:0.5,shuffled_label:0.5"
    fi
    local -a command=(
        "$H5_PYTHON" "${SCRIPT_DIR}/manage_pilot_run_spec.py"
        --output "$RUN_SPEC_FILE"
        --value "run_id=${RUN_ID}"
        --value "dataset=${DATASET}"
        --value "stage=${STAGE}"
        --value "seeds=${SEEDS}"
        --value "source_client_ids=${CLIENT_IDS}"
        --value "logical_client_count=${LOGICAL_CLIENT_COUNT}"
        --value "clients_per_round=${CLIENTS_PER_ROUND}"
        --value "mpi_workers=${MPI_WORKERS}"
        --value "rounds=${ROUNDS}"
        --value "eval_every=${EVAL_EVERY}"
        --value "evaluation_mode=${EVALUATION_MODE}"
        --value "locked_config=${LOCKED_CONFIG}"
        --value "dev_client_ids=${DEV_CLIENT_IDS}"
        --value "dev_per_label=${DEV_PER_LABEL}"
        --value "dev_seed=${DEV_SEED}"
        --value "real_matched_client_ids=${REAL_MATCHED_CLIENT_IDS}"
        --value "real_matched_limit_per_client=${REAL_MATCHED_LIMIT}"
        --value "shuffled_label_min_agreement=${SHUFFLE_MIN_LABEL_AGREEMENT}"
        --value "shuffled_label_max_agreement=${SHUFFLE_MAX_LABEL_AGREEMENT}"
        --value "generator_model=${GENERATOR_MODEL}"
        --value "task_model=${TASK_MODEL}"
        --value "generator_quota_flag=${GENERATOR_QUOTA_FLAG}"
        --value "generator_quota_value=${GENERATOR_QUOTA_VALUE}"
        --value "generator_sample_limit_per_client=${sample_limit_value}"
        --value "generator_epochs=${GENERATOR_EPOCHS}"
        --value "generator_mode=${GENERATOR_MODE}"
        --value "dp_target_epsilon=${DP_TARGET_EPSILON:-none}"
        --value "dp_noise_multiplier=${DP_NOISE_MULTIPLIER:-calibrated}"
        --value "dp_delta=${DP_DELTA}"
        --value "dp_max_grad_norm=${DP_MAX_GRAD_NORM}"
        --value "adapter_init_seed=${ADAPTER_INIT_SEED}"
        --value "prompt_template=${PROMPT_TEMPLATE}"
        --value "generator_learning_rate=5e-4"
        --value "generator_weight_decay=0.01"
        --value "generator_batch_size=4"
        --value "generator_max_length=${GENERATOR_MAX_LENGTH}"
        --value "generator_label_names_json=${LABEL_NAMES_JSON}"
        --value "include_public_synthetic=${INCLUDE_PUBLIC_SYNTHETIC}"
        --value "generator_generation_batch_size=4"
        --value "generator_max_new_tokens=${GENERATOR_MAX_NEW_TOKENS}"
        --value "generator_temperature=0.8"
        --value "generator_top_p=0.9"
        --value "generator_top_k=0"
        --value "generator_repetition_penalty=1.05"
        --value "generator_lora_r=8"
        --value "generator_lora_alpha=16"
        --value "generator_lora_dropout=0.05"
        --value "downstream_learning_rate=0.01"
        --value "downstream_train_batch_size=8"
        --value "downstream_eval_batch_size=8"
        --value "downstream_max_seq_length=64"
        --value "downstream_arms=${downstream_arms}"
        --value "downstream_peft_method=adapter"
        --value "downstream_var_control=true"
        --value "downstream_perturbation_sampling=true"
        --value "downstream_max_var_retries=${MAX_VAR_RETRIES}"
        --value "fixed_zo_query_budget=${FIXED_ZO_QUERY_BUDGET}"
        --value "downstream_guidance_first_round=0"
        --value "downstream_pre_update_evaluation=true"
        --value "downstream_logical_client_schedule_logging=true"
        --value "downstream_guidance_schedule=every_round_dense_feasibility_diagnostic"
        --value "downstream_client_fd_dropout=disabled_eval_mode"
        --value "downstream_cloud_train_batch_size=8"
        --value "downstream_direction_parameterization=normalized_z_h"
        --value "downstream_direction_formula=beta*sqrt(alpha/n)*z+beta*sqrt((1-alpha)/m)*Vz_g"
        --value "downstream_estimator=central_fd_times_z_h"
        --value "downstream_estimator_outer_n_multiplier=none"
        --value "downstream_finite_difference_step=0.01"
        --value "source_partition_method=${SOURCE_PARTITION_METHOD}"
        --value "h5_python=${H5_PYTHON}"
        --value "generator_python=${GENERATOR_PYTHON}"
        --value "fed_python=${FED_PYTHON}"
        --value "mpi_launcher=${MPI_LAUNCHER}"
        --value "gpu_ids_request=${GPU_IDS}"
        --value "generator_device_request=${GENERATOR_DEVICE}"
        --value "min_free_mib_per_process=${MIN_FREE_MIB_PER_PROCESS}"
        --value "min_generator_free_mib=${MIN_GENERATOR_FREE_MIB}"
        --value "allow_gpu_sharing=${ALLOW_GPU_SHARING}"
        --file "real_data=${REAL_DATA}"
        --file "real_partition=${REAL_PARTITION}"
        --file "runner=${SCRIPT_DIR}/run_sst2_nondp_feasibility.sh"
        --file "partition_builder=${SCRIPT_DIR}/prepare_sst2_pilot_partition.py"
        --file "exporter=${SCRIPT_DIR}/export_client_train_jsonl.py"
        --file "same_source_real_builder=${SCRIPT_DIR}/build_matched_real_control.py"
        --file "generator=${GENERATOR_SCRIPT}"
        --file "packer=${SCRIPT_DIR}/pack_synthetic_h5.py"
        --file "label_shuffler=${SCRIPT_DIR}/shuffle_synthetic_labels.py"
        --file "control_validator=${SCRIPT_DIR}/validate_guidance_controls.py"
        --file "public_control_validator=${SCRIPT_DIR}/validate_public_generator_control.py"
        --file "cross_arm_validator=${SCRIPT_DIR}/validate_nondp_smoke.py"
        --file "summarizer=${SCRIPT_DIR}/summarize_fed_pilot.py"
        --file "fed_entrypoint=${RUN_TC_DIR}/fedavg_main_tc.py"
        --file "fed_initializer=${PROJECT_ROOT}/experiments/distributed/transformer_exps/initializer.py"
        --file "base_data_manager=${PROJECT_ROOT}/data_manager/base_data_manager.py"
        --file "fed_server_manager=${PROJECT_ROOT}/FedML/fedml_api/distributed/fedsgd/FedSgdServerManager.py"
        --file "fed_cloud_manager=${PROJECT_ROOT}/FedML/fedml_api/distributed/fedsgd/FedSgdCloudManager.py"
        --file "fed_client_manager=${PROJECT_ROOT}/FedML/fedml_api/distributed/fedsgd/FedSgdClientManager.py"
        --file "fed_server=${PROJECT_ROOT}/FedML/fedml_api/distributed/fedsgd/FedSgdServer.py"
        --file "forward_trainer=${PROJECT_ROOT}/forward_training/tc_transformer_trainer_distribute.py"
    )
    if [[ "$RESUME" -eq 1 ]]; then
        command+=(--resume)
    fi
    run_logged_command "${SHARED_DIR}/run_spec.command.txt" \
        "${SHARED_DIR}/run_spec.log" "${command[@]}"
}

declare -a DETECTED_GPU_IDS=()
declare -a DETECTED_GPU_FREE=()

read_gpu_state() {
    [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || die "unset CUDA_VISIBLE_DEVICES so YAML ids and nvidia-smi ids agree"
    require_executable nvidia-smi
    while IFS=',' read -r raw_id raw_free; do
        raw_id="${raw_id//[[:space:]]/}"
        raw_free="${raw_free//[[:space:]]/}"
        [[ "$raw_id" =~ ^[0-9]+$ && "$raw_free" =~ ^[0-9]+$ ]] || continue
        DETECTED_GPU_IDS+=("$raw_id")
        DETECTED_GPU_FREE[raw_id]="$raw_free"
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    [[ "${#DETECTED_GPU_IDS[@]}" -gt 0 ]] || die "nvidia-smi returned no CUDA GPUs"
}

select_gpu_ids() {
    local requested_count="$1"
    local -a selected=()
    local id
    if [[ "$GPU_IDS" == "auto" ]]; then
        while read -r id; do
            [[ -n "$id" ]] && selected+=("$id")
        done < <(
            for id in "${DETECTED_GPU_IDS[@]}"; do
                printf '%s %s\n' "${DETECTED_GPU_FREE[id]}" "$id"
            done | sort -k1,1nr -k2,2n | awk '{print $2}'
        )
        if [[ "$requested_count" -lt "${#selected[@]}" ]]; then
            selected=("${selected[@]:0:requested_count}")
        fi
    else
        IFS=',' read -r -a selected <<< "$GPU_IDS"
        for id in "${selected[@]}"; do
            [[ "$id" =~ ^[0-9]+$ ]] || die "invalid GPU id: $id"
            [[ -n "${DETECTED_GPU_FREE[id]:-}" ]] || die "GPU id is not visible: $id"
        done
    fi
    [[ "${#selected[@]}" -gt 0 ]] || die "no GPUs selected"
    printf '%s\n' "${selected[@]}"
}

prepare_gpu_mapping() {
    read_gpu_state
    local -a selected=()
    local id rank max_id=0
    while read -r id; do selected+=("$id"); done < <(select_gpu_ids "$PROCESS_COUNT")
    if [[ "$PROCESS_COUNT" -gt "${#selected[@]}" && "$ALLOW_GPU_SHARING" -ne 1 ]]; then
        die "${PROCESS_COUNT} MPI ranks need ${PROCESS_COUNT} GPUs by default; selected ${#selected[@]}. Keep --mpi-workers small or pass --allow-gpu-sharing after checking memory."
    fi

    for id in "${DETECTED_GPU_IDS[@]}"; do
        (( id > max_id )) && max_id="$id"
    done
    local -a counts=()
    for ((id=0; id<=max_id; id++)); do counts[id]=0; done
    for ((rank=0; rank<PROCESS_COUNT; rank++)); do
        id="${selected[rank % ${#selected[@]}]}"
        counts[id]=$((counts[id] + 1))
    done
    for id in "${selected[@]}"; do
        local required=$((counts[id] * MIN_FREE_MIB_PER_PROCESS))
        if [[ "${DETECTED_GPU_FREE[id]}" -lt "$required" ]]; then
            die "GPU ${id} has ${DETECTED_GPU_FREE[id]} MiB free; mapping ${counts[id]} process(es) requires at least ${required} MiB"
        fi
    done

    if [[ "$PREFLIGHT_ONLY" -ne 1 ]]; then
        mkdir -p "$SHARED_DIR"
        local hostname_value
        hostname_value="$(hostname)"
        {
            printf 'mapping_pilot:\n'
            printf '    %s: [' "$hostname_value"
            for ((id=0; id<=max_id; id++)); do
                (( id > 0 )) && printf ', '
                printf '%d' "${counts[id]}"
            done
            printf ']\n'
        } > "$GPU_MAPPING_FILE"
        # Rental images may use Open MPI; the original server used MPICH Hydra.
        local mpi_version
        mpi_version="$("$MPI_LAUNCHER" --version 2>&1)"
        if [[ "$mpi_version" == *"Open MPI"* || "$mpi_version" == *"OpenRTE"* ]]; then
            printf '%s slots=%d\n' "$hostname_value" "$PROCESS_COUNT" > "$MPI_HOST_FILE"
        else
            printf '%s:%d\n' "$hostname_value" "$PROCESS_COUNT" > "$MPI_HOST_FILE"
        fi
    fi

    if [[ "$GENERATOR_DEVICE" == "auto" ]]; then
        GENERATOR_DEVICE="cuda:${selected[0]}"
    fi
    if [[ "$GENERATOR_DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
        id="${BASH_REMATCH[1]}"
        [[ -n "${DETECTED_GPU_FREE[id]:-}" ]] || die "generator GPU is not visible: $id"
        [[ "${DETECTED_GPU_FREE[id]}" -ge "$MIN_GENERATOR_FREE_MIB" ]] || die "generator GPU ${id} has less than ${MIN_GENERATOR_FREE_MIB} MiB free"
    elif [[ "$GENERATOR_DEVICE" != "cpu" ]]; then
        die "generator-device must be auto, cpu, or cuda:N"
    fi
}

write_command_file() {
    local destination="$1"
    shift
    {
        printf '# working_directory=%q\n' "$PWD"
        printf '%q ' "$@"
        printf '\n'
    } > "$destination"
}

run_logged_command() {
    local command_file="$1"
    local log_file="$2"
    shift 2
    write_command_file "$command_file" "$@"
    "$@" > "$log_file" 2>&1
}

prepare_fixed_partition() {
    if [[ -f "$PILOT_PARTITION" ]]; then
        [[ "$RESUME" -eq 1 ]] || die "fixed partition already exists: $PILOT_PARTITION"
        return
    fi
    mkdir -p "$SHARED_DIR"
    local -a command=(
        "$H5_PYTHON" "${SCRIPT_DIR}/prepare_sst2_pilot_partition.py"
        --data-file "$REAL_DATA"
        --partition-file "$REAL_PARTITION"
        --partition-method "$SOURCE_PARTITION_METHOD"
        --client-ids "$CLIENT_IDS"
        --output "$PILOT_PARTITION"
        --output-method "$PILOT_METHOD"
        --evaluation-mode "$EVALUATION_MODE"
        --dev-client-ids "$DEV_CLIENT_IDS"
        --dev-per-label "$DEV_PER_LABEL"
        --dev-seed "$DEV_SEED"
    )
    if [[ "$DATASET" == "sst_2" ]]; then
        command+=(--require-balanced --require-equal-train-size)
    fi
    run_logged_command "${SHARED_DIR}/prepare_partition.command.txt" \
        "${SHARED_DIR}/prepare_partition.log" "${command[@]}"
}

prepare_seed_synthetic() {
    local seed="$1"
    local seed_dir="${RUN_DIR}/seed_${seed}"
    local synthetic_dir="${seed_dir}/synthetic"
    local staging_dir="${PRIVATE_STAGING_RUN}/seed_${seed}"
    local generated_dir="${synthetic_dir}/generated"
    local cloud_data="${synthetic_dir}/${DATASET}_client_synthetic_data.h5"
    local cloud_partition="${synthetic_dir}/${DATASET}_client_synthetic_partition.h5"
    local cloud_manifest="${synthetic_dir}/pack_manifest.json"
    local public_dir="${seed_dir}/public_synthetic"
    local public_generated_dir="${public_dir}/generated"
    local public_data="${public_dir}/${DATASET}_public_synthetic_data.h5"
    local public_partition="${public_dir}/${DATASET}_public_synthetic_partition.h5"
    local public_manifest="${public_dir}/pack_manifest.json"
    local same_source_staging_dir="${PRIVATE_STAGING_RUN}/seed_${seed}_same_source_real"
    local same_source_control_dir="${seed_dir}/controls/same_source_real_matched"
    local same_source_jsonl="${same_source_staging_dir}/same_source_real_matched.jsonl"
    local same_source_build_manifest="${same_source_control_dir}/build_manifest.json"
    local same_source_data="${same_source_control_dir}/${DATASET}_same_source_real_matched_data.h5"
    local same_source_partition="${same_source_control_dir}/${DATASET}_same_source_real_matched_partition.h5"
    local same_source_pack_manifest="${same_source_control_dir}/pack_manifest.json"
    local real_staging_dir="${PRIVATE_STAGING_RUN}/seed_${seed}_real_matched"
    local real_control_dir="${seed_dir}/controls/real_matched"
    local real_data="${real_control_dir}/${DATASET}_real_matched_data.h5"
    local real_partition="${real_control_dir}/${DATASET}_real_matched_partition.h5"
    local real_manifest="${real_control_dir}/pack_manifest.json"
    local shuffled_dir="${seed_dir}/controls/shuffled_label"
    local shuffled_jsonl="${shuffled_dir}/synthetic_shuffled_labels.jsonl"
    local shuffle_manifest="${shuffled_dir}/shuffle_manifest.json"
    local shuffled_data="${shuffled_dir}/${DATASET}_shuffled_label_data.h5"
    local shuffled_partition="${shuffled_dir}/${DATASET}_shuffled_label_partition.h5"
    local shuffled_pack_manifest="${shuffled_dir}/pack_manifest.json"
    local controls_manifest="${seed_dir}/controls/validation_manifest.json"
    mkdir -p "$synthetic_dir" "$public_dir" "$same_source_control_dir" "$real_control_dir" \
        "$shuffled_dir" "$PRIVATE_STAGING_RUN" "$same_source_staging_dir"
    chmod 700 "$PRIVATE_STAGING_RUN"
    chmod 700 "$same_source_staging_dir"

    if [[ ! -f "${staging_dir}/manifest.json" ]]; then
        local -a export_command=(
            "$H5_PYTHON" "${SCRIPT_DIR}/export_client_train_jsonl.py"
            --data-file "$REAL_DATA"
            --partition-file "$REAL_PARTITION"
            --partition-method "$SOURCE_PARTITION_METHOD"
            --client-ids "$CLIENT_IDS"
            --output-dir "$staging_dir"
            --seed "$seed"
            --no-aggregate-output
        )
        if [[ -n "$SAMPLE_LIMIT_PER_CLIENT" ]]; then
            export_command+=(--sample-limit-per-client "$SAMPLE_LIMIT_PER_CLIENT")
        fi
        run_logged_command "${synthetic_dir}/export.command.txt" \
            "${synthetic_dir}/export.log" "${export_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "private staging already exists: $staging_dir"
    fi

    if [[ ! -f "${generated_dir}/manifest.json" ]]; then
        local -a generator_command
        if [[ "$GENERATOR_MODE" == "dp" ]]; then
            generator_command=(
                "$GENERATOR_PYTHON" "${SCRIPT_DIR}/dp_client_synthetic.py"
                --client-json-dir "$staging_dir"
                --model-path "$GENERATOR_MODEL"
                --output-dir "$generated_dir"
                --records-per-client "$SAMPLE_LIMIT_PER_CLIENT"
                --seed "$seed"
                --adapter-init-seed "$ADAPTER_INIT_SEED"
                --prompt-template "$PROMPT_TEMPLATE"
                "$GENERATOR_QUOTA_FLAG" "$GENERATOR_QUOTA_VALUE"
                --epochs "$GENERATOR_EPOCHS"
                --batch-size 4
                --learning-rate 5e-4
                --weight-decay 0.01
                --max-length "$GENERATOR_MAX_LENGTH"
                --label-names-json "$LABEL_NAMES_JSON"
                --delta "$DP_DELTA"
                --max-grad-norm "$DP_MAX_GRAD_NORM"
                --lora-r 8
                --lora-alpha 16
                --lora-dropout 0.05
                --generation-batch-size 4
                --max-new-tokens "$GENERATOR_MAX_NEW_TOKENS"
                --temperature 0.8
                --top-p 0.9
                --top-k 0
                --repetition-penalty 1.05
                --device "$GENERATOR_DEVICE"
                --dtype float32
                --min-free-mib "$MIN_GENERATOR_FREE_MIB"
                --memory-fraction 0.90
            )
            if [[ -n "$DP_TARGET_EPSILON" ]]; then
                generator_command+=(--target-epsilon "$DP_TARGET_EPSILON")
            else
                generator_command+=(--noise-multiplier "$DP_NOISE_MULTIPLIER")
            fi
        else
            generator_command=(
                "$GENERATOR_PYTHON" "${SCRIPT_DIR}/non_dp_client_synthetic.py"
                --client-json-dir "$staging_dir"
                --model-path "$GENERATOR_MODEL"
                --output-dir "$generated_dir"
                --seed "$seed"
                --adapter-init-seed "$ADAPTER_INIT_SEED"
                --prompt-template "$PROMPT_TEMPLATE"
                "$GENERATOR_QUOTA_FLAG" "$GENERATOR_QUOTA_VALUE"
                --require-all-labels
                --epochs "$GENERATOR_EPOCHS"
                --batch-size 4
                --learning-rate 5e-4
                --weight-decay 0.01
                --max-length "$GENERATOR_MAX_LENGTH"
                --label-names-json "$LABEL_NAMES_JSON"
                --lora-r 8
                --lora-alpha 16
                --lora-dropout 0.05
                --generation-batch-size 4
                --max-new-tokens "$GENERATOR_MAX_NEW_TOKENS"
                --temperature 0.8
                --top-p 0.9
                --top-k 0
                --repetition-penalty 1.05
                --device "$GENERATOR_DEVICE"
                --min-free-mib "$MIN_GENERATOR_FREE_MIB"
                --memory-fraction 0.90
            )
        fi
        run_logged_command "${synthetic_dir}/generate.command.txt" \
            "${synthetic_dir}/generate.log" "${generator_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "synthetic generation already exists: $generated_dir"
    fi

    if [[ "$GENERATOR_MODE" == "dp" ]]; then
        local dp_validation="${generated_dir}/validation_manifest.json"
        local -a dp_validate_command=(
            "$GENERATOR_PYTHON" "${SCRIPT_DIR}/validate_dp_release.py"
            --manifest "${generated_dir}/manifest.json"
            --output "$dp_validation"
        )
        run_logged_command "${synthetic_dir}/validate_dp.command.txt" \
            "${synthetic_dir}/validate_dp.log" "${dp_validate_command[@]}"
    fi

    if [[ ! -f "$cloud_manifest" ]]; then
        local -a pack_command=(
            "$H5_PYTHON" "${SCRIPT_DIR}/pack_synthetic_h5.py"
            --jsonl "${generated_dir}/synthetic.jsonl"
            --source-data-file "$REAL_DATA"
            --data-out "$cloud_data"
            --partition-out "$cloud_partition"
            --partition-method synthetic_cloud
            --cloud-clients 1
            --manifest-out "$cloud_manifest"
            --require-equal-label-counts
        )
        run_logged_command "${synthetic_dir}/pack.command.txt" \
            "${synthetic_dir}/pack.log" "${pack_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "packed synthetic data already exists: $cloud_manifest"
    fi

    # A single synthetic/no-cloud arm needs no real-text oracle controls.  This
    # keeps DP result directories free of the private same-source control data.
    if [[ "$CONDITION" == "client-syn" || "$CONDITION" == "no-cloud" ]]; then
        return 0
    fi

    if [[ "$INCLUDE_PUBLIC_SYNTHETIC" -eq 1 ]]; then
        if [[ ! -f "${public_generated_dir}/manifest.json" ]]; then
            local -a public_generator_command=(
                "$GENERATOR_PYTHON" "${SCRIPT_DIR}/non_dp_client_synthetic.py"
                --client-json-dir "$staging_dir"
                --model-path "$GENERATOR_MODEL"
                --output-dir "$public_generated_dir"
                --seed "$seed"
                --adapter-init-seed "$ADAPTER_INIT_SEED"
                --prompt-template "$PROMPT_TEMPLATE"
                "$GENERATOR_QUOTA_FLAG" "$GENERATOR_QUOTA_VALUE"
                --require-all-labels
                --public-generator-control
                --epochs "$GENERATOR_EPOCHS"
                --batch-size 4
                --learning-rate 5e-4
                --weight-decay 0.01
                --max-length "$GENERATOR_MAX_LENGTH"
                --label-names-json "$LABEL_NAMES_JSON"
                --lora-r 8
                --lora-alpha 16
                --lora-dropout 0.05
                --generation-batch-size 4
                --max-new-tokens "$GENERATOR_MAX_NEW_TOKENS"
                --temperature 0.8
                --top-p 0.9
                --top-k 0
                --repetition-penalty 1.05
                --device "$GENERATOR_DEVICE"
                --min-free-mib "$MIN_GENERATOR_FREE_MIB"
                --memory-fraction 0.90
            )
            run_logged_command "${public_dir}/generate.command.txt" \
                "${public_dir}/generate.log" "${public_generator_command[@]}"
        elif [[ "$RESUME" -ne 1 ]]; then
            die "public generation already exists: $public_generated_dir"
        fi

        if [[ ! -f "$public_manifest" ]]; then
            local -a public_pack_command=(
                "$H5_PYTHON" "${SCRIPT_DIR}/pack_synthetic_h5.py"
                --jsonl "${public_generated_dir}/synthetic.jsonl"
                --source-data-file "$REAL_DATA"
                --data-out "$public_data"
                --partition-out "$public_partition"
                --partition-method public_synthetic_cloud
                --cloud-clients 1
                --manifest-out "$public_manifest"
                --require-equal-label-counts
            )
            run_logged_command "${public_dir}/pack.command.txt" \
                "${public_dir}/pack.log" "${public_pack_command[@]}"
        elif [[ "$RESUME" -ne 1 ]]; then
            die "packed public synthetic data already exists: $public_manifest"
        fi
        require_file "$public_data"
        require_file "$public_partition"
        local public_validation="${public_dir}/validation_manifest.json"
        if [[ ! -f "$public_validation" || "$RESUME" -eq 1 ]]; then
            run_logged_command "${public_dir}/validate.command.txt" \
                "${public_dir}/validate.log" \
                "$H5_PYTHON" "${SCRIPT_DIR}/validate_public_generator_control.py" \
                --client-manifest "${generated_dir}/manifest.json" \
                --public-manifest "${public_generated_dir}/manifest.json" \
                --output "$public_validation"
        fi
        require_file "$public_validation"
    fi

    if [[ ! -f "$same_source_build_manifest" ]]; then
        local -a same_source_build_command=(
            "$H5_PYTHON" "${SCRIPT_DIR}/build_matched_real_control.py"
            --data-file "$REAL_DATA"
            --partition-file "$REAL_PARTITION"
            --partition-method "$SOURCE_PARTITION_METHOD"
            --client-ids "$CLIENT_IDS"
            --reference-jsonl "${generated_dir}/synthetic.jsonl"
            --output "$same_source_jsonl"
            --manifest-out "$same_source_build_manifest"
            --seed "$seed"
        )
        run_logged_command "${same_source_control_dir}/build.command.txt" \
            "${same_source_control_dir}/build.log" "${same_source_build_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "Same-Source-Real build already exists: $same_source_build_manifest"
    fi

    if [[ ! -f "$same_source_pack_manifest" ]]; then
        local -a same_source_pack_command=(
            "$H5_PYTHON" "${SCRIPT_DIR}/pack_synthetic_h5.py"
            --jsonl "$same_source_jsonl"
            --source-data-file "$REAL_DATA"
            --data-out "$same_source_data"
            --partition-out "$same_source_partition"
            --partition-method same_source_real_cloud
            --cloud-clients 1
            --manifest-out "$same_source_pack_manifest"
            --require-equal-label-counts
        )
        run_logged_command "${same_source_control_dir}/pack.command.txt" \
            "${same_source_control_dir}/pack.log" "${same_source_pack_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "Same-Source-Real packed data already exists: $same_source_pack_manifest"
    fi

    if [[ ! -f "${real_staging_dir}/manifest.json" ]]; then
        local -a real_export_command=(
            "$H5_PYTHON" "${SCRIPT_DIR}/export_client_train_jsonl.py"
            --data-file "$REAL_DATA"
            --partition-file "$REAL_PARTITION"
            --partition-method "$SOURCE_PARTITION_METHOD"
            --client-ids "$REAL_MATCHED_CLIENT_IDS"
            --output-dir "$real_staging_dir"
            --aggregate-output all_clients.jsonl
            --sample-limit-per-client "$REAL_MATCHED_LIMIT"
            --seed "$seed"
        )
        run_logged_command "${real_control_dir}/export.command.txt" \
            "${real_control_dir}/export.log" "${real_export_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "Real-Matched private staging already exists: $real_staging_dir"
    fi

    if [[ ! -f "$real_manifest" ]]; then
        local -a real_pack_command=(
            "$H5_PYTHON" "${SCRIPT_DIR}/pack_synthetic_h5.py"
            --jsonl "${real_staging_dir}/all_clients.jsonl"
            --source-data-file "$REAL_DATA"
            --data-out "$real_data"
            --partition-out "$real_partition"
            --partition-method real_matched_cloud
            --cloud-clients 1
            --manifest-out "$real_manifest"
            --require-equal-label-counts
        )
        run_logged_command "${real_control_dir}/pack.command.txt" \
            "${real_control_dir}/pack.log" "${real_pack_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "Real-Matched packed data already exists: $real_manifest"
    fi

    if [[ ! -f "$shuffle_manifest" ]]; then
        local shuffle_seed=$((seed + 700001))
        local -a shuffle_command=(
            "$H5_PYTHON" "${SCRIPT_DIR}/shuffle_synthetic_labels.py"
            --input "${generated_dir}/synthetic.jsonl"
            --output "$shuffled_jsonl"
            --manifest-out "$shuffle_manifest"
            --seed "$shuffle_seed"
            --min-label-agreement "$SHUFFLE_MIN_LABEL_AGREEMENT"
            --max-label-agreement "$SHUFFLE_MAX_LABEL_AGREEMENT"
        )
        run_logged_command "${shuffled_dir}/shuffle.command.txt" \
            "${shuffled_dir}/shuffle.log" "${shuffle_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "shuffled-label data already exists: $shuffle_manifest"
    fi

    if [[ ! -f "$shuffled_pack_manifest" ]]; then
        local -a shuffled_pack_command=(
            "$H5_PYTHON" "${SCRIPT_DIR}/pack_synthetic_h5.py"
            --jsonl "$shuffled_jsonl"
            --source-data-file "$REAL_DATA"
            --data-out "$shuffled_data"
            --partition-out "$shuffled_partition"
            --partition-method shuffled_label_cloud
            --cloud-clients 1
            --manifest-out "$shuffled_pack_manifest"
            --require-equal-label-counts
        )
        run_logged_command "${shuffled_dir}/pack.command.txt" \
            "${shuffled_dir}/pack.log" "${shuffled_pack_command[@]}"
    elif [[ "$RESUME" -ne 1 ]]; then
        die "packed shuffled-label data already exists: $shuffled_pack_manifest"
    fi

    local -a validate_command=(
        "$H5_PYTHON" "${SCRIPT_DIR}/validate_guidance_controls.py"
        --synthetic-pack-manifest "$cloud_manifest"
        --real-matched-pack-manifest "$real_manifest"
        --same-source-real-pack-manifest "$same_source_pack_manifest"
        --shuffled-pack-manifest "$shuffled_pack_manifest"
        --synthetic-jsonl "${generated_dir}/synthetic.jsonl"
        --same-source-real-jsonl "$same_source_jsonl"
        --same-source-real-build-manifest "$same_source_build_manifest"
        --shuffled-jsonl "$shuffled_jsonl"
        --shuffle-manifest "$shuffle_manifest"
        --output "$controls_manifest"
    )
    if [[ "$RESUME" -eq 1 ]]; then
        validate_command+=(--resume)
    fi
    run_logged_command "${seed_dir}/controls/validate.command.txt" \
        "${seed_dir}/controls/validate.log" "${validate_command[@]}"

    require_file "$real_data"
    require_file "$real_partition"
    require_file "$same_source_data"
    require_file "$same_source_partition"
    require_file "$shuffled_data"
    require_file "$shuffled_partition"
    require_file "$controls_manifest"
}

run_arm() {
    local seed="$1"
    local arm="$2"
    local alpha="$3"
    local seed_dir="${RUN_DIR}/seed_${seed}"
    local synthetic_dir="${seed_dir}/synthetic"
    local cloud_data
    local cloud_partition
    local cloud_partition_method
    local cloud_source
    case "$arm" in
        no_cloud|client_syn)
            cloud_data="${synthetic_dir}/${DATASET}_client_synthetic_data.h5"
            cloud_partition="${synthetic_dir}/${DATASET}_client_synthetic_partition.h5"
            cloud_partition_method="synthetic_cloud"
            if [[ "$GENERATOR_MODE" == "dp" ]]; then
                cloud_source="client_synthetic_record_dp"
            else
                cloud_source="client_synthetic_nondp"
            fi
            ;;
        public_syn)
            cloud_data="${seed_dir}/public_synthetic/${DATASET}_public_synthetic_data.h5"
            cloud_partition="${seed_dir}/public_synthetic/${DATASET}_public_synthetic_partition.h5"
            cloud_partition_method="public_synthetic_cloud"
            cloud_source="public_pretrained_synthetic"
            ;;
        same_source_real_matched)
            cloud_data="${seed_dir}/controls/same_source_real_matched/${DATASET}_same_source_real_matched_data.h5"
            cloud_partition="${seed_dir}/controls/same_source_real_matched/${DATASET}_same_source_real_matched_partition.h5"
            cloud_partition_method="same_source_real_cloud"
            cloud_source="nondeployable_same_source_client_real_matched"
            ;;
        real_matched)
            cloud_data="${seed_dir}/controls/real_matched/${DATASET}_real_matched_data.h5"
            cloud_partition="${seed_dir}/controls/real_matched/${DATASET}_real_matched_partition.h5"
            cloud_partition_method="real_matched_cloud"
            cloud_source="reserved_client_real_matched"
            ;;
        shuffled_label)
            cloud_data="${seed_dir}/controls/shuffled_label/${DATASET}_shuffled_label_data.h5"
            cloud_partition="${seed_dir}/controls/shuffled_label/${DATASET}_shuffled_label_partition.h5"
            cloud_partition_method="shuffled_label_cloud"
            cloud_source="client_synthetic_shuffled_labels"
            ;;
        *) die "unknown arm: $arm" ;;
    esac
    local arm_dir="${seed_dir}/arms/${arm}"
    local log_file="${arm_dir}/train.log"
    local command_file="${arm_dir}/command.txt"
    local exit_code_file="${arm_dir}/exit_code.txt"
    local metrics_file="${arm_dir}/metrics.json"
    local manifest_file="${arm_dir}/manifest.json"
    if [[ -f "$manifest_file" && "$RESUME" -eq 1 ]]; then
        if "$H5_PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["status"] == "complete" else 1)' "$manifest_file"; then
            printf 'Skipping completed arm: seed=%s arm=%s\n' "$seed" "$arm"
            return 0
        fi
        die "existing arm is incomplete; use a new run-id instead of overwriting: $arm_dir"
    fi
    [[ ! -e "$arm_dir" ]] || die "arm directory already exists: $arm_dir"
    mkdir -p "$arm_dir"
    require_file "$cloud_data"
    require_file "$cloud_partition"

    local python_path="${RUN_TC_DIR}:${PROJECT_ROOT}:${PROJECT_ROOT}/FedML${PYTHONPATH:+:${PYTHONPATH}}"
    local -a command=(
        env
        "PYTHONPATH=${python_path}"
        "PYTHONHASHSEED=${seed}"
        "HEIMDALLM_FAULTHANDLER=1"
        "WANDB_MODE=disabled"
        "WANDB_SILENT=true"
        "TOKENIZERS_PARALLELISM=false"
        "OMP_NUM_THREADS=1"
        timeout --signal=TERM --kill-after=60s "${TIMEOUT_MINUTES}m"
        "$MPI_LAUNCHER" --bind-to none -np "$PROCESS_COUNT" --hostfile "$MPI_HOST_FILE"
        "$FED_PYTHON" -m fedavg_main_tc
        --gpu_mapping_file "$GPU_MAPPING_FILE"
        --gpu_mapping_key mapping_pilot
        --dataset "$DATASET"
        --data_file_path "$REAL_DATA"
        --partition_file_path "$PILOT_PARTITION"
        --partition_method "$PILOT_METHOD"
        --cloud_dataset "$DATASET"
        --cloud_data_file_path "$cloud_data"
        --cloud_partition_file_path "$cloud_partition"
        --cloud_partition_method "$cloud_partition_method"
        --cloud_client_ids 0
        --cloud_max_seq_length 64
        --fl_algorithm FedFwd
        --model_type distilbert
        --model_name "$TASK_MODEL"
        --do_lower_case True
        --train_batch_size 8
        --eval_batch_size 8
        --max_seq_length 64
        --client_num_in_total "$LOGICAL_CLIENT_COUNT"
        --client_num_per_round "$CLIENTS_PER_ROUND"
        --worker_num "$MPI_WORKERS"
        --comm_round "$ROUNDS"
        --frequency_of_the_test "$EVAL_EVERY"
        --evaluate_before_training
        --manual_seed "$seed"
        --run_id "$seed"
        --epochs 1
        --lr 0.01
        --server_lr 0.1
        --learning_rate 0.01
        --peft_method adapter
        --use_adapter True
        --forward_mode
        --var_control
        --perturbation_sampling
        --max_var_retries "$MAX_VAR_RETRIES"
        --v_num 1
        --beta 1
        --pool_size 1
        --alpha "$alpha"
        --output_dir "${arm_dir}/model_output"
    )

    (
        cd "$arm_dir"
        write_command_file "$command_file" "${command[@]}"
        set +e
        "${command[@]}" > "$log_file" 2>&1
        local_status=$?
        set -e
        printf '%d\n' "$local_status" > "$exit_code_file"
    )
    local command_status
    command_status="$(cat "$exit_code_file")"

    local -a summary_command=(
        "$H5_PYTHON" "${SCRIPT_DIR}/summarize_fed_pilot.py"
        --log "$log_file"
        --metrics-out "$metrics_file"
        --manifest-out "$manifest_file"
        --command-file "$command_file"
        --condition "$arm"
        --cloud-source "$cloud_source"
        --evaluation-mode "$EVALUATION_MODE"
        --stage "$STAGE"
        --alpha "$alpha"
        --seed "$seed"
        --rounds "$ROUNDS"
        --eval-every "$EVAL_EVERY"
        --pre-update-eval
        --logical-clients "$LOGICAL_CLIENT_COUNT"
        --clients-per-round "$CLIENTS_PER_ROUND"
        --mpi-workers "$MPI_WORKERS"
        --exit-code "$command_status"
        --artifact "real_data=${REAL_DATA}"
        --artifact "real_source_partition=${REAL_PARTITION}"
        --artifact "fixed_real_partition=${PILOT_PARTITION}"
        --artifact "synthetic_jsonl=${synthetic_dir}/generated/synthetic.jsonl"
        --artifact "synthetic_manifest=${synthetic_dir}/generated/manifest.json"
        --artifact "arm_cloud_data=${cloud_data}"
        --artifact "arm_cloud_partition=${cloud_partition}"
        --artifact "run_spec=${RUN_SPEC_FILE}"
        --artifact "gpu_mapping=${GPU_MAPPING_FILE}"
        --artifact "fed_entrypoint=${RUN_TC_DIR}/fedavg_main_tc.py"
        --require-complete
    )
    if [[ -f "${seed_dir}/controls/same_source_real_matched/build_manifest.json" ]]; then
        summary_command+=(
            --artifact "same_source_real_build_manifest=${seed_dir}/controls/same_source_real_matched/build_manifest.json"
        )
    fi
    if [[ -f "${seed_dir}/controls/validation_manifest.json" ]]; then
        summary_command+=(
            --artifact "control_validation=${seed_dir}/controls/validation_manifest.json"
        )
    fi
    if [[ -f "${synthetic_dir}/generated/validation_manifest.json" ]]; then
        summary_command+=(
            --artifact "dp_validation=${synthetic_dir}/generated/validation_manifest.json"
        )
    fi
    if [[ "$INCLUDE_PUBLIC_SYNTHETIC" -eq 1 ]]; then
        summary_command+=(
            --artifact "public_synthetic_jsonl=${seed_dir}/public_synthetic/generated/synthetic.jsonl"
            --artifact "public_synthetic_manifest=${seed_dir}/public_synthetic/generated/manifest.json"
        )
    fi
    set +e
    "${summary_command[@]}" > "${arm_dir}/summary.log" 2>&1
    local summary_status=$?
    set -e
    if [[ "$command_status" -ne 0 || "$summary_status" -ne 0 ]]; then
        printf 'Arm failed validation: seed=%s arm=%s exit=%s; see %s\n' \
            "$seed" "$arm" "$command_status" "$arm_dir" >&2
        return 1
    fi
    printf 'Arm complete: seed=%s arm=%s metrics=%s\n' "$seed" "$arm" "$metrics_file"
}

preflight_dependencies
if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    if [[ "$PHASE" != "prepare" ]]; then
        prepare_gpu_mapping
    elif [[ "$GENERATOR_DEVICE" == "auto" ]]; then
        read_gpu_state
        while read -r selected_id; do
            GENERATOR_DEVICE="cuda:${selected_id}"
            break
        done < <(select_gpu_ids 1)
    fi
    print_resolved_plan
    printf 'generator_device=%s\n' "$GENERATOR_DEVICE"
    printf 'Preflight passed.\n'
    exit 0
fi

if [[ -e "$RUN_DIR" && "$RESUME" -ne 1 ]]; then
    die "result directory already exists: $RUN_DIR (use a new run-id or --resume)"
fi
mkdir -p "$SHARED_DIR"
write_or_check_run_spec
if [[ "$PHASE" != "prepare" ]]; then
    prepare_gpu_mapping
elif [[ "$GENERATOR_DEVICE" == "auto" ]]; then
    read_gpu_state
    while read -r selected_id; do
        GENERATOR_DEVICE="cuda:${selected_id}"
        break
    done < <(select_gpu_ids 1)
fi
print_resolved_plan
printf 'generator_device=%s\n' "$GENERATOR_DEVICE"
prepare_fixed_partition

if [[ "$PHASE" == "all" || "$PHASE" == "prepare" ]]; then
    for seed in "${SEED_ARRAY[@]}"; do
        prepare_seed_synthetic "$seed"
    done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "train" ]]; then
    overall_status=0
    for seed in "${SEED_ARRAY[@]}"; do
        case "$CONDITION" in
            paired)
                run_arm "$seed" no_cloud 1.0 || overall_status=1
                run_arm "$seed" client_syn 0.5 || overall_status=1
                if [[ "$INCLUDE_PUBLIC_SYNTHETIC" -eq 1 ]]; then
                    run_arm "$seed" public_syn 0.5 || overall_status=1
                fi
                run_arm "$seed" same_source_real_matched 0.5 || overall_status=1
                run_arm "$seed" real_matched 0.5 || overall_status=1
                run_arm "$seed" shuffled_label 0.5 || overall_status=1
                ;;
            no-cloud) run_arm "$seed" no_cloud 1.0 || overall_status=1 ;;
            client-syn) run_arm "$seed" client_syn 0.5 || overall_status=1 ;;
            public-syn) run_arm "$seed" public_syn 0.5 || overall_status=1 ;;
            same-source-real-matched) run_arm "$seed" same_source_real_matched 0.5 || overall_status=1 ;;
            real-matched) run_arm "$seed" real_matched 0.5 || overall_status=1 ;;
            shuffled-label) run_arm "$seed" shuffled_label 0.5 || overall_status=1 ;;
        esac
        if [[ "$CONDITION" == "paired" ]]; then
            set +e
            "$H5_PYTHON" "${SCRIPT_DIR}/validate_nondp_smoke.py" \
                --run-dir "$RUN_DIR" --seed "$seed" \
                > "${RUN_DIR}/seed_${seed}/cross_arm_validation.log" 2>&1
            validation_status=$?
            set -e
            if [[ "$validation_status" -ne 0 ]]; then
                printf 'Cross-arm validation failed: seed=%s; see %s\n' \
                    "$seed" "${RUN_DIR}/seed_${seed}/cross_arm_validation.json" >&2
                overall_status=1
            fi
        fi
    done
    if [[ "$overall_status" -ne 0 ]]; then
        die "one or more arms failed; incomplete runs are retained for diagnosis"
    fi
fi

printf 'Run complete: %s\n' "$RUN_DIR"
printf 'Private staging remains host-local at: %s\n' "$PRIVATE_STAGING_RUN"
