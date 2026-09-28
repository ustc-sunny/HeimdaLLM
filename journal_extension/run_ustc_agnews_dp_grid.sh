#!/usr/bin/env bash
# Sequential, resumable epsilon queue; saves each completed condition.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/home/zzkevin/heimdallm-journal-20260928}"
SUMMARY_DIR="${WORK_ROOT}/results/agnews_dp_v1_ustc_summary"
mkdir -p "${WORK_ROOT}/logs" "${WORK_ROOT}/backups"
exec 9>"${WORK_ROOT}/logs/dp-grid.lock"
flock -n 9 || { echo 'DP grid is already running.' >&2; exit 2; }
for epsilon in 8 4 2 1; do
    run_id="ustc_agnews_dp_eps${epsilon}_20260928"
    log="${WORK_ROOT}/logs/${run_id}.log"
    export HEIMDALLM_DP_EPSILON="$epsilon" HEIMDALLM_DP_RUN_ID="$run_id"
    arguments=()
    if [[ -d "${WORK_ROOT}/results/${run_id}" ]]; then
        arguments+=(--resume)
    fi
    echo "Starting ${run_id} at $(date -Is)"
    if ! bash "${SCRIPT_DIR}/run_ustc_agnews_dp_v1.sh" "${arguments[@]}" > "$log" 2>&1; then
        echo "Condition failed; retained artifacts: ${run_id}" >&2
        exit 1
    fi
    archive="${WORK_ROOT}/backups/${run_id}.tar.gz"
    tar -czf "${archive}.tmp" -C "${WORK_ROOT}/results" "$run_id"
    mv "${archive}.tmp" "$archive"
    sha256sum "$archive" > "${archive}.sha256"
    "${WORK_ROOT}/envs/kdd/bin/python" "${SCRIPT_DIR}/summarize_agnews_dp.py" \
        --results-root "${WORK_ROOT}/results" --output-dir "$SUMMARY_DIR" \
        --reference-curves "${SCRIPT_DIR}/results/agnews_nondp_v3/curves.csv"
    cp "$log" "${WORK_ROOT}/backups/"
    echo "Archived ${run_id} at $(date -Is)"
done
echo "All four DP conditions completed at $(date -Is)"
