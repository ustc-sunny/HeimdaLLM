#!/usr/bin/env bash
# Run the locked seed-57 pilot and continue all seeds only after validation.
set -Eeuo pipefail

WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/home/zzkevin/heimdallm-journal-20260928}"
REPO="${WORK_ROOT}/HeimdaLLM"
PYTHON="${WORK_ROOT}/envs/kdd/bin/python"
RUN_ID="ustc_agnews_noise_bridge_20260930"
ROOT="${WORK_ROOT}/results/${RUN_ID}"
RUNNER="${REPO}/journal_extension/run_ustc_agnews_matched.py"

if [[ "${1:-}" == "--follow-existing-pilot" ]]; then
    while true; do
        state="$($PYTHON - "$ROOT/state.json" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
print(json.loads(path.read_text()).get("status", "unknown") if path.is_file() else "pending")
PY
)"
        case "$state" in
            pilot_complete) break ;;
            complete) exit 0 ;;
            failed) echo 'Seed-57 pilot failed; preserving artifacts.' >&2; exit 1 ;;
            pending|running) sleep 30 ;;
            *) echo "Unexpected pilot state: $state" >&2; exit 1 ;;
        esac
    done
else
    "$PYTHON" "$RUNNER" --phase formal --arms zero_noise,dp_eps8 \
        --run-id "$RUN_ID" --seeds 57,58,59 --rounds 50 --pilot-only
fi

"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
summary=json.loads((root/'summary/summary.json').read_text())
expected=[('zero_noise',57),('dp_eps8',57)]
runs=[(r['arm'],r['seed']) for r in summary['runs']]
if summary['runs_completed'] != 2 or summary['runs_expected'] != 6 or runs != expected:
    raise SystemExit('Pilot summary is incomplete or out of order')
control=json.loads((root/'zero_noise/seed_57/generated/manifest.json').read_text())
if control.get('is_record_level_dp') is not False or control['privacy']['noise_added_to_clipped_gradient_sum'] is not False:
    raise SystemExit('Zero-noise control is mislabeled')
dp=json.loads((root/'dp_eps8/seed_57/generated/manifest.json').read_text())
validated=json.loads((root/'dp_eps8/seed_57/dp_validation.json').read_text())
if dp.get('is_record_level_dp') is not True or validated.get('status') != 'complete' or not all(validated['checks'].values()):
    raise SystemExit('DP release validation failed')
if dp['privacy']['achieved_epsilon_max'] > 8.011:
    raise SystemExit('DP epsilon exceeds the declared tolerance')
print('Seed-57 pilot passed accounting, labels, downstream metrics, and archiving checks.',flush=True)
PY

for arm in zero_noise dp_eps8; do
    (cd "${WORK_ROOT}/backups" && sha256sum --check "${RUN_ID}_${arm}_seed57.tar.gz.sha256")
done

exec "$PYTHON" "$RUNNER" --phase formal --arms zero_noise,dp_eps8 \
    --run-id "$RUN_ID" --seeds 57,58,59 --rounds 50 --resume
