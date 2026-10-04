#!/usr/bin/env bash
# Run the locked seed-57 pilot; continue only if filtered DP beats both controls.
set -Eeuo pipefail

WORK_ROOT="${HEIMDALLM_WORK_ROOT:-/home/zzkevin/heimdallm-journal-20260928}"
TOOLS="${WORK_ROOT}/diagnostics/agnews_dp_recovery_20261004"
RUNNER="${TOOLS}/run_ustc_agnews_dp_filter.py"
SELECTION="${TOOLS}/selection_v1"
ROOT="${WORK_ROOT}/results/ustc_agnews_dp_filter_20261004"
PYTHON="${WORK_ROOT}/envs/kdd/bin/python"

"$PYTHON" "$RUNNER" --work-root "$WORK_ROOT" --selection-dir "$SELECTION" \
    --gpu 2 --pilot-only

"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
summary=json.loads((root/'summary/summary.json').read_text())
values={run['arm']:run['final_accuracy'] for run in summary['runs'] if run['seed']==57}
expected={'filtered_dp','random_dp','filtered_public'}
if summary['runs_completed']!=3 or set(values)!=expected:
    raise SystemExit('Pilot did not produce the three validated cases')
decision={'seed':57,'values':values,
          'continue':values['filtered_dp']>values['random_dp'] and
                     values['filtered_dp']>values['filtered_public'],
          'criterion':'filtered_dp strictly exceeds both budget-matched controls'}
(root/'pilot_decision.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
print(json.dumps(decision,sort_keys=True),flush=True)
PY

if [[ "$("$PYTHON" -c 'import json,sys;print(int(json.load(open(sys.argv[1]))["continue"]))' "$ROOT/pilot_decision.json")" != 1 ]]; then
    echo 'Pilot did not beat both controls; stopping before seeds 58/59.'
    exit 0
fi

exec "$PYTHON" "$RUNNER" --work-root "$WORK_ROOT" --selection-dir "$SELECTION" \
    --gpu 2 --resume
