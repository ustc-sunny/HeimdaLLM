#!/usr/bin/env bash
# Fixed follow-up: semantic category names and longer generator context.
# Preserve v1 client data, generation count and downstream training settings.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_matpool_agnews.sh" \
  --run-id matpool_agnews_nondp_v2 \
  --seeds 57,58,59 \
  --generator-max-length 192 --generator-max-new-tokens 80 \
  --label-names-json '{"1":"World","2":"Sports","3":"Business","4":"Science and Technology"}' \
  --prompt-template 'Category: {label}\nNews article:\n' "$@"
