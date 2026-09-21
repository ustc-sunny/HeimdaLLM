#!/usr/bin/env bash
# Long-horizon Non-DP validation with a pretrained-only public generator arm.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_matpool_agnews.sh" \
  --run-id matpool_agnews_nondp_v3 \
  --seeds 57,58,59 \
  --rounds 50 --eval-every 1 --timeout-minutes 180 \
  --include-public-synthetic \
  --generator-max-length 192 --generator-max-new-tokens 80 \
  --label-names-json '{"1":"World","2":"Sports","3":"Business","4":"Science and Technology"}' \
  --prompt-template 'Category: {label}\nNews article:\n' "$@"
