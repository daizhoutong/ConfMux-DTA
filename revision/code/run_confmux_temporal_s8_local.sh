#!/usr/bin/env bash
set -euo pipefail
ROOT="${CONFMUX_SUPPLEMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -f "$ROOT/PATH_CONFIG.local.sh" ]; then source "$ROOT/PATH_CONFIG.local.sh"; fi
PYTHON_BIN="${PY_HIST:-python}"
cd "$ROOT"
# Default: audit the included frozen predictions. Regeneration requires the
# separately deposited, compatible historical bundle.
TASK="${TEMPORAL_TASK:-temporal_audit}"
if [ "$TASK" != temporal_audit ] && [ "$TASK" != temporal_predict ]; then
  echo "ERROR: TEMPORAL_TASK must be temporal_audit or temporal_predict" >&2
  exit 2
fi
exec "$PYTHON_BIN" -u "$ROOT/code/run_release.py" \
  --task "$TASK" --root "$ROOT" --threads "${CONFMUX_THREADS:-4}" \
  --out-dir "${OUT_DIR:-$ROOT/reproduction_runs/$TASK}" --execute
