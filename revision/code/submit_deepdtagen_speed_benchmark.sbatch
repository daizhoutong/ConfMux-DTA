#!/usr/bin/env bash
#SBATCH --job-name=dtagen_speed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dtagen_speed-%j.out
#SBATCH --error=logs/dtagen_speed-%j.err
set -uo pipefail
ROOT="${CONFMUX_SUPPLEMENT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/run_release.py" ]; then
  echo "ERROR: source PATH_CONFIG.local.sh and submit from the supplement root." >&2
  exit 2
fi
if [ -f "$ROOT/PATH_CONFIG.local.sh" ]; then source "$ROOT/PATH_CONFIG.local.sh"; fi
PYTHON_BIN="${PY_DEEPDTAGEN:-${PY_MAIN:-python}}"
THREADS="${SLURM_CPUS_PER_TASK:-${CONFMUX_THREADS:-4}}"
OUT="${DEEPDTAGEN_RUN_DIR:-$ROOT/reproduction_runs/deepdtagen_${SLURM_JOB_ID:-manual}}"
cd "$ROOT" || exit 2
"$PYTHON_BIN" "$ROOT/code/run_release.py" --task deepdtagen --root "$ROOT" \
  --batch-size "${BATCH_SIZE:-128}" --out-dir "$OUT" --threads "$THREADS" \
  --check-inputs --dry-run || exit $?
mkdir -p "$OUT" || exit 2
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 10 > "$OUT/gpu_trace.csv" 2>&1 &
  MONITOR_PID=$!
  cleanup_monitor() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
  trap cleanup_monitor EXIT
fi
# Put tracing outside the initially empty measurement result directory.
srun "$PYTHON_BIN" -u "$ROOT/code/run_release.py" \
  --task deepdtagen --root "$ROOT" --out-dir "$OUT/measurement" \
  --batch-size "${BATCH_SIZE:-128}" --threads "$THREADS" --execute
exit $?
