#!/usr/bin/env bash
#SBATCH --job-name=deepdta_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=23:50:00
#SBATCH --signal=USR1@600
#SBATCH --output=logs/deepdta_full-%j.out
#SBATCH --error=logs/deepdta_full-%j.err
set -uo pipefail
# Slurm copies this script to its spool; use the exported or submission root.
ROOT="${CONFMUX_SUPPLEMENT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "$ROOT" ] || [ ! -f "$ROOT/code/run_release.py" ]; then
  echo "ERROR: source PATH_CONFIG.local.sh and submit from the supplement root." >&2
  exit 2
fi
if [ -f "$ROOT/PATH_CONFIG.local.sh" ]; then source "$ROOT/PATH_CONFIG.local.sh"; fi
PYTHON_BIN="${PY_DEEPDTA:-${PY_MAIN:-python}}"
THREADS="${SLURM_CPUS_PER_TASK:-${CONFMUX_THREADS:-4}}"
RUN_DIR="${DEEPDTA_RUN_DIR:-$ROOT/reproduction_runs/deepdta}"
cd "$ROOT" || exit 2
# logs/ must exist BEFORE sbatch opens the log files.
srun --ntasks=1 "$PYTHON_BIN" -u "$ROOT/code/run_release.py" \
  --task deepdta --root "$ROOT" --out-dir "$RUN_DIR" \
  --threads "$THREADS" --device cuda --resume --execute
STATUS=$?
# Checkpoints can resume; automatic resubmission is opt-in.
if { [ "$STATUS" -eq 75 ] || [ "$STATUS" -eq 138 ]; } && [ "${AUTO_REQUEUE:-0}" = 1 ]; then
  scontrol requeue "$SLURM_JOB_ID" && exit 0
fi
exit "$STATUS"
