#!/usr/bin/env bash
# Copy to PATH_CONFIG.local.sh, edit needed values, then source it.
export CONFMUX_SUPPLEMENT_ROOT="${CONFMUX_SUPPLEMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PY_MAIN="${PY_MAIN:-python}"
export PY_DEEPDTA="${PY_DEEPDTA:-$PY_MAIN}"
export PY_DEEPDTAGEN="${PY_DEEPDTAGEN:-$PY_MAIN}"
export PY_HIST="${PY_HIST:-python}"
# Canonical data is a separate deposit, not part of this supplement.
export CONFMUX_DATA="${CONFMUX_DATA:-$CONFMUX_SUPPLEMENT_ROOT/../preprocessed/confmux_dta_bindingdb_agg.csv}"
export CONFMUX_THREADS="${CONFMUX_THREADS:-4}"
export REFERENCE_CACHE_DIR="${REFERENCE_CACHE_DIR:-$CONFMUX_SUPPLEMENT_ROOT/manifests/full_controlled_split}"
export CONFMUX_HISTORICAL_BUNDLE="${CONFMUX_HISTORICAL_BUNDLE:-}"
export CONFMUX_FULL_RUN="${CONFMUX_FULL_RUN:-$CONFMUX_SUPPLEMENT_ROOT/reproduction_runs/full}"
export CONFMUX_FEATURE_CACHE="${CONFMUX_FEATURE_CACHE:-}"
# Directory must contain model.py, utils.py and FetterGrad.py.
export DEEPDTAGEN_ROOT="${DEEPDTAGEN_ROOT:-}"
export DEEPDTAGEN_BINDINGDB_PT="${DEEPDTAGEN_BINDINGDB_PT:-}"
export TOKENIZER_FILE="${TOKENIZER_FILE:-}"
# Compatibility aliases.
export PYTHON_BIN="$PY_MAIN"
export WORK_DIR="$CONFMUX_SUPPLEMENT_ROOT"
export WORKDIR="$CONFMUX_SUPPLEMENT_ROOT"
export DATA_FILE="$CONFMUX_DATA"
export PT_FILE="$DEEPDTAGEN_BINDINGDB_PT"
