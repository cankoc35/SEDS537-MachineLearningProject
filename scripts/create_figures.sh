#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-outputs/.matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-outputs/.cache}"

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

"${PYTHON_BIN}" -m src.evaluation.visualize_results \
  --output-dir outputs/figures
