#!/usr/bin/env bash

set -euo pipefail

DATASET="${1:-all}"

python3 -m src.data.load_data --dataset "${DATASET}" --halueval-tasks qa
python3 -m src.data.preprocess --dataset "${DATASET}" --halueval-tasks qa
