#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" -m src.evaluation.ablation \
  --halueval-features data/processed/halueval_uncertainty_entropy_features.csv \
  --truthfulqa-features data/processed/truthfulqa_uncertainty_entropy_features.csv \
  --halueval-output outputs/tables/halueval_ablation_results.csv \
  --truthfulqa-output outputs/tables/truthfulqa_ablation_results.csv \
  --external-output outputs/tables/halueval_to_truthfulqa_ablation_results.csv \
  --test-size 0.2 \
  --random-state 42
