#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" -m src.evaluation.ablation \
  --halueval-features data/processed/halueval_uncertainty_entropy_qwen05b_features.csv \
  --truthfulqa-features data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv \
  --halueval-output outputs/tables/halueval_qwen05b_ablation_results.csv \
  --truthfulqa-output outputs/tables/truthfulqa_qwen05b_ablation_results.csv \
  --external-output outputs/tables/halueval_to_truthfulqa_qwen05b_ablation_results.csv \
  --test-size 0.2 \
  --random-state 42
