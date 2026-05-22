#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
FOLDS="${FOLDS:-5}"

"$PYTHON_BIN" -m src.evaluation.cross_validation \
  --input data/processed/halueval_uncertainty_entropy_features.csv \
  --folds "$FOLDS" \
  --fold-metrics-output outputs/tables/halueval_qwen3b_cv_fold_metrics.csv \
  --summary-output outputs/tables/halueval_qwen3b_cv_summary.csv

"$PYTHON_BIN" -m src.evaluation.cross_validation \
  --input data/processed/truthfulqa_uncertainty_entropy_features.csv \
  --folds "$FOLDS" \
  --fold-metrics-output outputs/tables/truthfulqa_qwen3b_cv_fold_metrics.csv \
  --summary-output outputs/tables/truthfulqa_qwen3b_cv_summary.csv

"$PYTHON_BIN" -m src.evaluation.cross_validation \
  --input data/processed/halueval_uncertainty_entropy_qwen05b_features.csv \
  --folds "$FOLDS" \
  --fold-metrics-output outputs/tables/halueval_qwen05b_cv_fold_metrics.csv \
  --summary-output outputs/tables/halueval_qwen05b_cv_summary.csv

"$PYTHON_BIN" -m src.evaluation.cross_validation \
  --input data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv \
  --folds "$FOLDS" \
  --fold-metrics-output outputs/tables/truthfulqa_qwen05b_cv_fold_metrics.csv \
  --summary-output outputs/tables/truthfulqa_qwen05b_cv_summary.csv
