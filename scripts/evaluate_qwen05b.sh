#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"$PYTHON_BIN" -m src.evaluation.evaluate_uncertainty_classifiers \
  --input data/processed/halueval_uncertainty_entropy_qwen05b_features.csv \
  --metrics-output outputs/tables/halueval_qwen05b_classifier_metrics.csv \
  --predictions-output outputs/predictions/halueval_qwen05b_classifier_predictions.csv

"$PYTHON_BIN" -m src.evaluation.evaluate_uncertainty_classifiers \
  --input data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv \
  --metrics-output outputs/tables/truthfulqa_qwen05b_classifier_metrics.csv \
  --predictions-output outputs/predictions/truthfulqa_qwen05b_classifier_predictions.csv

"$PYTHON_BIN" -m src.evaluation.evaluate_uncertainty_classifiers \
  --train-input data/processed/halueval_uncertainty_entropy_qwen05b_features.csv \
  --test-input data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv \
  --metrics-output outputs/tables/halueval_to_truthfulqa_qwen05b_classifier_metrics.csv \
  --predictions-output outputs/predictions/halueval_to_truthfulqa_qwen05b_classifier_predictions.csv
