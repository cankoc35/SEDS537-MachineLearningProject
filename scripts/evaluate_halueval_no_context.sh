#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" -m src.evaluation.evaluate_uncertainty_classifiers \
  --input data/processed/halueval_no_context_uncertainty_entropy_features.csv \
  --metrics-output outputs/tables/halueval_no_context_qwen3b_classifier_metrics.csv \
  --predictions-output outputs/predictions/halueval_no_context_qwen3b_classifier_predictions.csv

"${PYTHON_BIN}" -m src.evaluation.evaluate_uncertainty_classifiers \
  --input data/processed/halueval_no_context_uncertainty_entropy_qwen05b_features.csv \
  --metrics-output outputs/tables/halueval_no_context_qwen05b_classifier_metrics.csv \
  --predictions-output outputs/predictions/halueval_no_context_qwen05b_classifier_predictions.csv

"${PYTHON_BIN}" -m src.evaluation.evaluate_uncertainty_classifiers \
  --train-input data/processed/halueval_no_context_uncertainty_entropy_features.csv \
  --test-input data/processed/truthfulqa_uncertainty_entropy_features.csv \
  --metrics-output outputs/tables/halueval_no_context_to_truthfulqa_qwen3b_classifier_metrics.csv \
  --predictions-output outputs/predictions/halueval_no_context_to_truthfulqa_qwen3b_classifier_predictions.csv

"${PYTHON_BIN}" -m src.evaluation.evaluate_uncertainty_classifiers \
  --train-input data/processed/halueval_no_context_uncertainty_entropy_qwen05b_features.csv \
  --test-input data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv \
  --metrics-output outputs/tables/halueval_no_context_to_truthfulqa_qwen05b_classifier_metrics.csv \
  --predictions-output outputs/predictions/halueval_no_context_to_truthfulqa_qwen05b_classifier_predictions.csv
