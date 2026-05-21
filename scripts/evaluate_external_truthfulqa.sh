#!/usr/bin/env bash

set -euo pipefail

python -m src.evaluation.evaluate_uncertainty_classifiers \
  --train-input data/processed/halueval_uncertainty_entropy_features.csv \
  --test-input data/processed/truthfulqa_uncertainty_entropy_features.csv \
  --metrics-output outputs/tables/halueval_to_truthfulqa_classifier_metrics.csv \
  --predictions-output outputs/predictions/halueval_to_truthfulqa_classifier_predictions.csv \
  --random-state 42
