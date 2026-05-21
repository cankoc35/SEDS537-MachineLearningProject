#!/usr/bin/env bash

set -euo pipefail

python -m src.evaluation.evaluate_uncertainty_classifiers \
  --input data/processed/truthfulqa_uncertainty_entropy_features.csv \
  --metrics-output outputs/tables/truthfulqa_classifier_metrics.csv \
  --predictions-output outputs/predictions/truthfulqa_classifier_predictions.csv \
  --test-size 0.2 \
  --random-state 42
