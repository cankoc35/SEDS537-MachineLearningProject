#!/usr/bin/env bash

set -euo pipefail

python -m src.evaluation.evaluate_uncertainty_classifiers \
  --input data/processed/halueval_uncertainty_entropy_features.csv \
  --metrics-output outputs/tables/halueval_classifier_metrics.csv \
  --predictions-output outputs/predictions/halueval_classifier_predictions.csv \
  --test-size 0.2 \
  --random-state 42
