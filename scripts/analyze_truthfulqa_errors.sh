#!/usr/bin/env bash

set -euo pipefail

python -m src.evaluation.error_analysis \
  --halueval-features data/processed/halueval_uncertainty_entropy_features.csv \
  --truthfulqa-features data/processed/truthfulqa_uncertainty_entropy_features.csv \
  --truthfulqa-predictions outputs/predictions/truthfulqa_classifier_predictions.csv \
  --truthfulqa-processed data/processed/truthfulqa.jsonl
