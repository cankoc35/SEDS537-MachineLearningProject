# LLM Hallucination Detection via Uncertainty

This repository contains the term project for `SEDS 537 - Machine Learning`.

The project studies hallucination detection as a binary classification task. Given a prompt/context and an answer, the system predicts whether the answer is supported or hallucinated using uncertainty features extracted from an open-source LLM.

## Current Approach

The current implemented pipeline uses `Qwen/Qwen2.5-3B` as a feature extractor. Qwen is not prompted to judge whether an answer is hallucinated. Instead, the project reads Qwen's token-level probability distribution for a provided answer and computes uncertainty features.

The extracted features include:

- token confidence features such as mean/min/max token probability
- log-probability features such as mean logprob and negative mean logprob
- low-confidence token ratio
- entropy features such as mean/max/sum entropy
- high-entropy token ratio

These features are then used to train classical binary classifiers:

- Logistic Regression
- Linear SVM
- Random Forest

## Datasets

The project currently uses:

- `HaluEval QA`: main training and evaluation dataset
- `TruthfulQA generation`: prepared for later external/generalization evaluation

Processed datasets use this shared schema:

```text
id, dataset, task, prompt, context, answer, label, source, metadata
```

where:

```text
label = 0 -> supported / truthful
label = 1 -> hallucinated / unsupported
```

## Repository Structure

```text
.
├── data/
│   ├── raw/                         # downloaded raw datasets
│   ├── processed/                   # processed JSONL and feature tables
│   └── README.md
├── docs/
│   ├── development_log.md
│   ├── literature/
│   └── progress-report/
├── models/                          # local downloaded LLM weights, ignored by Git
│   └── qwen2.5-3b/
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── tables/
├── scripts/
│   ├── prepare_data.sh
│   └── evaluate.sh
├── src/
│   ├── data/                        # dataset download and preprocessing
│   ├── evaluation/                  # metrics and classifier evaluation
│   ├── features/                    # uncertainty, consistency, RAG feature modules
│   ├── generation/                  # Qwen scoring and feature extraction
│   ├── models/                      # Logistic Regression, SVM, Random Forest
│   └── utils/
├── proposal/
├── requirements.txt
└── README.md
```

## Setup

Create and activate a Python environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project was developed with Python 3.11.

## Data Preparation

Download and preprocess HaluEval QA and TruthfulQA generation:

```bash
./scripts/prepare_data.sh
```

Main processed files:

```text
data/processed/halueval.jsonl
data/processed/truthfulqa.jsonl
```

## Local LLM

The current feature extraction uses a local Qwen model:

```text
models/qwen2.5-3b/
```

This directory contains model weights and should not be committed to Git.

If the model is not present locally, download it with Hugging Face:

```bash
hf download Qwen/Qwen2.5-3B \
  --repo-type model \
  --local-dir models/qwen2.5-3b
```

## Feature Extraction

Extract uncertainty and entropy features for HaluEval:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/halueval.jsonl \
  --limit 0 \
  --model models/qwen2.5-3b \
  --output data/processed/halueval_uncertainty_entropy_features.csv
```

Use a small test run before the full dataset:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/halueval.jsonl \
  --limit 10 \
  --model models/qwen2.5-3b \
  --output data/processed/halueval_uncertainty_entropy_10.csv
```

`--limit 0` processes all examples.

## Classification Evaluation

Run the grouped 80/20 HaluEval evaluation:

```bash
./scripts/evaluate.sh
```

or directly:

```bash
.venv/bin/python -m src.evaluation.evaluate_uncertainty_classifiers
```

The evaluator:

- loads `data/processed/halueval_uncertainty_entropy_features.csv`
- uses a group-aware 80/20 train/test split
- keeps paired supported/hallucinated answers from the same question in the same split
- trains Logistic Regression, Linear SVM, and Random Forest
- writes metrics and predictions to `outputs/`

Output files:

```text
outputs/tables/halueval_classifier_metrics.csv
outputs/predictions/halueval_classifier_predictions.csv
```

Current HaluEval 80/20 results:

```text
Logistic Regression: accuracy 0.9875, F1 0.9875, ROC-AUC 0.9986
Linear SVM:          accuracy 0.9885, F1 0.9885, ROC-AUC 0.9986
Random Forest:       accuracy 0.9883, F1 0.9883, ROC-AUC 0.9988
```

These results show strong separation on HaluEval QA, but they should not yet be interpreted as general hallucination detection performance across all domains.

## Next Steps

Planned next development steps:

1. Extract the same uncertainty and entropy features for TruthfulQA.
2. Train on HaluEval and test on TruthfulQA as an external generalization check.
3. Add ablation experiments comparing:
   - confidence/logprob features only
   - entropy features only
   - all uncertainty features
4. Add error analysis for false positives and false negatives.
5. Optionally add self-consistency and retrieval/evidence-agreement features if time allows.

## Current Status

Completed:

- HaluEval and TruthfulQA data loading
- normalized JSONL preprocessing
- Qwen-based token probability/logprob extraction
- entropy feature extraction
- full HaluEval uncertainty feature table
- grouped 80/20 classification evaluation
- Logistic Regression, Linear SVM, and Random Forest baselines

In progress / remaining:

- TruthfulQA external evaluation
- ablation study
- error analysis
- final report tables and discussion
