# LLM Hallucination Detection via Uncertainty

This repository contains the term project for `SEDS 537 - Machine Learning`.

The project studies hallucination detection as a binary classification task. Given a question, a candidate answer, and optionally a context passage, the system predicts whether the answer is supported/truthful or hallucinated/unsupported.

## Project Aim

The aim is to detect hallucinated LLM answers using uncertainty signals extracted from an open-source LLM. The current implementation uses `Qwen/Qwen2.5-3B` as a feature extractor. Qwen is not asked to judge hallucination directly. Instead, the code reads Qwen's token-level probability distribution for a given answer and computes numeric uncertainty features.

Current feature groups:

- token probability features
- token log-probability features
- low-confidence token ratio
- token entropy features
- high-entropy token ratio

These features are used to train:

- Logistic Regression
- Linear SVM
- Random Forest

## Current Finding

The method works very well on context-grounded HaluEval QA, but performance is weaker on no-context TruthfulQA. This suggests that the current detector is strongest when context/evidence is available.

Important limitation:

```text
The classifier does not directly use raw context text.
However, context affects the uncertainty features because Qwen scores:
context + question + answer
```

For open-domain question-answer pairs without context, the method depends more heavily on Qwen's internal knowledge and calibration.

## Datasets

The project uses:

- `HaluEval QA`: main context-grounded hallucination dataset
- `TruthfulQA generation`: no-context truthfulness dataset and external evaluation dataset

Processed schema:

```text
id, dataset, task, prompt, context, answer, label, source, metadata
```

Labels:

```text
0 = supported / truthful
1 = hallucinated / unsupported / incorrect
```

Processed files:

```text
data/processed/halueval.jsonl
data/processed/truthfulqa.jsonl
```

Feature tables:

```text
data/processed/halueval_uncertainty_entropy_features.csv
data/processed/truthfulqa_uncertainty_entropy_features.csv
```

## Repository Structure

```text
.
├── data/
│   ├── raw/                         # raw downloaded datasets
│   ├── processed/                   # processed JSONL and feature tables
│   └── README.md
├── docs/
│   ├── development_log.md
│   ├── literature/
│   └── progress-report/
├── models/                          # local LLM weights, ignored by Git
│   └── qwen2.5-3b/
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── tables/
├── scripts/
│   ├── prepare_data.sh
│   ├── evaluate.sh
│   ├── evaluate_truthfulqa.sh
│   ├── evaluate_external_truthfulqa.sh
│   └── analyze_truthfulqa_errors.sh
├── src/
│   ├── data/                        # download and preprocessing
│   ├── evaluation/                  # metrics, evaluation, error analysis
│   ├── features/                    # uncertainty, consistency, RAG placeholders
│   ├── generation/                  # Qwen scoring and feature extraction
│   ├── models/                      # classifier definitions
│   └── utils/
├── proposal/
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

Download and preprocess the default datasets:

```bash
./scripts/prepare_data.sh
```

This creates:

```text
data/processed/halueval.jsonl
data/processed/truthfulqa.jsonl
```

## Local Qwen Model

The feature extractor expects local Qwen weights under:

```text
models/qwen2.5-3b/
```

If needed, download with:

```bash
hf download Qwen/Qwen2.5-3B \
  --repo-type model \
  --local-dir models/qwen2.5-3b
```

The root `models/` directory should not be committed to Git.

## Feature Extraction

HaluEval:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/halueval.jsonl \
  --limit 0 \
  --model models/qwen2.5-3b \
  --output data/processed/halueval_uncertainty_entropy_features.csv
```

TruthfulQA:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/truthfulqa.jsonl \
  --limit 0 \
  --model models/qwen2.5-3b \
  --output data/processed/truthfulqa_uncertainty_entropy_features.csv
```

`--limit 0` means process all rows.

## Evaluation Commands

### HaluEval 80/20 Evaluation

```bash
./scripts/evaluate.sh
```

Outputs:

```text
outputs/tables/halueval_classifier_metrics.csv
outputs/predictions/halueval_classifier_predictions.csv
```

Current HaluEval results:

```text
Logistic Regression: accuracy 0.9875, F1 0.9875, ROC-AUC 0.9986
Linear SVM:          accuracy 0.9885, F1 0.9885, ROC-AUC 0.9986
Random Forest:       accuracy 0.9883, F1 0.9883, ROC-AUC 0.9988
```

### TruthfulQA 80/20 Evaluation

```bash
./scripts/evaluate_truthfulqa.sh
```

Outputs:

```text
outputs/tables/truthfulqa_classifier_metrics.csv
outputs/predictions/truthfulqa_classifier_predictions.csv
```

Current TruthfulQA-only results:

```text
Logistic Regression: accuracy 0.6254, F1 0.6928, ROC-AUC 0.6539
Linear SVM:          accuracy 0.6254, F1 0.6941, ROC-AUC 0.6537
Random Forest:       accuracy 0.6352, F1 0.6850, ROC-AUC 0.6894
```

### External Evaluation: HaluEval to TruthfulQA

```bash
./scripts/evaluate_external_truthfulqa.sh
```

Outputs:

```text
outputs/tables/halueval_to_truthfulqa_classifier_metrics.csv
outputs/predictions/halueval_to_truthfulqa_classifier_predictions.csv
```

Current external evaluation results:

```text
Logistic Regression: accuracy 0.4838, F1 0.6252, ROC-AUC 0.3752
Linear SVM:          accuracy 0.4836, F1 0.6248, ROC-AUC 0.3719
Random Forest:       accuracy 0.4902, F1 0.6311, ROC-AUC 0.3985
```

Interpretation: HaluEval-trained models do not transfer well to TruthfulQA. This is evidence of dataset shift.

## Error Analysis

Run:

```bash
./scripts/analyze_truthfulqa_errors.sh
```

Outputs:

```text
outputs/tables/feature_distribution_by_dataset.csv
outputs/tables/feature_distribution_shift.csv
outputs/tables/truthfulqa_error_summary.csv
outputs/predictions/truthfulqa_error_examples.csv
```

Main finding:

TruthfulQA examples are more uncertain overall than HaluEval examples. Compared with HaluEval, TruthfulQA has:

- higher `max_token_entropy`
- lower `min_token_probability`
- higher `mean_token_entropy`
- lower `mean_token_probability`
- higher `low_confidence_token_ratio`

This explains why many correct TruthfulQA answers are predicted as hallucinated.

## Current Status

Completed:

- HaluEval and TruthfulQA preprocessing
- Qwen-based token probability/logprob extraction
- entropy feature extraction
- HaluEval feature table
- TruthfulQA feature table
- HaluEval grouped 80/20 evaluation
- TruthfulQA grouped 80/20 evaluation
- HaluEval-to-TruthfulQA external evaluation
- feature distribution analysis
- TruthfulQA error analysis
- progress report

Missing or remaining for final completion:

- ablation study
- final visualizations
- final error analysis write-up
- final report discussion and limitations
- optional improvement experiment

## Remaining Work

Recommended next steps:

1. Run ablation experiments:
   - confidence/logprob features only
   - entropy features only
   - all uncertainty features
2. Add visualizations:
   - confusion matrices
   - ROC curves
   - feature importance
   - feature distribution plots
3. Write final discussion:
   - strong HaluEval performance
   - weaker TruthfulQA performance
   - context-grounded vs no-context detection
   - dataset shift
4. Optional future improvement:
   - automatic retrieval of evidence/context before feature extraction
   - testing a smaller or different LLM as the feature extractor

## Scope Note

The current model should be described as a context-grounded hallucination detector. It performs best when context is available. For no-context use cases, a future version may retrieve evidence automatically from Wikipedia, web search, or trusted domain documents before extracting uncertainty features.
