# LLM Hallucination Detection via Uncertainty

Final project repository for `SEDS 537 - Machine Learning`.

This project studies hallucination detection as a binary classification task. Given a question, a candidate answer, and optionally a context passage, the system predicts whether the answer is supported/truthful or hallucinated/unsupported.

The central idea is to represent each answer with uncertainty features extracted from an open-source language model, then train classical machine learning classifiers on those numeric features.

## Final Submission Contents

This repository contains:

- source code for preprocessing, feature extraction, evaluation, ablation, error analysis, and visualization
- processed dataset files and extracted uncertainty feature tables
- evaluation outputs, prediction files, tables, and figures
- final report files under `docs/final-report/`
- development notes under `docs/development_log.md`

Local model weights and virtual environments are intentionally excluded from Git:

```text
models/
.venv/
```

## Project Aim

The aim is to detect hallucinated LLM answers using uncertainty signals extracted from open-source LLMs. The implementation uses:

- `Qwen/Qwen2.5-3B` as the main feature extractor
- `Qwen/Qwen2.5-0.5B` as a smaller comparison feature extractor

Qwen is not asked directly whether an answer is hallucinated. Instead, Qwen is used as a feature extractor. The code reads Qwen's token-level probability distribution for the given answer and computes numerical uncertainty features.

The classification models are:

- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest

## Dataset Representation

Raw examples are normalized into the following structure:

```text
prompt, context, answer, label
```

The classifiers do not receive raw text directly. Each example is represented as a 15-dimensional numerical uncertainty feature vector:

```text
X = uncertainty feature vector
y = binary hallucination label
```

The target labels are:

```text
0 = supported / truthful
1 = hallucinated / unsupported / incorrect
```

The extracted independent variables are:

```text
answer_length_tokens
mean_token_probability
min_token_probability
max_token_probability
mean_token_logprob
min_token_logprob
max_token_logprob
sum_token_logprob
negative_mean_logprob
low_confidence_token_ratio
mean_token_entropy
min_token_entropy
max_token_entropy
sum_token_entropy
high_entropy_token_ratio
```

## Datasets

The project uses two benchmark datasets:

- HaluEval QA: context-grounded hallucination detection dataset
- TruthfulQA generation: no-context truthfulness dataset

Dataset links:

- HaluEval: https://github.com/RUCAIBox/HaluEval
- TruthfulQA: https://huggingface.co/datasets/truthfulqa/truthful_qa
- TruthfulQA paper: https://arxiv.org/abs/2109.07958

Processed files in this repository:

```text
data/processed/halueval.jsonl
data/processed/halueval_no_context.jsonl
data/processed/truthfulqa.jsonl
```

Extracted feature tables:

```text
data/processed/halueval_uncertainty_entropy_features.csv
data/processed/truthfulqa_uncertainty_entropy_features.csv
data/processed/halueval_uncertainty_entropy_qwen05b_features.csv
data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv
data/processed/halueval_no_context_uncertainty_entropy_features.csv
data/processed/halueval_no_context_uncertainty_entropy_qwen05b_features.csv
```

## Method Summary

For each example, the pipeline performs the following steps:

1. Build a scoring prompt from the question, optional context, and answer.
2. Tokenize the prompt and the answer.
3. Run Qwen on the combined prompt and answer.
4. Extract probability, log-probability, and entropy for each answer token.
5. Aggregate token-level scores into answer-level uncertainty features.
6. Train binary classifiers using the extracted feature table.

Important distinction:

```text
The classifier does not directly use raw context text.
However, context can affect the uncertainty features because Qwen scores:
context + question + answer
```

This is why the method performs best in context-grounded settings.

## Main Findings

The method works very well on HaluEval QA and performs moderately on TruthfulQA. This suggests that uncertainty features are useful for hallucination detection, but generalization across dataset types is difficult.

Best grouped 5-fold cross-validation results:

| Experiment | Best Model | F1 | ROC-AUC |
|---|---:|---:|---:|
| HaluEval QA, Qwen2.5-3B | RBF SVM | 0.9883 | 0.9979 |
| HaluEval QA, Qwen2.5-0.5B | RBF SVM | 0.9759 | 0.9939 |
| HaluEval no-context, Qwen2.5-3B | Random Forest | 0.9277 | 0.9711 |
| HaluEval no-context, Qwen2.5-0.5B | Random Forest | 0.9271 | 0.9701 |
| TruthfulQA, Qwen2.5-3B | RBF SVM | 0.7173 | 0.6826 |
| TruthfulQA, Qwen2.5-0.5B | RBF SVM | 0.7181 | 0.6820 |

External transfer from HaluEval to TruthfulQA was weak:

| Experiment | Best Model | F1 | ROC-AUC |
|---|---:|---:|---:|
| HaluEval 3B to TruthfulQA | Random Forest | 0.6311 | 0.3985 |
| HaluEval 0.5B to TruthfulQA | Random Forest | 0.6388 | 0.4000 |
| HaluEval no-context 3B to TruthfulQA | Random Forest | 0.5927 | 0.4710 |
| HaluEval no-context 0.5B to TruthfulQA | Random Forest | 0.6024 | 0.4890 |

Main interpretation:

- HaluEval is highly separable using uncertainty features.
- Removing context lowers HaluEval performance but does not collapse it.
- TruthfulQA is harder because many correct and incorrect answers produce overlapping uncertainty patterns.
- Dataset shift is a major limitation for external transfer.

## Repository Structure

```text
.
├── data/
│   ├── raw/                         # raw downloaded datasets
│   ├── processed/                   # processed JSONL and feature tables
│   └── README.md
├── docs/
│   ├── development_log.md
│   ├── final-report/                # final LaTeX report and PDF
│   ├── literature/
│   ├── presentation/
│   └── progress-report/
├── outputs/
│   ├── figures/                     # generated plots
│   ├── predictions/                 # model predictions
│   └── tables/                      # metrics and analysis tables
├── scripts/                         # reproducibility scripts
├── src/
│   ├── data/                        # download and preprocessing
│   ├── evaluation/                  # metrics, evaluation, error analysis
│   ├── features/                    # uncertainty feature aggregation
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

Create the no-context HaluEval variant:

```bash
.venv/bin/python -m src.data.create_no_context_dataset \
  --input data/processed/halueval.jsonl \
  --output data/processed/halueval_no_context.jsonl
```

## Local Qwen Models

The feature extraction scripts expect local Qwen weights under:

```text
models/qwen2.5-3b/
models/qwen2.5-0.5b/
```

Download Qwen2.5-3B:

```bash
hf download Qwen/Qwen2.5-3B \
  --repo-type model \
  --local-dir models/qwen2.5-3b
```

Download Qwen2.5-0.5B:

```bash
hf download Qwen/Qwen2.5-0.5B \
  --repo-type model \
  --local-dir models/qwen2.5-0.5b
```

The `models/` directory is ignored by Git and should not be committed.

## Feature Extraction

HaluEval with Qwen2.5-3B:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/halueval.jsonl \
  --limit 0 \
  --model models/qwen2.5-3b \
  --output data/processed/halueval_uncertainty_entropy_features.csv
```

TruthfulQA with Qwen2.5-3B:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/truthfulqa.jsonl \
  --limit 0 \
  --model models/qwen2.5-3b \
  --output data/processed/truthfulqa_uncertainty_entropy_features.csv
```

For Qwen2.5-0.5B, use the same command structure with:

```text
--model models/qwen2.5-0.5b
```

For HaluEval no-context:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/halueval_no_context.jsonl \
  --limit 0 \
  --model models/qwen2.5-3b \
  --output data/processed/halueval_no_context_uncertainty_entropy_features.csv
```

`--limit 0` means process all rows.

## Evaluation

Run HaluEval 80/20 evaluation:

```bash
./scripts/evaluate.sh
```

Run TruthfulQA 80/20 evaluation:

```bash
./scripts/evaluate_truthfulqa.sh
```

Run HaluEval-to-TruthfulQA external evaluation:

```bash
./scripts/evaluate_external_truthfulqa.sh
```

Run Qwen2.5-0.5B evaluations:

```bash
./scripts/evaluate_qwen05b.sh
```

Run HaluEval no-context evaluations:

```bash
./scripts/evaluate_halueval_no_context.sh
```

Run grouped 5-fold cross-validation:

```bash
./scripts/run_cross_validation.sh
```

Run HaluEval no-context grouped 5-fold cross-validation:

```bash
./scripts/run_cross_validation_no_context.sh
```

Grouped splitting keeps all answers from the same original question in the same train/test fold. This prevents paired examples from leaking across train and test.

## Ablation, Error Analysis, and Figures

Run ablation study:

```bash
./scripts/run_ablation.sh
./scripts/run_ablation_qwen05b.sh
```

Run TruthfulQA error analysis:

```bash
./scripts/analyze_truthfulqa_errors.sh
```

Generate figures:

```bash
./scripts/create_figures.sh
```

Main output directories:

```text
outputs/tables/
outputs/predictions/
outputs/figures/
```

## Important Output Files

Cross-validation summaries:

```text
outputs/tables/halueval_qwen3b_cv_summary.csv
outputs/tables/halueval_qwen05b_cv_summary.csv
outputs/tables/truthfulqa_qwen3b_cv_summary.csv
outputs/tables/truthfulqa_qwen05b_cv_summary.csv
outputs/tables/halueval_no_context_qwen3b_cv_summary.csv
outputs/tables/halueval_no_context_qwen05b_cv_summary.csv
```

Ablation outputs:

```text
outputs/tables/halueval_ablation_results.csv
outputs/tables/truthfulqa_ablation_results.csv
outputs/tables/halueval_qwen05b_ablation_results.csv
outputs/tables/truthfulqa_qwen05b_ablation_results.csv
```

Visual outputs:

```text
outputs/figures/confusion_matrices.png
outputs/figures/roc_curves.png
outputs/figures/feature_distribution_shift.png
outputs/figures/context_effect_comparison.png
outputs/figures/halueval_qa_pca_decision_boundary_rbf_svm.png
outputs/figures/truthfulqa_pca_decision_boundary_rbf_svm.png
```

## Final Report

The final report files are stored under:

```text
docs/final-report/
```

The editable LaTeX source should also be shared through Overleaf according to the course submission instructions.

## Scope and Limitations

The current system is best described as an uncertainty-based hallucination detector. It performs strongest on context-grounded HaluEval examples and weaker on no-context TruthfulQA examples.

The main limitation is dataset shift: patterns learned from HaluEval do not transfer reliably to TruthfulQA. A future version could add automatic evidence retrieval before uncertainty feature extraction, or test additional LLM families as feature extractors.
