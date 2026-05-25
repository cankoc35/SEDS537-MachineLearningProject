# LLM Hallucination Detection via Uncertainty

This repository contains the term project for `SEDS 537 - Machine Learning`.

The project studies hallucination detection as a binary classification task. Given a question, a candidate answer, and optionally a context passage, the system predicts whether the answer is supported/truthful or hallucinated/unsupported.

## Project Aim

The aim is to detect hallucinated LLM answers using uncertainty signals extracted from open-source LLMs. The current implementation uses `Qwen/Qwen2.5-3B` as the main feature extractor and `Qwen/Qwen2.5-0.5B` as a smaller comparison model. Qwen is not asked to judge hallucination directly. Instead, the code reads Qwen's token-level probability distribution for a given answer and computes numeric uncertainty features.

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
data/processed/halueval_no_context.jsonl
data/processed/truthfulqa.jsonl
```

Feature tables:

```text
data/processed/halueval_uncertainty_entropy_features.csv
data/processed/truthfulqa_uncertainty_entropy_features.csv
data/processed/halueval_uncertainty_entropy_qwen05b_features.csv
data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv
data/processed/halueval_no_context_uncertainty_entropy_features.csv
data/processed/halueval_no_context_uncertainty_entropy_qwen05b_features.csv
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
│   ├── evaluate_halueval_no_context.sh
│   ├── evaluate.sh
│   ├── evaluate_truthfulqa.sh
│   ├── evaluate_external_truthfulqa.sh
│   ├── evaluate_qwen05b.sh
│   ├── run_ablation.sh
│   ├── run_ablation_qwen05b.sh
│   ├── run_cross_validation.sh
│   ├── run_cross_validation_no_context.sh
│   ├── create_figures.sh
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

Create the no-context HaluEval variant:

```bash
.venv/bin/python -m src.data.create_no_context_dataset \
  --input data/processed/halueval.jsonl \
  --output data/processed/halueval_no_context.jsonl
```

## Local Qwen Model

The feature extractor expects local Qwen weights under:

```text
models/qwen2.5-3b/
models/qwen2.5-0.5b/
```

If needed, download with:

```bash
hf download Qwen/Qwen2.5-3B \
  --repo-type model \
  --local-dir models/qwen2.5-3b
```

Smaller comparison model:

```bash
hf download Qwen/Qwen2.5-0.5B \
  --repo-type model \
  --local-dir models/qwen2.5-0.5b
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

For the 0.5B comparison, use the same commands with:

```text
--model models/qwen2.5-0.5b
```

and write to:

```text
data/processed/halueval_uncertainty_entropy_qwen05b_features.csv
data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv
```

For the no-context HaluEval variant:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/halueval_no_context.jsonl \
  --limit 0 \
  --model models/qwen2.5-3b \
  --output data/processed/halueval_no_context_uncertainty_entropy_features.csv
```

For Qwen2.5-0.5B:

```bash
.venv/bin/python -m src.generation.extract_logprobs \
  --input data/processed/halueval_no_context.jsonl \
  --limit 0 \
  --model models/qwen2.5-0.5b \
  --output data/processed/halueval_no_context_uncertainty_entropy_qwen05b_features.csv
```

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

### Qwen2.5-0.5B Evaluation

```bash
./scripts/evaluate_qwen05b.sh
```

Outputs:

```text
outputs/tables/halueval_qwen05b_classifier_metrics.csv
outputs/tables/truthfulqa_qwen05b_classifier_metrics.csv
outputs/tables/halueval_to_truthfulqa_qwen05b_classifier_metrics.csv
```

Main 0.5B results:

```text
HaluEval 80/20 best:        Random Forest, F1 0.9745, ROC-AUC 0.9938
TruthfulQA 80/20 best:      Random Forest, F1 0.6981, ROC-AUC 0.6898
HaluEval -> TruthfulQA:     Random Forest, F1 0.6388, ROC-AUC 0.4000
```

The smaller feature extractor remains useful, but it is weaker than 3B on HaluEval.

### HaluEval No-Context Evaluation

```bash
./scripts/evaluate_halueval_no_context.sh
```

Outputs:

```text
outputs/tables/halueval_no_context_qwen3b_classifier_metrics.csv
outputs/tables/halueval_no_context_qwen05b_classifier_metrics.csv
outputs/tables/halueval_no_context_to_truthfulqa_qwen3b_classifier_metrics.csv
outputs/tables/halueval_no_context_to_truthfulqa_qwen05b_classifier_metrics.csv
```

Main no-context results:

```text
HaluEval no-context 3B 80/20:        Random Forest, F1 0.9276, ROC-AUC 0.9699
HaluEval no-context 0.5B 80/20:      Random Forest, F1 0.9283, ROC-AUC 0.9681
HaluEval no-context 3B -> TruthfulQA:   Random Forest, F1 0.5927, ROC-AUC 0.4710
HaluEval no-context 0.5B -> TruthfulQA: Random Forest, F1 0.6024, ROC-AUC 0.4890
```

Context effect:

```text
HaluEval with context:    about 0.987--0.989 F1
HaluEval without context: about 0.927--0.928 F1
TruthfulQA no context:    about 0.697--0.701 F1
```

Removing context lowers HaluEval performance, which confirms that context helps. However, HaluEval no-context still performs much better than TruthfulQA, suggesting that HaluEval contains dataset-specific signals beyond evidence grounding.

## Cross-Validation

Run grouped 5-fold cross-validation:

```bash
./scripts/run_cross_validation.sh
```

Outputs:

```text
outputs/tables/halueval_qwen3b_cv_summary.csv
outputs/tables/truthfulqa_qwen3b_cv_summary.csv
outputs/tables/halueval_qwen05b_cv_summary.csv
outputs/tables/truthfulqa_qwen05b_cv_summary.csv
```

For HaluEval no-context CV:

```bash
./scripts/run_cross_validation_no_context.sh
```

Outputs:

```text
outputs/tables/halueval_no_context_qwen3b_cv_summary.csv
outputs/tables/halueval_no_context_qwen05b_cv_summary.csv
```

Grouped cross-validation keeps all answers from the same original question in the same fold. This prevents train-test leakage.

Main CV findings:

```text
Qwen2.5-3B   HaluEval:   best ROC-AUC 0.9986, F1 about 0.9871
Qwen2.5-0.5B HaluEval:   best ROC-AUC 0.9945, F1 about 0.9746
Qwen2.5-3B   TruthfulQA: best ROC-AUC 0.6836, F1 about 0.6968
Qwen2.5-0.5B TruthfulQA: best ROC-AUC 0.6841, F1 about 0.7006
Qwen2.5-3B   HaluEval no-context:   best ROC-AUC 0.9711, F1 about 0.9277
Qwen2.5-0.5B HaluEval no-context:   best ROC-AUC 0.9701, F1 about 0.9271
```

The CV results confirm the original 80/20 pattern: HaluEval is much easier than TruthfulQA.

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

## Ablation Study

Run:

```bash
./scripts/run_ablation.sh
```

For the 0.5B feature tables:

```bash
./scripts/run_ablation_qwen05b.sh
```

Outputs:

```text
outputs/tables/halueval_ablation_results.csv
outputs/tables/truthfulqa_ablation_results.csv
outputs/tables/halueval_to_truthfulqa_ablation_results.csv
outputs/tables/halueval_qwen05b_ablation_results.csv
outputs/tables/truthfulqa_qwen05b_ablation_results.csv
outputs/tables/halueval_to_truthfulqa_qwen05b_ablation_results.csv
```

Feature groups:

```text
confidence_logprob = answer length + probability/log-probability features
entropy = answer length + entropy features
all_features = confidence/logprob + entropy features
```

Main ablation findings:

- HaluEval: all features perform best.
- TruthfulQA: all features are generally best, but performance remains moderate.
- HaluEval -> TruthfulQA: entropy-only transfers slightly better, but all transfer results are weak.
- The same pattern appears with the 0.5B feature extractor.

Final decision:

```text
Use all features as the main method.
Use ablation results to explain feature contribution and dataset shift.
```

## Visualizations

Generate report figures:

```bash
./scripts/create_figures.sh
```

Outputs:

```text
outputs/figures/cv_model_size_comparison.png
outputs/figures/context_effect_comparison.png
outputs/figures/ablation_feature_group_comparison.png
outputs/figures/confusion_matrices.png
outputs/figures/roc_curves.png
outputs/figures/feature_distribution_shift.png
```

These figures summarize model-size comparison, context effect, feature ablation, classification errors, ROC curves, and feature distribution shift.

## Current Status

Completed:

- HaluEval and TruthfulQA preprocessing
- HaluEval no-context dataset
- Qwen-based token probability/logprob extraction
- entropy feature extraction
- HaluEval feature table
- TruthfulQA feature table
- Qwen2.5-0.5B feature tables
- HaluEval no-context feature tables
- HaluEval grouped 80/20 evaluation
- HaluEval no-context grouped 80/20 evaluation
- TruthfulQA grouped 80/20 evaluation
- HaluEval-to-TruthfulQA external evaluation
- HaluEval no-context-to-TruthfulQA external evaluation
- Qwen2.5-0.5B grouped and external evaluation
- grouped 5-fold cross-validation
- HaluEval no-context grouped 5-fold cross-validation
- feature distribution analysis
- TruthfulQA error analysis
- ablation study
- Qwen2.5-0.5B ablation study
- visualizations
- progress report

Missing or remaining for final completion:

- final error analysis write-up
- final report discussion and limitations
- optional improvement experiment or future-work discussion

## Remaining Work

Recommended next steps:

1. Write final discussion:
   - strong HaluEval performance
   - effect of removing HaluEval context
   - weaker TruthfulQA performance
   - context-grounded vs no-context detection
   - dataset shift
2. Use generated figures in the final report and presentation.
3. Optional future improvement:
   - automatic retrieval of evidence/context before feature extraction
   - testing a different LLM family as the feature extractor

## Scope Note

The current model should be described as a context-grounded hallucination detector. It performs best when context is available. For no-context use cases, a future version may retrieve evidence automatically from Wikipedia, web search, or trusted domain documents before extracting uncertainty features.
