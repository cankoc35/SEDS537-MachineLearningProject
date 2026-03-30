# LLM Hallucination Detection via Uncertainty

This repository contains the term project for `SEDS 537 - Machine Learning`.

The project studies how to detect hallucinated large language model (LLM) outputs using uncertainty-related signals derived from generation. The main idea is to treat hallucination detection as a binary classification problem: given a prompt and a generated answer, predict whether the answer is supported or hallucinated.

## Project Aim

Large language models can generate fluent responses that are factually incorrect or unsupported. These hallucinations reduce reliability in question answering and other knowledge-intensive tasks. This project aims to build and evaluate a reproducible framework for hallucination detection using uncertainty signals such as token confidence and entropy, together with additional signals like self-consistency and retrieval-based evidence agreement.

## Main Objective

The project will:

- generate answers with a fixed open-source LLM
- extract uncertainty-related features from token probabilities
- compare several baseline hallucination detection methods
- propose a stronger multi-signal detection method
- evaluate results with ablation study and error analysis

## Planned Methods

### Baseline Methods

- Entropy thresholding
- Token confidence thresholding
- Logistic Regression on handcrafted uncertainty features
- Support Vector Machine (SVM) on handcrafted uncertainty features

### Proposed Method

A multi-signal hallucination detector that combines:

- token-level uncertainty features
- self-consistency across multiple sampled responses
- retrieval-based evidence agreement from a lightweight RAG component

## Datasets

The main datasets planned for this project are:

- `TruthfulQA`
- `HaluEval`

One dataset can be used for development and the other for external validation to test generalization.

## Evaluation Plan

The models will be evaluated as hallucination detectors using:

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- PR-AUC

The project will also include:

- ablation study on different feature groups
- error analysis of failure cases
- visualizations where useful

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── docs/
│   └── seds537_termProject.pdf
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── tables/
├── proposal/
│   ├── proposal.tex
│   └── proposal.pdf
├── scripts/
│   ├── evaluate.sh
│   ├── run_baselines.sh
│   ├── run_generation.sh
│   └── run_proposed.sh
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── generation/
│   ├── models/
│   └── utils/
├── requirements.txt
└── README.md
```

## What We Are Going To Do

The practical workflow of the project is:

1. Prepare `TruthfulQA` and `HaluEval` in a consistent format.
2. Select and fix one base LLM for answer generation.
3. Generate answers and collect token-level confidence or log-probability information.
4. Extract uncertainty features such as mean entropy, max entropy, average token confidence, and low-confidence token ratio.
5. Implement threshold-based baselines.
6. Train classical classifiers such as Logistic Regression and SVM on the extracted features.
7. Build the proposed multi-signal method by adding self-consistency and retrieval-based support features.
8. Run experiments, ablation study, and error analysis.
9. Produce figures, tables, and the final LNCS-style paper.

## Roadmap

### Phase 1: Project Setup

- finalize proposal
- organize repository and experiment pipeline
- define dataset format and labels

### Phase 2: Data and Generation

- download and inspect datasets
- implement data loaders and preprocessing
- generate answers with the selected LLM
- collect token probabilities or logits

### Phase 3: Feature Engineering

- compute entropy-based features
- compute confidence-based features
- prepare response-level feature tables

### Phase 4: Baselines

- implement entropy threshold baseline
- implement token confidence threshold baseline
- train Logistic Regression
- train SVM

### Phase 5: Proposed Method

- add self-consistency features
- add retrieval and evidence-agreement features
- train the final multi-signal hallucination detector

### Phase 6: Evaluation

- compare all methods on both datasets
- run ablation study
- perform error analysis
- generate tables and plots

### Phase 7: Final Delivery

- clean the codebase for reproducibility
- document setup and execution steps
- write the final paper and prepare presentation materials

## Reproducibility Goal

The final repository should include:

- clean project structure
- dataset acquisition instructions
- dependency file
- generation, training, and evaluation scripts
- enough documentation to reproduce the experiments

## Current Status

- proposal drafted in LaTeX
- repository structure initialized
- implementation modules and script entrypoints created
- experimental pipeline still to be implemented
