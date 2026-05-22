# Development Log

## May 7, 2026

Today we moved the project from an initialized scaffold to a working data-preparation stage.

### Project Direction Clarified

- Reviewed the main project README and proposal materials.
- Confirmed the project topic: LLM hallucination detection using uncertainty-aware multi-signal fusion.
- Clarified the task formulation as binary classification:
  - `label = 0`: supported, truthful, or non-hallucinated answer.
  - `label = 1`: unsupported, false, or hallucinated answer.
- Decided to use both planned datasets:
  - HaluEval QA as the main development and training dataset.
  - TruthfulQA generation as an external validation and generalization dataset.

### Dataset Roles Defined

- HaluEval QA will be used for the main hallucination-detection workflow because it directly provides:
  - `knowledge`
  - `question`
  - `right_answer`
  - `hallucinated_answer`
- HaluEval examples are converted into two binary rows:
  - `question + context + right_answer` with `label = 0`.
  - `question + context + hallucinated_answer` with `label = 1`.
- TruthfulQA generation will be used to test whether the detector generalizes to misconception-style falsehoods.
- TruthfulQA examples are converted into multiple binary rows:
  - `question + correct_answer` with `label = 0`.
  - `question + incorrect_answer` with `label = 1`.
- We decided not to add HaluEval dialogue or summarization yet because QA-only scope is cleaner and more consistent for the first full experiment.

### Data Loading Implemented

- Implemented dataset acquisition logic in `src/data/load_data.py`.
- Added direct download support for official HaluEval files from the RUCAIBox GitHub repository.
- Added TruthfulQA loading through the Hugging Face `datasets` package.
- Saved a local raw TruthfulQA JSONL copy at:
  - `data/raw/truthfulqa/truthfulqa_generation.jsonl`
- Saved the raw HaluEval QA file at:
  - `data/raw/halueval/qa_data.json`
- Added defensive loading support for HaluEval files that are named `.json` but formatted as JSONL.

### Data Preprocessing Implemented

- Implemented preprocessing logic in `src/data/preprocess.py`.
- Created a shared normalized schema for both datasets:
  - `id`
  - `dataset`
  - `task`
  - `prompt`
  - `context`
  - `answer`
  - `label`
  - `source`
  - `metadata`
- Generated processed HaluEval QA data:
  - `data/processed/halueval.jsonl`
  - 20,000 rows
- Generated processed TruthfulQA data:
  - `data/processed/truthfulqa.jsonl`
  - 5,918 rows
- Verified example records manually to confirm label correctness.

### Data Preparation Script Added

- Added `scripts/prepare_data.sh`.
- The script downloads and preprocesses the default dataset setup:
  - HaluEval QA
  - TruthfulQA generation
- Verified the script works with:

```bash
./scripts/prepare_data.sh
```

- Confirmed successful output:

```text
Wrote 20000 records to data/processed/halueval.jsonl
Wrote 5918 records to data/processed/truthfulqa.jsonl
```

### Dependency Setup Updated

- Updated `requirements.txt` with the initial project dependencies:
  - `datasets`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `transformers`
  - `torch`
  - `matplotlib`
- Discussed using Python 3.11 for the project virtual environment because it has strong compatibility with the planned ML stack.

### Documentation Updated

- Updated `data/README.md` with data preparation commands and the processed schema.
- Added this development log so progress can be tracked across future work sessions.

### Important Notes

- HaluEval's `qa_data.json` is not standard single-document JSON. It is newline-delimited JSON, so VS Code may highlight it in red even though the data is usable.
- The processed files use the `.jsonl` extension intentionally because each line is one independent JSON record.
- The rest of the project should use files from `data/processed/`, not the raw files directly.

### Next Step

The next development milestone is feature extraction:

- Add utilities to load processed JSONL files.
- Implement token scoring or log-probability extraction with a fixed open-source LLM.
- Compute uncertainty features such as token confidence, mean log probability, entropy, and low-confidence token ratio.
- Save feature tables for HaluEval and TruthfulQA.

### Literature Review Triage

- Reviewed the PDFs added under `docs/literature/`.
- Identified the strongest papers for the project scope: hallucination detection, uncertainty estimation, HaluEval, semantic entropy, calibration, and multi-signal detection.
- Created a selected paper folder:
  - `docs/literature/selected/`
- Copied 10 core papers into the selected folder while preserving the original PDFs.
- Created concise review notes:
  - `docs/literature/literature_review_notes.md`
- Noted that the original TruthfulQA benchmark paper and SelfCheckGPT paper should be added if possible because they are important for the final literature review.

## May 13, 2026

Today we extended the project from HaluEval-only evaluation to TruthfulQA evaluation and error analysis.

### TruthfulQA Feature Table Ready

- Confirmed that uncertainty and entropy features were extracted for TruthfulQA:
  - `data/processed/truthfulqa_uncertainty_entropy_features.csv`
- The feature table contains 5,918 examples:
  - 2,600 correct/truthful answers with `label = 0`
  - 3,318 incorrect/false answers with `label = 1`
- The feature columns match the HaluEval uncertainty feature table, so the same classifiers can be reused.

### External Evaluation: Train on HaluEval, Test on TruthfulQA

- Updated `src/evaluation/evaluate_uncertainty_classifiers.py` to support external evaluation with:
  - `--train-input`
  - `--test-input`
- Added:
  - `scripts/evaluate_external_truthfulqa.sh`
- Ran the experiment:
  - train set: HaluEval, 20,000 rows
  - test set: TruthfulQA, 5,918 rows
- Results:

```text
Logistic Regression: accuracy 0.4838, F1 0.6252, ROC-AUC 0.3752
Linear SVM:          accuracy 0.4836, F1 0.6248, ROC-AUC 0.3719
Random Forest:       accuracy 0.4902, F1 0.6311, ROC-AUC 0.3985
```

- Interpretation:
  - The HaluEval-trained detector does not transfer well to TruthfulQA.
  - This suggests that the very high HaluEval results are partly dataset-specific.
  - TruthfulQA uses a different task structure: no context passage, misconception-style questions, and multiple correct/incorrect answer candidates per question.

### TruthfulQA-Only Evaluation

- Updated grouped splitting so TruthfulQA answers from the same original question stay in the same split.
  - Example: `truthfulqa_000000_correct_0` and `truthfulqa_000000_incorrect_6` are grouped as `truthfulqa_000000`.
- Added:
  - `scripts/evaluate_truthfulqa.sh`
- Ran grouped 80/20 train-test evaluation on TruthfulQA only:
  - train rows: 4,786
  - test rows: 1,132
- Results:

```text
Logistic Regression: accuracy 0.6254, F1 0.6928, ROC-AUC 0.6539
Linear SVM:          accuracy 0.6254, F1 0.6941, ROC-AUC 0.6537
Random Forest:       accuracy 0.6352, F1 0.6850, ROC-AUC 0.6894
```

- Interpretation:
  - Uncertainty features are useful on TruthfulQA, but much weaker than on HaluEval.
  - Random Forest gives the best ROC-AUC on TruthfulQA-only evaluation.
  - The results support the conclusion that TruthfulQA is a harder and structurally different evaluation dataset.

### Feature Distribution and Error Analysis

- Replaced the placeholder `src/evaluation/error_analysis.py` with a working analysis script.
- Added:
  - `scripts/analyze_truthfulqa_errors.sh`
- Generated:
  - `outputs/tables/feature_distribution_by_dataset.csv`
  - `outputs/tables/feature_distribution_shift.csv`
  - `outputs/tables/truthfulqa_error_summary.csv`
  - `outputs/predictions/truthfulqa_error_examples.csv`

### Main Distribution-Shift Finding

- TruthfulQA examples are more uncertain overall than HaluEval examples.
- Largest shifts:

```text
max_token_entropy:          HaluEval 2.3049 -> TruthfulQA 3.3386
min_token_probability:      HaluEval 0.2433 -> TruthfulQA 0.0321
mean_token_entropy:         HaluEval 0.8602 -> TruthfulQA 1.3642
mean_token_probability:     HaluEval 0.6854 -> TruthfulQA 0.5267
low_confidence_token_ratio: HaluEval 0.1502 -> TruthfulQA 0.2977
negative_mean_logprob:      HaluEval 1.0772 -> TruthfulQA 1.9594
```

- This explains why TruthfulQA performance is lower:
  - many correct TruthfulQA answers still look uncertain to Qwen
  - some incorrect misconception answers look plausible and receive relatively confident scores
  - the decision boundary learned from HaluEval does not match TruthfulQA well

### TruthfulQA Error Pattern

- TruthfulQA-only error analysis showed many false positives.
- False positive rates on the TruthfulQA test split:

```text
Logistic Regression: 285 / 1132
Linear SVM:          288 / 1132
Random Forest:       245 / 1132
```

- False positives are correct answers predicted as hallucinated.
- This supports the finding that correct TruthfulQA answers often have higher uncertainty than supported HaluEval answers.

### Current Conclusion

- HaluEval-only performance is very strong, but it should not be overclaimed.
- TruthfulQA experiments show limited cross-dataset generalization.
- The project now has a more realistic story:
  - uncertainty features work very well on HaluEval QA
  - they are moderately useful on TruthfulQA
  - dataset shift is a major challenge

### Next Step

- Run ablation experiments:
  - confidence/logprob features only
  - entropy features only
  - all uncertainty features
- Use ablation to understand which feature groups are responsible for performance on HaluEval and TruthfulQA.

### Context Requirement and Future Retrieval Direction

- Discussed an important limitation of the current strongest setup:
  - the HaluEval-trained model works best when a context/evidence passage is available
  - the classifier itself does not directly use raw context text
  - however, context affects the uncertainty features because Qwen scores the answer conditioned on `context + question + answer`
- This means the current high-performing detector is best described as a context-grounded hallucination detector.
- For use cases where only `question + answer` is available, performance is weaker, as seen in TruthfulQA.
- A no-context detector is possible, but it becomes open-domain truthfulness detection and depends heavily on the feature extractor model's internal knowledge and calibration.

Possible future solution:

- Add automatic evidence retrieval before feature extraction.
- Pipeline idea:

```text
question + answer
-> retrieve evidence/context from a trusted source
-> build context + question + answer input
-> extract Qwen uncertainty features
-> classify supported vs hallucinated
```

- The context should ideally come from an external source, not be invented by the LLM.
- Possible evidence sources:
  - Wikipedia for general factual questions
  - web search for broader but noisier coverage
  - domain-specific trusted documents for specialized tasks
  - dataset-provided context when available
- This would make the system more convenient because the user would not need to manually provide context, while still keeping the detector evidence-grounded.
- Current project scope will likely keep retrieval as future work unless time allows.

## May 21, 2026

Today we implemented and ran the ablation study for the uncertainty feature groups.

### Ablation Implementation

- Replaced the placeholder `src/evaluation/ablation.py` with a working ablation pipeline.
- Added:
  - `scripts/run_ablation.sh`
- The ablation compares three feature groups:
  - `confidence_logprob`: answer length plus token probability/log-probability features
  - `entropy`: answer length plus token entropy features
  - `all_features`: confidence/log-probability plus entropy features
- The ablation is run for three settings:
  - HaluEval grouped 80/20
  - TruthfulQA grouped 80/20
  - train on HaluEval, test on TruthfulQA

### Ablation Output Files

- Generated:
  - `outputs/tables/halueval_ablation_results.csv`
  - `outputs/tables/truthfulqa_ablation_results.csv`
  - `outputs/tables/halueval_to_truthfulqa_ablation_results.csv`

### HaluEval Ablation Findings

- Best HaluEval result:

```text
All features + Linear SVM
Accuracy: 0.9885
F1:       0.9885
ROC-AUC:  0.9986
```

- Summary:
  - confidence/logprob-only features are already very strong
  - entropy-only features are useful but weaker
  - all features together give the best overall HaluEval performance

### TruthfulQA Ablation Findings

- Best TruthfulQA ROC-AUC:

```text
All features + Random Forest
Accuracy: 0.6352
F1:       0.6850
ROC-AUC:  0.6894
```

- Summary:
  - TruthfulQA remains harder than HaluEval
  - confidence/logprob and entropy features both provide useful signal
  - all features are the safest main setting

### HaluEval to TruthfulQA Transfer Ablation

- Best transfer F1:

```text
Entropy only + Random Forest
Accuracy: 0.5035
F1:       0.6451
ROC-AUC:  0.4391
```

- Summary:
  - entropy-only transfers slightly better than all features in this setting
  - however, all transfer results remain weak
  - the main issue is still dataset shift between HaluEval and TruthfulQA

### Current Ablation Conclusion

- We should continue using all features as the main method.
- Ablation shows that:
  - confidence/logprob features are the strongest signal on HaluEval
  - entropy adds useful complementary information
  - all features perform best within the same dataset
  - cross-dataset generalization remains weak because HaluEval and TruthfulQA have different uncertainty patterns

### Updated Completion Status

- Ablation study is now completed.
- Remaining work:
  - visualizations
  - final error analysis write-up
  - final report discussion and limitations
  - optional improvement experiment if time allows

### Presentation Note: Feature Extractor Model Size

- Started testing a smaller feature extractor, `Qwen/Qwen2.5-0.5B`, in addition to the main `Qwen/Qwen2.5-3B` setup.
- Initial one-example check shows that the smaller model is more uncertain even on a supported HaluEval answer:

```text
Qwen2.5-3B:   Arthur token probability about 0.92
Qwen2.5-0.5B: Arthur token probability about 0.71
```

- This suggests that the quality and calibration of the LLM used for feature extraction can affect the downstream hallucination detector.
- Presentation wording:

```text
The feature extractor matters. A smaller LLM can be less confident and less calibrated, which may make hallucination detection harder. Larger or better-calibrated LLMs may provide stronger uncertainty features, but this should be verified experimentally.
```
