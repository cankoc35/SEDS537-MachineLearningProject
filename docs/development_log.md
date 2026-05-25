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

### Presentation Note: 0.5B Ablation Findings

- Ran the same ablation setup with `Qwen/Qwen2.5-0.5B` feature tables.
- Same-dataset results still favor using all uncertainty features:

```text
HaluEval 0.5B:   all features + Random Forest, F1 0.9745, ROC-AUC 0.9938
TruthfulQA 0.5B: all features + Random Forest, F1 0.6981, ROC-AUC 0.6898
```

- Cross-dataset transfer still remains weak, but entropy-only features transferred slightly better:

```text
HaluEval -> TruthfulQA 0.5B: entropy only + Random Forest, F1 0.6557, ROC-AUC 0.4527
```

- Presentation wording:

```text
The ablation results show that all features are best for same-dataset evaluation, while entropy-only features are slightly more robust when transferring from HaluEval to TruthfulQA. However, transfer performance is still weak, so dataset shift remains the main limitation.
```

## May 22, 2026

Today we completed the model-size comparison, grouped cross-validation, 0.5B ablation, and visualization stage.

### Qwen2.5-0.5B Feature Tables

- Confirmed that uncertainty and entropy feature extraction was completed with the smaller `Qwen/Qwen2.5-0.5B` model.
- Added feature tables:
  - `data/processed/halueval_uncertainty_entropy_qwen05b_features.csv`
  - `data/processed/truthfulqa_uncertainty_entropy_qwen05b_features.csv`
- This allows comparison between the main `Qwen/Qwen2.5-3B` feature extractor and a much smaller model.

### Qwen2.5-0.5B Evaluation

- Added:
  - `scripts/evaluate_qwen05b.sh`
- Ran grouped 80/20 evaluation for:
  - HaluEval with 0.5B features
  - TruthfulQA with 0.5B features
  - HaluEval-to-TruthfulQA transfer with 0.5B features
- Generated:
  - `outputs/tables/halueval_qwen05b_classifier_metrics.csv`
  - `outputs/tables/truthfulqa_qwen05b_classifier_metrics.csv`
  - `outputs/tables/halueval_to_truthfulqa_qwen05b_classifier_metrics.csv`

Main 0.5B results:

```text
HaluEval 0.5B:              Random Forest, F1 0.9745, ROC-AUC 0.9938
TruthfulQA 0.5B:            Random Forest, F1 0.6981, ROC-AUC 0.6898
HaluEval -> TruthfulQA 0.5B: Random Forest, F1 0.6388, ROC-AUC 0.4000
```

### Grouped Cross-Validation

- Added:
  - `src/evaluation/cross_validation.py`
  - `scripts/run_cross_validation.sh`
- Ran grouped 5-fold cross-validation for:
  - Qwen2.5-3B HaluEval
  - Qwen2.5-3B TruthfulQA
  - Qwen2.5-0.5B HaluEval
  - Qwen2.5-0.5B TruthfulQA
- Grouped CV keeps all answers from the same original question in the same fold, which prevents train-test leakage.

Main grouped CV findings:

```text
Qwen2.5-3B   HaluEval:   best ROC-AUC 0.9986, F1 about 0.9871
Qwen2.5-0.5B HaluEval:   best ROC-AUC 0.9945, F1 about 0.9746
Qwen2.5-3B   TruthfulQA: best ROC-AUC 0.6836, F1 about 0.6968
Qwen2.5-0.5B TruthfulQA: best ROC-AUC 0.6841, F1 about 0.7006
```

Interpretation:

- Cross-validation confirms the earlier 80/20 result pattern.
- HaluEval remains much easier than TruthfulQA.
- The 3B extractor is better on HaluEval.
- On TruthfulQA, 3B and 0.5B are very similar, so the main issue is dataset structure and lack of context rather than only model size.

### Qwen2.5-0.5B Ablation

- Added:
  - `scripts/run_ablation_qwen05b.sh`
- Ran the same ablation feature groups for the 0.5B feature tables:
  - confidence/logprob
  - entropy
  - all features
- Generated:
  - `outputs/tables/halueval_qwen05b_ablation_results.csv`
  - `outputs/tables/truthfulqa_qwen05b_ablation_results.csv`
  - `outputs/tables/halueval_to_truthfulqa_qwen05b_ablation_results.csv`

Main 0.5B ablation findings:

```text
HaluEval 0.5B:   all features + Random Forest, F1 0.9745, ROC-AUC 0.9938
TruthfulQA 0.5B: all features + Random Forest, F1 0.6981, ROC-AUC 0.6898
Transfer 0.5B:   entropy only + Random Forest, F1 0.6557, ROC-AUC 0.4527
```

Conclusion:

- Same-dataset evaluation still favors all features.
- Entropy-only features transfer slightly better from HaluEval to TruthfulQA.
- Transfer performance remains weak overall, so dataset shift is still the main limitation.

### Visualizations

- Added:
  - `src/evaluation/visualize_results.py`
  - `scripts/create_figures.sh`
- Generated report/presentation figures:
  - `outputs/figures/cv_model_size_comparison.png`
  - `outputs/figures/ablation_feature_group_comparison.png`
  - `outputs/figures/confusion_matrices.png`
  - `outputs/figures/roc_curves.png`
  - `outputs/figures/feature_distribution_shift.png`
- Updated `.gitignore` to ignore local Matplotlib cache folders.

### Current Status

- The experimental pipeline is now largely complete.
- Completed:
  - preprocessing
  - uncertainty and entropy feature extraction
  - 3B and 0.5B feature extraction
  - grouped 80/20 evaluation
  - grouped 5-fold cross-validation
  - external HaluEval-to-TruthfulQA evaluation
  - ablation
  - visualizations
- Remaining work:
  - final report write-up
  - final presentation preparation
  - limitations and future-work discussion

## May 25, 2026

Today we added and evaluated a no-context version of HaluEval to directly measure how much the context passage helps the uncertainty-based detector.

### HaluEval No-Context Dataset

- Added:
  - `src/data/create_no_context_dataset.py`
- Created:
  - `data/processed/halueval_no_context.jsonl`
- This file keeps the same prompts, answers, labels, and IDs as HaluEval, but clears the `context` field.
- The purpose is to compare:
  - HaluEval with context
  - HaluEval without context
  - TruthfulQA without context

### HaluEval No-Context Feature Tables

- Confirmed no-context feature extraction was completed for both feature extractors:
  - `data/processed/halueval_no_context_uncertainty_entropy_features.csv`
  - `data/processed/halueval_no_context_uncertainty_entropy_qwen05b_features.csv`

### HaluEval No-Context Evaluation

- Added:
  - `scripts/evaluate_halueval_no_context.sh`
  - `scripts/run_cross_validation_no_context.sh`
- Ran grouped 80/20 evaluation, HaluEval-to-TruthfulQA transfer, and grouped 5-fold cross-validation.

Main results:

```text
HaluEval no-context 3B 80/20:   Random Forest, F1 0.9276, ROC-AUC 0.9699
HaluEval no-context 0.5B 80/20: Random Forest, F1 0.9283, ROC-AUC 0.9681
HaluEval no-context 3B CV:      Random Forest, F1 0.9277, ROC-AUC 0.9711
HaluEval no-context 0.5B CV:    Random Forest, F1 0.9271, ROC-AUC 0.9701
```

Transfer results:

```text
HaluEval no-context 3B -> TruthfulQA:   Random Forest, F1 0.5927, ROC-AUC 0.4710
HaluEval no-context 0.5B -> TruthfulQA: Random Forest, F1 0.6024, ROC-AUC 0.4890
```

### Context Effect Finding

- With context, HaluEval achieved about 0.987--0.989 F1.
- Without context, HaluEval dropped to about 0.928 F1.
- This confirms that context helps, but the drop is not as large as expected.

Interpretation:

- HaluEval remains relatively easy even without context.
- Possible reasons:
  - HaluEval supported and hallucinated answers are paired and often stylistically different.
  - Many questions contain useful clues even without the context passage.
  - Some hallucinated answers are longer, vague, generic, or less likely under the language model.
  - The classifier may learn HaluEval-specific uncertainty and answer-style patterns, not only factual grounding.

### TruthfulQA Difficulty Explanation

- TruthfulQA still performs much worse because:
  - it has no supporting context
  - it is designed around misconceptions and tricky questions
  - false answers may be common myths and therefore look likely to the language model
  - correct answers may reject common myths and therefore look less familiar
  - TruthfulQA has a different structure from HaluEval, with multiple correct and incorrect answers per question

Classroom explanation:

```text
TruthfulQA is harder because it tests open-domain truthfulness without supporting context, and many false answers are common misconceptions that can still look likely to the language model. At the same time, correct answers often sound less common because they reject those misconceptions. Therefore, token confidence and entropy are less separable than in HaluEval.
```

### Updated Conclusion

- The project should clearly distinguish:
  - context-grounded hallucination detection
  - no-context truthfulness detection
- The method works best in context-grounded HaluEval.
- HaluEval no-context shows that context matters, but dataset-specific patterns also make HaluEval easier than TruthfulQA.
- TruthfulQA remains the stronger test of open-domain truthfulness generalization.

### Presentation Note: Why TruthfulQA Is Tricky

- The original TruthfulQA paper shows that TruthfulQA is intentionally difficult because it targets common misconceptions, myths, stereotypes, and conspiracy-like false beliefs.
- The benchmark is not normal factual QA. It is designed to test whether models imitate falsehoods that are common in human text.
- Important paper findings:

```text
Best main model: GPT-3 175B with helpful prompt, about 58% truthful
Human baseline: about 94% truthful
Default GPT-3 scaling: larger models often became more informative but less truthful
```

- This supports our lower TruthfulQA scores.
- In TruthfulQA, false answers can be familiar and likely under the language model.
- Correct answers often reject common myths, so they may sound less familiar and receive lower confidence.
- Therefore, token confidence and entropy separate labels less cleanly than in HaluEval.

Presentation wording:

```text
TruthfulQA is difficult because it was designed to trigger imitative falsehoods. A false answer can be a common myth that the model has seen many times, so it may receive high confidence. A correct answer may reject the myth and sound less familiar. This explains why uncertainty-only features perform much worse on TruthfulQA than on HaluEval.
```
