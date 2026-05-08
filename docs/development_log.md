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
