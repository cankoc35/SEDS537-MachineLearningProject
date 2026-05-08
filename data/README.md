# Data

This directory stores datasets used in the hallucination detection project.

- `raw/`: downloaded datasets in their original form
- `processed/`: cleaned or transformed data ready for experiments

Recommended datasets:
- `TruthfulQA`
- `HaluEval`

Keep large dataset files out of Git unless they are small enough to version safely.

## Preparing the Data

Install dependencies first:

```bash
pip install -r requirements.txt
```

Download and preprocess the default setup:

```bash
./scripts/prepare_data.sh
```

The default setup downloads the HaluEval QA task and TruthfulQA generation data,
then writes normalized JSONL files to `data/processed/`.

You can also run datasets separately:

```bash
./scripts/prepare_data.sh halueval
./scripts/prepare_data.sh truthfulqa
```

The processed files use the shared schema:

```text
id, dataset, task, prompt, context, answer, label, source, metadata
```

where `label = 0` means supported/truthful and `label = 1` means
hallucinated/unsupported.
