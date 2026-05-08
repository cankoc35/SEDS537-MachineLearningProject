"""Preprocess raw datasets into a shared binary hallucination schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            file.seek(0)
            return [json.loads(line) for line in file if line.strip()]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    _ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_as_text(item) for item in value if _as_text(item)).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _base_record(
    *,
    record_id: str,
    dataset: str,
    task: str,
    prompt: Any,
    context: Any,
    answer: Any,
    label: int,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "dataset": dataset,
        "task": task,
        "prompt": _as_text(prompt),
        "context": _as_text(context),
        "answer": _as_text(answer),
        "label": label,
        "source": source,
        "metadata": metadata or {},
    }


def preprocess_example(example: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single already-labeled example."""

    required = {"id", "dataset", "task", "prompt", "answer", "label", "source"}
    missing = required.difference(example)
    if missing:
        raise ValueError(f"Example missing required fields: {sorted(missing)}")

    return _base_record(
        record_id=example["id"],
        dataset=example["dataset"],
        task=example["task"],
        prompt=example["prompt"],
        context=example.get("context", ""),
        answer=example["answer"],
        label=int(example["label"]),
        source=example["source"],
        metadata=example.get("metadata", {}),
    )


def preprocess_halueval_task(task: str, raw_path: Path) -> list[dict[str, Any]]:
    """Convert one HaluEval task file into binary classification rows."""

    raw_records = _read_json(raw_path)
    processed: list[dict[str, Any]] = []

    for index, item in enumerate(raw_records):
        base_id = f"halueval_{task}_{index:06d}"

        if task == "qa":
            prompt = item.get("question", "")
            context = item.get("knowledge", "")
            supported_answer = item.get("right_answer", "")
            hallucinated_answer = item.get("hallucinated_answer", "")
            supported_source = "right_answer"
            hallucinated_source = "hallucinated_answer"
        elif task == "dialogue":
            prompt = item.get("dialogue_history", "")
            context = item.get("knowledge", "")
            supported_answer = item.get("right_response", "")
            hallucinated_answer = item.get("hallucinated_response", "")
            supported_source = "right_response"
            hallucinated_source = "hallucinated_response"
        elif task == "summarization":
            prompt = "Summarize the document."
            context = item.get("document", "")
            supported_answer = item.get("right_summary", "")
            hallucinated_answer = item.get("hallucinated_summary", "")
            supported_source = "right_summary"
            hallucinated_source = "hallucinated_summary"
        elif task == "general":
            label_text = _as_text(item.get("hallucination_label", "")).lower()
            label = 1 if label_text in {"yes", "true", "1", "hallucinated"} else 0
            processed.append(
                _base_record(
                    record_id=f"{base_id}_chatgpt_response",
                    dataset="halueval",
                    task="general",
                    prompt=item.get("user_query", ""),
                    context="",
                    answer=item.get("chatgpt_response", ""),
                    label=label,
                    source="chatgpt_response",
                    metadata={"raw_index": index, "hallucination_label": item.get("hallucination_label")},
                )
            )
            continue
        else:
            raise ValueError(f"Unsupported HaluEval task: {task}")

        metadata = {"raw_index": index}
        processed.append(
            _base_record(
                record_id=f"{base_id}_supported",
                dataset="halueval",
                task=task,
                prompt=prompt,
                context=context,
                answer=supported_answer,
                label=0,
                source=supported_source,
                metadata=metadata,
            )
        )
        processed.append(
            _base_record(
                record_id=f"{base_id}_hallucinated",
                dataset="halueval",
                task=task,
                prompt=prompt,
                context=context,
                answer=hallucinated_answer,
                label=1,
                source=hallucinated_source,
                metadata=metadata,
            )
        )

    return processed


def preprocess_halueval(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    tasks: Iterable[str] | None = None,
) -> Path:
    """Preprocess available HaluEval raw files into data/processed/halueval.jsonl."""

    halueval_dir = raw_dir / "halueval"
    selected_tasks = list(tasks) if tasks else [
        path.name.removesuffix("_data.json") for path in sorted(halueval_dir.glob("*_data.json"))
    ]

    if not selected_tasks:
        raise FileNotFoundError(
            f"No HaluEval raw files found in {halueval_dir}. "
            "Run: python -m src.data.load_data --dataset halueval"
        )

    records: list[dict[str, Any]] = []
    for task in selected_tasks:
        raw_path = halueval_dir / f"{task}_data.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing HaluEval task file: {raw_path}")
        records.extend(preprocess_halueval_task(task, raw_path))

    output_path = processed_dir / "halueval.jsonl"
    count = _write_jsonl(records, output_path)
    print(f"Wrote {count} records to {output_path}")
    return output_path


def preprocess_truthfulqa(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    """Preprocess TruthfulQA generation data into binary classification rows."""

    raw_path = raw_dir / "truthfulqa" / "truthfulqa_generation.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing TruthfulQA raw file: {raw_path}. "
            "Run: python -m src.data.load_data --dataset truthfulqa"
        )

    records: list[dict[str, Any]] = []
    raw_records = _read_jsonl(raw_path)

    for index, item in enumerate(raw_records):
        question = item.get("question", "")
        category = item.get("category", "")
        split = item.get("_split", "")
        raw_index = item.get("_index", index)
        base_id = f"truthfulqa_{index:06d}"
        metadata = {"raw_index": raw_index, "split": split, "category": category}

        correct_answers = item.get("correct_answers") or []
        incorrect_answers = item.get("incorrect_answers") or []

        best_answer = item.get("best_answer")
        if best_answer and best_answer not in correct_answers:
            correct_answers = [best_answer, *correct_answers]

        for answer_index, answer in enumerate(correct_answers):
            records.append(
                _base_record(
                    record_id=f"{base_id}_correct_{answer_index}",
                    dataset="truthfulqa",
                    task="qa",
                    prompt=question,
                    context="",
                    answer=answer,
                    label=0,
                    source="correct_answer",
                    metadata=metadata,
                )
            )

        for answer_index, answer in enumerate(incorrect_answers):
            records.append(
                _base_record(
                    record_id=f"{base_id}_incorrect_{answer_index}",
                    dataset="truthfulqa",
                    task="qa",
                    prompt=question,
                    context="",
                    answer=answer,
                    label=1,
                    source="incorrect_answer",
                    metadata=metadata,
                )
            )

    output_path = processed_dir / "truthfulqa.jsonl"
    count = _write_jsonl(records, output_path)
    print(f"Wrote {count} records to {output_path}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess project datasets.")
    parser.add_argument(
        "--dataset",
        choices=["truthfulqa", "halueval", "all"],
        required=True,
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--halueval-tasks",
        nargs="+",
        default=None,
        help="Optional HaluEval tasks to preprocess, e.g. qa dialogue summarization general.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.dataset in {"halueval", "all"}:
        preprocess_halueval(tasks=args.halueval_tasks)

    if args.dataset in {"truthfulqa", "all"}:
        preprocess_truthfulqa()


if __name__ == "__main__":
    main(sys.argv[1:])
