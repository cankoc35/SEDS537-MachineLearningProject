"""Download and load raw datasets used by the project."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

HALUEVAL_URLS = {
    "qa": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
    "dialogue": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json",
    "summarization": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json",
    "general": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json",
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, destination: Path, force: bool = False) -> Path:
    _ensure_dir(destination.parent)
    if destination.exists() and not force:
        print(f"Using existing file: {destination}")
        return destination

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved: {destination}")
    return destination


def _read_json_or_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            file.seek(0)
            return [json.loads(line) for line in file if line.strip()]


def download_halueval(
    tasks: Iterable[str] = ("qa",),
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> dict[str, Path]:
    """Download selected HaluEval task files into data/raw/halueval."""

    output_dir = raw_dir / "halueval"
    paths: dict[str, Path] = {}

    for task in tasks:
        if task not in HALUEVAL_URLS:
            valid = ", ".join(sorted(HALUEVAL_URLS))
            raise ValueError(f"Unknown HaluEval task '{task}'. Valid tasks: {valid}")

        destination = output_dir / f"{task}_data.json"
        paths[task] = _download_file(HALUEVAL_URLS[task], destination, force=force)

    return paths


def download_truthfulqa(
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
    config: str = "generation",
) -> Path:
    """Download TruthfulQA from Hugging Face and save a local JSONL copy."""

    output_dir = raw_dir / "truthfulqa"
    output_path = output_dir / f"truthfulqa_{config}.jsonl"
    _ensure_dir(output_dir)

    if output_path.exists() and not force:
        print(f"Using existing file: {output_path}")
        return output_path

    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise ImportError(
            "TruthfulQA loading requires the 'datasets' package. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc

    dataset = hf_load_dataset("truthful_qa", config)

    with output_path.open("w", encoding="utf-8") as file:
        for split_name, split_data in dataset.items():
            for index, row in enumerate(split_data):
                record = dict(row)
                record["_split"] = split_name
                record["_index"] = index
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved: {output_path}")
    return output_path


def load_dataset(name: str, raw_dir: Path = RAW_DATA_DIR):
    """Load a raw local dataset file after it has been downloaded."""

    normalized_name = name.lower()

    if normalized_name == "truthfulqa":
        path = raw_dir / "truthfulqa" / "truthfulqa_generation.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing TruthfulQA raw file: {path}. "
                "Run: python -m src.data.load_data --dataset truthfulqa"
            )
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    if normalized_name == "halueval":
        directory = raw_dir / "halueval"
        if not directory.exists():
            raise FileNotFoundError(
                f"Missing HaluEval raw directory: {directory}. "
                "Run: python -m src.data.load_data --dataset halueval"
            )
        data = {}
        for path in sorted(directory.glob("*_data.json")):
            task = path.name.removesuffix("_data.json")
            data[task] = _read_json_or_jsonl(path)
        return data

    raise ValueError("Dataset must be one of: truthfulqa, halueval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw project datasets.")
    parser.add_argument(
        "--dataset",
        choices=["truthfulqa", "halueval", "all"],
        required=True,
        help="Dataset to download.",
    )
    parser.add_argument(
        "--halueval-tasks",
        nargs="+",
        default=["qa"],
        choices=sorted(HALUEVAL_URLS),
        help="HaluEval task files to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if local copies already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.dataset in {"halueval", "all"}:
        download_halueval(tasks=args.halueval_tasks, force=args.force)

    if args.dataset in {"truthfulqa", "all"}:
        download_truthfulqa(force=args.force)


if __name__ == "__main__":
    main(sys.argv[1:])
