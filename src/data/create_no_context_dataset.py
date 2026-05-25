"""Create a copy of a processed JSONL dataset with empty context fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/processed/halueval.jsonl")
DEFAULT_OUTPUT = Path("data/processed/halueval_no_context.jsonl")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a no-context version of a processed JSONL dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input processed JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL file with context cleared.",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_context(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    no_context_records = []
    for record in records:
        updated = dict(record)
        updated["context"] = ""
        metadata = dict(updated.get("metadata", {}))
        metadata["context_removed"] = True
        updated["metadata"] = metadata
        no_context_records.append(updated)
    return no_context_records


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = read_jsonl(args.input)
    no_context_records = clear_context(records)
    write_jsonl(no_context_records, args.output)
    print(f"Wrote {len(no_context_records)} no-context records to {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
