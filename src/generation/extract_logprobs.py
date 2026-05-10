"""CLI for previewing and scoring answer tokens."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from src.features.uncertainty_features import compute_uncertainty_features
from src.generation.model_scoring import (
    load_model,
    load_tokenizer,
    score_answer_tokens,
    tokenize_for_scoring,
)
from src.generation.prompting import build_preview_text
from src.utils.jsonl import iter_jsonl


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B"
FEATURE_COLUMNS = [
    "answer_length_tokens",
    "mean_token_probability",
    "min_token_probability",
    "max_token_probability",
    "mean_token_logprob",
    "min_token_logprob",
    "max_token_logprob",
    "sum_token_logprob",
    "negative_mean_logprob",
    "low_confidence_token_ratio",
    "mean_token_entropy",
    "min_token_entropy",
    "max_token_entropy",
    "sum_token_entropy",
    "high_entropy_token_ratio",
]
METADATA_COLUMNS = ["id", "dataset", "task", "label", "source", "model", "device"]


def extract_logprobs(generation_output):
    """Return token-level confidence information."""
    raise NotImplementedError("Implement logprob extraction for the selected model stack.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview processed examples before logprob extraction."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a processed JSONL file, e.g. data/processed/halueval.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of examples to process. Use 0 to process all examples.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model/tokenizer name.",
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="Load tokenizer and print token counts plus answer tokens.",
    )
    parser.add_argument(
        "--score-one",
        action="store_true",
        help="Load model and print probability/logprob for one example's answer tokens.",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Print response-level uncertainty features. Requires --score-one.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write batch uncertainty features to a .csv or .jsonl file.",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.1,
        help="Probability threshold used for low_confidence_token_ratio.",
    )
    parser.add_argument(
        "--high-entropy-threshold",
        type=float,
        default=2.0,
        help="Entropy threshold used for high_entropy_token_ratio.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print batch progress every N examples when --output is used.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for --score-one: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args(argv)


def should_stop(index: int, limit: int) -> bool:
    """Return True when the requested limit has been reached."""

    return limit > 0 and index >= limit


def build_feature_row(
    example: dict[str, Any],
    token_scores: list[dict[str, Any]],
    model_name: str,
    device: str,
    low_confidence_threshold: float,
    high_entropy_threshold: float,
) -> dict[str, Any]:
    """Create one response-level feature row for model training/evaluation."""

    features = compute_uncertainty_features(
        token_scores,
        low_confidence_threshold=low_confidence_threshold,
        high_entropy_threshold=high_entropy_threshold,
    )
    return {
        "id": example.get("id"),
        "dataset": example.get("dataset"),
        "task": example.get("task"),
        "label": example.get("label"),
        "source": example.get("source"),
        "model": model_name,
        "device": device,
        **features,
    }


def write_feature_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write feature rows as CSV or JSONL based on the output suffix."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        fieldnames = METADATA_COLUMNS + FEATURE_COLUMNS
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return

    if output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    raise ValueError("Output path must end with .csv or .jsonl.")


def write_batch_features(args: argparse.Namespace) -> None:
    """Score examples and write response-level uncertainty features."""

    tokenizer = load_tokenizer(args.model)
    model, device = load_model(args.model, device=args.device)
    rows: list[dict[str, Any]] = []

    for index, example in enumerate(iter_jsonl(args.input)):
        if should_stop(index, args.limit):
            break

        token_scores = score_answer_tokens(example, tokenizer, model, device)
        rows.append(
            build_feature_row(
                example=example,
                token_scores=token_scores,
                model_name=args.model,
                device=device,
                low_confidence_threshold=args.low_confidence_threshold,
                high_entropy_threshold=args.high_entropy_threshold,
            )
        )

        processed = index + 1
        if args.progress_every > 0 and processed % args.progress_every == 0:
            print(f"Processed {processed} examples...")

    write_feature_rows(rows, args.output)
    print(f"Wrote {len(rows)} feature rows to {args.output}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.show_features and not args.score_one:
        raise SystemExit("--show-features requires --score-one.")

    if args.output is not None:
        write_batch_features(args)
        return

    tokenizer = load_tokenizer(args.model) if args.show_tokens or args.score_one else None
    model = None
    device = None

    if args.score_one:
        model, device = load_model(args.model, device=args.device)

    for index, example in enumerate(iter_jsonl(args.input)):
        if args.score_one and index >= 1:
            break
        if should_stop(index, args.limit):
            break

        print("=" * 80)
        print(f"id: {example.get('id')}")
        print(f"label: {example.get('label')}")
        print(f"dataset: {example.get('dataset')}")
        print("-" * 80)
        print(build_preview_text(example))

        if tokenizer is not None:
            tokenized = tokenize_for_scoring(example, tokenizer)
            print("-" * 80)
            print(f"model: {args.model}")
            print(f"prompt_tokens: {len(tokenized['prompt_ids'])}")
            print(f"answer_tokens: {len(tokenized['answer_ids'])}")
            print(f"answer_token_ids: {tokenized['answer_ids']}")
            print(f"answer_tokens_text: {tokenized['answer_tokens']}")

        if args.score_one:
            token_scores = score_answer_tokens(example, tokenizer, model, device)
            print("-" * 80)
            print(f"device: {device}")
            print("answer token scores:")
            for token_score in token_scores:
                print(
                    f"{token_score['token_index']:>3} "
                    f"{token_score['token_text']!r:<18} "
                    f"prob={token_score['probability']:.8f} "
                    f"logprob={token_score['logprob']:.6f} "
                    f"entropy={token_score['entropy']:.6f}"
                )

            if args.show_features:
                features = compute_uncertainty_features(
                    token_scores,
                    low_confidence_threshold=args.low_confidence_threshold,
                    high_entropy_threshold=args.high_entropy_threshold,
                )
                print("-" * 80)
                print("uncertainty features:")
                for feature_name, feature_value in features.items():
                    print(f"{feature_name}: {feature_value}")


if __name__ == "__main__":
    main(sys.argv[1:])
