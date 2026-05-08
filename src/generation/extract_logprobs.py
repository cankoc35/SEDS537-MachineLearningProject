"""CLI for previewing and scoring answer tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        help="Number of examples to preview.",
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
        "--device",
        default="auto",
        help="Torch device for --score-one: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.show_features and not args.score_one:
        raise SystemExit("--show-features requires --score-one.")

    tokenizer = load_tokenizer(args.model) if args.show_tokens or args.score_one else None
    model = None
    device = None

    if args.score_one:
        model, device = load_model(args.model, device=args.device)

    for index, example in enumerate(iter_jsonl(args.input)):
        if args.score_one and index >= 1:
            break
        if index >= args.limit:
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
                    f"logprob={token_score['logprob']:.6f}"
                )

            if args.show_features:
                features = compute_uncertainty_features(token_scores)
                print("-" * 80)
                print("uncertainty features:")
                for feature_name, feature_value in features.items():
                    print(f"{feature_name}: {feature_value}")


if __name__ == "__main__":
    main(sys.argv[1:])
