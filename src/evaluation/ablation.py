"""Run ablation experiments over uncertainty feature groups."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluate_uncertainty_classifiers import (
    build_models,
    get_positive_scores,
    split_train_test,
    validate_columns,
)
from src.evaluation.metrics import compute_metrics


DEFAULT_HALUEVAL_FEATURES = Path("data/processed/halueval_uncertainty_entropy_features.csv")
DEFAULT_TRUTHFULQA_FEATURES = Path(
    "data/processed/truthfulqa_uncertainty_entropy_features.csv"
)
DEFAULT_HALUEVAL_OUTPUT = Path("outputs/tables/halueval_ablation_results.csv")
DEFAULT_TRUTHFULQA_OUTPUT = Path("outputs/tables/truthfulqa_ablation_results.csv")
DEFAULT_EXTERNAL_OUTPUT = Path("outputs/tables/halueval_to_truthfulqa_ablation_results.csv")

ANSWER_LENGTH_FEATURES = ["answer_length_tokens"]
CONFIDENCE_LOGPROB_FEATURES = [
    "mean_token_probability",
    "min_token_probability",
    "max_token_probability",
    "mean_token_logprob",
    "min_token_logprob",
    "max_token_logprob",
    "sum_token_logprob",
    "negative_mean_logprob",
    "low_confidence_token_ratio",
]
ENTROPY_FEATURES = [
    "mean_token_entropy",
    "min_token_entropy",
    "max_token_entropy",
    "sum_token_entropy",
    "high_entropy_token_ratio",
]
FEATURE_GROUPS = {
    "confidence_logprob": ANSWER_LENGTH_FEATURES + CONFIDENCE_LOGPROB_FEATURES,
    "entropy": ANSWER_LENGTH_FEATURES + ENTROPY_FEATURES,
    "all_features": (
        ANSWER_LENGTH_FEATURES + CONFIDENCE_LOGPROB_FEATURES + ENTROPY_FEATURES
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation studies for uncertainty feature groups."
    )
    parser.add_argument(
        "--halueval-features",
        type=Path,
        default=DEFAULT_HALUEVAL_FEATURES,
    )
    parser.add_argument(
        "--truthfulqa-features",
        type=Path,
        default=DEFAULT_TRUTHFULQA_FEATURES,
    )
    parser.add_argument(
        "--halueval-output",
        type=Path,
        default=DEFAULT_HALUEVAL_OUTPUT,
    )
    parser.add_argument(
        "--truthfulqa-output",
        type=Path,
        default=DEFAULT_TRUTHFULQA_OUTPUT,
    )
    parser.add_argument(
        "--external-output",
        type=Path,
        default=DEFAULT_EXTERNAL_OUTPUT,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of groups used for test splits.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splits and classifiers.",
    )
    return parser.parse_args(argv)


def validate_feature_groups(data: pd.DataFrame) -> None:
    """Check that all ablation feature columns exist."""

    required_columns = sorted(
        {"id", "label", *[feature for group in FEATURE_GROUPS.values() for feature in group]}
    )
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required ablation columns: {missing}")


def evaluate_feature_group(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list[str],
    feature_group: str,
    evaluation: str,
    random_state: int,
) -> list[dict[str, Any]]:
    """Train all models for one feature group and return metric rows."""

    x_train = train_data[features]
    y_train = train_data["label"].astype(int)
    x_test = test_data[features]
    y_test = test_data["label"].astype(int)
    rows = []

    for model_name, model in build_models(random_state=random_state).items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_score = get_positive_scores(model, x_test)
        metrics = compute_metrics(y_test, y_pred, y_score=y_score)
        rows.append(
            {
                "evaluation": evaluation,
                "feature_group": feature_group,
                "feature_count": len(features),
                "model": model_name,
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                **metrics,
            }
        )

    return rows


def run_ablation_for_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    evaluation: str,
    random_state: int,
) -> pd.DataFrame:
    """Run all feature-group ablations for a train/test pair."""

    rows = []
    for feature_group, features in FEATURE_GROUPS.items():
        rows.extend(
            evaluate_feature_group(
                train_data=train_data,
                test_data=test_data,
                features=features,
                feature_group=feature_group,
                evaluation=evaluation,
                random_state=random_state,
            )
        )
    return pd.DataFrame(rows)


def write_results(results: pd.DataFrame, output_path: Path) -> None:
    """Write ablation results to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Wrote {len(results)} rows to {output_path}")


def run_ablation(args: argparse.Namespace) -> None:
    """Run HaluEval, TruthfulQA, and external ablation experiments."""

    halueval_data = pd.read_csv(args.halueval_features)
    truthfulqa_data = pd.read_csv(args.truthfulqa_features)
    validate_columns(halueval_data)
    validate_columns(truthfulqa_data)
    validate_feature_groups(halueval_data)
    validate_feature_groups(truthfulqa_data)

    halueval_train, halueval_test = split_train_test(
        halueval_data,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    truthfulqa_train, truthfulqa_test = split_train_test(
        truthfulqa_data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    halueval_results = run_ablation_for_split(
        train_data=halueval_train,
        test_data=halueval_test,
        evaluation="halueval_grouped_80_20",
        random_state=args.random_state,
    )
    truthfulqa_results = run_ablation_for_split(
        train_data=truthfulqa_train,
        test_data=truthfulqa_test,
        evaluation="truthfulqa_grouped_80_20",
        random_state=args.random_state,
    )
    external_results = run_ablation_for_split(
        train_data=halueval_data,
        test_data=truthfulqa_data,
        evaluation="halueval_to_truthfulqa",
        random_state=args.random_state,
    )

    write_results(halueval_results, args.halueval_output)
    write_results(truthfulqa_results, args.truthfulqa_output)
    write_results(external_results, args.external_output)

    print("\nHaluEval ablation:")
    print(halueval_results.to_string(index=False))
    print("\nTruthfulQA ablation:")
    print(truthfulqa_results.to_string(index=False))
    print("\nHaluEval -> TruthfulQA ablation:")
    print(external_results.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_ablation(args)


if __name__ == "__main__":
    main(sys.argv[1:])
