"""Feature distribution and error analysis utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluate_uncertainty_classifiers import FEATURE_COLUMNS, get_group_id


DEFAULT_HALUEVAL_FEATURES = Path("data/processed/halueval_uncertainty_entropy_features.csv")
DEFAULT_TRUTHFULQA_FEATURES = Path(
    "data/processed/truthfulqa_uncertainty_entropy_features.csv"
)
DEFAULT_TRUTHFULQA_PREDICTIONS = Path(
    "outputs/predictions/truthfulqa_classifier_predictions.csv"
)
DEFAULT_TRUTHFULQA_PROCESSED = Path("data/processed/truthfulqa.jsonl")
DEFAULT_DISTRIBUTION_OUTPUT = Path(
    "outputs/tables/feature_distribution_by_dataset.csv"
)
DEFAULT_SHIFT_OUTPUT = Path("outputs/tables/feature_distribution_shift.csv")
DEFAULT_ERROR_SUMMARY_OUTPUT = Path("outputs/tables/truthfulqa_error_summary.csv")
DEFAULT_ERROR_EXAMPLES_OUTPUT = Path(
    "outputs/predictions/truthfulqa_error_examples.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare feature distributions and inspect TruthfulQA errors."
    )
    parser.add_argument("--halueval-features", type=Path, default=DEFAULT_HALUEVAL_FEATURES)
    parser.add_argument(
        "--truthfulqa-features",
        type=Path,
        default=DEFAULT_TRUTHFULQA_FEATURES,
    )
    parser.add_argument(
        "--truthfulqa-predictions",
        type=Path,
        default=DEFAULT_TRUTHFULQA_PREDICTIONS,
    )
    parser.add_argument(
        "--truthfulqa-processed",
        type=Path,
        default=DEFAULT_TRUTHFULQA_PROCESSED,
    )
    parser.add_argument(
        "--distribution-output",
        type=Path,
        default=DEFAULT_DISTRIBUTION_OUTPUT,
    )
    parser.add_argument("--shift-output", type=Path, default=DEFAULT_SHIFT_OUTPUT)
    parser.add_argument(
        "--error-summary-output",
        type=Path,
        default=DEFAULT_ERROR_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--error-examples-output",
        type=Path,
        default=DEFAULT_ERROR_EXAMPLES_OUTPUT,
    )
    parser.add_argument(
        "--examples-per-error-type",
        type=int,
        default=8,
        help="Number of confident false positives/negatives to save per model.",
    )
    return parser.parse_args(argv)


def ensure_parent(path: Path) -> None:
    """Create output parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)


def load_processed_jsonl(path: Path) -> pd.DataFrame:
    """Load processed JSONL records with prompt and answer text."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                metadata = record.get("metadata") or {}
                records.append(
                    {
                        "id": record.get("id"),
                        "prompt": record.get("prompt"),
                        "answer": record.get("answer"),
                        "category": metadata.get("category"),
                    }
                )
    return pd.DataFrame(records)


def summarize_feature_distributions(
    halueval_features: pd.DataFrame,
    truthfulqa_features: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize feature distributions by dataset and label."""

    combined = pd.concat([halueval_features, truthfulqa_features], ignore_index=True)
    rows = []

    for (dataset, label), group in combined.groupby(["dataset", "label"]):
        for feature in FEATURE_COLUMNS:
            rows.append(
                {
                    "dataset": dataset,
                    "label": int(label),
                    "feature": feature,
                    "mean": group[feature].mean(),
                    "std": group[feature].std(),
                    "median": group[feature].median(),
                    "min": group[feature].min(),
                    "max": group[feature].max(),
                }
            )

    return pd.DataFrame(rows)


def compare_dataset_shift(
    halueval_features: pd.DataFrame,
    truthfulqa_features: pd.DataFrame,
) -> pd.DataFrame:
    """Compare overall feature distributions between HaluEval and TruthfulQA."""

    rows = []
    for feature in FEATURE_COLUMNS:
        halueval_mean = halueval_features[feature].mean()
        truthfulqa_mean = truthfulqa_features[feature].mean()
        halueval_std = halueval_features[feature].std()
        truthfulqa_std = truthfulqa_features[feature].std()
        pooled_std = ((halueval_std**2 + truthfulqa_std**2) / 2) ** 0.5
        standardized_difference = (
            (truthfulqa_mean - halueval_mean) / pooled_std if pooled_std else 0.0
        )

        rows.append(
            {
                "feature": feature,
                "halueval_mean": halueval_mean,
                "truthfulqa_mean": truthfulqa_mean,
                "mean_difference": truthfulqa_mean - halueval_mean,
                "halueval_std": halueval_std,
                "truthfulqa_std": truthfulqa_std,
                "standardized_difference": standardized_difference,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="standardized_difference",
        key=lambda column: column.abs(),
        ascending=False,
    )


def label_error_type(row: pd.Series) -> str:
    """Label prediction outcome."""

    if row["label"] == row["prediction"]:
        return "correct"
    if row["label"] == 0 and row["prediction"] == 1:
        return "false_positive"
    return "false_negative"


def summarize_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Count error types by model."""

    predictions = predictions.copy()
    predictions["error_type"] = predictions.apply(label_error_type, axis=1)
    summary = (
        predictions.groupby(["model", "error_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "error_type"])
    )
    totals = predictions.groupby("model").size().rename("total")
    summary = summary.merge(totals, on="model")
    summary["rate"] = summary["count"] / summary["total"]
    return summary


def select_error_examples(
    predictions: pd.DataFrame,
    processed_examples: pd.DataFrame,
    examples_per_error_type: int,
) -> pd.DataFrame:
    """Select confident false positives and false negatives with text."""

    predictions = predictions.copy()
    predictions["error_type"] = predictions.apply(label_error_type, axis=1)
    errors = predictions[predictions["error_type"] != "correct"].copy()
    examples = []

    for (model, error_type), group in errors.groupby(["model", "error_type"]):
        if error_type == "false_positive":
            selected = group.sort_values("score", ascending=False).head(
                examples_per_error_type
            )
        else:
            selected = group.sort_values("score", ascending=True).head(
                examples_per_error_type
            )
        examples.append(selected)

    if not examples:
        return pd.DataFrame()

    selected_errors = pd.concat(examples, ignore_index=True)
    selected_errors["group_id"] = selected_errors["id"].astype(str).map(get_group_id)
    return selected_errors.merge(processed_examples, on="id", how="left")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    halueval_features = pd.read_csv(args.halueval_features)
    truthfulqa_features = pd.read_csv(args.truthfulqa_features)
    truthfulqa_predictions = pd.read_csv(args.truthfulqa_predictions)
    truthfulqa_processed = load_processed_jsonl(args.truthfulqa_processed)

    distributions = summarize_feature_distributions(
        halueval_features=halueval_features,
        truthfulqa_features=truthfulqa_features,
    )
    shift = compare_dataset_shift(
        halueval_features=halueval_features,
        truthfulqa_features=truthfulqa_features,
    )
    error_summary = summarize_errors(truthfulqa_predictions)
    error_examples = select_error_examples(
        predictions=truthfulqa_predictions,
        processed_examples=truthfulqa_processed,
        examples_per_error_type=args.examples_per_error_type,
    )

    for output_path, output_data in [
        (args.distribution_output, distributions),
        (args.shift_output, shift),
        (args.error_summary_output, error_summary),
        (args.error_examples_output, error_examples),
    ]:
        ensure_parent(output_path)
        output_data.to_csv(output_path, index=False)

    print(f"Wrote feature distributions to {args.distribution_output}")
    print(f"Wrote feature shift comparison to {args.shift_output}")
    print(f"Wrote TruthfulQA error summary to {args.error_summary_output}")
    print(f"Wrote TruthfulQA error examples to {args.error_examples_output}")
    print("\nLargest feature shifts:")
    print(shift.head(8).to_string(index=False))
    print("\nTruthfulQA error summary:")
    print(error_summary.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
