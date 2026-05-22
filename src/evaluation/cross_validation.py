"""Grouped cross-validation for uncertainty-feature classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from src.evaluation.evaluate_uncertainty_classifiers import (
    FEATURE_COLUMNS,
    evaluate_models,
    get_group_id,
    validate_columns,
)


DEFAULT_INPUT = Path("data/processed/halueval_uncertainty_entropy_features.csv")
DEFAULT_FOLD_METRICS_OUTPUT = Path("outputs/tables/halueval_cv_fold_metrics.csv")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/tables/halueval_cv_summary.csv")
SUMMARY_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped k-fold cross-validation on uncertainty features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Feature CSV used for grouped cross-validation.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of grouped cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for classifiers.",
    )
    parser.add_argument(
        "--fold-metrics-output",
        type=Path,
        default=DEFAULT_FOLD_METRICS_OUTPUT,
        help="Path for per-fold metrics CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path for mean/std cross-validation summary CSV.",
    )
    return parser.parse_args(argv)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation for each model."""

    summary = (
        fold_metrics.groupby("model")[SUMMARY_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(column).strip("_")
        for column in summary.columns
    ]
    return summary


def run_grouped_cross_validation(
    data: pd.DataFrame,
    folds: int,
    random_state: int,
) -> pd.DataFrame:
    """Run grouped k-fold CV and return per-fold metrics."""

    groups = data["id"].astype(str).map(get_group_id)
    unique_groups = groups.nunique()
    if folds > unique_groups:
        raise ValueError(
            f"Requested {folds} folds, but only {unique_groups} groups are available."
        )

    splitter = GroupKFold(n_splits=folds)
    fold_metric_frames = []

    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(data[FEATURE_COLUMNS], data["label"], groups=groups),
        start=1,
    ):
        train_data = data.iloc[train_index].copy()
        test_data = data.iloc[test_index].copy()
        metrics, _ = evaluate_models(
            train_data=train_data,
            test_data=test_data,
            random_state=random_state,
            evaluation_name=f"grouped_{folds}_fold_cv",
        )
        metrics.insert(1, "fold", fold_index)
        fold_metric_frames.append(metrics)
        print(f"Completed fold {fold_index}/{folds}")

    return pd.concat(fold_metric_frames, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data = pd.read_csv(args.input)
    validate_columns(data)

    fold_metrics = run_grouped_cross_validation(
        data=data,
        folds=args.folds,
        random_state=args.random_state,
    )
    summary = summarize_fold_metrics(fold_metrics)

    args.fold_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(args.fold_metrics_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    print(f"Loaded {len(data)} rows from {args.input}")
    print(summary.to_string(index=False))
    print(f"Wrote fold metrics to {args.fold_metrics_output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main(sys.argv[1:])
