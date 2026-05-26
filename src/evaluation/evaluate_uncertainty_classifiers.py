"""Train and evaluate classifiers on uncertainty feature tables."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.evaluation.metrics import compute_metrics
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.svm import build_linear_svm, build_rbf_svm


DEFAULT_INPUT = Path("data/processed/halueval_uncertainty_entropy_features.csv")
DEFAULT_METRICS_OUTPUT = Path("outputs/tables/halueval_classifier_metrics.csv")
DEFAULT_PREDICTIONS_OUTPUT = Path("outputs/predictions/halueval_classifier_predictions.csv")
DEFAULT_EXTERNAL_METRICS_OUTPUT = Path(
    "outputs/tables/halueval_to_truthfulqa_classifier_metrics.csv"
)
DEFAULT_EXTERNAL_PREDICTIONS_OUTPUT = Path(
    "outputs/predictions/halueval_to_truthfulqa_classifier_predictions.csv"
)
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination classifiers on uncertainty features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Feature CSV used for grouped train/test evaluation.",
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        help="Feature CSV used for external evaluation training.",
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        help="Feature CSV used for external evaluation testing.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Path for the model metrics CSV.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_OUTPUT,
        help="Path for per-example predictions CSV.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of question groups used for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the grouped 80/20 split and classifiers.",
    )
    return parser.parse_args(argv)


def get_group_id(example_id: str) -> str:
    """Group answers from the same original question."""

    group_id = re.sub(r"_(supported|hallucinated)$", "", example_id)
    group_id = re.sub(r"_(correct|incorrect)_\d+$", "", group_id)
    return group_id


def validate_columns(data: pd.DataFrame) -> None:
    """Fail early if the feature table is missing required columns."""

    required_columns = ["id", "label", *FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in feature table: {missing}")


def split_train_test(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a group-aware train/test split."""

    groups = data["id"].astype(str).map(get_group_id)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(splitter.split(data, data["label"], groups=groups))
    return data.iloc[train_index].copy(), data.iloc[test_index].copy()


def build_models(random_state: int) -> dict[str, Any]:
    """Create the classifiers used in the first experiment."""

    return {
        "Logistic Regression": build_logistic_regression(random_state=random_state),
        "Linear SVM": build_linear_svm(random_state=random_state),
        "RBF SVM": build_rbf_svm(random_state=random_state),
        "Random Forest": build_random_forest(random_state=random_state),
    }


def get_positive_scores(model: Any, features: pd.DataFrame) -> list[float]:
    """Return positive-class scores for ROC-AUC and ranking."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1].tolist()

    if hasattr(model, "decision_function"):
        return model.decision_function(features).tolist()

    return model.predict(features).tolist()


def evaluate_models(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    random_state: int,
    evaluation_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train all classifiers and return metrics plus predictions."""

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["label"].astype(int)
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["label"].astype(int)

    metric_rows = []
    prediction_rows = []

    for model_name, model in build_models(random_state).items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_score = get_positive_scores(model, x_test)
        metrics = compute_metrics(y_test, y_pred, y_score=y_score)

        metric_rows.append(
            {
                "model": model_name,
                "evaluation": evaluation_name,
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                **metrics,
            }
        )

        for row_index, (_, row) in enumerate(test_data.iterrows()):
            prediction_rows.append(
                {
                    "model": model_name,
                    "evaluation": evaluation_name,
                    "id": row["id"],
                    "dataset": row.get("dataset"),
                    "source": row.get("source"),
                    "label": int(row["label"]),
                    "prediction": int(y_pred[row_index]),
                    "score": float(y_score[row_index]),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if (args.train_input is None) != (args.test_input is None):
        raise SystemExit("--train-input and --test-input must be used together.")

    if args.train_input and args.test_input:
        train_data = pd.read_csv(args.train_input)
        test_data = pd.read_csv(args.test_input)
        validate_columns(train_data)
        validate_columns(test_data)
        evaluation_name = f"{train_data['dataset'].iloc[0]}_to_{test_data['dataset'].iloc[0]}"

        if args.metrics_output == DEFAULT_METRICS_OUTPUT:
            args.metrics_output = DEFAULT_EXTERNAL_METRICS_OUTPUT
        if args.predictions_output == DEFAULT_PREDICTIONS_OUTPUT:
            args.predictions_output = DEFAULT_EXTERNAL_PREDICTIONS_OUTPUT

        print(f"Loaded {len(train_data)} train rows from {args.train_input}")
        print(f"Loaded {len(test_data)} test rows from {args.test_input}")
    else:
        data = pd.read_csv(args.input)
        validate_columns(data)
        train_data, test_data = split_train_test(
            data=data,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        evaluation_name = "grouped_80_20"
        print(f"Loaded {len(data)} rows from {args.input}")

    metrics, predictions = evaluate_models(
        train_data=train_data,
        test_data=test_data,
        random_state=args.random_state,
        evaluation_name=evaluation_name,
    )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False)
    predictions.to_csv(args.predictions_output, index=False)

    print(f"Train rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print(metrics.to_string(index=False))
    print(f"Wrote metrics to {args.metrics_output}")
    print(f"Wrote predictions to {args.predictions_output}")


if __name__ == "__main__":
    main(sys.argv[1:])
