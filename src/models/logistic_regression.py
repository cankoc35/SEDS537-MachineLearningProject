"""Logistic Regression baseline."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression(random_state: int = 42) -> Pipeline:
    """Build a scaled Logistic Regression hallucination classifier."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )


def train_logistic_regression(X_train, y_train, random_state: int = 42) -> Pipeline:
    """Train a logistic regression hallucination detector."""

    model = build_logistic_regression(random_state=random_state)
    model.fit(X_train, y_train)
    return model
