"""Support Vector Machine baseline."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_linear_svm(random_state: int = 42) -> Pipeline:
    """Build a scaled linear SVM hallucination classifier."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(max_iter=10000, random_state=random_state)),
        ]
    )


def train_svm(X_train, y_train, random_state: int = 42) -> Pipeline:
    """Train a linear SVM hallucination detector."""

    model = build_linear_svm(random_state=random_state)
    model.fit(X_train, y_train)
    return model
