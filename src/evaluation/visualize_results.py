"""Create report figures from evaluation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation.evaluate_uncertainty_classifiers import FEATURE_COLUMNS

DEFAULT_OUTPUT_DIR = Path("outputs/figures")
CV_SUMMARIES = {
    ("Qwen2.5-3B", "HaluEval"): Path("outputs/tables/halueval_qwen3b_cv_summary.csv"),
    ("Qwen2.5-3B", "TruthfulQA"): Path(
        "outputs/tables/truthfulqa_qwen3b_cv_summary.csv"
    ),
    ("Qwen2.5-0.5B", "HaluEval"): Path(
        "outputs/tables/halueval_qwen05b_cv_summary.csv"
    ),
    ("Qwen2.5-0.5B", "TruthfulQA"): Path(
        "outputs/tables/truthfulqa_qwen05b_cv_summary.csv"
    ),
}
CONTEXT_EFFECT_SUMMARIES = {
    ("Qwen2.5-3B", "HaluEval\nwith context"): Path(
        "outputs/tables/halueval_qwen3b_cv_summary.csv"
    ),
    ("Qwen2.5-3B", "HaluEval\nno context"): Path(
        "outputs/tables/halueval_no_context_qwen3b_cv_summary.csv"
    ),
    ("Qwen2.5-3B", "TruthfulQA\nno context"): Path(
        "outputs/tables/truthfulqa_qwen3b_cv_summary.csv"
    ),
    ("Qwen2.5-0.5B", "HaluEval\nwith context"): Path(
        "outputs/tables/halueval_qwen05b_cv_summary.csv"
    ),
    ("Qwen2.5-0.5B", "HaluEval\nno context"): Path(
        "outputs/tables/halueval_no_context_qwen05b_cv_summary.csv"
    ),
    ("Qwen2.5-0.5B", "TruthfulQA\nno context"): Path(
        "outputs/tables/truthfulqa_qwen05b_cv_summary.csv"
    ),
}
ABLATION_FILES = {
    ("Qwen2.5-3B", "HaluEval"): Path("outputs/tables/halueval_ablation_results.csv"),
    ("Qwen2.5-3B", "TruthfulQA"): Path(
        "outputs/tables/truthfulqa_ablation_results.csv"
    ),
    ("Qwen2.5-0.5B", "HaluEval"): Path(
        "outputs/tables/halueval_qwen05b_ablation_results.csv"
    ),
    ("Qwen2.5-0.5B", "TruthfulQA"): Path(
        "outputs/tables/truthfulqa_qwen05b_ablation_results.csv"
    ),
}
METRIC_FILES = {
    ("Qwen2.5-3B", "HaluEval"): Path("outputs/tables/halueval_classifier_metrics.csv"),
    ("Qwen2.5-3B", "TruthfulQA"): Path(
        "outputs/tables/truthfulqa_classifier_metrics.csv"
    ),
    ("Qwen2.5-0.5B", "HaluEval"): Path(
        "outputs/tables/halueval_qwen05b_classifier_metrics.csv"
    ),
    ("Qwen2.5-0.5B", "TruthfulQA"): Path(
        "outputs/tables/truthfulqa_qwen05b_classifier_metrics.csv"
    ),
}
PREDICTION_FILES = {
    ("Qwen2.5-3B", "HaluEval"): Path(
        "outputs/predictions/halueval_classifier_predictions.csv"
    ),
    ("Qwen2.5-3B", "TruthfulQA"): Path(
        "outputs/predictions/truthfulqa_classifier_predictions.csv"
    ),
    ("Qwen2.5-0.5B", "HaluEval"): Path(
        "outputs/predictions/halueval_qwen05b_classifier_predictions.csv"
    ),
    ("Qwen2.5-0.5B", "TruthfulQA"): Path(
        "outputs/predictions/truthfulqa_qwen05b_classifier_predictions.csv"
    ),
}
FEATURE_DISTRIBUTION_FILE = Path("outputs/tables/feature_distribution_by_dataset.csv")
FEATURE_FILES = {
    "HaluEval QA": Path("data/processed/halueval_uncertainty_entropy_features.csv"),
    "TruthfulQA": Path("data/processed/truthfulqa_uncertainty_entropy_features.csv"),
}
SELECTED_FEATURES = [
    "mean_token_probability",
    "negative_mean_logprob",
    "mean_token_entropy",
    "low_confidence_token_ratio",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visualization figures for the hallucination project."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG figures will be written.",
    )
    return parser.parse_args(argv)


def save_current_figure(path: Path) -> None:
    """Save the current matplotlib figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")


def load_best_cv_rows() -> pd.DataFrame:
    """Load the best ROC-AUC row from each cross-validation summary."""

    rows = []
    for (extractor, dataset), path in CV_SUMMARIES.items():
        data = pd.read_csv(path)
        best = data.sort_values("roc_auc_mean", ascending=False).iloc[0]
        rows.append(
            {
                "feature_extractor": extractor,
                "dataset": dataset,
                "model": best["model"],
                "f1": best["f1_mean"],
                "roc_auc": best["roc_auc_mean"],
            }
        )
    return pd.DataFrame(rows)


def plot_cv_comparison(output_dir: Path) -> None:
    """Plot best grouped CV F1 and ROC-AUC by feature extractor."""

    data = load_best_cv_rows()
    labels = [f"{row.dataset}\n{row.feature_extractor}" for row in data.itertuples()]
    x_positions = range(len(data))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(
        [x - width / 2 for x in x_positions],
        data["f1"],
        width=width,
        label="F1",
        color="#4C78A8",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        data["roc_auc"],
        width=width,
        label="ROC-AUC",
        color="#F58518",
    )
    plt.xticks(list(x_positions), labels)
    plt.ylim(0.55, 1.02)
    plt.ylabel("Score")
    plt.title("Best Grouped 5-Fold CV Results")
    plt.legend()
    save_current_figure(output_dir / "cv_model_size_comparison.png")


def load_context_effect_rows() -> pd.DataFrame:
    """Load best ROC-AUC rows for context-effect comparison."""

    rows = []
    for (extractor, setting), path in CONTEXT_EFFECT_SUMMARIES.items():
        data = pd.read_csv(path)
        best = data.sort_values("roc_auc_mean", ascending=False).iloc[0]
        rows.append(
            {
                "feature_extractor": extractor,
                "setting": setting,
                "model": best["model"],
                "f1": best["f1_mean"],
                "roc_auc": best["roc_auc_mean"],
            }
        )
    return pd.DataFrame(rows)


def plot_context_effect_comparison(output_dir: Path) -> None:
    """Plot HaluEval with context, HaluEval no-context, and TruthfulQA."""

    data = load_context_effect_rows()
    setting_order = [
        "HaluEval\nwith context",
        "HaluEval\nno context",
        "TruthfulQA\nno context",
    ]
    colors = {
        "Qwen2.5-3B": "#4C78A8",
        "Qwen2.5-0.5B": "#F58518",
    }
    x_positions = range(len(setting_order))
    width = 0.36

    plt.figure(figsize=(9, 5))
    for offset, extractor in [(-width / 2, "Qwen2.5-3B"), (width / 2, "Qwen2.5-0.5B")]:
        subset = data[data["feature_extractor"] == extractor].set_index("setting")
        scores = [subset.loc[setting, "f1"] for setting in setting_order]
        plt.bar(
            [x + offset for x in x_positions],
            scores,
            width=width,
            label=extractor,
            color=colors[extractor],
        )

    plt.xticks(list(x_positions), setting_order)
    plt.ylim(0.55, 1.02)
    plt.ylabel("Best grouped CV F1")
    plt.title("Effect of Context on Hallucination Detection")
    plt.legend()
    save_current_figure(output_dir / "context_effect_comparison.png")


def load_ablation_rows() -> pd.DataFrame:
    """Load best ROC-AUC row per feature group for each ablation file."""

    rows = []
    for (extractor, dataset), path in ABLATION_FILES.items():
        data = pd.read_csv(path)
        for feature_group, group_data in data.groupby("feature_group"):
            best = group_data.sort_values("roc_auc", ascending=False).iloc[0]
            rows.append(
                {
                    "feature_extractor": extractor,
                    "dataset": dataset,
                    "feature_group": feature_group,
                    "model": best["model"],
                    "f1": best["f1"],
                    "roc_auc": best["roc_auc"],
                }
            )
    return pd.DataFrame(rows)


def plot_ablation_comparison(output_dir: Path) -> None:
    """Plot ablation F1 scores by feature group."""

    data = load_ablation_rows()
    feature_order = ["confidence_logprob", "entropy", "all_features"]
    panel_order = [
        ("Qwen2.5-3B", "HaluEval"),
        ("Qwen2.5-0.5B", "HaluEval"),
        ("Qwen2.5-3B", "TruthfulQA"),
        ("Qwen2.5-0.5B", "TruthfulQA"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for axis, (extractor, dataset) in zip(axes.flatten(), panel_order):
        panel = data[
            (data["feature_extractor"] == extractor) & (data["dataset"] == dataset)
        ].set_index("feature_group")
        scores = [panel.loc[group, "f1"] for group in feature_order]
        axis.bar(feature_order, scores, color=["#72B7B2", "#E45756", "#54A24B"])
        axis.set_title(f"{dataset} / {extractor}")
        axis.set_ylim(0.55, 1.02)
        axis.tick_params(axis="x", rotation=20)

    fig.suptitle("Ablation Results: Best F1 by Feature Group")
    fig.supylabel("F1")
    save_current_figure(output_dir / "ablation_feature_group_comparison.png")


def get_best_model_name(metric_path: Path) -> str:
    """Return the best ROC-AUC model from a metrics file."""

    data = pd.read_csv(metric_path)
    return str(data.sort_values("roc_auc", ascending=False).iloc[0]["model"])


def get_model_f1(metric_path: Path, model_name: str) -> float:
    """Return the grouped 80/20 F1 score for a specific model."""

    data = pd.read_csv(metric_path)
    model_row = data[data["model"] == model_name]
    if model_row.empty:
        raise ValueError(f"Model '{model_name}' not found in {metric_path}")
    return float(model_row.iloc[0]["f1"])


def plot_confusion_matrices(output_dir: Path) -> None:
    """Plot confusion matrices for the best 80/20 model in each setting."""

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for axis, key in zip(axes.flatten(), PREDICTION_FILES):
        extractor, dataset = key
        model_name = get_best_model_name(METRIC_FILES[key])
        predictions = pd.read_csv(PREDICTION_FILES[key])
        predictions = predictions[predictions["model"] == model_name]
        matrix = confusion_matrix(
            predictions["label"],
            predictions["prediction"],
            labels=[0, 1],
        )
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{dataset} / {extractor}\n{model_name}")
        axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], labels=["True 0", "True 1"])
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    int(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="black",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion Matrices for Best 80/20 Models")
    save_current_figure(output_dir / "confusion_matrices.png")


def plot_roc_curves(output_dir: Path) -> None:
    """Plot ROC curves for the best 80/20 model in each setting."""

    plt.figure(figsize=(8, 6))
    for key, prediction_path in PREDICTION_FILES.items():
        extractor, dataset = key
        model_name = get_best_model_name(METRIC_FILES[key])
        predictions = pd.read_csv(prediction_path)
        predictions = predictions[predictions["model"] == model_name]
        false_positive_rate, true_positive_rate, _ = roc_curve(
            predictions["label"],
            predictions["score"],
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)
        label = f"{dataset} / {extractor} / {model_name} (AUC={roc_auc:.3f})"
        plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Best 80/20 Models")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    save_current_figure(output_dir / "roc_curves.png")


def plot_feature_distribution_shift(output_dir: Path) -> None:
    """Plot selected feature means by dataset and label."""

    data = pd.read_csv(FEATURE_DISTRIBUTION_FILE)
    data = data[data["feature"].isin(SELECTED_FEATURES)].copy()
    data["group"] = data["dataset"] + " label=" + data["label"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, feature in zip(axes.flatten(), SELECTED_FEATURES):
        feature_data = data[data["feature"] == feature]
        axis.bar(feature_data["group"], feature_data["mean"], color="#4C78A8")
        axis.set_title(feature)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel("Mean")

    fig.suptitle("Feature Distribution Shift by Dataset and Label")
    save_current_figure(output_dir / "feature_distribution_shift.png")


def plot_pca_decision_boundary(
    feature_path: Path,
    title: str,
    output_path: Path,
    classifier_name: str = "Logistic Regression",
    max_scatter_points: int = 6000,
) -> None:
    """Plot a 2D PCA decision boundary for visualization only."""

    data = pd.read_csv(feature_path)
    features = data[FEATURE_COLUMNS]
    labels = data["label"].astype(int)

    scaled_features = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled_features)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }
    classifier = classifiers[classifier_name]
    classifier.fit(coordinates, labels)

    x_min, x_max = coordinates[:, 0].min() - 0.5, coordinates[:, 0].max() + 0.5
    y_min, y_max = coordinates[:, 1].min() - 0.5, coordinates[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = pd.DataFrame({"pc1": xx.ravel(), "pc2": yy.ravel()}).to_numpy()
    predictions = classifier.predict(grid).reshape(xx.shape)

    plot_data = pd.DataFrame(
        {
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
            "label": labels,
        }
    )
    if len(plot_data) > max_scatter_points:
        plot_data = plot_data.sample(max_scatter_points, random_state=42)

    plt.figure(figsize=(7, 5.5))
    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.18,
        levels=[-0.5, 0.5, 1.5],
        colors=["#4C78A8", "#E45756"],
    )
    for label, color, name in [(0, "#4C78A8", "label 0"), (1, "#E45756", "label 1")]:
        subset = plot_data[plot_data["label"] == label]
        plt.scatter(
            subset["pc1"],
            subset["pc2"],
            s=12,
            alpha=0.35,
            color=color,
            edgecolors="none",
            label=name,
        )

    explained = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    save_current_figure(output_path)


def plot_pca_decision_boundaries(output_dir: Path) -> None:
    """Create PCA decision-boundary plots for HaluEval QA and TruthfulQA."""

    for dataset_name, path in FEATURE_FILES.items():
        metrics_key = ("Qwen2.5-3B", "HaluEval" if dataset_name == "HaluEval QA" else dataset_name)
        metric_path = METRIC_FILES[metrics_key]
        output_name = dataset_name.lower().replace(" ", "_").replace("qa", "qa")
        logistic_f1 = get_model_f1(metric_path, "Logistic Regression")
        rbf_f1 = get_model_f1(metric_path, "RBF SVM")
        random_forest_f1 = get_model_f1(metric_path, "Random Forest")
        plot_pca_decision_boundary(
            feature_path=path,
            title=f"{dataset_name}: Logistic Regression PCA Boundary (F1={logistic_f1:.3f})",
            output_path=output_dir / f"{output_name}_pca_decision_boundary.png",
        )
        plot_pca_decision_boundary(
            feature_path=path,
            title=f"{dataset_name}: RBF SVM PCA Boundary (F1={rbf_f1:.3f})",
            output_path=output_dir / f"{output_name}_pca_decision_boundary_rbf_svm.png",
            classifier_name="RBF SVM",
        )
        plot_pca_decision_boundary(
            feature_path=path,
            title=f"{dataset_name}: Random Forest PCA Boundary (F1={random_forest_f1:.3f})",
            output_path=output_dir
            / f"{output_name}_pca_decision_boundary_random_forest.png",
            classifier_name="Random Forest",
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plot_cv_comparison(args.output_dir)
    plot_context_effect_comparison(args.output_dir)
    plot_ablation_comparison(args.output_dir)
    plot_confusion_matrices(args.output_dir)
    plot_roc_curves(args.output_dir)
    plot_feature_distribution_shift(args.output_dir)
    plot_pca_decision_boundaries(args.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
