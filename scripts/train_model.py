#!/usr/bin/env python3
"""Train baseline ML models on extracted bearing features.

This script extends the previous baseline by:
- performing train-only feature ranking / feature selection
- comparing multiple feature subset sizes
- adding a neural-network baseline (MLP)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML baselines on extracted bearing features.")
    parser.add_argument("--input", type=Path, required=True, help="Path to combined_features_with_outputs.csv")
    parser.add_argument("--output", type=Path, required=True, help="Directory for reports and plots")
    parser.add_argument(
        "--feature-counts",
        default="10,20,all",
        help="Comma-separated feature subset sizes to compare, for example: 10,20,all",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="GroupShuffleSplit test size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading features from {csv_path}...")
    frame = pd.read_csv(csv_path)
    if "output_fault_status" not in frame.columns:
        raise ValueError("Expected output_fault_status column in input CSV.")
    return frame.dropna(subset=["output_fault_status"]).copy()


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in frame.columns
        if column.startswith(("td_", "fd_", "tf_")) and not column.endswith("_path")
    ]
    if not feature_columns:
        raise ValueError("No td_/fd_/tf_ feature columns found in the input CSV.")
    return feature_columns


def parse_feature_counts(raw_value: str, total_feature_count: int) -> list[int | str]:
    selections: list[int | str] = []
    seen: set[int | str] = set()
    for token in (part.strip().lower() for part in raw_value.split(",")):
        if not token:
            continue
        value: int | str
        if token == "all":
            value = "all"
        else:
            value = max(1, min(int(token), total_feature_count))
        if value not in seen:
            selections.append(value)
            seen.add(value)
    if "all" not in seen:
        selections.append("all")
    return selections


def format_subset_name(feature_count: int | str) -> str:
    return "all_features" if feature_count == "all" else f"top_{feature_count}"


def save_feature_importance_artifacts(
    ranking: pd.Series, output_dir: Path, task_name: str, top_n_plot: int = 20
) -> None:
    ranking_df = ranking.reset_index()
    ranking_df.columns = ["feature", "importance"]
    ranking_df.to_csv(output_dir / f"feature_ranking_{task_name}.csv", index=False)

    plot_count = min(top_n_plot, len(ranking_df))
    plot_df = ranking_df.head(plot_count)
    plt.figure(figsize=(10, max(6, 0.35 * plot_count)))
    sns.barplot(data=plot_df, x="importance", y="feature", hue="feature", dodge=False, palette="viridis")
    plt.title(f"Top {plot_count} Feature Importances - {task_name}")
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_{task_name}.png", dpi=150)
    plt.close()


def build_models(random_state: int) -> dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=random_state),
        "SVM (SVC)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=False, random_state=random_state)),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ]
        ),
        "Neural Network (MLP)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size="auto",
                        learning_rate="adaptive",
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=15,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_accuracy_comparison(results_df: pd.DataFrame, output_path: Path, task_name: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="feature_subset", y="accuracy", hue="model", palette="tab10")
    plt.title(f"Accuracy Comparison - {task_name}")
    plt.xlabel("Feature Subset")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def rank_features(X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> pd.Series:
    ranker = RandomForestClassifier(
        n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced"
    )
    ranker.fit(X_train, y_train)
    ranking = pd.Series(ranker.feature_importances_, index=X_train.columns, dtype=float)
    return ranking.sort_values(ascending=False)


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    task_name: str,
    feature_subset_name: str,
    labels: list[str],
    random_state: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for model_name, model in build_models(random_state).items():
        print(f"[{task_name} / {feature_subset_name}] Training {model_name}...")
        if model_name == "Neural Network (MLP)":
            encoder = LabelEncoder()
            y_train_encoded = encoder.fit_transform(y_train)
            model.fit(X_train, y_train_encoded)
            predictions_encoded = model.predict(X_test)
            predictions = encoder.inverse_transform(predictions_encoded)
            report_dict = classification_report(
                y_test,
                predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(y_test, predictions, labels=labels)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            report_dict = classification_report(
                y_test,
                predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(y_test, predictions, labels=labels)

        cm_path = (
            output_dir
            / f"cm_{task_name}_{feature_subset_name}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        )
        plot_confusion_matrix(cm, labels, cm_path, f"Confusion Matrix: {model_name}\nTask: {task_name} | {feature_subset_name}")

        records.append(
            {
                "task": task_name,
                "feature_subset": feature_subset_name,
                "model": model_name,
                "num_features": int(X_train.shape[1]),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                "classification_report": report_dict,
                "confusion_matrix_path": str(cm_path),
            }
        )
    return records


def run_task(
    task_name: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: pd.Series,
    output_dir: Path,
    feature_counts: list[int | str],
    test_size: float,
    random_state: int,
) -> dict[str, object]:
    X_full = frame[feature_columns].copy()
    X_full = X_full.fillna(X_full.median(numeric_only=True))
    groups = frame["source_basename"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X_full, target, groups=groups))
    X_train_full = X_full.iloc[train_idx].copy()
    X_test_full = X_full.iloc[test_idx].copy()
    y_train = target.iloc[train_idx].reset_index(drop=True)
    y_test = target.iloc[test_idx].reset_index(drop=True)
    X_train_full = X_train_full.reset_index(drop=True)
    X_test_full = X_test_full.reset_index(drop=True)
    labels = sorted(target.astype(str).unique().tolist())

    ranking = rank_features(X_train_full, y_train, random_state)
    save_feature_importance_artifacts(ranking, output_dir, task_name)

    task_records: list[dict[str, object]] = []
    selected_features_map: dict[str, list[str]] = {}
    for feature_count in feature_counts:
        subset_name = format_subset_name(feature_count)
        selected_features = ranking.index.tolist() if feature_count == "all" else ranking.index[: int(feature_count)].tolist()
        selected_features_map[subset_name] = selected_features

        X_train_subset = X_train_full[selected_features]
        X_test_subset = X_test_full[selected_features]
        task_records.extend(
            evaluate_models(
                X_train=X_train_subset,
                X_test=X_test_subset,
                y_train=y_train,
                y_test=y_test,
                output_dir=output_dir,
                task_name=task_name,
                feature_subset_name=subset_name,
                labels=labels,
                random_state=random_state,
            )
        )

    results_df = pd.DataFrame(
        [
            {
                "task": record["task"],
                "feature_subset": record["feature_subset"],
                "model": record["model"],
                "num_features": record["num_features"],
                "accuracy": record["accuracy"],
                "macro_f1": record["macro_f1"],
                "weighted_f1": record["weighted_f1"],
                "confusion_matrix_path": record["confusion_matrix_path"],
            }
            for record in task_records
        ]
    )
    results_df.to_csv(output_dir / f"accuracy_summary_{task_name}.csv", index=False)
    plot_accuracy_comparison(results_df, output_dir / f"accuracy_comparison_{task_name}.png", task_name)

    with (output_dir / f"selected_features_{task_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(selected_features_map, handle, indent=2)

    return {
        "task_name": task_name,
        "train_samples": int(len(X_train_full)),
        "test_samples": int(len(X_test_full)),
        "labels": labels,
        "feature_ranking_path": str(output_dir / f"feature_ranking_{task_name}.csv"),
        "selected_features_path": str(output_dir / f"selected_features_{task_name}.json"),
        "results": task_records,
    }


def build_multiclass_target(frame: pd.DataFrame) -> pd.Series:
    def _format_row(row: pd.Series) -> str:
        diameter = row["output_fault_diameter_in"]
        if pd.isna(diameter):
            return f"{row['output_fault_location']}_UNKNOWN"
        return f"{row['output_fault_location']}_{float(diameter):.3f}"

    return frame.apply(_format_row, axis=1)


def main() -> int:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_data(args.input.expanduser().resolve())
    feature_columns = get_feature_columns(frame)
    feature_counts = parse_feature_counts(args.feature_counts, len(feature_columns))

    print(f"Total available features: {len(feature_columns)}")
    print(f"Evaluated feature subsets: {[format_subset_name(value) for value in feature_counts]}")

    binary_mask = frame["output_fault_status"].isin(["FAULT", "NO_FAULT"])
    binary_frame = frame.loc[binary_mask].copy().reset_index(drop=True)
    binary_target = binary_frame["output_fault_status"].astype(str)

    fault_frame = frame.loc[frame["output_fault_status"] == "FAULT"].copy().reset_index(drop=True)
    diagnosis_target = build_multiclass_target(fault_frame)

    report = {
        "input_csv": str(args.input.expanduser().resolve()),
        "output_dir": str(output_dir),
        "feature_columns": feature_columns,
        "feature_counts": [value if value == "all" else int(value) for value in feature_counts],
        "tasks": {},
    }

    report["tasks"]["Binary_Detection"] = run_task(
        task_name="Binary_Detection",
        frame=binary_frame,
        feature_columns=feature_columns,
        target=binary_target,
        output_dir=output_dir,
        feature_counts=feature_counts,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    report["tasks"]["Multi_Diagnosis"] = run_task(
        task_name="Multi_Diagnosis",
        frame=fault_frame,
        feature_columns=feature_columns,
        target=diagnosis_target,
        output_dir=output_dir,
        feature_counts=feature_counts,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    with (output_dir / "ml_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved ML reports and plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
