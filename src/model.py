from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data import FEATURE_COLUMNS, TARGET_COLUMN, load_crop_data


def _preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(transformers=[("num", numeric_pipeline, FEATURE_COLUMNS)])


def _candidate_models() -> dict[str, object]:
    return {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(max_iter=1200),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
    }


def _evaluate_kmeans(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, object]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", _preprocessor()),
            ("model", KMeans(n_clusters=y_train.nunique(), random_state=42, n_init=10)),
        ]
    )
    pipeline.fit(X_train)
    train_clusters = pipeline.predict(X_train)
    mapping = {}
    for cluster in sorted(set(train_clusters)):
        labels = y_train[train_clusters == cluster]
        mapping[cluster] = labels.mode().iat[0]

    predicted = [mapping.get(cluster, y_train.mode().iat[0]) for cluster in pipeline.predict(X_test)]
    return {
        "name": "K-Means Clustering",
        "accuracy": accuracy_score(y_test, predicted),
        "report": classification_report(y_test, predicted, zero_division=0),
    }


def train_and_select_model(base_dir: Path) -> dict[str, object]:
    df = load_crop_data(base_dir)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    best = {"name": None, "accuracy": -1.0, "pipeline": None, "report": ""}

    for name, estimator in _candidate_models().items():
        pipeline = Pipeline(steps=[("preprocess", _preprocessor()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)
        results.append({"name": name, "accuracy": accuracy, "report": report})

        if accuracy > best["accuracy"]:
            best = {"name": name, "accuracy": accuracy, "pipeline": pipeline, "report": report}

    results.append(_evaluate_kmeans(X_train, X_test, y_train, y_test))

    return {
        "best_model_name": best["name"],
        "best_accuracy": best["accuracy"],
        "best_pipeline": best["pipeline"],
        "best_report": best["report"],
        "results": sorted(results, key=lambda item: item["accuracy"], reverse=True),
        "features": FEATURE_COLUMNS,
    }


def save_model(payload: dict[str, object], model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "model_name": payload["best_model_name"],
        "accuracy": payload["best_accuracy"],
        "pipeline": payload["best_pipeline"],
        "features": payload["features"],
    }
    with model_path.open("wb") as file:
        pickle.dump(serializable, file)


def load_model(model_path: Path) -> dict[str, object]:
    with model_path.open("rb") as file:
        return pickle.load(file)


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return
    base_dir = model_path.resolve().parent.parent
    payload = train_and_select_model(base_dir)
    save_model(payload, model_path)
