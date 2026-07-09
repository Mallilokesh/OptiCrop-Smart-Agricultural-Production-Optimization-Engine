from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.model import load_model


def predict_crop(values: dict[str, float], model_path: Path) -> tuple[str, list[tuple[str, float]]]:
    artifact = load_model(model_path)
    pipeline = artifact["pipeline"]
    features = artifact["features"]
    frame = pd.DataFrame([{feature: values[feature] for feature in features}])
    crop = str(pipeline.predict(frame)[0])

    probabilities = []
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        scores = pipeline.predict_proba(frame)[0]
        classes = pipeline.classes_
        probabilities = sorted(
            [(str(label), round(float(score) * 100, 2)) for label, score in zip(classes, scores)],
            key=lambda item: item[1],
            reverse=True,
        )[:3]

    return crop, probabilities
