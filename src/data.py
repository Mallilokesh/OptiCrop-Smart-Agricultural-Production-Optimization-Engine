from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

CROP_PROFILES = {
    "rice": {"N": (70, 105), "P": (35, 60), "K": (35, 55), "temperature": (23, 32), "humidity": (75, 95), "ph": (5.4, 7.0), "rainfall": (160, 280)},
    "maize": {"N": (55, 95), "P": (35, 65), "K": (20, 45), "temperature": (20, 32), "humidity": (45, 75), "ph": (5.6, 7.5), "rainfall": (55, 125)},
    "chickpea": {"N": (15, 45), "P": (50, 80), "K": (60, 85), "temperature": (17, 28), "humidity": (15, 45), "ph": (6.0, 8.2), "rainfall": (35, 95)},
    "cotton": {"N": (95, 140), "P": (35, 65), "K": (35, 70), "temperature": (24, 36), "humidity": (45, 75), "ph": (5.8, 8.0), "rainfall": (65, 140)},
    "banana": {"N": (80, 120), "P": (70, 100), "K": (45, 75), "temperature": (25, 35), "humidity": (70, 92), "ph": (5.5, 7.2), "rainfall": (90, 180)},
    "mango": {"N": (10, 45), "P": (15, 40), "K": (25, 55), "temperature": (27, 38), "humidity": (45, 70), "ph": (5.2, 7.8), "rainfall": (70, 160)},
    "grapes": {"N": (15, 45), "P": (120, 145), "K": (190, 205), "temperature": (18, 32), "humidity": (75, 90), "ph": (5.5, 7.5), "rainfall": (55, 95)},
}


def dataset_path(base_dir: Path) -> Path:
    preferred = base_dir / "data" / "crop_recommendation.csv"
    if preferred.exists():
        return preferred
    return base_dir / "data" / "crop_recommendation_sample.csv"


def create_sample_dataset(path: Path, samples_per_crop: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for crop, profile in CROP_PROFILES.items():
        for _ in range(samples_per_crop):
            row = {}
            for feature, (low, high) in profile.items():
                center = (low + high) / 2
                spread = (high - low) / 5
                value = rng.normal(center, spread)
                row[feature] = round(float(np.clip(value, low, high)), 2)
            row[TARGET_COLUMN] = crop
            rows.append(row)

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_crop_data(base_dir: Path) -> pd.DataFrame:
    path = dataset_path(base_dir)
    if not path.exists():
        return create_sample_dataset(path)

    df = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required.difference(df.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {names}")

    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
