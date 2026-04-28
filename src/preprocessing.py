"""
Feature preprocessing: validation, missing values, scaling.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, SCALER_PATH
from src.utils import get_logger

logger = get_logger(__name__)

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "skin_L": (0.0, 255.0),
    "skin_a": (0.0, 255.0),
    "skin_b": (0.0, 255.0),
    "skin_H": (0.0, 180.0),
    "skin_S": (0.0, 255.0),
    "skin_V": (0.0, 255.0),
    "brightness": (0.0, 255.0),
    "contrast": (0.0, 128.0),
    "saturation": (0.0, 255.0),
    "hair_H": (0.0, 180.0),
    "hair_S": (0.0, 255.0),
    "hair_V": (0.0, 255.0),
}


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
    checked = df[FEATURE_COLUMNS].copy()
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        checked[col] = checked[col].clip(lo, hi)
    return checked


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].isnull().any():
            median = out[col].median()
            out[col] = out[col].fillna(0.0 if np.isnan(median) else median)
    return out


class FeatureScaler:
    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        self._scaler.fit(df[FEATURE_COLUMNS].values)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted or loaded before transform.")
        return self._scaler.transform(df[FEATURE_COLUMNS].values)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def save(self, path: Path = SCALER_PATH) -> None:
        joblib.dump(self._scaler, path)
        logger.info("Scaler saved to %s", path)

    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> "FeatureScaler":
        if not path.exists():
            raise FileNotFoundError(f"Scaler not found: {path}")
        instance = cls()
        instance._scaler = joblib.load(path)
        instance._fitted = True
        return instance


def preprocess_pipeline(df: pd.DataFrame, scaler: FeatureScaler | None = None, fit_scaler: bool = False) -> tuple[np.ndarray, FeatureScaler]:
    checked = handle_missing(validate_features(df))
    if scaler is None or fit_scaler:
        scaler = FeatureScaler()
        X = scaler.fit_transform(checked)
    else:
        X = scaler.transform(checked)
    return X, scaler
