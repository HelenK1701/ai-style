# src/preprocessing.py
"""
Feature preprocessing: validation, missing-value handling, and scaling.

StandardScaler is used rather than MinMaxScaler because our colour features
(LAB, HSV) are not strictly bounded across all possible inputs after
processing, and StandardScaler is more robust to outliers in small datasets
typical of a personal-colour project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, RANDOM_SEED, SCALER_PATH
from src.utils import get_logger

logger = get_logger(__name__)


# ── Validation ─────────────────────────────────────────────────────────────────

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    # CIE LAB: L in [0,100], a/b in [-128, 127] (OpenCV encodes as uint8 offset)
    "skin_L": (0.0, 100.0),
    "skin_a": (0.0, 255.0),
    "skin_b": (0.0, 255.0),
    # HSV in OpenCV: H in [0,180], S/V in [0,255]
    "skin_H": (0.0, 180.0),
    "skin_S": (0.0, 255.0),
    "skin_V": (0.0, 255.0),
    "brightness": (0.0, 255.0),
    "contrast": (0.0, 128.0),   # std dev of grayscale rarely exceeds 128
    "saturation": (0.0, 255.0),
    "hair_H": (0.0, 180.0),
    "hair_S": (0.0, 255.0),
    "hair_V": (0.0, 255.0),
}


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check that all required columns are present and values fall within
    expected physical ranges. Clips out-of-range values with a warning.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature DataFrame (one row per sample).

    Returns
    -------
    pd.DataFrame
        Validated (and range-clipped) DataFrame.

    Raises
    ------
    ValueError
        If a required column is entirely missing.
    """
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    df = df[FEATURE_COLUMNS].copy()

    for col, (lo, hi) in FEATURE_BOUNDS.items():
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            logger.warning(
                "Column '%s': %d value(s) outside [%.1f, %.1f] — clipping.",
                col, out_of_range, lo, hi,
            )
            df[col] = df[col].clip(lo, hi)

    return df


# ── Missing-value handling ─────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with column medians.

    Median imputation is preferred over mean here because some colour
    channels (e.g., hue for very low-saturation regions) can be noisy,
    and the median is more robust to those outlier readings.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame, potentially containing NaN values.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaN values replaced.
    """
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        logger.warning("Imputing %d missing value(s) with column medians.", n_missing)
        for col in df.columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                # If all values are NaN (e.g., degenerate crop), use 0.0
                fill_val = median_val if not np.isnan(median_val) else 0.0
                df[col] = df[col].fillna(fill_val)
    return df


# ── Scaling ────────────────────────────────────────────────────────────────────

class FeatureScaler:
    """
    Thin wrapper around sklearn StandardScaler with save/load helpers.

    Keeping scaling separate from the KMeans model allows us to update
    the scaler independently (e.g., after collecting more training data)
    without retraining the whole clustering model.
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """Fit scaler on training data."""
        self._scaler.fit(df[FEATURE_COLUMNS].values)
        self._fitted = True
        logger.info("Scaler fitted on %d samples.", len(df))
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Scale features.  Raises if called before ``fit`` or ``load``.
        """
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before calling transform().")
        return self._scaler.transform(df[FEATURE_COLUMNS].values)

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on and transform ``df`` in one step."""
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------
    def save(self, path: Path = SCALER_PATH) -> None:
        """Persist the fitted scaler to disk via joblib."""
        joblib.dump(self._scaler, path)
        logger.info("Scaler saved → %s", path)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> "FeatureScaler":
        """
        Load a previously-saved scaler from disk.

        Raises
        ------
        FileNotFoundError
            If the scaler artifact does not exist.
        """
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Scaler not found at {path}. Run train.py first."
            )
        instance = cls()
        instance._scaler = joblib.load(path)
        instance._fitted = True
        logger.info("Scaler loaded ← %s", path)
        return instance

    # ------------------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        return FEATURE_COLUMNS

    @property
    def mean_(self) -> np.ndarray:
        return self._scaler.mean_

    @property
    def scale_(self) -> np.ndarray:
        return self._scaler.scale_


# ── Pipeline helper ────────────────────────────────────────────────────────────

def preprocess_pipeline(
    df: pd.DataFrame,
    scaler: Optional[FeatureScaler] = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, FeatureScaler]:
    """
    Full preprocessing pipeline: validate → impute → scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature DataFrame.
    scaler : FeatureScaler | None
        Pre-fitted scaler. If None and ``fit_scaler`` is False,
        a new scaler is created and fit on ``df``.
    fit_scaler : bool
        If True, always fit a new scaler on ``df`` (training mode).

    Returns
    -------
    (X_scaled, scaler)
        Scaled feature array and the scaler used.
    """
    df = validate_features(df)
    df = handle_missing(df)

    if scaler is None or fit_scaler:
        scaler = FeatureScaler()
        X_scaled = scaler.fit_transform(df)
    else:
        X_scaled = scaler.transform(df)

    return X_scaled, scaler
