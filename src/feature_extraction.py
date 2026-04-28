"""
Feature extraction from face crop.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import (
    FEATURE_COLUMNS,
    HAIR_KMEANS_CLUSTERS,
    HAIR_REGION_FRACTION,
    RANDOM_SEED,
    SKIN_REGION_FRACTION_BOTTOM,
    SKIN_REGION_FRACTION_TOP,
)
from src.utils import bgr_to_hsv, bgr_to_lab, crop_region, get_logger, safe_mean

logger = get_logger(__name__)


def _skin_zone(face_crop_rgb: np.ndarray) -> np.ndarray:
    return crop_region(face_crop_rgb, top_frac=SKIN_REGION_FRACTION_TOP, bottom_frac=SKIN_REGION_FRACTION_BOTTOM)


def _mean_lab_skin(face_crop_rgb: np.ndarray) -> tuple[float, float, float]:
    bgr = cv2.cvtColor(_skin_zone(face_crop_rgb), cv2.COLOR_RGB2BGR)
    lab = bgr_to_lab(bgr)
    return float(safe_mean(lab[:, :, 0])), float(safe_mean(lab[:, :, 1])), float(safe_mean(lab[:, :, 2]))


def _mean_hsv_skin(face_crop_rgb: np.ndarray) -> tuple[float, float, float]:
    bgr = cv2.cvtColor(_skin_zone(face_crop_rgb), cv2.COLOR_RGB2BGR)
    hsv = bgr_to_hsv(bgr)
    return float(safe_mean(hsv[:, :, 0])), float(safe_mean(hsv[:, :, 1])), float(safe_mean(hsv[:, :, 2]))


def _brightness_contrast(face_crop_rgb: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.mean(gray)), float(np.std(gray))


def _mean_saturation(face_crop_rgb: np.ndarray) -> float:
    bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = bgr_to_hsv(bgr)
    return float(safe_mean(hsv[:, :, 1]))


def _dominant_hair_hsv(face_crop_rgb: np.ndarray) -> tuple[float, float, float]:
    hair_zone = crop_region(face_crop_rgb, 0.0, HAIR_REGION_FRACTION)
    if hair_zone.size == 0:
        return 0.0, 0.0, 0.0
    pixels = hair_zone.reshape(-1, 3).astype(np.float32)
    k = max(1, min(HAIR_KMEANS_CLUSTERS, len(pixels)))
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    labels = model.fit_predict(pixels)
    counts = np.bincount(labels)
    rgb = model.cluster_centers_[int(np.argmax(counts))].astype(np.uint8)
    hsv = cv2.cvtColor(rgb[::-1].reshape(1, 1, 3), cv2.COLOR_BGR2HSV)
    return float(hsv[0, 0, 0]), float(hsv[0, 0, 1]), float(hsv[0, 0, 2])


def extract_features(face_crop_rgb: np.ndarray) -> dict[str, float]:
    h, w = face_crop_rgb.shape[:2]
    if h < 20 or w < 20:
        raise ValueError("Face crop too small for reliable features.")

    skin_L, skin_a, skin_b = _mean_lab_skin(face_crop_rgb)
    skin_H, skin_S, skin_V = _mean_hsv_skin(face_crop_rgb)
    brightness, contrast = _brightness_contrast(face_crop_rgb)
    saturation = _mean_saturation(face_crop_rgb)
    hair_H, hair_S, hair_V = _dominant_hair_hsv(face_crop_rgb)

    return {
        "skin_L": skin_L,
        "skin_a": skin_a,
        "skin_b": skin_b,
        "skin_H": skin_H,
        "skin_S": skin_S,
        "skin_V": skin_V,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "hair_H": hair_H,
        "hair_S": hair_S,
        "hair_V": hair_V,
    }


def features_to_dataframe(features: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([features])[FEATURE_COLUMNS]
