# src/feature_extraction.py
"""
Feature extraction from a cropped face image.

Extracts a fixed-length numerical feature vector capturing:
  - Skin tone in CIE LAB + HSV colour spaces
  - Overall image brightness, contrast, and saturation
  - Dominant hair colour using KMeans on the top region of the crop

Design notes
------------
- CIE LAB is used for skin tone because it is perceptually uniform:
  equal Euclidean distances correspond to equal perceived colour differences.
  This makes downstream clustering more meaningful than using raw RGB or HSV.
- Hair colour is extracted from the *top 25 %* of the face crop, which
  reliably corresponds to the forehead / hairline area in a portrait.
- KMeans (k=3) on the hair region finds the dominant colour cluster,
  ignoring minority clusters (skin bleeding into the region, etc.).
"""

from __future__ import annotations

import logging
from typing import Optional

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
from src.utils import (
    bgr_to_hsv,
    bgr_to_lab,
    crop_region,
    get_logger,
    safe_mean,
)

logger = get_logger(__name__)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _mean_lab_skin(face_crop_rgb: np.ndarray) -> tuple[float, float, float]:
    """
    Average CIE LAB values for the central 'skin zone' of the face crop.

    We use the vertical band [SKIN_REGION_FRACTION_TOP, SKIN_REGION_FRACTION_BOTTOM]
    which corresponds roughly to the forehead-to-chin region, excluding the
    hairline and chin/neck boundary which carry less representative skin signal.
    """
    skin_zone = crop_region(
        face_crop_rgb,
        top_frac=SKIN_REGION_FRACTION_TOP,
        bottom_frac=SKIN_REGION_FRACTION_BOTTOM,
    )
    # OpenCV LAB conversion expects BGR input
    bgr = cv2.cvtColor(skin_zone, cv2.COLOR_RGB2BGR)
    lab = bgr_to_lab(bgr)
    L = float(safe_mean(lab[:, :, 0]))
    a = float(safe_mean(lab[:, :, 1]))
    b_val = float(safe_mean(lab[:, :, 2]))
    return L, a, b_val


def _mean_hsv_skin(face_crop_rgb: np.ndarray) -> tuple[float, float, float]:
    """
    Average HSV values for the central skin zone.

    HSV captures saturation and value separately from hue, which is useful
    for distinguishing warm/cool undertones (hue) from pale/deep skin (value).
    """
    skin_zone = crop_region(
        face_crop_rgb,
        top_frac=SKIN_REGION_FRACTION_TOP,
        bottom_frac=SKIN_REGION_FRACTION_BOTTOM,
    )
    bgr = cv2.cvtColor(skin_zone, cv2.COLOR_RGB2BGR)
    hsv = bgr_to_hsv(bgr)
    H = float(safe_mean(hsv[:, :, 0]))
    S = float(safe_mean(hsv[:, :, 1]))
    V = float(safe_mean(hsv[:, :, 2]))
    return H, S, V


def _brightness_contrast(face_crop_rgb: np.ndarray) -> tuple[float, float]:
    """
    Global brightness (mean grayscale) and contrast (std grayscale).

    Contrast defined as pixel std deviation is a common proxy in colour
    analysis: high-contrast individuals have greater light/dark variation
    (e.g., dark hair + light skin), while low-contrast individuals are
    more monochromatic overall.
    """
    gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return brightness, contrast


def _mean_saturation(face_crop_rgb: np.ndarray) -> float:
    """
    Mean saturation across the full face crop (HSV S channel).

    People with vivid, strongly-coloured features score higher here.
    """
    bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = bgr_to_hsv(bgr)
    return float(safe_mean(hsv[:, :, 1]))


def _dominant_hair_hsv(face_crop_rgb: np.ndarray) -> tuple[float, float, float]:
    """
    Dominant hair colour in HSV, derived from the top fraction of the crop.

    Algorithm
    ---------
    1. Slice the top HAIR_REGION_FRACTION of the image (hairline zone).
    2. Flatten pixels into a (N, 3) array in RGB.
    3. Run KMeans(k=HAIR_KMEANS_CLUSTERS) to find colour clusters.
    4. Return HSV of the cluster with the most members (dominant colour).

    This is more robust than a simple mean because it ignores minor
    colour contamination (sky, clothing, skin bleeding into the frame).
    """
    hair_zone = crop_region(face_crop_rgb, top_frac=0.0,
                             bottom_frac=HAIR_REGION_FRACTION)

    if hair_zone.size == 0:
        logger.warning("Hair zone is empty — returning zeros for hair features.")
        return 0.0, 0.0, 0.0

    pixels = hair_zone.reshape(-1, 3).astype(np.float32)

    k = min(HAIR_KMEANS_CLUSTERS, len(pixels))  # guard against tiny crops
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(pixels)

    # Pick the cluster with the most pixels
    counts = np.bincount(labels)
    dominant_rgb = kmeans.cluster_centers_[np.argmax(counts)].astype(np.uint8)

    # Convert dominant colour to HSV
    pixel_bgr = dominant_rgb[::-1].reshape(1, 1, 3)  # RGB → BGR
    hsv_pixel = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
    H = float(hsv_pixel[0, 0, 0])
    S = float(hsv_pixel[0, 0, 1])
    V = float(hsv_pixel[0, 0, 2])
    return H, S, V


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_features(face_crop_rgb: np.ndarray) -> dict[str, float]:
    """
    Extract a full feature vector from a cropped face image.

    Parameters
    ----------
    face_crop_rgb : np.ndarray
        RGB image array of the cropped face region.

    Returns
    -------
    dict[str, float]
        Dictionary mapping feature names to float values.
        Keys match FEATURE_COLUMNS in config.py.

    Raises
    ------
    ValueError
        If the crop is too small to reliably extract features.
    """
    h, w = face_crop_rgb.shape[:2]
    if h < 20 or w < 20:
        raise ValueError(
            f"Face crop too small ({w}x{h}) for feature extraction. "
            "Ensure a clear frontal portrait is supplied."
        )

    logger.debug("Extracting features from crop %dx%d", w, h)

    skin_L, skin_a, skin_b = _mean_lab_skin(face_crop_rgb)
    skin_H, skin_S, skin_V = _mean_hsv_skin(face_crop_rgb)
    brightness, contrast = _brightness_contrast(face_crop_rgb)
    saturation = _mean_saturation(face_crop_rgb)
    hair_H, hair_S, hair_V = _dominant_hair_hsv(face_crop_rgb)

    features = {
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

    logger.info("Features extracted: %s", {k: f"{v:.2f}" for k, v in features.items()})
    return features


def features_to_dataframe(features: dict[str, float]) -> pd.DataFrame:
    """
    Wrap a single feature dict into a one-row DataFrame with the canonical
    column ordering expected by the preprocessing / clustering pipeline.
    """
    df = pd.DataFrame([features])[FEATURE_COLUMNS]
    return df


def batch_extract(
    crops: list[np.ndarray],
    ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Extract features from a list of face crops and return a DataFrame.

    Parameters
    ----------
    crops : list[np.ndarray]
        RGB face crop arrays.
    ids : list[str] | None
        Optional row identifiers. Defaults to integer indices.

    Returns
    -------
    pd.DataFrame
        Shape (N, len(FEATURE_COLUMNS)), indexed by ``ids``.
    """
    records = []
    failed = 0
    for i, crop in enumerate(crops):
        try:
            records.append(extract_features(crop))
        except ValueError as exc:
            logger.warning("Skipping crop %d: %s", i, exc)
            failed += 1

    if failed:
        logger.warning("%d/%d crops failed feature extraction.", failed, len(crops))

    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    if ids is not None:
        df.index = ids[: len(df)]
    return df
