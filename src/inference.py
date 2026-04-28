"""
High-level inference pipeline for one portrait.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import CLUSTER_MODEL_PATH, METADATA_PATH
from src.face_detection import FaceDetector
from src.feature_extraction import extract_features, features_to_dataframe
from src.preprocessing import FeatureScaler, preprocess_pipeline
from src.recommender import explain_prediction, get_recommendations
from src.utils import get_logger, load_image_rgb

logger = get_logger(__name__)


def _load_label_map() -> dict[int, str]:
    if not METADATA_PATH.exists():
        return {}
    data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    raw = data.get("cluster_label_map", {})
    return {int(k): str(v) for k, v in raw.items()}


def run_inference(image_path: str | Path) -> dict[str, Any]:
    if not Path(CLUSTER_MODEL_PATH).exists():
        raise FileNotFoundError("Cluster model not found. Run `python train.py` first.")

    img_rgb = load_image_rgb(image_path)
    detector = FaceDetector()
    face_crop = detector.crop_face(img_rgb)

    features = extract_features(face_crop)
    feature_df = features_to_dataframe(features)

    scaler = FeatureScaler.load()
    X_scaled, _ = preprocess_pipeline(feature_df, scaler=scaler, fit_scaler=False)

    model = joblib.load(CLUSTER_MODEL_PATH)
    cluster_id = int(model.predict(X_scaled)[0])
    label_map = _load_label_map()
    label = label_map.get(cluster_id, f"Cluster {cluster_id}")

    recommendations = get_recommendations(label)
    explanation = explain_prediction(label, features)

    return {
        "cluster_id": cluster_id,
        "cluster_label": label,
        "features": features,
        "recommendations": recommendations,
        "explanation": explanation,
        "face_crop_rgb": face_crop,
        "X_scaled": X_scaled,
    }
