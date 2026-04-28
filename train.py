"""
Train clustering pipeline on images from data/raw.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.clustering import evaluate_k_range, fit_kmeans, save_model_and_metadata, semantic_labels_from_centroids, select_best_k
from src.config import RAW_DIR
from src.face_detection import FaceDetector
from src.feature_extraction import extract_features
from src.preprocessing import preprocess_pipeline
from src.utils import get_logger, load_image_rgb

logger = get_logger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_feature_table(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    detector = FaceDetector()
    rows: list[dict[str, float]] = []
    image_files = [p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if not image_files:
        raise FileNotFoundError(f"No training images found in {raw_dir}")

    for path in image_files:
        try:
            img_rgb = load_image_rgb(path)
            crop = detector.crop_face(img_rgb)
            rows.append(extract_features(crop))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
    if not rows:
        raise RuntimeError("No valid images for training.")
    return pd.DataFrame(rows)


def main() -> None:
    feature_df = collect_feature_table()
    X_scaled, scaler = preprocess_pipeline(feature_df, fit_scaler=True)
    scaler.save()

    metrics = evaluate_k_range(X_scaled)
    best_k = select_best_k(metrics)
    model = fit_kmeans(X_scaled, best_k)
    label_map = semantic_labels_from_centroids(model.cluster_centers_)
    save_model_and_metadata(model, metrics, label_map)

    logger.info("Training complete. Samples=%d, chosen_k=%d", len(feature_df), best_k)


if __name__ == "__main__":
    main()
