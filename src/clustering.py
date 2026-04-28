"""
Unsupervised clustering utilities.
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import (
    CLUSTER_LABEL_CANDIDATES,
    CLUSTER_MODEL_PATH,
    KMEANS_MAX_CLUSTERS,
    KMEANS_MAX_ITER,
    KMEANS_MIN_CLUSTERS,
    KMEANS_N_INIT,
    METADATA_PATH,
    RANDOM_SEED,
)
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_k_range(X_scaled: np.ndarray) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    for k in range(KMEANS_MIN_CLUSTERS, KMEANS_MAX_CLUSTERS + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITER)
        labels = model.fit_predict(X_scaled)
        inertia = float(model.inertia_)
        sil = float(silhouette_score(X_scaled, labels)) if len(np.unique(labels)) > 1 else -1.0
        results[k] = {"inertia": inertia, "silhouette": sil}
    return results


def select_best_k(metrics: dict[int, dict[str, float]]) -> int:
    best_k = max(metrics, key=lambda k: metrics[k]["silhouette"])
    return int(best_k)


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int) -> KMeans:
    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_SEED,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    model.fit(X_scaled)
    return model


def semantic_labels_from_centroids(centroids: np.ndarray) -> dict[int, str]:
    brightness_idx = 6
    contrast_idx = 7
    saturation_idx = 8

    order = sorted(
        range(len(centroids)),
        key=lambda i: (centroids[i][contrast_idx] * 0.5 + centroids[i][brightness_idx] * 0.2 + centroids[i][saturation_idx] * 0.3),
    )
    labels = {}
    for rank, cluster_id in enumerate(order):
        labels[cluster_id] = CLUSTER_LABEL_CANDIDATES[rank % len(CLUSTER_LABEL_CANDIDATES)]
    return labels


def save_model_and_metadata(model: KMeans, metrics: dict[int, dict[str, float]], label_map: dict[int, str]) -> None:
    joblib.dump(model, CLUSTER_MODEL_PATH)
    payload = {
        "n_clusters": model.n_clusters,
        "metrics": metrics,
        "cluster_label_map": {str(k): v for k, v in label_map.items()},
    }
    METADATA_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved model to %s and metadata to %s", CLUSTER_MODEL_PATH, METADATA_PATH)


def plot_elbow_curve(metrics: dict[int, dict[str, float]], save_path: str | None = None) -> None:
    ks = sorted(metrics)
    inertias = [metrics[k]["inertia"] for k in ks]
    plt.figure(figsize=(7, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Curve")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
