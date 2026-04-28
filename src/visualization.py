"""
Visualization helpers for palettes and feature charts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def create_color_swatch(colors: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 1.6))
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, max(1, len(colors)))
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    return fig


def create_feature_radar(features: dict[str, float]) -> plt.Figure:
    keys = ["skin_L", "skin_S", "skin_V", "brightness", "contrast", "saturation", "hair_S", "hair_V"]
    values = [float(features.get(k, 0.0)) for k in keys]
    max_val = max(values) if max(values) > 0 else 1.0
    norm = [v / max_val for v in values]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    norm += norm[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, norm, linewidth=2)
    ax.fill(angles, norm, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_yticklabels([])
    fig.tight_layout()
    return fig


def create_cluster_projection(X_scaled: np.ndarray, labels: np.ndarray) -> px.scatter:
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_scaled)
    df = pd.DataFrame({"pc1": proj[:, 0], "pc2": proj[:, 1], "cluster": labels.astype(str)})
    return px.scatter(df, x="pc1", y="pc2", color="cluster", title="Cluster Projection (PCA)")
