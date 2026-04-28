"""
Recommendation engine from predicted cluster.
"""

from __future__ import annotations

from src.config import RECOMMENDATIONS


def explain_prediction(label: str, features: dict[str, float]) -> str:
    contrast = features.get("contrast", 0.0)
    saturation = features.get("saturation", 0.0)
    skin_h = features.get("skin_H", 0.0)

    undertone = "cool" if 90 <= skin_h <= 140 else "warm" if 10 <= skin_h < 90 else "neutral"
    contrast_txt = "high contrast" if contrast > 55 else "soft contrast" if contrast < 35 else "balanced contrast"
    sat_txt = "vivid saturation" if saturation > 120 else "muted saturation"
    return f"Recommended because your profile matches '{label}' with {contrast_txt}, {undertone} undertones and {sat_txt}."


def get_recommendations(label: str) -> dict[str, object]:
    return RECOMMENDATIONS.get(label, RECOMMENDATIONS["Neutral Balanced"])
