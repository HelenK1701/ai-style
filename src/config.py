"""
Centralized configuration for the Personal Color & Style Recommender.
"""

from pathlib import Path
from typing import Final

ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
SAMPLE_DIR: Final[Path] = DATA_DIR / "sample_images"
MODELS_DIR: Final[Path] = ROOT_DIR / "models"
NOTEBOOKS_DIR: Final[Path] = ROOT_DIR / "notebooks"

for _d in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

SCALER_PATH: Final[Path] = MODELS_DIR / "scaler.pkl"
CLUSTER_MODEL_PATH: Final[Path] = MODELS_DIR / "cluster_model.pkl"
METADATA_PATH: Final[Path] = MODELS_DIR / "metadata.json"

RANDOM_SEED: Final[int] = 42
MEDIAPIPE_CONFIDENCE: Final[float] = 0.5
HAAR_SCALE_FACTOR: Final[float] = 1.1
HAAR_MIN_NEIGHBORS: Final[int] = 5
HAAR_MIN_SIZE: Final[tuple[int, int]] = (80, 80)
FACE_PADDING: Final[float] = 0.30

HAIR_KMEANS_CLUSTERS: Final[int] = 3
HAIR_REGION_FRACTION: Final[float] = 0.25
SKIN_REGION_FRACTION_TOP: Final[float] = 0.35
SKIN_REGION_FRACTION_BOTTOM: Final[float] = 0.75

KMEANS_MIN_CLUSTERS: Final[int] = 3
KMEANS_MAX_CLUSTERS: Final[int] = 6
KMEANS_N_INIT: Final[int] = 20
KMEANS_MAX_ITER: Final[int] = 500

FEATURE_COLUMNS: Final[list[str]] = [
    "skin_L",
    "skin_a",
    "skin_b",
    "skin_H",
    "skin_S",
    "skin_V",
    "brightness",
    "contrast",
    "saturation",
    "hair_H",
    "hair_S",
    "hair_V",
]

CLUSTER_LABEL_CANDIDATES: Final[list[str]] = [
    "Warm Soft",
    "Cool High Contrast",
    "Neutral Balanced",
    "Deep Rich",
    "Light Luminous",
    "Muted Earthy",
]

RECOMMENDATIONS: Final[dict[str, dict[str, object]]] = {
    "Warm Soft": {
        "clothing_palette": ["#F5CBA7", "#FAD7A0", "#A9CCE3", "#ABEBC6", "#F9E79F", "#D7BDE2"],
        "hair_colors": ["Strawberry Blonde", "Golden Honey", "Warm Caramel", "Copper Highlights"],
        "jewelry": ["Rose Gold", "Yellow Gold", "Warm Copper"],
        "makeup_tones": ["Peachy blush", "Coral lip", "Bronze highlighter"],
        "style_notes": "Earthy warm colors and soft textures suit your profile.",
    },
    "Cool High Contrast": {
        "clothing_palette": ["#1A1A2E", "#C0C0C0", "#FDFEFE", "#E74C3C", "#2C3E50", "#85C1E9"],
        "hair_colors": ["Jet Black", "Cool Ash Brown", "Platinum Blonde", "Espresso"],
        "jewelry": ["Silver", "White Gold", "Platinum"],
        "makeup_tones": ["Berry lip", "Cool blush", "Silver highlight"],
        "style_notes": "Bold contrast and clear cool shades work best.",
    },
    "Neutral Balanced": {
        "clothing_palette": ["#BDC3C7", "#D5D8DC", "#A8D8B9", "#F0D9B5", "#AED6F1", "#E8DAEF"],
        "hair_colors": ["Ash Brown", "Neutral Blonde", "Soft Chestnut", "Natural Brunette"],
        "jewelry": ["Mixed metals", "Rose Gold", "Brushed Silver"],
        "makeup_tones": ["Nude lip", "Taupe eyeshadow", "Champagne highlight"],
        "style_notes": "Balanced tones make both warm and cool muted palettes wearable.",
    },
    "Deep Rich": {
        "clothing_palette": ["#6C3483", "#1B4F72", "#78281F", "#1E8449", "#784212", "#F39C12"],
        "hair_colors": ["Blue Black", "Dark Espresso", "Rich Mahogany", "Deep Auburn"],
        "jewelry": ["Yellow Gold", "Dark Bronze", "Oxidized Silver"],
        "makeup_tones": ["Deep plum lip", "Smoky eye", "Gold highlight"],
        "style_notes": "Jewel tones and rich deep shades amplify your natural contrast.",
    },
    "Light Luminous": {
        "clothing_palette": ["#FDFEFE", "#FDEBD0", "#D6EAF8", "#D5F5E3", "#FDEDEC", "#FEF9E7"],
        "hair_colors": ["Platinum", "Champagne Blonde", "Light Golden Blonde", "Strawberry Highlights"],
        "jewelry": ["Delicate Gold", "Pearl", "Rose Gold"],
        "makeup_tones": ["Sheer blush", "Soft peach lip", "Iridescent highlight"],
        "style_notes": "Airy, light and soft tones keep your complexion fresh.",
    },
    "Muted Earthy": {
        "clothing_palette": ["#C9A96E", "#8D9E87", "#B07A5A", "#7D8471", "#C4A882", "#A0785A"],
        "hair_colors": ["Warm Medium Brown", "Muted Copper", "Olive Brown", "Natural Brunette"],
        "jewelry": ["Matte Gold", "Brass", "Tortoiseshell"],
        "makeup_tones": ["Terracotta blush", "Warm brown lip", "Bronze eyeshadow"],
        "style_notes": "Natural earthy tones and low-chroma shades suit you best.",
    },
}

LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
