# src/config.py
"""
Centralized configuration for the Personal Color & Style Recommender.

All magic numbers, paths, and hyperparameters are defined here
to keep the rest of the codebase clean and easy to tune.
"""

from pathlib import Path
from typing import Final

# ── Project Paths ──────────────────────────────────────────────────────────────
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
SAMPLE_DIR: Final[Path] = DATA_DIR / "sample_images"
MODELS_DIR: Final[Path] = ROOT_DIR / "models"
NOTEBOOKS_DIR: Final[Path] = ROOT_DIR / "notebooks"

# Ensure critical directories exist at import time
for _d in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Model Artifacts ────────────────────────────────────────────────────────────
SCALER_PATH: Final[Path] = MODELS_DIR / "scaler.pkl"
CLUSTER_MODEL_PATH: Final[Path] = MODELS_DIR / "cluster_model.pkl"
METADATA_PATH: Final[Path] = MODELS_DIR / "metadata.json"

# ── Random Seed (reproducibility) ─────────────────────────────────────────────
RANDOM_SEED: Final[int] = 42

# ── Face Detection ─────────────────────────────────────────────────────────────
# MediaPipe confidence threshold — lower catches more faces but may false-positive
MEDIAPIPE_CONFIDENCE: Final[float] = 0.5
# Haar Cascade fallback scale / neighbors
HAAR_SCALE_FACTOR: Final[float] = 1.1
HAAR_MIN_NEIGHBORS: Final[int] = 5
HAAR_MIN_SIZE: Final[tuple[int, int]] = (80, 80)

# How much padding (fraction of face size) to add when cropping
FACE_PADDING: Final[float] = 0.30

# ── Feature Extraction ─────────────────────────────────────────────────────────
# Number of KMeans clusters used to find the dominant hair color
HAIR_KMEANS_CLUSTERS: Final[int] = 3
# Fraction of the face crop that is used as the "hair region" (top portion)
HAIR_REGION_FRACTION: Final[float] = 0.25
# Fraction of the face crop used as the "skin region" (center)
SKIN_REGION_FRACTION_TOP: Final[float] = 0.35
SKIN_REGION_FRACTION_BOTTOM: Final[float] = 0.75

# ── Clustering (unsupervised learning) ────────────────────────────────────────
KMEANS_MIN_CLUSTERS: Final[int] = 3
KMEANS_MAX_CLUSTERS: Final[int] = 6
# joblib cache: number of different initialisations per k
KMEANS_N_INIT: Final[int] = 20
KMEANS_MAX_ITER: Final[int] = 500

# ── Feature Column Names ───────────────────────────────────────────────────────
FEATURE_COLUMNS: Final[list[str]] = [
    "skin_L",        # LAB lightness of skin
    "skin_a",        # LAB a* (green–red axis)
    "skin_b",        # LAB b* (blue–yellow axis)
    "skin_H",        # HSV hue of skin
    "skin_S",        # HSV saturation
    "skin_V",        # HSV value (brightness)
    "brightness",    # mean pixel brightness (grayscale)
    "contrast",      # pixel standard deviation (grayscale)
    "saturation",    # mean saturation across full face crop
    "hair_H",        # dominant hair color hue
    "hair_S",        # dominant hair color saturation
    "hair_V",        # dominant hair color value
]

# ── Cluster Label Mapping ──────────────────────────────────────────────────────
# Semantic labels assigned after fitting — indexed 0..N-1
# The actual assignment is done dynamically in clustering.py
CLUSTER_LABEL_CANDIDATES: Final[list[str]] = [
    "Warm Soft",
    "Cool High Contrast",
    "Neutral Balanced",
    "Deep Rich",
    "Light Luminous",
    "Muted Earthy",
]

# ── Recommendation Database ────────────────────────────────────────────────────
# Each cluster label maps to a full recommendation bundle.
RECOMMENDATIONS: Final[dict[str, dict]] = {
    "Warm Soft": {
        "clothing_palette": [
            "#F5CBA7",  # Peach
            "#FAD7A0",  # Warm cream
            "#A9CCE3",  # Soft sky blue
            "#ABEBC6",  # Mint green
            "#F9E79F",  # Butter yellow
            "#D7BDE2",  # Lavender blush
        ],
        "clothing_palette_names": [
            "Peach", "Warm Cream", "Soft Sky Blue",
            "Mint Green", "Butter Yellow", "Lavender Blush",
        ],
        "hair_colors": [
            "Strawberry Blonde",
            "Golden Honey",
            "Warm Caramel",
            "Copper Highlights",
        ],
        "jewelry": ["Rose Gold", "Yellow Gold", "Warm Copper"],
        "makeup_tones": [
            "Peachy blush", "Coral lip", "Bronze highlighter",
        ],
        "style_notes": (
            "Your warm, soft palette shines in earthy tones and pastels. "
            "Embrace flowy fabrics, bohemian silhouettes, and natural textures "
            "like linen and suede. Avoid stark cool whites or icy blues."
        ),
        "season": "Spring / Soft Autumn",
    },

    "Cool High Contrast": {
        "clothing_palette": [
            "#1A1A2E",  # Navy
            "#C0C0C0",  # Silver
            "#FDFEFE",  # Pure white
            "#E74C3C",  # True red
            "#2C3E50",  # Charcoal
            "#85C1E9",  # Ice blue
        ],
        "clothing_palette_names": [
            "Navy", "Silver", "Pure White",
            "True Red", "Charcoal", "Ice Blue",
        ],
        "hair_colors": [
            "Jet Black",
            "Cool Ash Brown",
            "Platinum Blonde",
            "Espresso",
        ],
        "jewelry": ["Silver", "White Gold", "Platinum"],
        "makeup_tones": [
            "Berry lip", "Cool-toned blush", "Silver highlighter",
        ],
        "style_notes": (
            "Your high-contrast cool palette commands attention. You look "
            "exceptional in bold, graphic pieces — structured silhouettes, "
            "monochrome looks, and high-shine fabrics. Avoid muddy earth tones."
        ),
        "season": "True Winter",
    },

    "Neutral Balanced": {
        "clothing_palette": [
            "#BDC3C7",  # Stone grey
            "#D5D8DC",  # Soft greige
            "#A8D8B9",  # Sage green
            "#F0D9B5",  # Warm sand
            "#AED6F1",  # Dusty blue
            "#E8DAEF",  # Soft mauve
        ],
        "clothing_palette_names": [
            "Stone Grey", "Soft Greige", "Sage Green",
            "Warm Sand", "Dusty Blue", "Soft Mauve",
        ],
        "hair_colors": [
            "Ash Brown",
            "Neutral Blonde",
            "Soft Chestnut",
            "Balayage (warm-to-cool)",
        ],
        "jewelry": ["Mixed metals", "Rose gold", "Brushed silver"],
        "makeup_tones": [
            "Nude lip", "Taupe eyeshadow", "Champagne highlight",
        ],
        "style_notes": (
            "Your balanced neutral profile is incredibly versatile. You can "
            "bridge warm and cool tones effortlessly. Invest in quality basics "
            "in greige, sage, and dusty blue — they'll form a timeless wardrobe."
        ),
        "season": "True Summer / Soft Summer",
    },

    "Deep Rich": {
        "clothing_palette": [
            "#6C3483",  # Deep plum
            "#1B4F72",  # Midnight blue
            "#78281F",  # Burgundy
            "#1E8449",  # Forest green
            "#784212",  # Rich chocolate
            "#F39C12",  # Saffron gold
        ],
        "clothing_palette_names": [
            "Deep Plum", "Midnight Blue", "Burgundy",
            "Forest Green", "Rich Chocolate", "Saffron Gold",
        ],
        "hair_colors": [
            "Blue-Black",
            "Dark Espresso",
            "Rich Mahogany",
            "Deep Auburn",
        ],
        "jewelry": ["Yellow Gold", "Dark Bronze", "Oxidized Silver"],
        "makeup_tones": [
            "Deep plum lip", "Smoky eye", "Gold highlighter",
        ],
        "style_notes": (
            "Your deep, rich coloring is dramatic and striking. You are made "
            "for jewel tones, luxurious fabrics like velvet and silk, and bold "
            "statement pieces. Avoid washed-out pastels — they can drain you."
        ),
        "season": "Deep Autumn / Dark Winter",
    },

    "Light Luminous": {
        "clothing_palette": [
            "#FDFEFE",  # Bright white
            "#FDEBD0",  # Blush ivory
            "#D6EAF8",  # Baby blue
            "#D5F5E3",  # Pale mint
            "#FDEDEC",  # Blush pink
            "#FEF9E7",  # Cream yellow
        ],
        "clothing_palette_names": [
            "Bright White", "Blush Ivory", "Baby Blue",
            "Pale Mint", "Blush Pink", "Cream Yellow",
        ],
        "hair_colors": [
            "Platinum",
            "Champagne Blonde",
            "Light Golden Blonde",
            "Strawberry Highlights",
        ],
        "jewelry": ["Delicate gold", "Pearl", "Rose gold"],
        "makeup_tones": [
            "Sheer blush", "Soft peach lip", "Iridescent highlight",
        ],
        "style_notes": (
            "Your light, luminous coloring looks best in soft, delicate tones. "
            "Embrace airy fabrics, light layering, and romantic details like "
            "lace and chiffon. Heavy darks or very bright neons may overwhelm."
        ),
        "season": "Light Spring / Light Summer",
    },

    "Muted Earthy": {
        "clothing_palette": [
            "#C9A96E",  # Camel
            "#8D9E87",  # Muted sage
            "#B07A5A",  # Terracotta
            "#7D8471",  # Olive drab
            "#C4A882",  # Warm taupe
            "#A0785A",  # Rust
        ],
        "clothing_palette_names": [
            "Camel", "Muted Sage", "Terracotta",
            "Olive Drab", "Warm Taupe", "Rust",
        ],
        "hair_colors": [
            "Warm Medium Brown",
            "Muted Copper",
            "Olive Brown",
            "Natural Brunette",
        ],
        "jewelry": ["Matte gold", "Brass", "Tortoiseshell"],
        "makeup_tones": [
            "Terracotta blush", "Warm brown lip", "Bronze eyeshadow",
        ],
        "style_notes": (
            "Your muted, earthy tones are grounded and effortlessly stylish. "
            "You shine in natural materials — linen, leather, suede — in toned-"
            "down, complex colors. Steer clear of neon or icy cool shades."
        ),
        "season": "Soft Autumn / Muted Autumn",
    },
}

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
