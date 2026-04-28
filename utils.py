# src/utils.py
"""
Shared utility functions used across the pipeline.

Covers logging setup, image I/O, colour conversions,
and small helpers that don't belong to any single module.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.config import LOG_FORMAT, LOG_DATE_FORMAT


# ── Logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently-formatted logger.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int
        Logging level (default INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Image I/O ─────────────────────────────────────────────────────────────────

def load_image_bgr(path: str | Path) -> np.ndarray:
    """
    Load an image from disk in BGR format (OpenCV convention).

    Parameters
    ----------
    path : str | Path
        Filesystem path to the image.

    Returns
    -------
    np.ndarray
        BGR image array of shape (H, W, 3).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If OpenCV fails to decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not decode: {path}")
    return img


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    return cv2.cvtColor(load_image_bgr(path), cv2.COLOR_BGR2RGB)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to a BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_image(
    img: np.ndarray,
    max_dim: int = 800,
) -> np.ndarray:
    """
    Proportionally resize an image so its largest dimension ≤ ``max_dim``.

    Keeping images at a reasonable size speeds up every downstream step
    without meaningfully impacting color analysis accuracy.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── Colour Conversions ─────────────────────────────────────────────────────────

def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to CIE LAB colour space."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV colour space."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert a CSS hex string (e.g. '#F5CBA7') to an (R, G, B) tuple.
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert (R, G, B) integers to a '#RRGGBB' hex string."""
    return f"#{r:02X}{g:02X}{b:02X}"


# ── Array Helpers ──────────────────────────────────────────────────────────────

def safe_mean(arr: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """
    Return the mean of ``arr``, defaulting to 0.0 if ``arr`` is empty or NaN.
    """
    if arr is None or arr.size == 0:
        return 0.0
    result = np.nanmean(arr, axis=axis)
    return result


def crop_region(
    img: np.ndarray,
    top_frac: float,
    bottom_frac: float,
    left_frac: float = 0.0,
    right_frac: float = 1.0,
) -> np.ndarray:
    """
    Crop a rectangular region from an image using fractional coordinates.

    Parameters
    ----------
    img : np.ndarray
        Source image.
    top_frac, bottom_frac : float
        Vertical crop boundaries as fraction of image height [0, 1].
    left_frac, right_frac : float
        Horizontal crop boundaries as fraction of image width [0, 1].
    """
    h, w = img.shape[:2]
    y0 = int(h * top_frac)
    y1 = int(h * bottom_frac)
    x0 = int(w * left_frac)
    x1 = int(w * right_frac)
    return img[y0:y1, x0:x1]
