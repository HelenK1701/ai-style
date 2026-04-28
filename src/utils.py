"""
Shared utility helpers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.config import LOG_DATE_FORMAT, LOG_FORMAT


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_image_bgr(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def load_image_rgb(path: str | Path) -> np.ndarray:
    return cv2.cvtColor(load_image_bgr(path), cv2.COLOR_BGR2RGB)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_image(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)


def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def safe_mean(arr: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    if arr is None or arr.size == 0:
        return 0.0
    return np.nanmean(arr, axis=axis)


def crop_region(
    img: np.ndarray,
    top_frac: float,
    bottom_frac: float,
    left_frac: float = 0.0,
    right_frac: float = 1.0,
) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = int(h * top_frac)
    y1 = int(h * bottom_frac)
    x0 = int(w * left_frac)
    x1 = int(w * right_frac)
    return img[y0:y1, x0:x1]
