# src/face_detection.py
"""
Face detection with MediaPipe (preferred) and OpenCV Haar Cascade fallback.

Design decision: MediaPipe's short-range model is far more accurate than
Haar Cascades on modern portrait photos, but requires an optional dependency.
We gracefully fall back to OpenCV so the pipeline stays runnable even in
minimal environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import (
    FACE_PADDING,
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    HAAR_SCALE_FACTOR,
    MEDIAPIPE_CONFIDENCE,
)
from src.utils import get_logger, resize_image

logger = get_logger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class FaceBox:
    """Bounding box for a detected face in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    detector: str = "unknown"

    @property
    def area(self) -> int:
        return self.w * self.h

    def padded(self, img_h: int, img_w: int, padding: float = FACE_PADDING) -> "FaceBox":
        """
        Return a new FaceBox expanded by ``padding`` fraction on each side,
        clamped to image boundaries.

        Padding is applied relative to the *smaller* of width/height to keep
        the crop proportional regardless of face aspect ratio.
        """
        pad = int(min(self.w, self.h) * padding)
        x = max(0, self.x - pad)
        y = max(0, self.y - pad)
        x2 = min(img_w, self.x + self.w + pad)
        y2 = min(img_h, self.y + self.h + pad)
        return FaceBox(x=x, y=y, w=x2 - x, h=y2 - y,
                       confidence=self.confidence, detector=self.detector)


# ── Detector classes ───────────────────────────────────────────────────────────

class MediaPipeDetector:
    """
    Wraps MediaPipe FaceDetection for single-image inference.

    The model is loaded lazily on first call so import-time cost stays low.
    """

    def __init__(self, min_confidence: float = MEDIAPIPE_CONFIDENCE) -> None:
        self._min_confidence = min_confidence
        self._detector = None  # lazy-loaded
        self._available = self._check_availability()

    # ------------------------------------------------------------------
    @staticmethod
    def _check_availability() -> bool:
        try:
            import mediapipe  # noqa: F401
            return True
        except ImportError:
            logger.warning("mediapipe not installed — will use Haar Cascade fallback.")
            return False

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Initialise the MediaPipe detector (called once)."""
        import mediapipe as mp
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,          # 1 = full-range model (better for portraits)
            min_detection_confidence=self._min_confidence,
        )

    # ------------------------------------------------------------------
    def detect(self, img_rgb: np.ndarray) -> list[FaceBox]:
        """
        Detect faces in an RGB image.

        Parameters
        ----------
        img_rgb : np.ndarray
            RGB image array, shape (H, W, 3).

        Returns
        -------
        list[FaceBox]
            Detected face bounding boxes, sorted by area descending.
        """
        if not self._available:
            return []
        if self._detector is None:
            self._load()

        h, w = img_rgb.shape[:2]
        results = self._detector.process(img_rgb)

        if not results.detections:  # type: ignore[union-attr]
            return []

        boxes: list[FaceBox] = []
        for det in results.detections:  # type: ignore[union-attr]
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(int(bb.width * w), w - x)
            bh = min(int(bb.height * h), h - y)
            score = det.score[0] if det.score else 1.0
            boxes.append(FaceBox(x=x, y=y, w=bw, h=bh,
                                 confidence=score, detector="mediapipe"))

        return sorted(boxes, key=lambda b: b.area, reverse=True)


class HaarCascadeDetector:
    """
    Fallback detector using OpenCV's frontal-face Haar Cascade.

    Less accurate than MediaPipe, especially for non-frontal poses or
    unusual lighting, but has zero extra dependencies.
    """

    def __init__(self) -> None:
        # cv2.data.haarcascades is the canonical path to bundled cascades
        cascade_path = str(
            Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
        )
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            # Try the cv2.data helper attribute (newer OpenCV builds)
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, img_rgb: np.ndarray) -> list[FaceBox]:
        """Detect faces using Haar Cascade; returns boxes sorted by area."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )
        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return []

        boxes = [
            FaceBox(x=int(x), y=int(y), w=int(w), h=int(h),
                    confidence=1.0, detector="haar")
            for x, y, w, h in faces
        ]
        return sorted(boxes, key=lambda b: b.area, reverse=True)


# ── Public API ─────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    High-level face detector that tries MediaPipe first and falls back
    to Haar Cascade automatically.

    Usage
    -----
    >>> detector = FaceDetector()
    >>> box = detector.detect_primary(img_rgb)
    >>> crop = detector.crop_face(img_rgb, box)
    """

    def __init__(self) -> None:
        self._mp = MediaPipeDetector()
        self._haar = HaarCascadeDetector()

    # ------------------------------------------------------------------
    def detect_all(self, img_rgb: np.ndarray) -> list[FaceBox]:
        """
        Return all detected faces (largest first).

        Tries MediaPipe; if no results, falls back to Haar Cascade.
        """
        img_rgb = resize_image(img_rgb, max_dim=1024)  # cap resolution for speed
        boxes = self._mp.detect(img_rgb)
        if boxes:
            logger.info("MediaPipe detected %d face(s).", len(boxes))
            return boxes

        logger.info("MediaPipe found nothing — trying Haar Cascade.")
        boxes = self._haar.detect(img_rgb)
        logger.info("Haar Cascade detected %d face(s).", len(boxes))
        return boxes

    # ------------------------------------------------------------------
    def detect_primary(self, img_rgb: np.ndarray) -> Optional[FaceBox]:
        """
        Return the largest detected face, or ``None`` if none found.
        """
        boxes = self.detect_all(img_rgb)
        return boxes[0] if boxes else None

    # ------------------------------------------------------------------
    def crop_face(
        self,
        img_rgb: np.ndarray,
        box: Optional[FaceBox] = None,
        padding: float = FACE_PADDING,
    ) -> np.ndarray:
        """
        Crop the face region from ``img_rgb``.

        If ``box`` is not provided, detection is run automatically.
        If detection fails, returns the full image (graceful degradation).

        Parameters
        ----------
        img_rgb : np.ndarray
            Source RGB image.
        box : FaceBox | None
            Pre-detected bounding box, or None to auto-detect.
        padding : float
            Fractional padding applied to each side of the bounding box.

        Returns
        -------
        np.ndarray
            Cropped (padded) face region in RGB.
        """
        if box is None:
            box = self.detect_primary(img_rgb)

        if box is None:
            logger.warning(
                "No face detected — returning full image for feature extraction."
            )
            return img_rgb

        h, w = img_rgb.shape[:2]
        padded = box.padded(img_h=h, img_w=w, padding=padding)
        crop = img_rgb[
            padded.y : padded.y + padded.h,
            padded.x : padded.x + padded.w,
        ]
        logger.debug(
            "Face crop: (%d,%d) %dx%d [detector=%s, conf=%.2f]",
            padded.x, padded.y, padded.w, padded.h,
            box.detector, box.confidence,
        )
        return crop
