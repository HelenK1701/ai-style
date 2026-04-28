"""
Face detection with MediaPipe and OpenCV fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import FACE_PADDING, HAAR_MIN_NEIGHBORS, HAAR_MIN_SIZE, HAAR_SCALE_FACTOR, MEDIAPIPE_CONFIDENCE
from src.utils import get_logger, resize_image

logger = get_logger(__name__)


@dataclass
class FaceBox:
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
        pad = int(min(self.w, self.h) * padding)
        x = max(0, self.x - pad)
        y = max(0, self.y - pad)
        x2 = min(img_w, self.x + self.w + pad)
        y2 = min(img_h, self.y + self.h + pad)
        return FaceBox(x=x, y=y, w=x2 - x, h=y2 - y, confidence=self.confidence, detector=self.detector)


class MediaPipeDetector:
    def __init__(self, min_confidence: float = MEDIAPIPE_CONFIDENCE) -> None:
        self._min_confidence = min_confidence
        self._detector = None
        try:
            import mediapipe as mp

            self._mp = mp
            self._available = True
        except ImportError:
            self._available = False
            self._mp = None
            logger.warning("mediapipe is unavailable, fallback to Haar Cascade.")

    def _load(self) -> None:
        if self._mp is None:
            return
        self._detector = self._mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=self._min_confidence
        )

    def detect(self, img_rgb: np.ndarray) -> list[FaceBox]:
        if not self._available:
            return []
        if self._detector is None:
            self._load()
        if self._detector is None:
            return []
        h, w = img_rgb.shape[:2]
        results = self._detector.process(img_rgb)
        detections = getattr(results, "detections", None)
        if not detections:
            return []

        boxes: list[FaceBox] = []
        for det in detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(int(bb.width * w), w - x)
            bh = min(int(bb.height * h), h - y)
            score = det.score[0] if det.score else 1.0
            boxes.append(FaceBox(x, y, bw, bh, score, "mediapipe"))
        return sorted(boxes, key=lambda box: box.area, reverse=True)


class HaarCascadeDetector:
    def __init__(self) -> None:
        cascade_path = str(Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml")
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, img_rgb: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=HAAR_SCALE_FACTOR, minNeighbors=HAAR_MIN_NEIGHBORS, minSize=HAAR_MIN_SIZE
        )
        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return []
        boxes = [FaceBox(int(x), int(y), int(w), int(h), 1.0, "haar") for x, y, w, h in faces]
        return sorted(boxes, key=lambda box: box.area, reverse=True)


class FaceDetector:
    def __init__(self) -> None:
        self._mp = MediaPipeDetector()
        self._haar = HaarCascadeDetector()

    def detect_primary(self, img_rgb: np.ndarray) -> FaceBox | None:
        img_rgb = resize_image(img_rgb, max_dim=1024)
        boxes = self._mp.detect(img_rgb)
        if not boxes:
            boxes = self._haar.detect(img_rgb)
        return boxes[0] if boxes else None

    def crop_face(self, img_rgb: np.ndarray, box: FaceBox | None = None, padding: float = FACE_PADDING) -> np.ndarray:
        box = box or self.detect_primary(img_rgb)
        if box is None:
            logger.warning("No face detected, returning full image.")
            return img_rgb
        h, w = img_rgb.shape[:2]
        p = box.padded(h, w, padding)
        return img_rgb[p.y : p.y + p.h, p.x : p.x + p.w]
