"""Pixel-level redaction primitives (blur / pixelate / solid mask)."""

import cv2
import numpy as np

from .config import normalize_method


def blur(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    h, w = roi.shape[:2]
    kw = max(3, (w // 10) * 2 + 1)
    kh = max(3, (h // 10) * 2 + 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)
    return frame


def pixelate(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // 15), max(1, h // 15)), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame


def solid(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    frame[y1:y2, x1:x2] = 0
    return frame


def redact_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, method: str = "Blur") -> np.ndarray:
    method = normalize_method(method)
    if method == "Pixelate":
        return pixelate(frame, x1, y1, x2, y2)
    if method == "Solid Mask":
        return solid(frame, x1, y1, x2, y2)
    return blur(frame, x1, y1, x2, y2)
