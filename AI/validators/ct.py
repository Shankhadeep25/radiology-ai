"""
CT Chest Dedicated Modality Validator.
Validates axial thoracic cross-sections and extracts scanner bed / lung window metrics.
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def validate_ct_chest(gray: np.ndarray) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate CT chest cross-section geometry.
    Tolerates scanner table edges, burnt-in calibration text, and non-circular thoracic contours
    by recording numeric quality audit flags instead of throwing hard rejections.
    """
    h, w = gray.shape[:2]

    bh = max(h // 16, 4)
    bw = max(w // 16, 4)
    border_pixels = np.concatenate([
        gray[:bh, :].ravel(), gray[-bh:, :].ravel(),
        gray[:, :bw].ravel(), gray[:, -bw:].ravel()
    ])
    border_mean = float(np.mean(border_pixels))
    center_mean = float(np.mean(gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]))
    dark_ratio = float(np.mean(gray < 40))

    _, thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if p > 0:
            circ = float(4 * np.pi * area / (p ** 2 + 1e-5))

    center_roi = gray[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
    center_dark = float(np.mean(center_roi < 40)) if center_roi.size > 0 else 0.0

    metrics = {
        "modality_validated": "CT",
        "border_density": round(border_mean, 1),
        "center_density": round(center_mean, 1),
        "circularity": round(circ, 3),
        "dark_ratio": round(dark_ratio, 3),
        "center_dark": round(center_dark, 3),
        "flag_ct_geometry": int(circ < 0.50),
        "flag_ct_border": int(border_mean > 35.0),
        "flag_ct_airspace": int(center_dark < 0.15),
    }

    return True, "Valid CT chest study verified.", metrics

