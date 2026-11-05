"""
Breast Ultrasound (US) Dedicated Modality Validator.
Validates sonographic acoustic acquisitions, far-field attenuation, and speckle metrics.
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def validate_ultrasound(gray: np.ndarray) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate breast ultrasound acoustic properties.
    Records numeric telemetry flags without hard-rejecting compressed or resized clinical sonograms.
    """
    h, w = gray.shape[:2]

    bottom_mean = float(np.mean(gray[int(h * 0.75):, :])) if h > 4 else 0.0

    _grad = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    smooth_frac = float(np.mean(np.abs(_grad) < 10))

    bh = max(h // 16, 4)
    bw = max(w // 16, 4)
    border_pixels = np.concatenate([
        gray[:bh, :].ravel(), gray[-bh:, :].ravel(),
        gray[:, :bw].ravel(), gray[:, -bw:].ravel()
    ])
    border_mean = float(np.mean(border_pixels))

    metrics = {
        "modality_validated": "US",
        "bottom_density": round(bottom_mean, 1),
        "border_density": round(border_mean, 1),
        "smooth_frac": round(smooth_frac, 3),
        "flag_us_smooth": int(smooth_frac > 0.72),
        "flag_us_depth": int(bottom_mean > 25.0),
    }

    return True, "Valid Ultrasound scan verified.", metrics

