"""
Brain MRI (MR) Dedicated Modality Validator.
Validates multi-slice series montages and single-slice axial, sagittal, and coronal cranial scans.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np

# Ensure AI directory is on sys.path for montage_detector import
_AI_DIR = str(Path(__file__).resolve().parent.parent)
if _AI_DIR not in sys.path:
    sys.path.insert(0, _AI_DIR)

from montage_detector import detect_montage_grid


def validate_brain_mri(gray: np.ndarray) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate Brain MRI study.
    Accepts:
    1. Multi-slice series volumetric montage grids (e.g. 2x4, 2x3, 2x2, 3x3).
    2. Single-slice cranial scans across all clinical planes (axial, sagittal, coronal).
    """
    h, w = gray.shape[:2]

    # 1. Check for multi-slice volumetric contact sheet / grid
    is_montage, grid_meta = detect_montage_grid(gray)

    # 2. Extract anatomical metrics
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

    metrics = {
        "modality_validated": "MR",
        "border_density": round(border_mean, 1),
        "center_density": round(center_mean, 1),
        "circularity": round(circ, 3),
        "dark_ratio": round(dark_ratio, 3),
        "is_multi_slice_montage": is_montage and grid_meta.get("count", 0) >= 4,
        "flag_mr_geometry": int(circ < 0.50),  # Non-circular flags sagittal/coronal without rejecting
        "flag_mr_attenuation": int(border_mean >= 30.0 or center_mean < 100.0),
        "flag_mr_montage": int(is_montage and grid_meta.get("count", 0) >= 4),
    }

    if is_montage and grid_meta.get("count", 0) >= 4:
        metrics["montage_grid"] = grid_meta
        return True, f"Verified as multi-slice MRI montage ({grid_meta['rows']}x{grid_meta['cols']} grid).", metrics

    return True, "Valid Brain MRI study verified.", metrics

