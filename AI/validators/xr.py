"""
Chest X-Ray (XR) Dedicated Modality Validator.
Validates frontal chest radiographs and records radiographic geometry metrics.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def validate_chest_xray(gray: np.ndarray) -> tuple[bool, str, dict[str, Any]]:
    """
    Validate radiographic thoracic anatomy for chest radiographs.
    Hard-rejects images lacking a cardiomediastinal density pattern (flag_cxr_cardiac_density);
    all other flags are recorded as non-blocking telemetry for quality auditing.
    """
    h, w = gray.shape[:2]
    aspect_ratio = float(w) / float(h)

    center_roi = gray[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
    center_dark = float(np.mean(center_roi < 40)) if center_roi.size > 0 else 0.0

    bottom_quarter = gray[int(h * 0.75):, :]
    mean_bot = float(np.mean(bottom_quarter)) if bottom_quarter.size > 0 else 0.0

    center_bottom = gray[int(h * 0.70):int(h * 0.95), int(w * 0.35):int(w * 0.65)]
    mean_cl = float(np.mean(center_bottom)) if center_bottom.size > 0 else 0.0

    roi = gray[int(h * 0.15) : int(h * 0.85), int(w * 0.10) : int(w * 0.90)]
    rh, rw = roi.shape[:2]
    if rh < 50 or rw < 50:
        return True, "Valid chest X-ray study (reduced resolution).", {
            "aspect_ratio": round(aspect_ratio, 3),
            "flag_cxr_low_res": 1,
        }

    left_lung = roi[:, : int(rw * 0.38)]
    right_lung = roi[:, int(rw * 0.62) :]
    mediastinum = roi[:, int(rw * 0.40) : int(rw * 0.60)]

    mean_l = float(np.mean(left_lung)) if left_lung.size > 0 else 0.0
    mean_r = float(np.mean(right_lung)) if right_lung.size > 0 else 0.0
    mean_m = float(np.mean(mediastinum)) if mediastinum.size > 0 else 0.0
    std_roi = float(np.std(roi)) if roi.size > 0 else 0.0

    denom = max(mean_l, mean_r, 1.0)
    sym_diff = abs(mean_l - mean_r) / denom
    contrast_spread = abs(mean_m - ((mean_l + mean_r) / 2.0))

    metrics = {
        "modality_validated": "XR",
        "aspect_ratio": round(aspect_ratio, 3),
        "center_dark": round(center_dark, 3),
        "mean_left_lung": round(mean_l, 1),
        "mean_right_lung": round(mean_r, 1),
        "mean_mediastinum": round(mean_m, 1),
        "symmetry_difference": round(sym_diff, 3),
        "bottom_density": round(mean_bot, 1),
        "cardiac_base_density": round(mean_cl, 1),
        "roi_std": round(std_roi, 1),
        "flag_cxr_aspect": int(aspect_ratio < 0.45 or aspect_ratio > 2.2),
        "flag_cxr_airspace": int(center_dark > 0.20),
        "flag_cxr_base_density": int(mean_bot < 80.0),
        "flag_cxr_cardiac_density": int(mean_cl < 90.0),
        "flag_cxr_symmetry": int(sym_diff > 0.40),
        "flag_cxr_contrast": int(std_roi < 18.0 or (contrast_spread < 2.0 and std_roi < 25.0)),
    }

    if metrics.get("flag_cxr_cardiac_density"):
        return False, "Image rejected: Lacks chest radiograph cardiomediastinal density pattern.", metrics

    return True, "Valid chest X-ray anatomy verified.", metrics

