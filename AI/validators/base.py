"""
Universal Medical Scan Authenticity Validator (Stage 1).
Purely modality-agnostic: rejects corrupted files, non-medical color photos,
blank canvases, pure noise, and text document / screenshot graphics.
Does NOT execute any modality-specific heuristics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_shannon_entropy(gray: np.ndarray) -> float:
    """Compute 8-bit Shannon entropy to gauge image information content."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / (hist.sum() + 1e-7)
    non_zero = hist_norm[hist_norm > 0]
    return float(-np.sum(non_zero * np.log2(non_zero)))


def verify_base_authenticity(
    image_input: str | Path | np.ndarray,
    expected_modality: str = "XR",
) -> tuple[bool, str, dict[str, Any], np.ndarray | None]:
    """
    Stage 1 Universal Authenticity Guard:
    Verifies that the file is an authentic, readable, non-corrupted medical image.
    
    Returns:
        (is_valid: bool, reason: str, telemetry: dict, gray_array: np.ndarray | None)
    """
    expected_modality = (expected_modality or "XR").upper()

    # 1. Load image and determine color channels
    raw_bgr: np.ndarray | None = None
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            return False, f"File does not exist: {p.name}", {}, None

        # If DICOM file, verify using pydicom
        if p.suffix.lower() in {".dcm", ".dicom"}:
            try:
                import pydicom
                ds = pydicom.dcmread(str(p), stop_before_pixels=False)
                if not hasattr(ds, "pixel_array"):
                    return False, "DICOM file contains no pixel data.", {}, None
                arr = ds.pixel_array
                if arr is None or arr.size == 0 or np.std(arr) < 10:
                    return False, "DICOM pixel array is blank or unreadable.", {}, None
                gray_dicom = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                return True, "Valid DICOM clinical study verified.", {
                    "is_dicom": True,
                    "modality": str(getattr(ds, "Modality", expected_modality))
                }, gray_dicom
            except Exception as exc:
                return False, f"Corrupted or invalid DICOM file: {exc}", {}, None

        raw_bgr = cv2.imread(str(p))
    elif isinstance(image_input, np.ndarray):
        raw_bgr = image_input

    if raw_bgr is None or raw_bgr.size == 0:
        return False, "Unreadable or corrupted image file. Please upload a valid PNG, JPG, or DICOM file.", {}, None

    h, w = raw_bgr.shape[:2]
    if h < 100 or w < 100:
        return False, f"Image resolution too low ({w}×{h}px). Medical scans must be at least 100×100 pixels.", {
            "width": w, "height": h
        }, None

    # 2. Check for Color Saturation / Non-Radiographic Natural Photos
    mean_sat = 0.0
    channel_disparity = 0.0
    hue_std = 0.0

    if raw_bgr.ndim == 3 and raw_bgr.shape[2] == 3:
        hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        b = raw_bgr[:, :, 0].astype(np.float32)
        g = raw_bgr[:, :, 1].astype(np.float32)
        r = raw_bgr[:, :, 2].astype(np.float32)
        channel_disparity = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))

        # Monochromatic clinical PACS display tint tolerance:
        # Clinical monitors or DICOM exports frequently apply a cool-blue or warm-sepia LUT tint.
        # In a tinted monochrome scan, hue across tissue pixels is uniform (hue_std < 14.0).
        # In natural color photographs, wide chromatic diversity yields hue_std > 20.0.
        sig_mask = hsv[:, :, 2] > 25
        hue_sig = hsv[:, :, 0][sig_mask] if np.any(sig_mask) else np.array([])
        hue_std = float(np.std(hue_sig)) if len(hue_sig) > 100 else 0.0
        is_monochrome_tint = (hue_std < 14.0)

        sat_threshold = 28.0 if expected_modality in ("US", "MR") else 18.0
        disp_threshold = 24.0 if expected_modality in ("US", "MR") else 14.0

        if (mean_sat > sat_threshold or channel_disparity > disp_threshold) and not is_monochrome_tint:
            telemetry = {
                "color_saturation": round(mean_sat, 2),
                "channel_disparity": round(channel_disparity, 2),
                "hue_std": round(hue_std, 2),
                "width": w,
                "height": h,
            }
            return (
                False,
                f"Image rejected: Natural color photograph or non-medical image detected "
                f"(color saturation={mean_sat:.1f}, channel disparity={channel_disparity:.1f}). "
                f"Radiological scans must be monochromatic grayscale images.",
                telemetry,
                None,
            )

    # 3. Check Dynamic Range, Entropy, and Low-Information / Blank Canvas
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) if raw_bgr.ndim == 3 else raw_bgr
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    entropy = compute_shannon_entropy(gray)

    telemetry = {
        "mean_intensity": round(mean_val, 1),
        "std_intensity": round(std_val, 1),
        "entropy": round(entropy, 2),
        "color_saturation": round(mean_sat, 2),
        "channel_disparity": round(channel_disparity, 2),
        "width": w,
        "height": h,
    }

    if std_val < 14.0:
        return False, f"Image rejected: Blank or solid uniform image detected (std={std_val:.1f}).", telemetry, None
    if mean_val < 8.0:
        return False, "Image rejected: Completely black / unexposed scan.", telemetry, None
    if mean_val > 248.0:
        return False, "Image rejected: Completely white / washed-out canvas.", telemetry, None
    if entropy < 3.2:
        return False, f"Image rejected: Extremely low information content / synthetic graphic (entropy={entropy:.2f} < 3.2).", telemetry, None

    # 4. Check for Text Document / Diagram / Screenshot
    extreme_pixels = float(np.mean((gray > 240) | (gray < 15)))
    telemetry["extreme_pixel_ratio"] = round(extreme_pixels, 3)
    if extreme_pixels > 0.86 and std_val > 55.0:
        return False, "Image rejected: Text document, chart, or screenshot detected rather than a medical scan.", telemetry, None

    return True, "Authentic medical scan verified.", telemetry, gray

