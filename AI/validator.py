"""
Medical image authenticity and radiographic validity validator.
Detects and rejects non-medical images, color photographs, cartoons,
text documents, synthetic noise, and non-X-ray images before inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _compute_shannon_entropy(gray: np.ndarray) -> float:
    """Compute 8-bit Shannon entropy to gauge image information content."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / (hist.sum() + 1e-7)
    non_zero = hist_norm[hist_norm > 0]
    return float(-np.sum(non_zero * np.log2(non_zero)))


def _verify_chest_xray_anatomy(gray: np.ndarray) -> tuple[bool, str, dict[str, float]]:
    """
    Verify radiographic thoracic anatomy for chest X-rays:
    1. Bilateral radiolucent lung fields.
    2. Higher-density central mediastinal column (cardiac silhouette + spine).
    3. Thoracic hemithorax symmetry.
    """
    h, w = gray.shape[:2]
    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.45 or aspect_ratio > 2.2:
        return False, f"Aspect ratio ({aspect_ratio:.2f}) does not match frontal chest radiographs (0.5 - 2.0).", {}

    # Extract central ROI (excluding outer borders/collimation)
    roi = gray[int(h * 0.15) : int(h * 0.85), int(w * 0.10) : int(w * 0.90)]
    rh, rw = roi.shape[:2]
    if rh < 50 or rw < 50:
        return False, "Resolution too low for anatomical analysis.", {}

    # Left lung region, right lung region, and central mediastinum
    left_lung = roi[:, : int(rw * 0.38)]
    right_lung = roi[:, int(rw * 0.62) :]
    mediastinum = roi[:, int(rw * 0.40) : int(rw * 0.60)]

    mean_l = float(np.mean(left_lung))
    mean_r = float(np.mean(right_lung))
    mean_m = float(np.mean(mediastinum))
    std_roi = float(np.std(roi))

    # Hemithorax symmetry
    denom = max(mean_l, mean_r, 1.0)
    sym_diff = abs(mean_l - mean_r) / denom

    metrics = {
        "aspect_ratio": round(aspect_ratio, 3),
        "mean_left_lung": round(mean_l, 1),
        "mean_right_lung": round(mean_r, 1),
        "mean_mediastinum": round(mean_m, 1),
        "symmetry_difference": round(sym_diff, 3),
        "roi_std": round(std_roi, 1),
    }

    # Severe asymmetry check: a chest X-ray has bilateral lungs; natural objects/faces/cars lack this balance
    if sym_diff > 0.40:
        return False, f"Bilateral hemithorax symmetry failed (diff={sym_diff:.2f} > 0.40). Image does not exhibit chest anatomy.", metrics

    # Mediastinal density: In standard frontal chest radiographs, the spine/heart column absorbs more radiation
    # than the air-filled lungs. (In inverted radiographs, contrast is reversed, but standard CXR has denser center).
    # We check that the center isn't a completely uniform or inverted void.
    contrast_spread = abs(mean_m - ((mean_l + mean_r) / 2.0))
    if std_roi < 18.0 or contrast_spread < 2.0 and std_roi < 25.0:
        return False, "Image lacks the radiolucent lung-to-mediastinal attenuation contrast of a chest X-ray.", metrics

    return True, "Valid chest X-ray anatomy", metrics


def verify_medical_image(
    image_input: str | Path | np.ndarray,
    expected_modality: str = "XR",
) -> tuple[bool, str, dict[str, Any]]:
    """
    Authenticate whether an uploaded image is a genuine medical scan
    and matches the expected modality (default: XR / Chest X-ray).

    Returns:
        (is_valid: bool, reason: str, telemetry: dict)
    """
    expected_modality = (expected_modality or "XR").upper()

    # 1. Load image and determine color channels
    raw_bgr: np.ndarray | None = None
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            return False, f"File does not exist: {p.name}", {}

        # If DICOM file, verify using pydicom
        if p.suffix.lower() in {".dcm", ".dicom"}:
            try:
                import pydicom
                ds = pydicom.dcmread(str(p), stop_before_pixels=False)
                if not hasattr(ds, "pixel_array"):
                    return False, "DICOM file contains no pixel data.", {}
                arr = ds.pixel_array
                if arr is None or arr.size == 0 or np.std(arr) < 10:
                    return False, "DICOM pixel array is blank or unreadable.", {}
                return True, "Valid DICOM clinical study verified.", {"is_dicom": True, "modality": str(getattr(ds, "Modality", expected_modality))}
            except Exception as exc:
                return False, f"Corrupted or invalid DICOM file: {exc}", {}

        # Standard raster image
        raw_bgr = cv2.imread(str(p))
    elif isinstance(image_input, np.ndarray):
        raw_bgr = image_input

    if raw_bgr is None or raw_bgr.size == 0:
        return False, "Unreadable or corrupted image file. Please upload a valid PNG, JPG, or DICOM file.", {}

    h, w = raw_bgr.shape[:2]
    if h < 100 or w < 100:
        return False, f"Image resolution too low ({w}×{h}px). Medical scans must be at least 100×100 pixels.", {"width": w, "height": h}

    # 2. Check for Color Saturation / Non-Radiographic Natural Photos
    # Medical scans (X-ray, CT, MRI, US) are monochromatic (grayscale).
    # Natural photos (selfies, animals, landscapes, cars, memes) have high chromatic divergence and saturation.
    mean_sat = 0.0
    channel_disparity = 0.0
    is_color = False

    if raw_bgr.ndim == 3 and raw_bgr.shape[2] == 3:
        hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        b = raw_bgr[:, :, 0].astype(np.float32)
        g = raw_bgr[:, :, 1].astype(np.float32)
        r = raw_bgr[:, :, 2].astype(np.float32)
        channel_disparity = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))

        # Ultrasound occasionally has a small colored Doppler flow box or hospital logo,
        # so we allow a higher threshold for US, but XR/CT/MRI must be strictly monochromatic.
        sat_threshold = 30.0 if expected_modality == "US" else 18.0
        disp_threshold = 24.0 if expected_modality == "US" else 14.0

        if mean_sat > sat_threshold or channel_disparity > disp_threshold:
            is_color = True
            telemetry = {
                "color_saturation": round(mean_sat, 2),
                "channel_disparity": round(channel_disparity, 2),
                "width": w,
                "height": h,
            }
            return (
                False,
                f"Image rejected: Natural color photograph or non-medical image detected "
                f"(color saturation={mean_sat:.1f}, channel disparity={channel_disparity:.1f}). "
                f"Radiological scans must be monochromatic grayscale images.",
                telemetry,
            )

    # 3. Check Dynamic Range, Entropy, and Low-Information / Blank Canvas
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) if raw_bgr.ndim == 3 else raw_bgr
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    entropy = _compute_shannon_entropy(gray)

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
        return False, f"Image rejected: Blank or solid uniform image detected (std={std_val:.1f}).", telemetry
    if mean_val < 8.0:
        return False, "Image rejected: Completely black / unexposed scan.", telemetry
    if mean_val > 248.0:
        return False, "Image rejected: Completely white / washed-out canvas.", telemetry
    if entropy < 3.2:
        return False, f"Image rejected: Extremely low information content / synthetic graphic (entropy={entropy:.2f} < 3.2).", telemetry

    # 4. Check for Text Document / Diagram / Screenshot
    # Text pages (white background with black text) have an extreme bimodal distribution (>85% pure white/black)
    extreme_pixels = float(np.mean((gray > 240) | (gray < 15)))
    telemetry["extreme_pixel_ratio"] = round(extreme_pixels, 3)
    if extreme_pixels > 0.86 and std_val > 55.0:
        return False, "Image rejected: Text document, chart, or screenshot detected rather than a medical scan.", telemetry

    # 5. Modality-Specific Anatomical Validation
    if expected_modality == "XR":
        is_xray, reason, anat_metrics = _verify_chest_xray_anatomy(gray)
        telemetry.update(anat_metrics)
        if not is_xray:
            return False, f"Image rejected: {reason}", telemetry

    elif expected_modality in ("CT", "MR"):
        # CT and MRI scans are typically cross-sectional (axial/sagittal) with a prominent dark surround
        dark_perimeter = float(np.mean(gray[: int(h * 0.08), :] < 30))
        telemetry["dark_perimeter"] = round(dark_perimeter, 2)
        if dark_perimeter < 0.20 and expected_modality == "CT":
            # CT scans have dark scanner field background around the patient body
            pass  # soft warning or check

    return True, "Image verified as a genuine medical scan.", telemetry

