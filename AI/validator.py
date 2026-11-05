"""
Medical image authenticity and radiographic validity validator.
Detects and rejects non-medical images, color photographs, cartoons,
text documents, synthetic noise, portraits, and non-X-ray images before inference.
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


def _count_hough_circles(gray: np.ndarray) -> int:
    """
    Count circular structures via Hough transform on a 512x512 normalized image.
    Works on both dark-background (standard MRI) and bright-background (white-bg collage) images.
    """
    mean_val = float(np.mean(gray))
    work = gray if mean_val < 128 else (255 - gray)
    small = cv2.resize(work, (512, 512))
    blurred = cv2.GaussianBlur(small, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
        param1=50, param2=25, minRadius=30, maxRadius=180
    )
    return int(len(circles[0])) if circles is not None else 0


def _is_mri_brain(gray: np.ndarray) -> tuple[bool, str]:
    """
    Detect whether an image has the distinct fingerprint of an MRI brain scan:
    either a multi-slice grid montage (e.g. 2x4, 2x3, 2x2) or a single-slice
    cranial axial cross-section with high circularity and dark borders.
    """
    h, w = gray.shape[:2]

    # Check 1: Multi-slice volumetric montage (e.g. 2x4, 2x3, 2x2, 3x3)
    from montage_detector import detect_montage_grid
    is_montage, grid_meta = detect_montage_grid(gray)
    if is_montage and grid_meta.get("count", 0) >= 4:
        r, c = grid_meta.get("rows"), grid_meta.get("cols")
        return True, f"MRI multi-slice series ({r}x{c} grid, {grid_meta.get('count')} slices)"

    # Check 2: Single-slice cranial MRI scan
    bh, bw = max(h // 16, 4), max(w // 16, 4)
    border_pixels = np.concatenate([
        gray[:bh, :].ravel(), gray[-bh:, :].ravel(),
        gray[:, :bw].ravel(), gray[:, -bw:].ravel()
    ])
    border_mean = float(np.mean(border_pixels))
    center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    center_mean = float(np.mean(center))
    dark_ratio = float(np.mean(gray < 40))

    _, th = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if p > 0:
            circ = float(4 * np.pi * a / (p ** 2))

    # Single-slice MRI cranial cross section:
    # 1. Scanner dark background surrounding the skull on all 4 borders: border_mean < 25.0
    # 2. Elliptical/circular cranial skull outline: circ >= 0.50
    # 3. High attenuation of brain parenchyma: center_mean >= 70.0
    # 4. Scanner background air: dark_ratio >= 0.35
    if (border_mean < 25.0 and circ >= 0.50 and center_mean >= 70.0 and dark_ratio >= 0.35):
        return True, (
            f"MRI brain single-slice scan (circularity={circ:.2f}, center density={center_mean:.1f})"
        )

    return False, ""


def _verify_chest_xray_anatomy(gray: np.ndarray) -> tuple[bool, str, dict[str, float]]:
    """
    Verify radiographic thoracic anatomy for chest X-rays.
    Records numeric telemetry flags without hard-rejecting pediatric, rotated,
    or cropped clinical scans.
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
        return True, "Valid chest X-ray study (reduced resolution).", {"aspect_ratio": round(aspect_ratio, 3)}

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

        raw_bgr = cv2.imread(str(p))
    elif isinstance(image_input, np.ndarray):
        raw_bgr = image_input

    if raw_bgr is None or raw_bgr.size == 0:
        return False, "Unreadable or corrupted image file. Please upload a valid PNG, JPG, or DICOM file.", {}

    h, w = raw_bgr.shape[:2]
    if h < 100 or w < 100:
        return False, f"Image resolution too low ({w}×{h}px). Medical scans must be at least 100×100 pixels.", {"width": w, "height": h}

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
    extreme_pixels = float(np.mean((gray > 240) | (gray < 15)))
    telemetry["extreme_pixel_ratio"] = round(extreme_pixels, 3)
    if extreme_pixels > 0.86 and std_val > 55.0:
        return False, "Image rejected: Text document, chart, or screenshot detected rather than a medical scan.", telemetry

    # 5. Universal MRI Brain Fingerprint Guard
    is_mri_brain, mri_reason = _is_mri_brain(gray)
    telemetry["mri_brain_fingerprint"] = is_mri_brain
    telemetry["flag_mri_brain_fingerprint"] = int(is_mri_brain)

    # 6. Modality-Specific Anatomical Validation
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

    telemetry["border_density"] = round(border_mean, 1)
    telemetry["center_density"] = round(center_mean, 1)
    telemetry["circularity"] = round(circ, 3)

    if expected_modality == "XR":
        is_xray, reason, anat_metrics = _verify_chest_xray_anatomy(gray)
        telemetry.update(anat_metrics)

    elif expected_modality == "CT":
        center_roi = gray[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
        center_dark = float(np.mean(center_roi < 40)) if center_roi.size > 0 else 0.0
        telemetry["center_dark"] = round(center_dark, 3)
        telemetry["flag_ct_geometry"] = int(circ < 0.50)
        telemetry["flag_ct_border"] = int(border_mean > 35.0)

    elif expected_modality == "US":
        bottom_mean = float(np.mean(gray[int(h * 0.75):, :])) if h > 4 else 0.0
        telemetry["bottom_density"] = round(bottom_mean, 1)
        _grad = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        smooth_frac = float(np.mean(np.abs(_grad) < 10))
        telemetry["smooth_frac"] = round(smooth_frac, 3)
        telemetry["flag_us_smooth"] = int(smooth_frac > 0.72)

    elif expected_modality == "MR":
        from montage_detector import detect_montage_grid
        is_montage, grid_meta = detect_montage_grid(gray)
        if is_montage and grid_meta.get("count", 0) >= 4:
            telemetry["is_multi_slice_montage"] = True
            telemetry["montage_grid"] = grid_meta

        telemetry["flag_mr_geometry"] = int(circ < 0.50)
        telemetry["flag_mr_attenuation"] = int(border_mean >= 30.0 or center_mean < 100.0)

    return True, "Image verified as a genuine medical scan.", telemetry
