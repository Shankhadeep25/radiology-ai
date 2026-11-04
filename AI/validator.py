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
    # Invert bright-background images so brain structures become bright blobs
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
    Detect MRI brain scan fingerprint — works for single-slice and multi-slice collages
    on both dark and bright (white-padded) backgrounds.

    Returns (is_mri_brain: bool, reason: str).

    Discriminating features (empirically derived from real sample analysis):
      Single-slice MRI brain      : mean~62, border<25, center>100, dark>0.35, circles>=15, circ>0.70
      Multi-slice bright collage  : mean>160, bright_ratio>0.50, circles>=30  (circ irrelevant — merged blobs)
      Multi-slice dark collage    : mean<128, dark_ratio>0.25, circles>=30, circ<0.50 (many blobs, irregular)
      CT chest (look-alike)       : circles~16, center~87 (dark lung airspace)
      US breast                   : circles=1 (fan-shaped acquisition, not round)
      Chest XR                    : circles~11, border>100 (no dark skull border)
    """
    h, w = gray.shape[:2]
    mean_val = float(np.mean(gray))
    dark_ratio = float(np.mean(gray < 40))
    bright_ratio = float(np.mean(gray > 200))

    bh = max(h // 16, 4)
    bw = max(w // 16, 4)
    border_pixels = np.concatenate([
        gray[:bh, :].ravel(), gray[-bh:, :].ravel(),
        gray[:, :bw].ravel(), gray[:, -bw:].ravel()
    ])
    border_mean = float(np.mean(border_pixels))
    center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    center_mean = float(np.mean(center))

    # Whole-image circularity on the original gray (dark-bg) or inverted (bright-bg)
    work = gray if mean_val < 128 else (255 - gray)
    _, th = cv2.threshold(work, 30, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ = 0.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        if p > 0:
            circ = float(4 * np.pi * a / (p ** 2))

    n_circles = _count_hough_circles(gray)

    # --- Case 1: Bright-background multi-slice collage (white-padded contact sheet) ---
    # Key signal: many Hough circles (8+ brain slices each contribute 5-6 detected circles).
    # XR has ~11, CT ~16, US ~1. Brain collages have 30-55.
    # NOTE: overall circularity is NOT reliable here — merged blobs give low circ.
    if mean_val > 160 and bright_ratio > 0.50 and n_circles >= 30:
        return True, (
            f"MRI brain collage (bright bg) detected: {n_circles} circular brain structures, "
            f"mean={mean_val:.0f}"
        )

    # --- Case 2: Dark-background multi-slice collage ---
    # Many circular structures but on a dark background.
    # Circ is low because individual brain ovals merge into an irregular mass.
    # Distinguishes from single-slice MRI by having many circles but low composite circularity.
    if mean_val < 128 and dark_ratio > 0.25 and n_circles >= 30 and circ < 0.55:
        return True, (
            f"MRI brain collage (dark bg) detected: {n_circles} circular brain structures, "
            f"mean={mean_val:.0f}, dark_ratio={dark_ratio:.2f}"
        )

    # --- Case 3: Standard dark-background single-slice MRI ---
    # Characteristics: very dark border (skull surround), bright center (grey matter),
    # high circularity of brain oval, significant dark background ratio, many circular structures.
    # Key CT differentiator: CT center_mean~87 (dark lung airspace), MRI center_mean~129 (bright grey matter).
    if (dark_ratio >= 0.35 and border_mean < 25 and center_mean > 100
            and circ >= 0.70 and n_circles >= 15):
        return True, (
            f"MRI brain single-slice detected: border={border_mean:.0f}, center={center_mean:.0f}, "
            f"circ={circ:.2f}, circles={n_circles}"
        )

    return False, ""


def _verify_chest_xray_anatomy(gray: np.ndarray) -> tuple[bool, str, dict[str, float]]:
    """
    Verify radiographic thoracic anatomy for chest X-rays:
    1. Dense abdominal dome / diaphragm at the lower base.
    2. Bilateral radiolucent lung fields.
    3. Higher-density central mediastinal column (cardiac silhouette + spine).
    4. Thoracic hemithorax symmetry.
    """
    h, w = gray.shape[:2]
    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.45 or aspect_ratio > 2.2:
        return False, f"Aspect ratio ({aspect_ratio:.2f}) does not match frontal chest radiographs (0.5 - 2.0).", {}

    # Check 1: Abdominal / Diaphragmatic Base Density
    # In upright frontal chest radiographs, the lower 25% represents the dense diaphragm and abdominal viscera.
    # Studio portraits or clothing photos feature dark clothing/lapels/ties at the base (mean < 70).
    bottom_quarter = gray[int(h * 0.75):, :]
    mean_bot = float(np.mean(bottom_quarter))
    if mean_bot < 80.0:
        return (
            False,
            f"Image rejected: Lower abdominal/diaphragmatic base density is too low ({mean_bot:.1f} < 80.0). "
            f"Chest radiographs require a dense subdiaphragmatic base, not dark clothing or empty space.",
            {"bottom_density": round(mean_bot, 1)},
        )

    # Check 2: Central cardiac-diaphragmatic attenuation
    # In an authentic chest X-ray, the cardiac silhouette rests on the diaphragm, forming a dense central lower zone.
    # In a portrait of a person in a suit, the center bottom is a dark necktie and jacket.
    center_bottom = gray[int(h * 0.70):int(h * 0.95), int(w * 0.35):int(w * 0.65)]
    mean_cl = float(np.mean(center_bottom))
    if mean_cl < 90.0:
        return (
            False,
            f"Image rejected: Lacks central cardiac-diaphragmatic tissue attenuation ({mean_cl:.1f} < 90.0).",
            {"cardiac_base_density": round(mean_cl, 1)},
        )

    # Check 3: Extract central ROI (excluding outer borders/collimation)
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
        "bottom_density": round(mean_bot, 1),
        "cardiac_base_density": round(mean_cl, 1),
        "roi_std": round(std_roi, 1),
    }

    # Severe asymmetry check: a chest X-ray has bilateral lungs; natural objects/faces/cars lack this balance
    if sym_diff > 0.40:
        return False, f"Bilateral hemithorax symmetry failed (diff={sym_diff:.2f} > 0.40). Image does not exhibit chest anatomy.", metrics

    contrast_spread = abs(mean_m - ((mean_l + mean_r) / 2.0))
    if std_roi < 18.0 or (contrast_spread < 2.0 and std_roi < 25.0):
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
    mean_sat = 0.0
    channel_disparity = 0.0

    if raw_bgr.ndim == 3 and raw_bgr.shape[2] == 3:
        hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        b = raw_bgr[:, :, 0].astype(np.float32)
        g = raw_bgr[:, :, 1].astype(np.float32)
        r = raw_bgr[:, :, 2].astype(np.float32)
        channel_disparity = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))

        sat_threshold = 30.0 if expected_modality in ("US", "MR") else 18.0
        disp_threshold = 40.0 if expected_modality == "MR" else (24.0 if expected_modality == "US" else 14.0)

        if mean_sat > sat_threshold or channel_disparity > disp_threshold:
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
    extreme_pixels = float(np.mean((gray > 240) | (gray < 15)))
    telemetry["extreme_pixel_ratio"] = round(extreme_pixels, 3)
    if extreme_pixels > 0.86 and std_val > 55.0:
        return False, "Image rejected: Text document, chart, or screenshot detected rather than a medical scan.", telemetry

    # 5. Universal MRI Brain Fingerprint Guard
    # Runs BEFORE modality-specific checks for ALL modalities.
    # Prevents MRI brain scans from being accepted as CT / XR / US and producing false diagnoses.
    is_mri_brain, mri_reason = _is_mri_brain(gray)
    telemetry["mri_brain_fingerprint"] = is_mri_brain

    if expected_modality in ("XR", "CT", "US"):
        if is_mri_brain:
            modality_names = {"XR": "Chest X-Ray", "CT": "CT Chest Scan", "US": "Breast Ultrasound"}
            return (
                False,
                f"Image rejected: This appears to be an MRI Brain scan, not a "
                f"{modality_names[expected_modality]}. "
                f"Please select the 'MRI Brain' modality. ({mri_reason})",
                telemetry,
            )

    # 6. Modality-Specific Anatomical Validation
    if expected_modality == "XR":
        is_xray, reason, anat_metrics = _verify_chest_xray_anatomy(gray)
        telemetry.update(anat_metrics)
        if not is_xray:
            return False, reason, telemetry

    elif expected_modality == "MR":
        # MRI Brain: check for single-slice circularity OR multi-slice montage.
        # First, run the MRI fingerprint detection (already computed above).
        if is_mri_brain:
            # Confirmed MRI brain — check if it's a montage or single slice
            from montage_detector import detect_montage_grid
            is_montage, grid_meta = detect_montage_grid(gray)
            if is_montage:
                telemetry["is_multi_slice_montage"] = True
                telemetry["montage_grid"] = grid_meta
                return True, f"Verified as a multi-slice volumetric MRI montage ({grid_meta['rows']}x{grid_meta['cols']} grid, {grid_meta['count']} slices).", telemetry
            else:
                # Single-slice MRI confirmed by fingerprint — also do circularity check
                _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                circ = 0.0
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    perim = cv2.arcLength(c, True)
                    circ = float(4 * np.pi * area / (perim ** 2 + 1e-5))
                telemetry["cranial_circularity"] = round(circ, 3)
                return True, "Verified as MRI Brain scan.", telemetry
        else:
            # Not confirmed by fingerprint — try legacy circularity check + montage fallback
            _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circ = 0.0
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                perim = cv2.arcLength(c, True)
                circ = float(4 * np.pi * area / (perim ** 2 + 1e-5))
            telemetry["cranial_circularity"] = round(circ, 3)

            if circ >= 0.50:
                return True, "Verified as MRI Brain scan (circularity check passed).", telemetry

            # Try montage detection as last resort
            from montage_detector import detect_montage_grid
            is_montage, grid_meta = detect_montage_grid(gray)
            if is_montage:
                telemetry["is_multi_slice_montage"] = True
                telemetry["montage_grid"] = grid_meta
                return True, f"Verified as a multi-slice volumetric MRI montage ({grid_meta['rows']}x{grid_meta['cols']} grid, {grid_meta['count']} slices).", telemetry

            return (
                False,
                f"Image rejected: Does not match MRI brain scan characteristics "
                f"(circularity={circ:.2f} < 0.50, MRI fingerprint not detected). "
                f"Please upload a brain MRI scan.",
                telemetry,
            )

    return True, "Image verified as a genuine medical scan.", telemetry
