"""
Math, segmentation, and anatomy-recognition utilities for multi-modal radiology AI.

Separation of concerns:
- preprocessing.py  → I/O, windowing, normalization, CLAHE
- utils.py          → voxel math, masks, anatomy classification, volume/mass integration
- modalities.py     → modality-specific inference orchestration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from preprocessing import load_image, normalize_to_uint8

# Declared modality codes → expected anatomy label from image content
MODALITY_TO_ANATOMY: dict[str, str] = {
    "XR": "chest_xray",
    "CT": "ct_chest",
    "MR": "mri_brain",
    "US": "us_breast",
}

ANATOMY_TO_MODALITY: dict[str, str] = {v: k for k, v in MODALITY_TO_ANATOMY.items()}

# Approximate soft-tissue density when HU map unavailable (g/mL)
DEFAULT_NODULE_DENSITY_G_PER_ML = 1.05


def voxel_volume_mm3(spacing_mm: Sequence[float]) -> float:
    """Volume of a single voxel: ΔV = Δx · Δy · Δz."""
    dx, dy, dz = (float(spacing_mm[0]), float(spacing_mm[1]), float(spacing_mm[2]))
    return dx * dy * dz


def hu_to_density_g_per_ml(hu: np.ndarray) -> np.ndarray:
    """
    Convert Hounsfield units to physical density (g/mL) relative to water.
    ρ ≈ 1 + HU/1000  (standard CT calibration).
    """
    return 1.0 + np.asarray(hu, dtype=np.float64) / 1000.0


def compute_region_stats(gray: np.ndarray) -> dict[str, float]:
    """
    Region statistics for lesion candidate detection.
    Reused by modality pipelines — do not duplicate in modalities.py.
    """
    h, w = gray.shape[:2]
    roi = gray[int(h * 0.1) : int(h * 0.9), int(w * 0.1) : int(w * 0.9)]
    blur = cv2.GaussianBlur(roi, (9, 9), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 40]
    max_area = max(areas) if areas else 0.0
    return {
        "mean": float(np.mean(roi)),
        "std": float(np.std(roi)),
        "blob_count": float(len(areas)),
        "max_blob": float(max_area),
        "texture": float(np.std(roi)) / 80.0,
    }


def classify_image_anatomy(gray: np.ndarray) -> dict[str, Any]:
    """
    Content-based anatomy / modality recognition from scale-invariant pixel statistics.
    Prevents routing a brain MRI through the chest X-ray DenseNet pipeline.
    """
    thumb = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    h, w = thumb.shape[:2]
    roi = thumb[int(h * 0.08) : int(h * 0.92), int(w * 0.08) : int(w * 0.92)]
    mean = float(np.mean(roi))
    std = float(np.std(roi))
    dark_ratio = float(np.mean(roi < 70))
    bright_ratio = float(np.mean(roi > 190))

    lap = cv2.Laplacian(roi, cv2.CV_64F)
    speckle = float(np.std(lap)) / 25.0

    cy, cx = roi.shape[0] // 2, roi.shape[1] // 2
    r = min(cy, cx) // 2
    center = roi[cy - r : cy + r, cx - r : cy + r]
    center_bright = float(np.mean(center)) / 255.0
    periphery = roi.copy()
    periphery[cy - r : cy + r, cx - r : cy + r] = int(mean)
    periphery_contrast = abs(float(np.mean(center)) - float(np.mean(periphery))) / 255.0

    # Circularity of bright object in center (MRI skull is oval/circular; portraits are not)
    _, thresh = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        circularity = float(4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2 + 1e-5))

    mri_brain = 0.0
    if dark_ratio >= 0.28 and circularity >= 0.48:
        mri_brain = min(
            1.0,
            (dark_ratio - 0.20) * 1.2 + center_bright * 0.45 + (1 - mean / 255) * 0.35,
        )

    us_breast = 0.0
    _grad = cv2.Sobel(roi, cv2.CV_64F, 1, 1, ksize=3)
    _smooth = float(np.mean(np.abs(_grad) < 10))
    if speckle > 0.45 and dark_ratio > 0.15 and _smooth <= 0.70:
        us_breast = min(1.0, (speckle - 0.40) * 0.8 + (dark_ratio * 0.5) + (std / 100.0) * 0.3)

    ct_chest = 0.0
    if dark_ratio >= 0.35:
        ct_chest = min(1.0, dark_ratio * 0.6 + periphery_contrast * 0.6 + bright_ratio * 0.4)

    chest_xray = min(
        1.0,
        (1.0 - dark_ratio) * 0.5
        + min(0.35, std / 120.0)
        + (0.25 if mean >= 120 else 0.0)
        + max(0.0, 0.2 - speckle * 0.15),
    )

    scores = {
        "mri_brain": round(mri_brain, 4),
        "us_breast": round(us_breast, 4),
        "ct_chest": round(ct_chest, 4),
        "chest_xray": round(chest_xray, 4),
    }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_anatomy, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    return {
        "anatomy": best_anatomy,
        "confidence": round(best_score, 4),
        "modality": ANATOMY_TO_MODALITY.get(best_anatomy, "XR"),
        "scores": scores,
        "margin": round(best_score - second_score, 4),
    }


def anatomy_blocks_pipeline(modality: str, anatomy_result: dict[str, Any]) -> str | None:
    """
    Return an error message if the image anatomy clearly conflicts with the selected pipeline.
    Prevents e.g. chest X-ray → Brain Tumor when MRI is selected.
    """
    modality = (modality or "XR").upper()
    expected = MODALITY_TO_ANATOMY.get(modality, "chest_xray")
    detected = anatomy_result.get("anatomy", "chest_xray")
    scores = anatomy_result.get("scores", {})
    chest_score = float(scores.get("chest_xray", 0))

    labels = {
        "chest_xray": "chest X-ray",
        "mri_brain": "MRI brain scan",
        "ct_chest": "CT chest scan",
        "us_breast": "breast ultrasound",
    }
    selected = {"XR": "X-Ray (MVP)", "CT": "CT Chest", "MR": "MRI Brain", "US": "Ultrasound"}

    if detected == expected:
        return None

    # Chest X-ray sent through CT/MRI/US pipeline
    if modality in ("MR", "CT", "US") and chest_score >= 0.42:
        if detected == "chest_xray" or chest_score >= float(scores.get(detected, 0)):
            return (
                f"This image is a {labels['chest_xray']}, but you selected {selected.get(modality, modality)}. "
                f"Please switch to **X-Ray (MVP)** or upload the correct {selected.get(modality, modality)} scan."
            )

    # Wrong organ scan on X-ray pipeline
    if modality == "XR" and detected in ("mri_brain", "ct_chest", "us_breast"):
        det_score = float(scores.get(detected, anatomy_result.get("confidence", 0)))
        if det_score >= 0.60:
            correct = ANATOMY_TO_MODALITY.get(detected, "XR")
            return (
                f"This image appears to be a {labels.get(detected, detected)}, not a chest X-ray. "
                f"Please select **{selected.get(correct, correct)}** and analyze again."
            )

    return None


def validate_modality_match(
    declared_modality: str,
    anatomy_result: dict[str, Any],
    filename_hint: str | None = None,
) -> dict[str, Any]:
    """Compare user-declared modality with image-content anatomy detection."""
    declared = (declared_modality or "XR").upper()
    expected_anatomy = MODALITY_TO_ANATOMY.get(declared, "chest_xray")
    detected_anatomy = anatomy_result.get("anatomy", "chest_xray")
    detected_modality = anatomy_result.get("modality", "XR")
    confidence = float(anatomy_result.get("confidence", 0.0))
    margin = float(anatomy_result.get("margin", 0.0))

    mismatch = detected_anatomy != expected_anatomy
    strong_signal = confidence >= 0.60 and margin >= 0.10

    effective_modality = declared
    warning = None

    if mismatch and confidence >= 0.88 and margin >= 0.30:
        effective_modality = detected_modality
        warning = (
            f"Image content appears to be {detected_anatomy.replace('_', ' ')} "
            f"(confidence {confidence:.0%}) but {declared} was selected. "
            f"Auto-routed to {effective_modality} pipeline for correct organ analysis."
        )
    elif mismatch and not strong_signal:
        warning = (
            f"Selected modality is {declared}; anatomy classifier is uncertain. "
            "Confirm the correct organ/modality for best results."
        )

    return {
        "declared_modality": declared,
        "detected_anatomy": detected_anatomy,
        "detected_modality": detected_modality,
        "effective_modality": effective_modality,
        "mismatch": mismatch,
        "warning": warning,
        "filename_modality": None,
    }


def segment_lesion_mask(
    gray: np.ndarray,
    modality: str = "CT",
    min_area: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Extract precise lesion candidate mask and spatial metrics.
    Works for CT lung nodules, MRI brain tumors, and Ultrasound breast lesions.
    """
    h, w = gray.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)
    Y, X = np.ogrid[:h, :w]
    bbox = {
        "x": 0,
        "y": 0,
        "width": 0,
        "height": 0,
        "area_px": 0,
        "diameter_mm": 0.0,
        "anatomical_location": "Visualized parenchyma",
    }

    if modality == "CT":
        # Thoracic interior cavity (excluding outer chest wall and spine)
        thorax = (
            (((X - w // 2) / (w * 0.38)) ** 2 + ((Y - h // 2) / (h * 0.30)) ** 2 < 1.0)
            & (np.abs(X - w // 2) > int(w * 0.08))
            & (Y < int(h * 0.72))
        )
        # Nodule candidate: soft-tissue focal opacity in lung parenchyma
        candidates = ((gray >= 120) & (gray <= 225) & thorax).astype(np.uint8)
        contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt, best_score = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area <= 3500:
                peri = cv2.arcLength(c, True)
                circ = (4 * np.pi * area) / (peri ** 2 + 1e-6)
                if circ >= 0.40:
                    score = circ * np.log1p(area)
                    if score > best_score:
                        best_score, best_cnt = score, c

        if best_cnt is not None:
            cv2.drawContours(full, [best_cnt], -1, 1, thickness=-1)
            x, y, bw, bh = cv2.boundingRect(best_cnt)
            area = int(cv2.contourArea(best_cnt))
            d_mm = round(2.0 * np.sqrt(area / np.pi) * 0.7, 1)
            side = "Right" if (x + bw / 2) < w / 2 else "Left"
            zone = "Upper" if (y + bh / 2) < h / 3 else "Lower" if (y + bh / 2) > 2 * h / 3 else "Middle"
            bbox = {
                "x": int(x),
                "y": int(y),
                "width": int(bw),
                "height": int(bh),
                "area_px": area,
                "diameter_mm": d_mm,
                "anatomical_location": f"{side} {zone} Lobe",
            }
            return full, bbox

    elif modality == "MR":
        # Brain parenchyma mask
        brain = ((gray > 30) & (gray < 240)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        brain = cv2.morphologyEx(brain, cv2.MORPH_OPEN, kernel)

        mean_brain = float(np.mean(gray[brain > 0])) if np.any(brain > 0) else 100.0
        # Focal hyperintense lesion / edema
        hyper = ((gray >= min(190, int(mean_brain * 1.30))) & (brain > 0)).astype(np.uint8)
        contours, _ = cv2.findContours(hyper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        for c in contours:
            area = cv2.contourArea(c)
            if 40 <= area <= 8000:
                best_cnt = c
                break

        if best_cnt is not None:
            cv2.drawContours(full, [best_cnt], -1, 1, thickness=-1)
            x, y, bw, bh = cv2.boundingRect(best_cnt)
            area = int(cv2.contourArea(best_cnt))
            d_mm = round(2.0 * np.sqrt(area / np.pi) * 0.9, 1)
            side = "Right" if (x + bw / 2) < w / 2 else "Left"
            lobe = "Frontal Lobe" if (y + bh / 2) < h * 0.45 else "Temporoparietal region"
            bbox = {
                "x": int(x),
                "y": int(y),
                "width": int(bw),
                "height": int(bh),
                "area_px": area,
                "diameter_mm": d_mm,
                "anatomical_location": f"{side} {lobe}",
            }
            return full, bbox

    elif modality == "US":
        # Hypoechoic breast lesion in mid-depth fibroglandular parenchyma
        mid = gray[int(h * 0.15) : int(h * 0.8), :]
        blur = cv2.GaussianBlur(mid, (7, 7), 0)
        hypo = ((blur >= 15) & (blur <= 90)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hypo = cv2.morphologyEx(hypo, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(hypo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        for c in contours:
            area = cv2.contourArea(c)
            if 50 <= area <= 7000:
                best_cnt = c
                break

        if best_cnt is not None:
            shifted = best_cnt + np.array([[0, int(h * 0.15)]])
            cv2.drawContours(full, [shifted], -1, 1, thickness=-1)
            x, y, bw, bh = cv2.boundingRect(shifted)
            area = int(cv2.contourArea(shifted))
            d_mm = round(2.0 * np.sqrt(area / np.pi) * 0.15, 1)
            ar = round(bh / max(bw, 1), 2)
            bbox = {
                "x": int(x),
                "y": int(y),
                "width": int(bw),
                "height": int(bh),
                "area_px": area,
                "diameter_mm": d_mm,
                "aspect_ratio": ar,
                "anatomical_location": "Upper Outer Quadrant",
            }
            return full, bbox

    return full, bbox


def promote_slice_to_volume(
    slice_mask: np.ndarray,
    slice_hu: np.ndarray | None = None,
    slice_thickness_mm: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Promote a 2D segmentation to a minimal 3D voxel grid for volumetric integration."""
    mask_3d = slice_mask[np.newaxis, ...].astype(np.uint8)
    hu_3d = None
    if slice_hu is not None:
        hu_3d = np.asarray(slice_hu, dtype=np.float64)[np.newaxis, ...]
    spacing = (slice_thickness_mm, 1.0, 1.0)
    return mask_3d, hu_3d if hu_3d is None else (mask_3d, hu_3d, spacing)


def compute_volumetric_mass_from_mask(
    mask: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_volume: Optional[np.ndarray] = None,
    default_density_g_per_ml: float = DEFAULT_NODULE_DENSITY_G_PER_ML,
) -> dict[str, Any]:
    """Discrete triple integration over segmented voxel space."""
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[np.newaxis, ...]

    binary = mask_arr > 0
    n_voxels = int(np.count_nonzero(binary))
    if n_voxels == 0:
        return {
            "volume_mm3": 0.0,
            "volume_ml": 0.0,
            "mass_g": 0.0,
            "voxel_count": 0,
            "voxel_spacing_mm": tuple(float(s) for s in voxel_spacing_mm),
            "integration_method": "discrete_voxel_sum",
        }

    dz, dy, dx = (float(voxel_spacing_mm[0]), float(voxel_spacing_mm[1]), float(voxel_spacing_mm[2]))
    dv_mm3 = dz * dy * dx
    volume_mm3 = n_voxels * dv_mm3
    volume_ml = round(volume_mm3 / 1000.0, 2)

    if hu_volume is not None:
        hu_vol = np.asarray(hu_volume, dtype=np.float64)
        if hu_vol.ndim == 2:
            hu_vol = hu_vol[np.newaxis, ...]
        rho = hu_to_density_g_per_ml(hu_vol[binary])
        mass_g = float(np.sum(rho * (dv_mm3 / 1000.0)))
    else:
        mass_g = float(volume_mm3 / 1000.0 * default_density_g_per_ml)

    diameter_mm = float(2.0 * ((3.0 * volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0)) if volume_mm3 > 0 else 0.0

    return {
        "volume_mm3": round(volume_mm3, 2),
        "volume_ml": volume_ml,
        "mass_g": round(mass_g, 3),
        "voxel_count": n_voxels,
        "voxel_spacing_mm": (dz, dy, dx),
        "equivalent_diameter_mm": round(diameter_mm, 2),
        "integration_method": "discrete_voxel_sum",
    }


def compute_nodule_volumetric_mass(
    gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pixel_spacing_mm: Tuple[float, float] = (0.7, 0.7),
    slice_thickness_mm: float = 1.0,
    default_hu: float = -100.0,
) -> dict[str, Any]:
    """End-to-end lung nodule volumetric mass from CT slice."""
    if mask is None:
        mask, bbox = segment_lesion_mask(gray, modality="CT")
    else:
        bbox = {}

    hu_slice = (gray.astype(np.float64) / 255.0) * 400.0 - 200.0
    dz, dy, dx = slice_thickness_mm, pixel_spacing_mm[0], pixel_spacing_mm[1]
    mask_3d = mask[np.newaxis, ...] if mask.ndim == 2 else mask
    hu_3d = hu_slice[np.newaxis, ...] if hu_slice.ndim == 2 else hu_slice

    metrics = compute_volumetric_mass_from_mask(
        mask_3d,
        voxel_spacing_mm=(dz, dy, dx),
        hu_volume=hu_3d,
    )
    metrics["bounding_box"] = bbox
    return metrics


def filename_hint_modality(filename: str | None) -> str | None:
    """Infer intended modality from filename."""
    name = (filename or "").lower()
    if any(k in name for k in ("ct", "nodule", "lidc")):
        return "CT"
    if any(k in name for k in ("mri", "mr_", "brats", "brain", "tumor")):
        return "MR"
    if any(k in name for k in ("us_", "ultrasound", "breast", "busi", "birads")):
        return "US"
    if any(k in name for k in ("pneumonia", "normal", "chest", "xray", "cxr")):
        return "XR"
    return None


def load_gray_for_analysis(image_path: str) -> np.ndarray:
    return load_image(image_path)
