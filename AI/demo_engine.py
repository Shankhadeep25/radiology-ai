"""
Fallback computer-vision chest X-ray feature engine.
Extracts anatomical region density, symmetry, and gradient textures.
Does NOT use filename shortcuts or synthetic random numbers.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from preprocessing import load_image
from conditions import POSITIVE_THRESHOLD, CONDITION_INFO


def _analyze_regions(gray: np.ndarray) -> dict:
    """Extract region-based features for multi-condition scoring."""
    h, w = gray.shape[:2]
    roi = gray[int(h * 0.12) : int(h * 0.88), int(w * 0.08) : int(w * 0.92)]

    upper = roi[: int(roi.shape[0] * 0.4), :]
    middle = roi[int(roi.shape[0] * 0.3) : int(roi.shape[0] * 0.65), :]
    lower = roi[int(roi.shape[0] * 0.55) :, :]
    left = roi[:, : int(roi.shape[1] * 0.48)]
    right = roi[:, int(roi.shape[1] * 0.52) :]
    heart = roi[
        int(roi.shape[0] * 0.35) : int(roi.shape[0] * 0.72),
        int(roi.shape[1] * 0.38) : int(roi.shape[1] * 0.62),
    ]
    costophrenic_l = roi[int(roi.shape[0] * 0.75) :, : int(roi.shape[1] * 0.35)]
    costophrenic_r = roi[int(roi.shape[0] * 0.75) :, int(roi.shape[1] * 0.65) :]

    return {
        "mean": float(np.mean(roi)),
        "std": float(np.std(roi)),
        "upper_dark": float(1.0 - np.mean(upper) / 255.0),
        "lower_dark": float(1.0 - np.mean(lower) / 255.0),
        "middle_dark": float(1.0 - np.mean(middle) / 255.0),
        "left_dark": float(1.0 - np.mean(left) / 255.0),
        "right_dark": float(1.0 - np.mean(right) / 255.0),
        "heart_dark": float(1.0 - np.mean(heart) / 255.0),
        "heart_width": float(heart.shape[1] / max(roi.shape[1], 1)),
        "costo_blunt_l": float(1.0 - np.mean(costophrenic_l) / 255.0),
        "costo_blunt_r": float(1.0 - np.mean(costophrenic_r) / 255.0),
        "bilateral_diff": float(abs(np.mean(left) - np.mean(right)) / 255.0),
        "texture": float(np.std(roi)) / 80.0,
    }


def _score_all_conditions(features: dict) -> dict[str, float]:
    """Score radiological conditions from image morphology."""
    f = features
    scores = {}

    scores["Pneumonia"] = min(0.92, f["lower_dark"] * 0.55 + f["texture"] * 0.30 + f["bilateral_diff"] * 0.25)
    scores["Tuberculosis"] = min(0.92, f["upper_dark"] * 0.65 + f["texture"] * 0.20)
    scores["COVID-19"] = min(0.90, f["middle_dark"] * 0.40 + (1.0 - f["bilateral_diff"]) * 0.30 + f["lower_dark"] * 0.25)
    scores["Pulmonary Edema"] = min(0.90, f["middle_dark"] * 0.45 + f["heart_dark"] * 0.30)
    scores["Mass / nodular opacity"] = min(0.90, max(f["upper_dark"], f["lower_dark"]) * 0.45 + f["texture"] * 0.35)
    scores["Pulmonary Nodule"] = min(0.85, f["texture"] * 0.45 + f["bilateral_diff"] * 0.35)
    scores["Pleural Effusion"] = min(0.92, max(f["costo_blunt_l"], f["costo_blunt_r"]) * 0.70 + f["lower_dark"] * 0.25)
    scores["Cardiomegaly"] = min(0.92, max(0.0, (f["heart_width"] - 0.40) * 3.5) + f["heart_dark"] * 0.25)
    scores["Atelectasis"] = min(0.88, f["bilateral_diff"] * 0.45 + f["lower_dark"] * 0.35)
    scores["Consolidation"] = min(0.90, f["lower_dark"] * 0.50 + f["middle_dark"] * 0.35)
    scores["Pneumothorax"] = min(0.85, f["bilateral_diff"] * 0.45 + (1.0 - f["upper_dark"]) * 0.25)
    opacity_score = min(0.92, (f["lower_dark"] + f["middle_dark"]) * 0.40 + f["texture"] * 0.20)
    scores["Pneumonia"] = max(scores["Pneumonia"], opacity_score)

    abnormal_max = max(scores.values()) if scores else 0.0
    scores["Normal"] = max(0.10, min(0.95, 1.0 - abnormal_max * 0.85))

    return {k: round(v, 4) for k, v in scores.items()}


def _build_findings_list(scores: dict) -> list[dict]:
    ranked = sorted(
        [(k, v) for k, v in scores.items() if k != "Normal"],
        key=lambda x: x[1],
        reverse=True,
    )
    findings = []
    for condition, prob in ranked:
        if prob < 0.20:
            continue
        status = "detected" if prob >= POSITIVE_THRESHOLD else "possible"
        info = CONDITION_INFO.get(condition, {})
        findings.append({
            "condition": condition,
            "probability": round(prob, 3),
            "status": status,
            "category": info.get("category", "Other"),
            "urgency": info.get("urgency", "medium"),
            "icon": info.get("icon", "📋"),
        })
    return findings[:8]


def demo_predict(image_path: str, hint_name: str | None = None, modality: str = "XR") -> dict:
    gray = load_image(image_path)
    features = _analyze_regions(gray)
    scores = _score_all_conditions(features)

    abnormal = {k: v for k, v in scores.items() if k != "Normal"}
    if abnormal and max(abnormal.values()) >= POSITIVE_THRESHOLD:
        primary = max(abnormal, key=abnormal.get)
        confidence = abnormal[primary]
    else:
        primary = "Normal"
        confidence = scores.get("Normal", 0.85)

    confidence = min(0.96, confidence)
    severity = None
    if primary != "Normal":
        severity = "severe" if confidence > 0.78 else "moderate" if confidence > 0.58 else "mild"

    uncertainty = "high" if confidence >= 0.80 else "medium"
    all_findings = _build_findings_list(scores)
    differential = [{"condition": f["condition"], "probability": f["probability"]} for f in all_findings]

    return {
        "primary_label": primary,
        "confidence": round(confidence, 4),
        "severity": severity,
        "uncertainty_level": uncertainty,
        "differential_diagnosis": differential,
        "all_findings": all_findings,
        "model_version": "cv-feature-engine-v2",
        "raw_scores": {k: round(v, 3) for k, v in scores.items()},
    }


def demo_gradcam(image_path: str, study_id: int, output_dir: str, target_label: str = "Pneumonia") -> str:
    """Generate image saliency heatmap using Sobel gradients and morphological contrast."""
    gray = load_image(image_path)
    h, w = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    saliency = np.sqrt(grad_x ** 2 + grad_y ** 2)
    saliency = cv2.GaussianBlur(saliency, (25, 25), 0)

    # Normalize to [0, 1]
    s_min, s_max = float(saliency.min()), float(saliency.max())
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    else:
        saliency = np.zeros_like(saliency)

    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)
    combined = np.hstack([original_bgr, overlay])

    out_path = Path(output_dir) / f"gradcam_{study_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    return str(out_path)
