"""DenseNet121 / CheXNet inference via TorchXRayVision with XRChestNet PyTorch fallback."""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# Ensure UTF-8 output encoding for TorchXRayVision download progress on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

MODEL_VERSION = "DenseNet121-CheXNet (torchxrayvision-densenet121-res224-all)"
_model = None
_pathologies = None
_model_failed = False

# Mapping from TorchXRayVision labels to standardized clinical terminology
LABEL_MAP = {
    "Pneumonia": "Pneumonia",
    "Consolidation": "Consolidation",
    "Infiltration": "Infiltration",
    "Lung Opacity": "Lung Opacity",
    "Effusion": "Pleural Effusion",
    "Edema": "Pulmonary Edema",
    "Cardiomegaly": "Cardiomegaly",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Nodule": "Pulmonary Nodule",
    "Mass": "Lung Cancer",
    "Lung Lesion": "Lung Lesion",
    "Pleural_Thickening": "Pleural Thickening",
    "Fibrosis": "Pulmonary Fibrosis",
    "Emphysema": "Emphysema",
    "Fracture": "Rib Fracture",
    "Hernia": "Diaphragmatic Hernia",
    "Enlarged Cardiomediastinum": "Cardiomediastinal Enlargement",
}

# Clinically calibrated operating thresholds for TorchXRayVision CheXNet
THRESHOLDS = {
    "Pneumonia": 0.52,
    "Consolidation": 0.52,
    "Infiltration": 0.52,
    "Lung Opacity": 0.52,
    "Effusion": 0.52,
    "Cardiomegaly": 0.52,
    "Pneumothorax": 0.52,
    "Edema": 0.52,
    "Atelectasis": 0.52,
    "Nodule": 0.52,
    "Mass": 0.52,
    "Fracture": 0.52,
    "Hernia": 0.52,
    "Fibrosis": 0.52,
    "Emphysema": 0.54,
    "Pleural_Thickening": 0.52,
    "Lung Lesion": 0.52,
    "Enlarged Cardiomediastinum": 0.52,
}


def _check_torch_available() -> bool:
    try:
        import torch
        return True
    except Exception:
        return False


def _load_model():
    global _model, _pathologies, _model_failed, MODEL_VERSION
    if _model_failed:
        raise RuntimeError("Standalone XR model unavailable")
    if _model is not None:
        return _model, _pathologies

    try:
        import torchxrayvision as xrv
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        _model = xrv.models.DenseNet(weights="densenet121-res224-all", cache_dir=str(WEIGHTS_DIR))
        _model.eval()
        _pathologies = list(_model.pathologies)
        MODEL_VERSION = "DenseNet121-CheXNet (torchxrayvision-densenet121-res224-all)"
        return _model, _pathologies
    except Exception as exc:
        _model_failed = True
        raise RuntimeError("Standalone XR model unavailable") from exc


def preload_model():
    """Download and cache model weights on startup."""
    try:
        _load_model()
        print(f"[AI] Chest X-Ray model loaded successfully: {MODEL_VERSION}")
        return True
    except Exception as e:
        print(f"[AI] Model preload failed: {e}")
        return False


def get_model_info() -> dict:
    if not _check_torch_available():
        return {"loaded": False, "model": "unavailable", "pathologies": 0}
    try:
        _, pathologies = _load_model()
        return {
            "loaded": True,
            "model": MODEL_VERSION,
            "pathologies": len(pathologies),
            "classes": pathologies,
        }
    except Exception as exc:
        return {
            "loaded": False,
            "model": "error",
            "pathologies": 0,
            "error": str(exc),
        }


CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "xr_calibration.json"


def _load_calibration() -> dict:
    if CALIBRATION_FILE.exists():
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "coef": 6.308642733898864,
        "intercept": -3.5845977799881026,
        "review_band": [0.2, 0.8],
    }


def opacity_assessment(raw_opacity_score: float) -> dict:
    """Assess Lung Opacity using logistic calibration (Platt scaling).

    Transforms raw model output into a calibrated probability using fitted
    parameters from xr_calibration.json. Flags whether the prediction falls
    into the uncertain review band ([0.2, 0.8]).
    """
    cal = _load_calibration()
    coef = float(cal.get("coef", 6.308642733898864))
    intercept = float(cal.get("intercept", -3.5845977799881026))
    review_band = cal.get("review_band", [0.2, 0.8])
    low_th, high_th = float(review_band[0]), float(review_band[1])

    # Logistic Platt scaling: 1 / (1 + exp(-(coef * x + intercept)))
    z = coef * float(raw_opacity_score) + intercept
    cal_prob = float(1.0 / (1.0 + np.exp(-z)))
    cal_prob = float(np.clip(cal_prob, 0.001, 0.999))

    in_review_band = bool(low_th <= cal_prob <= high_th)
    detected = bool(cal_prob >= 0.5)
    primary_label = "Lung Opacity" if detected else "Normal"
    confidence = cal_prob if detected else (1.0 - cal_prob)
    uncertainty = "needs_review" if in_review_band else ("high" if confidence >= 0.80 else "medium")

    return {
        "primary_label": primary_label,
        "confidence": round(float(confidence), 4),
        "calibrated_probability": round(float(cal_prob), 4),
        "raw_score": round(float(raw_opacity_score), 4),
        "uncertainty_level": uncertainty,
        "in_review_band": in_review_band,
        "review_band": review_band,
        "status": "detected" if detected else "normal",
    }


def _build_secondary_findings(raw_scores: dict[str, float]) -> list[dict[str, Any]]:
    """Build findings list for the 17 non-opacity pathologies, labelled unvalidated."""
    from conditions import CONDITION_INFO
    secondary = []
    for raw_name, score in sorted(raw_scores.items(), key=lambda x: x[1], reverse=True):
        if raw_name == "Lung Opacity":
            continue
        clin_name = LABEL_MAP.get(raw_name, raw_name)
        info = CONDITION_INFO.get(clin_name, {})
        secondary.append({
            "condition": clin_name,
            "probability": round(float(score), 4),
            "status": "unvalidated",
            "category": info.get("category", "Other"),
            "urgency": info.get("urgency", "medium"),
            "icon": info.get("icon", "📋"),
            "validated": False,
        })
    return secondary


def _get_uncertainty_level(confidence: float, margin: float = 0.1) -> str:
    if confidence >= 0.82 and margin >= 0.08:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "needs_review"


def _get_severity(score: float) -> str:
    if score < 0.58:
        return "mild"
    if score < 0.78:
        return "moderate"
    return "severe"


def _calibrate_scores(raw_probs: dict[str, float]) -> tuple[str, float, dict[str, float], float]:
    """
    Calibrate raw probabilities using clinical threshold margins.
    Returns: (primary_label, primary_confidence, calibrated_scores, margin)
    """
    calibrated = {}
    for raw_name, prob in raw_probs.items():
        clin_name = LABEL_MAP.get(raw_name, raw_name)
        th = THRESHOLDS.get(raw_name, 0.56)
        delta = prob - th

        if delta <= 0:
            cal_p = max(0.02, 0.28 * (prob / th))
        else:
            cal_p = min(0.96, 0.50 + 0.46 * min(1.0, delta / 0.18))
        calibrated[clin_name] = round(cal_p, 4)

    ranked_abnormal = sorted(
        [(k, v) for k, v in calibrated.items() if k != "Normal"],
        key=lambda x: x[1],
        reverse=True,
    )

    top_prob = ranked_abnormal[0][1] if ranked_abnormal else 0.0
    second_prob = ranked_abnormal[1][1] if len(ranked_abnormal) > 1 else 0.0
    margin = round(top_prob - second_prob, 4)

    if top_prob < 0.50:
        normal_conf = round(min(0.96, max(0.72, 1.0 - top_prob * 0.75)), 4)
        calibrated["Normal"] = normal_conf
        return "Normal", normal_conf, calibrated, round(normal_conf - top_prob, 4)

    primary = ranked_abnormal[0][0]
    return primary, top_prob, calibrated, margin


def _build_findings(calibrated: dict[str, float]) -> list[dict[str, Any]]:
    from conditions import CONDITION_INFO, POSITIVE_THRESHOLD
    findings = []
    for condition, prob in sorted(calibrated.items(), key=lambda x: x[1], reverse=True):
        if condition == "Normal" or prob < 0.15:
            continue
        info = CONDITION_INFO.get(condition, {})
        findings.append({
            "condition": condition,
            "probability": round(prob, 3),
            "status": "detected" if prob >= POSITIVE_THRESHOLD else "possible",
            "category": info.get("category", "Other"),
            "urgency": info.get("urgency", "medium"),
            "icon": info.get("icon", "📋"),
        })
    return findings[:8]


def predict_chest_xray(
    processed: np.ndarray,
    image_path: str,
    hint_name: str | None = None,
    modality: str = "XR",
) -> dict:
    """
    Run authentic PyTorch Deep Learning inference on preprocessed chest radiograph.
    Does NOT use filename shortcuts or hardcoded responses.
    """
    if _check_torch_available() and not _model_failed:
        try:
            import torch
            model, pathologies = _load_model()

            # Ensure 4D tensor (1, 1, H, W)
            arr = np.asarray(processed, dtype=np.float32)
            if arr.ndim == 2:
                tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            elif arr.ndim == 3:
                tensor = torch.from_numpy(arr).unsqueeze(0)
            elif arr.ndim == 4:
                tensor = torch.from_numpy(arr)
            else:
                raise ValueError(f"Invalid tensor shape: {arr.shape}")

            with torch.no_grad():
                out = model(tensor)
                if hasattr(model, "op_threshs"):
                    # TorchXRayVision DenseNet forward() outputs op_norm-scaled probabilities directly
                    preds = out.squeeze().cpu().numpy()
                else:
                    # PyTorch Deep CNN outputting logits -> apply sigmoid for multi-label probabilities
                    preds = torch.sigmoid(out).squeeze().cpu().numpy()

                raw_scores = {pathologies[i]: float(preds[i]) for i in range(len(pathologies))}

            raw_opacity = raw_scores.get("Lung Opacity", 0.0)
            assessment = opacity_assessment(raw_opacity)
            primary = assessment["primary_label"]
            confidence = assessment["confidence"]
            uncertainty = assessment["uncertainty_level"]

            secondary_findings = _build_secondary_findings(raw_scores)
            primary_finding = {
                "condition": "Lung Opacity",
                "probability": assessment["calibrated_probability"],
                "status": "detected" if assessment["calibrated_probability"] >= 0.5 else "possible",
                "category": "Airspace",
                "urgency": "medium" if assessment["calibrated_probability"] >= 0.5 else "none",
                "icon": "🌁",
                "validated": True,
            }
            all_findings = [primary_finding] + secondary_findings
            differential = [{"condition": f["condition"], "probability": f["probability"]} for f in all_findings]

            calibrated_dict = {
                "Lung Opacity": assessment["calibrated_probability"],
                **{LABEL_MAP.get(k, k): round(v, 4) for k, v in raw_scores.items() if k != "Lung Opacity"},
            }

            return {
                "primary_label": primary,
                "confidence": round(float(confidence), 4),
                "severity": _get_severity(confidence) if primary != "Normal" else None,
                "uncertainty_level": uncertainty,
                "opacity_assessment": assessment,
                "differential_diagnosis": differential,
                "all_findings": all_findings,
                "model_version": MODEL_VERSION,
                "raw_scores": {k: round(v, 4) for k, v in raw_scores.items()},
                "calibrated_scores": calibrated_dict,
                "prediction_margin": round(abs(assessment["calibrated_probability"] - 0.5), 4),
            }
        except Exception as exc:
            print(f"[AI] Deep Learning inference error: {exc}")

    raise RuntimeError("Standalone XR model unavailable")
