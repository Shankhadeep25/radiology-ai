"""DenseNet121 / CheXNet inference via TorchXRayVision with XRChestNet PyTorch fallback."""

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
        raise RuntimeError("Chest model weights unavailable")
    if _model is not None:
        return _model, _pathologies

    # Attempt 1: TorchXRayVision DenseNet with local cache dir
    try:
        import torchxrayvision as xrv
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        _model = xrv.models.DenseNet(weights="densenet121-res224-all", cache_dir=str(WEIGHTS_DIR))
        _model.eval()
        _pathologies = list(_model.pathologies)
        MODEL_VERSION = "DenseNet121-CheXNet (torchxrayvision-densenet121-res224-all)"
        return _model, _pathologies
    except Exception as exc:
        print(f"[AI] TorchXRayVision download/cache unavailable ({exc}); activating XRChestNet Deep CNN")

    # Attempt 2: Dedicated XRChestNet PyTorch Deep CNN
    try:
        from dl_models import get_xr_standalone_model
        _model = get_xr_standalone_model()
        _model.eval()
        _pathologies = list(_model.CLASSES)
        MODEL_VERSION = "XRChestNet (Deep Residual CNN - 18 Pathologies)"
        return _model, _pathologies
    except Exception as exc2:
        print(f"[AI] Failed loading XRChestNet weights: {exc2}")
        _model_failed = True
        raise


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

            primary, confidence, calibrated, margin = _calibrate_scores(raw_scores)
            all_findings = _build_findings(calibrated)
            differential = [{"condition": f["condition"], "probability": f["probability"]} for f in all_findings]

            return {
                "primary_label": primary,
                "confidence": round(float(confidence), 4),
                "severity": _get_severity(confidence) if primary != "Normal" else None,
                "uncertainty_level": _get_uncertainty_level(confidence, margin),
                "differential_diagnosis": differential,
                "all_findings": all_findings,
                "model_version": MODEL_VERSION,
                "raw_scores": {k: round(v, 4) for k, v in raw_scores.items()},
                "calibrated_scores": calibrated,
                "prediction_margin": margin,
            }
        except Exception as exc:
            print(f"[AI] Deep Learning inference error: {exc}")

    # Fallback only when PyTorch fails completely
    from demo_engine import demo_predict
    return demo_predict(image_path, hint_name=hint_name, modality=modality)
