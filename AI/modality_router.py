"""Route studies to the correct organ/modality pipeline with content validation."""

from modalities import (
    MODALITY_MODELS,
    detect_modality_from_file,
    predict_ct,
    predict_mri,
    predict_ultrasound,
)
from predict import predict_chest_xray
from preprocessing import load_image, preprocess_image
from utils import anatomy_blocks_pipeline, classify_image_anatomy, validate_modality_match
from validator import verify_medical_image


def _invalid_image_result(
    reason: str,
    declared_modality: str,
    telemetry: dict,
) -> dict:
    return {
        "primary_label": "Invalid Image Rejected",
        "confidence": 0.0,
        "severity": None,
        "uncertainty_level": "rejected",
        "differential_diagnosis": [],
        "all_findings": [],
        "model_version": "authenticity-guard",
        "modality": declared_modality,
        "report_extra": reason,
        "is_valid": False,
        "rejection_reason": reason,
        "telemetry": telemetry,
        "model_catalog": MODALITY_MODELS,
    }


def _modality_mismatch_result(
    anatomy: dict,
    validation: dict,
    message: str,
    declared_modality: str,
) -> dict:
    return {
        "primary_label": "Modality Mismatch",
        "confidence": 0.0,
        "severity": None,
        "uncertainty_level": "needs_review",
        "differential_diagnosis": [],
        "all_findings": [],
        "model_version": "anatomy-guard",
        "modality": declared_modality,
        "report_extra": message,
        "anatomy_detection": anatomy,
        "modality_validation": {**validation, "warning": message, "blocked": True},
        "model_catalog": MODALITY_MODELS,
    }


def run_modality_inference(
    image_path: str,
    declared_modality: str | None = None,
    original_filename: str | None = None,
) -> dict:
    hint = original_filename or image_path
    declared = (declared_modality or detect_modality_from_file(image_path, declared_modality)).upper()

    # Medical authenticity and non-medical image verification
    is_valid, reject_reason, auth_telemetry = verify_medical_image(image_path, declared)
    if not is_valid:
        return _invalid_image_result(reject_reason, declared, auth_telemetry)

    gray = load_image(image_path)
    is_montage = auth_telemetry.get("is_multi_slice_montage", False)

    if is_montage and declared == "MR":
        anatomy = {"anatomy": "mri_brain" if declared == "MR" else "ct_chest", "confidence": 0.95, "scores": {}}
        validation = {
            "declared_modality": declared,
            "detected_modality": declared,
            "effective_modality": declared,
            "match": True,
            "warning": None,
            "blocked": False,
        }
        modality = declared
    else:
        anatomy = classify_image_anatomy(gray)
        validation = validate_modality_match(declared, anatomy, hint)
        modality = validation["effective_modality"]

        block_msg = anatomy_blocks_pipeline(modality, anatomy)
        if block_msg:
            return _modality_mismatch_result(anatomy, validation, block_msg, declared)

    if modality == "CT":
        result = predict_ct(image_path, hint_name=hint)
    elif modality == "MR":
        result = predict_mri(image_path, hint_name=hint)
    elif modality == "US":
        result = predict_ultrasound(image_path, hint_name=hint)
    else:
        block_msg = anatomy_blocks_pipeline("XR", anatomy)
        if block_msg:
            return _modality_mismatch_result(anatomy, validation, block_msg, declared)
        processed = preprocess_image(image_path)
        result = predict_chest_xray(processed, image_path, hint_name=hint, modality="XR")
        result["modality"] = "XR"

    result["modality"] = result.get("modality", modality)
    result["anatomy_detection"] = anatomy
    result["modality_validation"] = validation
    if validation.get("warning"):
        result["report_extra"] = (result.get("report_extra") or "") + " " + validation["warning"]
    result["model_catalog"] = MODALITY_MODELS
    result["authenticity_telemetry"] = auth_telemetry
    return result
