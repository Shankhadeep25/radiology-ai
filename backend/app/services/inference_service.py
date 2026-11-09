import sys
from pathlib import Path

AI_DIR = Path(__file__).resolve().parent.parent.parent.parent / "AI"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from predict import predict_chest_xray
from gradcam import generate_gradcam, generate_gradcam_with_telemetry
from report_generator import generate_report
from preprocessing import preprocess_image
from modality_router import run_modality_inference

from app.config import settings
from app.services.encryption_service import decrypt_to_temp


def _resolve_image_path(file_path: str) -> tuple[str, Path | None]:
    """Return readable path; decrypt .enc files to temp."""
    path = Path(file_path)
    if path.suffix == ".enc" or str(path).endswith(".enc"):
        tmp_dir = settings.upload_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        temp = decrypt_to_temp(path, tmp_dir)
        return str(temp), temp
    return file_path, None


def run_gradcam_only(image_path: str, study_id: int, target_label: str = "Pneumonia", modality: str = "XR") -> str:
    if (modality or "").upper() in {"CT", "MR", "MRI", "US"}:
        raise RuntimeError("Grad-CAM not available for baseline models")
    readable, temp_file = _resolve_image_path(image_path)
    try:
        processed = preprocess_image(readable)
        return generate_gradcam(
            image_path=readable,
            processed=processed,
            study_id=study_id,
            output_dir=str(settings.heatmap_dir),
            target_label=target_label,
            modality=modality,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Grad-CAM failed for unsupported or corrupted image") from exc
    finally:
        if temp_file is not None and temp_file.exists():
            temp_file.unlink(missing_ok=True)


def run_report_only(prediction_data: dict, study_id: int, report_meta: dict | None = None) -> dict:
    meta = report_meta or {}
    telemetry = prediction_data.get("telemetry") or prediction_data.get("raw_scores") or {}
    return generate_report(
        primary_label=prediction_data["primary_label"],
        confidence=prediction_data["confidence"],
        severity=prediction_data.get("severity"),
        differential=prediction_data["differential_diagnosis"],
        output_dir=str(settings.report_dir),
        study_id=study_id,
        all_findings=prediction_data.get("all_findings"),
        extra=prediction_data.get("report_extra", ""),
        modality=prediction_data.get("modality", "XR"),
        patient_name=meta.get("patient_name"),
        patient_age=meta.get("patient_age"),
        patient_gender=meta.get("patient_gender"),
        model_version=prediction_data.get("model_version"),
        uncertainty_level=prediction_data.get("uncertainty_level"),
        original_filename=meta.get("original_filename"),
        study_date=meta.get("study_date"),
        telemetry=telemetry,
    )


def run_prediction_only(image_path: str, modality: str | None = None, original_filename: str | None = None) -> dict:
    readable, temp_file = _resolve_image_path(image_path)
    try:
        prediction = run_modality_inference(readable, modality, original_filename)
        if not prediction.get("is_valid", True) or prediction.get("primary_label") == "Invalid Image Rejected":
            raise ValueError(prediction.get("rejection_reason") or "Uploaded file is not a valid medical scan.")
        return prediction
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise ValueError("Inference failed for unsupported or corrupted image") from exc
    finally:
        if temp_file is not None and temp_file.exists():
            temp_file.unlink(missing_ok=True)


def run_full_analysis(
    image_path: str,
    study_id: int,
    modality: str | None = None,
    original_filename: str | None = None,
    report_meta: dict | None = None,
) -> dict:
    readable, temp_file = _resolve_image_path(image_path)
    try:
        prediction = run_modality_inference(readable, modality, original_filename)
        if not prediction.get("is_valid", True) or prediction.get("primary_label") == "Invalid Image Rejected":
            raise ValueError(prediction.get("rejection_reason") or "Uploaded file is not a valid medical scan.")
        processed = preprocess_image(readable)
        effective_modality = prediction.get("modality", "XR")

        if (
            prediction.get("primary_label") in ("Modality Mismatch", "Invalid Image Rejected")
            or prediction.get("status") == "research_baseline"
        ):
            heatmap_path = None
            telemetry = prediction.get("telemetry", {})
        else:
            heatmap_path, grad_tel = generate_gradcam_with_telemetry(
                image_path=readable,
                processed=processed,
                study_id=study_id,
                output_dir=str(settings.heatmap_dir),
                target_label=prediction["primary_label"],
                modality=effective_modality,
            )
            telemetry = {**prediction.get("telemetry", {}), **grad_tel}

        prediction["telemetry"] = telemetry

        report = generate_report(
            primary_label=prediction["primary_label"],
            confidence=prediction["confidence"],
            severity=prediction["severity"],
            differential=prediction["differential_diagnosis"],
            output_dir=str(settings.report_dir),
            study_id=study_id,
            all_findings=prediction.get("all_findings"),
            extra=prediction.get("report_extra", ""),
            modality=effective_modality,
            patient_name=(report_meta or {}).get("patient_name"),
            patient_age=(report_meta or {}).get("patient_age"),
            patient_gender=(report_meta or {}).get("patient_gender"),
            model_version=prediction.get("model_version"),
            uncertainty_level=prediction.get("uncertainty_level"),
            original_filename=original_filename,
            study_date=(report_meta or {}).get("study_date"),
            telemetry=telemetry,
        )

        return {**prediction, "heatmap_path": heatmap_path, "report": report, "telemetry": telemetry}
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise ValueError("Analysis pipeline failed") from exc
    finally:
        if temp_file is not None and temp_file.exists():
            temp_file.unlink(missing_ok=True)
