import json
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.models.db_models import AuditLog, Patient, Prediction, Report, Study, User
from app.models.schemas import (
    AnalyzeResponse,
    DifferentialItem,
    FindingItem,
    HistoryItem,
    MetricsResponse,
    PatientCreate,
    PredictionResponse,
    ReportResponse,
    StudyResponse,
    UploadResponse,
)
from app.services.inference_service import run_full_analysis, run_gradcam_only, run_report_only, run_prediction_only
from app.services.encryption_service import encrypt_file
from app.services.orthanc_service import upload_to_orthanc, orthanc_available

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


def _log_action(db: Session, user: User | None, action: str, resource_type: str, resource_id: int, details: str = ""):
    db.add(AuditLog(
        user_id=user.id if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
    ))


def create_patient(db: Session, patient_data: PatientCreate) -> Patient:
    patient = Patient(**patient_data.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def save_upload(file: UploadFile, patient_id: int, db: Session, user: User | None = None, modality: str = "XR") -> UploadResponse:
    ext = Path(file.filename or "scan.png").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    file_type = "dicom" if ext in {".dcm", ".dicom"} else ext.lstrip(".")
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = settings.upload_dir / unique_name

    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Image authenticity and non-medical image verification
    from validator import verify_medical_image
    is_valid, reject_reason, _ = verify_medical_image(dest, modality)
    if not is_valid:
        dest.unlink(missing_ok=True)
        raise ValueError(reject_reason)

    # Encrypt at rest
    if settings.encrypt_uploads:
        dest = encrypt_file(dest)

    # Sync DICOM to Orthanc PACS if available
    orthanc_id = None
    if file_type == "dicom" and orthanc_available():
        try:
            from app.services.encryption_service import decrypt_file
            raw = decrypt_file(dest) if settings.encrypt_uploads else dest.read_bytes()
            result = upload_to_orthanc(raw, file.filename or unique_name)
            orthanc_id = result.get("ID")
        except Exception:
            pass

    study = Study(
        patient_id=patient_id,
        modality=modality.upper() if modality else "XR",
        file_path=str(dest),
        file_type=file_type,
        original_filename=file.filename or unique_name,
        uploaded_by=user.id if user else None,
    )
    db.add(study)
    db.commit()
    db.refresh(study)
    _log_action(db, user, "upload", "study", study.id, file.filename or "")
    db.commit()

    return UploadResponse(study_id=study.id, patient_id=patient_id, message="Upload successful")


def analyze_study(db: Session, study_id: int, user: User | None = None) -> AnalyzeResponse:
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise ValueError("Study not found")

    result = run_full_analysis(
        study.file_path,
        study_id,
        study.modality,
        study.original_filename,
        report_meta={
            "patient_name": study.patient.name if study.patient else None,
            "patient_age": study.patient.age if study.patient else None,
            "patient_gender": study.patient.gender if study.patient else None,
            "study_date": study.created_at,
        },
    )

    prediction = Prediction(
        study_id=study.id,
        primary_label=result["primary_label"],
        confidence=result["confidence"],
        severity=result["severity"],
        uncertainty_level=result["uncertainty_level"],
        differential_diagnosis=json.dumps({
            "differential": result["differential_diagnosis"],
            "all_findings": result.get("all_findings", []),
            "modality_validation": result.get("modality_validation"),
            "anatomy_detection": result.get("anatomy_detection"),
            "raw_scores": result.get("raw_scores"),
            "telemetry": result.get("telemetry"),
            "calibrated": result.get("calibrated"),
            "in_review_band": result.get("in_review_band"),
            "status": result.get("status"),
            "all_scores": result.get("all_scores"),
            "disclaimer": result.get("disclaimer"),
            "model_metrics": result.get("model_metrics"),
            "modality_warning": result.get("modality_warning"),
        }),
        heatmap_path=result.get("heatmap_path"),
        model_version=result.get("model_version"),
    )
    db.add(prediction)

    report = Report(
        study_id=study.id,
        findings=result["report"]["findings"],
        impression=result["report"]["impression"],
        pdf_path=result["report"].get("pdf_path"),
        generated_by="ai-clinical-engine",
    )
    db.add(report)
    db.commit()
    db.refresh(prediction)
    db.refresh(report)
    db.refresh(study)

    _log_action(db, user, "analyze", "study", study.id, result["primary_label"])
    db.commit()

    return AnalyzeResponse(
        study=_study_to_response(study),
        prediction=_prediction_to_response(prediction),
        report=_report_to_response(report),
    )


def generate_gradcam(db: Session, study_id: int, user: User | None = None) -> dict:
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise ValueError("Study not found")
    label = study.prediction.primary_label if study.prediction else "Pneumonia"
    modality = study.modality or "XR"
    heatmap_path = run_gradcam_only(study.file_path, study_id, label, modality=modality)
    if study.prediction:
        study.prediction.heatmap_path = heatmap_path
        db.commit()
    _log_action(db, user, "gradcam", "study", study_id)
    db.commit()
    return {"study_id": study_id, "heatmap_url": f"/api/v1/files/heatmaps/{Path(heatmap_path).name}"}


def generate_report_for_study(db: Session, study_id: int, user: User | None = None) -> ReportResponse:
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise ValueError("Study not found")
    if not study.prediction:
        pred_data = run_prediction_only(study.file_path, study.modality, study.original_filename)
    else:
        diff_raw, findings_raw, meta = _parse_diagnosis_json(study.prediction.differential_diagnosis)
        pred_data = {
            "primary_label": study.prediction.primary_label,
            "confidence": study.prediction.confidence,
            "severity": study.prediction.severity,
            "differential_diagnosis": diff_raw,
            "all_findings": findings_raw,
            "modality": study.modality,
            "telemetry": meta.get("telemetry"),
            "model_version": study.prediction.model_version,
            "uncertainty_level": study.prediction.uncertainty_level,
        }
    report_data = run_report_only(
        pred_data,
        study_id,
        report_meta={
            "patient_name": study.patient.name if study.patient else None,
            "patient_age": study.patient.age if study.patient else None,
            "patient_gender": study.patient.gender if study.patient else None,
            "original_filename": study.original_filename,
            "study_date": study.created_at,
        },
    )
    if study.report:
        study.report.findings = report_data["findings"]
        study.report.impression = report_data["impression"]
        study.report.pdf_path = report_data.get("pdf_path")
        study.report.generated_by = "ai-clinical-engine"
    else:
        db.add(Report(
            study_id=study.id,
            findings=report_data["findings"],
            impression=report_data["impression"],
            pdf_path=report_data.get("pdf_path"),
            generated_by="ai-clinical-engine",
        ))
    db.commit()
    db.refresh(study)
    _log_action(db, user, "report", "study", study_id)
    db.commit()
    return _report_to_response(study.report)


def get_patient_timeline(db: Session, patient_id: int) -> list[dict]:
    rows = (
        db.query(Study, Prediction)
        .outerjoin(Prediction, Prediction.study_id == Study.id)
        .filter(Study.patient_id == patient_id)
        .order_by(Study.created_at.asc())
        .all()
    )
    timeline = []
    for study, prediction in rows:
        timeline.append({
            "study_id": study.id,
            "date": study.created_at.isoformat(),
            "label": prediction.primary_label if prediction else "Pending",
            "confidence": prediction.confidence if prediction else 0,
            "severity": prediction.severity if prediction else None,
        })
    return timeline


def get_audit_logs(db: Session, limit: int = 100) -> list[dict]:
    rows = (
        db.query(AuditLog, User)
        .outerjoin(User, AuditLog.user_id == User.id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [{
        "id": log.id,
        "user": user.full_name if user else "System",
        "role": user.role if user else None,
        "action": log.action,
        "resource_type": log.resource_type,
        "resource_id": log.resource_id,
        "details": log.details,
        "created_at": log.created_at.isoformat(),
    } for log, user in rows]


def get_study(db: Session, study_id: int) -> StudyResponse:
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise ValueError("Study not found")
    return _study_to_response(study)


def get_history(db: Session, limit: int = 50) -> list[HistoryItem]:
    rows = (
        db.query(Study, Patient, Prediction)
        .join(Patient, Study.patient_id == Patient.id)
        .outerjoin(Prediction, Prediction.study_id == Study.id)
        .order_by(Study.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        HistoryItem(
            study_id=study.id,
            patient_name=patient.name,
            primary_label=prediction.primary_label if prediction else "Pending",
            confidence=prediction.confidence if prediction else 0.0,
            created_at=study.created_at,
        )
        for study, patient, prediction in rows
    ]


def get_metrics(db: Session) -> MetricsResponse:
    total_studies = db.query(Study).count()
    total_patients = db.query(Patient).count()
    predictions = db.query(Prediction).all()
    pneumonia_cases = sum(1 for p in predictions if "pneumonia" in p.primary_label.lower() or "opacity" in p.primary_label.lower())
    normal_cases = sum(1 for p in predictions if "normal" in p.primary_label.lower())
    avg_confidence = sum(p.confidence for p in predictions) / len(predictions) if predictions else 0.0
    needs_review = sum(1 for p in predictions if p.uncertainty_level == "needs_review")
    return MetricsResponse(
        total_studies=total_studies,
        total_patients=total_patients,
        pneumonia_cases=pneumonia_cases,
        normal_cases=normal_cases,
        avg_confidence=round(avg_confidence, 3),
        needs_review_count=needs_review,
    )


def _study_to_response(study: Study) -> StudyResponse:
    return StudyResponse(
        id=study.id,
        patient_id=study.patient_id,
        modality=study.modality,
        file_type=study.file_type,
        original_filename=study.original_filename,
        image_url=f"/api/v1/files/uploads/{Path(study.file_path).name}",
        created_at=study.created_at,
        prediction=_prediction_to_response(study.prediction) if study.prediction else None,
        report=_report_to_response(study.report) if study.report else None,
    )


def _parse_diagnosis_json(raw: str | None) -> tuple[list, list, dict]:
    if not raw:
        return [], [], {}
    data = json.loads(raw)
    if isinstance(data, list):
        return data, data, {}
    meta = {
        "modality_validation": data.get("modality_validation"),
        "anatomy_detection": data.get("anatomy_detection"),
        "raw_scores": data.get("raw_scores") or {},
        "telemetry": data.get("telemetry") or {},
    }
    return data.get("differential", []), data.get("all_findings", []), meta


def _prediction_to_response(prediction: Prediction) -> PredictionResponse:
    diff_raw, findings_raw, meta = _parse_diagnosis_json(prediction.differential_diagnosis)
    heatmap_url = None
    if prediction.heatmap_path:
        heatmap_url = f"/api/v1/files/heatmaps/{Path(prediction.heatmap_path).name}"
    raw_scores = meta.get("raw_scores") or {}
    telemetry = meta.get("telemetry") or raw_scores.get("volumetrics") or raw_scores
    volumetrics = telemetry if isinstance(telemetry, dict) else None
    return PredictionResponse(
        id=prediction.id,
        study_id=prediction.study_id,
        primary_label=prediction.primary_label,
        confidence=prediction.confidence,
        severity=prediction.severity,
        uncertainty_level=prediction.uncertainty_level,
        differential_diagnosis=[DifferentialItem(**d) for d in diff_raw],
        all_findings=[FindingItem(**f) for f in findings_raw],
        heatmap_url=heatmap_url,
        model_version=prediction.model_version,
        modality_validation=meta.get("modality_validation"),
        anatomy_detection=meta.get("anatomy_detection"),
        volumetrics=volumetrics,
        calibrated=meta.get("calibrated"),
        in_review_band=meta.get("in_review_band"),
        status=meta.get("status"),
        all_scores=meta.get("all_scores"),
        disclaimer=meta.get("disclaimer"),
        model_metrics=meta.get("model_metrics"),
        modality_warning=meta.get("modality_warning"),
        created_at=prediction.created_at,
    )


def _report_to_response(report: Report) -> ReportResponse:
    pdf_url = None
    if report.pdf_path:
        pdf_url = f"/api/v1/files/reports/{Path(report.pdf_path).name}"
    return ReportResponse(
        id=report.id,
        study_id=report.study_id,
        findings=report.findings,
        impression=report.impression,
        pdf_url=pdf_url,
        generated_by=report.generated_by,
        created_at=report.created_at,
    )
