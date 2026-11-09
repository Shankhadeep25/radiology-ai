import logging
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.config import settings
from app.database import get_db
from app.models.db_models import Patient, User
from app.models.schemas import (
    AnalyzeResponse,
    AuditLogItem,
    AsyncTaskResponse,
    GradcamResponse,
    HistoryItem,
    MetricsResponse,
    OrthancStatus,
    PatientCreate,
    PatientResponse,
    ReportResponse,
    StudyResponse,
    TimelinePoint,
    Token,
    UploadResponse,
    UserCreate,
    UserResponse,
    ChatRequest,
    ChatResponse,
)
from app.services import radiology_service
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
    require_role,
)
from app.services.orthanc_service import get_orthanc_status, list_orthanc_patients
from app.services.chat_service import generate_reply

router = APIRouter()


def _require_user(user: User = Depends(get_current_user)) -> User:
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# ── Auth ──────────────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        role=user_data.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return Token(access_token=create_access_token(data={"sub": user.email}))


@router.get("/auth/me", response_model=UserResponse)
def me(user: User = Depends(_require_user)):
    return user


# ── AI Chat Assistant ─────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat_assistant(body: ChatRequest, user: User = Depends(_require_user)):
    reply = generate_reply(body.message, body.study_context)
    return ChatResponse(reply=reply)


@router.post("/chat/public", response_model=ChatResponse)
def chat_assistant_public(body: ChatRequest):
    """Public demo chat — no login required."""
    reply = generate_reply(body.message, body.study_context)
    return ChatResponse(reply=reply)


# ── Patients ─────────────────────────────────────────────────────────────────

@router.post("/patients", response_model=PatientResponse)
def create_patient(
    patient: PatientCreate,
    db: Session = Depends(get_db),
    user: User = Depends(_require_user),
):
    return radiology_service.create_patient(db, patient)


@router.get("/patients", response_model=list[PatientResponse])
def list_patients(db: Session = Depends(get_db), user: User = Depends(_require_user)):
    return db.query(Patient).order_by(Patient.created_at.desc()).all()


@router.get("/patients/{patient_id}/timeline", response_model=list[TimelinePoint])
def patient_timeline(patient_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return radiology_service.get_patient_timeline(db, patient_id)


# ── Upload & Analysis ─────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_scan(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    modality: str = Form("XR"),
    analyze_async: bool = Form(False),
    db: Session = Depends(get_db),
    user: User = Depends(_require_user),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        result = radiology_service.save_upload(file, patient_id, db, user, modality)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not analyze_async:
        return result
    try:
        from app.celery_app import process_dicom_upload
        from app.services.job_status import set_job_status

        task = process_dicom_upload.delay(result.study_id)
        set_job_status(task.id, "queued", study_id=result.study_id)
        return UploadResponse(
            study_id=result.study_id,
            patient_id=result.patient_id,
            message="Upload successful; analysis queued — poll /tasks/{task_id}",
            task_id=task.id,
        )
    except Exception:
        return UploadResponse(
            study_id=result.study_id,
            patient_id=result.patient_id,
            message="Upload successful; async worker unavailable — use POST /predict/{study_id}",
        )


@router.post("/predict/{study_id}", response_model=AnalyzeResponse)
def predict(study_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    try:
        return radiology_service.analyze_study(db, study_id, user)
    except Exception as e:
        logger.exception("Inference failed for study %s: %s", study_id, e)
        err_msg = str(e)
        if "model unavailable" in err_msg.lower():
            raise HTTPException(status_code=503, detail=err_msg)
        if isinstance(e, ValueError) and "study not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail="Study not found")
        raise HTTPException(status_code=500, detail=err_msg or "Inference failed")


@router.post("/predict/{study_id}/async", response_model=AsyncTaskResponse)
def predict_async(study_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    from app.models.db_models import Study
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    try:
        from app.celery_app import process_dicom_upload
        from app.services.job_status import set_job_status
        task = process_dicom_upload.delay(study_id)
        set_job_status(task.id, status="queued", study_id=study_id)
        return AsyncTaskResponse(
            task_id=task.id,
            status="queued",
            message="Queued — poll GET /api/v1/tasks/{task_id}",
        )
    except Exception:
        raise HTTPException(status_code=503, detail="Async worker unavailable — use POST /predict/{study_id}")


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str, user: User = Depends(_require_user)):
    from app.services.job_status import get_job_status
    cached = get_job_status(task_id)
    if cached:
        return cached
    try:
        from celery.result import AsyncResult
        from app.celery_app import celery_app
        result = AsyncResult(task_id, app=celery_app)
        payload = {"task_id": task_id, "status": (result.status or "unknown").lower()}
        if result.successful() and isinstance(result.result, dict):
            for key in ("study_id", "primary_label"):
                if key in result.result:
                    payload[key] = result.result[key]
        elif result.failed():
            payload["error"] = "Task failed"
        return payload
    except Exception:
        raise HTTPException(status_code=404, detail="Task not found")


@router.post("/gradcam/{study_id}", response_model=GradcamResponse)
def gradcam(study_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    try:
        return radiology_service.generate_gradcam(db, study_id, user)
    except (RuntimeError, ValueError) as e:
        err_msg = str(e).lower()
        if "not available" in err_msg or "model unavailable" in err_msg:
            raise HTTPException(status_code=503, detail=str(e))
        if isinstance(e, ValueError):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        err_msg = str(e).lower()
        if "not available" in err_msg or "model unavailable" in err_msg:
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail="Grad-CAM failed")


@router.post("/report/{study_id}", response_model=ReportResponse)
def report(study_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    try:
        return radiology_service.generate_report_for_study(db, study_id, user)
    except (RuntimeError, ValueError) as e:
        if "model unavailable" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        if isinstance(e, ValueError):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        if "model unavailable" in str(e).lower():
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail="Report generation failed")


# ── Studies ───────────────────────────────────────────────────────────────────

@router.get("/studies/{study_id}", response_model=StudyResponse)
def get_study(study_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    try:
        return radiology_service.get_study(db, study_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/studies/{study_id}/image")
def get_study_image(study_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    from app.models.db_models import Study
    from app.services.encryption_service import decrypt_file
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    path = Path(study.file_path)
    if str(path).endswith(".enc"):
        data = decrypt_file(path)
        media = "application/dicom" if study.file_type == "dicom" else "image/jpeg"
        return Response(content=data, media_type=media)
    return FileResponse(path)


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/history", response_model=list[HistoryItem])
def history(db: Session = Depends(get_db), user: User = Depends(_require_user)):
    return radiology_service.get_history(db)


@router.get("/metrics", response_model=MetricsResponse)
def metrics(db: Session = Depends(get_db), user: User = Depends(_require_user)):
    return radiology_service.get_metrics(db)


# ── Audit Logs (admin/radiologist only) ──────────────────────────────────────

@router.get("/audit-logs", response_model=list[AuditLogItem])
def audit_logs(
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin", "radiologist")),
):
    return radiology_service.get_audit_logs(db)


# ── PACS / Orthanc ────────────────────────────────────────────────────────────

@router.get("/pacs/status", response_model=OrthancStatus)
def pacs_status(user: User = Depends(_require_user)):
    return get_orthanc_status()


@router.get("/pacs/patients")
def pacs_patients(user: User = Depends(require_role("admin", "radiologist", "doctor"))):
    return list_orthanc_patients()


@router.get("/dicomweb/studies")
def dicomweb_qido(patient_id: str | None = None, user: User = Depends(_require_user)):
    """DICOMweb QIDO-RS — query studies from Orthanc."""
    from app.services.dicomweb_service import qido_studies
    return {"resource": "QIDO-RS", "studies": qido_studies(patient_id)}


@router.get("/dicomweb/studies/{study_uid}")
def dicomweb_wado(study_uid: str, user: User = Depends(_require_user)):
    """DICOMweb WADO-RS — retrieve study metadata."""
    from app.services.dicomweb_service import wado_study_metadata
    return wado_study_metadata(study_uid)


@router.get("/fhir/ImagingStudy/{study_id}")
def fhir_imaging_study(study_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    from app.models.db_models import Study
    from app.services.fhir_service import imaging_study_resource
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    return imaging_study_resource(study.patient_id, study.id, study.modality, study.original_filename)


@router.get("/fhir/DiagnosticReport/{study_id}")
def fhir_diagnostic_report(study_id: int, db: Session = Depends(get_db), user: User = Depends(_require_user)):
    from app.models.db_models import Study
    from app.services.fhir_service import diagnostic_report_resource
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    return diagnostic_report_resource(study.patient_id, study.id, study.prediction)


@router.get("/modalities")
def list_modalities():
    import sys
    from pathlib import Path
    ai_dir = Path(__file__).resolve().parent.parent.parent.parent / "AI"
    if str(ai_dir) not in sys.path:
        sys.path.insert(0, str(ai_dir))
    from modalities import MODALITY_MODELS
    return {
        "mvp": "XR",
        "supported": MODALITY_MODELS,
        "note": "X-ray is production MVP (DenseNet121). CT, MRI, and Ultrasound are research baselines (not clinically validated), disabled by default unless ENABLE_BASELINE_MODELS=1.",
    }


# ── Baseline Models Status ───────────────────────────────────────────────────

@router.get("/baseline/status")
def baseline_status():
    import json
    import os
    import sys
    from pathlib import Path

    ai_dir = Path(__file__).resolve().parent.parent.parent.parent / "AI"
    if str(ai_dir) not in sys.path:
        sys.path.insert(0, str(ai_dir))
    from dl_models import normalize_meta, _BASELINE_CACHE

    flag_enabled = os.getenv("ENABLE_BASELINE_MODELS") == "1"
    weights_dir = Path(__file__).resolve().parent.parent.parent.parent / "weights"

    models_status = {}
    for name in ["mri", "us", "ct"]:
        weights_file = weights_dir / f"{name}_densenet121.pt"
        metrics_file = weights_dir / "metrics" / f"{name}_metrics.json"
        weights_present = weights_file.exists() and metrics_file.exists()
        loaded = name in _BASELINE_CACHE

        norm_meta = None
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                norm_meta = normalize_meta(name, raw_meta)
            except Exception:
                norm_meta = None

        models_status[name] = {
            "enabled": flag_enabled,
            "weights_present": weights_present,
            "loaded": loaded,
            "metrics": norm_meta,
        }

    return {
        "enabled": flag_enabled,
        "models": models_status,
        "disclaimer": "Research baselines, not clinically validated. Enabled only when ENABLE_BASELINE_MODELS=1.",
    }


# ── Demo Samples ──────────────────────────────────────────────────────────────

@router.get("/ai/model")
def ai_model_info():
    import sys
    from pathlib import Path
    ai_dir = Path(__file__).resolve().parent.parent.parent.parent / "AI"
    if str(ai_dir) not in sys.path:
        sys.path.insert(0, str(ai_dir))
    from predict import get_model_info
    return get_model_info()


@router.get("/demo/samples")
def demo_samples():
    samples_dir = settings.base_dir / "demo_samples"
    samples = []
    if samples_dir.exists():
        for f in sorted(samples_dir.glob("*")):
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                stem = f.stem.lower()
                if "pneumonia" in stem:
                    label = "X-Ray Pneumonia"
                elif "ct" in stem:
                    label = "CT Adenocarcinoma"
                elif "mri" in stem:
                    label = "MRI Glioma"
                elif "us" in stem:
                    label = "Ultrasound Benign"
                elif "normal" in stem:
                    label = "X-Ray Normal"
                else:
                    label = f.stem
                samples.append({"id": f.stem, "filename": f.name, "label": label,
                                "url": f"/api/v1/demo/samples/{f.name}"})
    return samples


@router.get("/demo/samples/{filename}")
def get_demo_sample(filename: str):
    path = settings.base_dir / "demo_samples" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path)
