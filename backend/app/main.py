from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path

from app.api.routes import router
from app.config import settings
from app.database import Base, engine, SessionLocal
from app.models.db_models import User
from app.services.auth_service import get_password_hash

AI_DIR = Path(__file__).resolve().parent.parent.parent / "AI"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

# Eagerly import AI validation layer so uvicorn --reload picks up AI
# changes when this file is touched.
try:
    import validator as _ai_validator  # noqa: F401
    import modality_router as _ai_router  # noqa: F401
except Exception:
    pass

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.app_name,
    description="Explainable AI Radiology Assistant — DenseNet121 + Grad-CAM + DICOM",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_prefix)
app.mount("/api/v1/files/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")
app.mount("/api/v1/files/heatmaps", StaticFiles(directory=str(settings.heatmap_dir)), name="heatmaps")
app.mount("/api/v1/files/reports", StaticFiles(directory=str(settings.report_dir)), name="reports")


def seed_default_users():
    db = SessionLocal()
    defaults = [
        ("doctor@radiology.ai", "Dr. Demo", "demo1234", "doctor"),
        ("radiologist@radiology.ai", "Dr. Radiologist", "demo1234", "radiologist"),
        ("admin@radiology.ai", "Admin", "demo1234", "admin"),
    ]
    try:
        for email, name, pwd, role in defaults:
            if not db.query(User).filter(User.email == email).first():
                db.add(User(email=email, full_name=name,
                            hashed_password=get_password_hash(pwd), role=role))
        db.commit()
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    seed_default_users()
    try:
        from predict import preload_model, get_model_info
        preload_model()
        print(f"[Startup] {get_model_info()}")
    except Exception as e:
        print(f"[Startup] AI model: demo engine fallback ({e})")


@app.get("/health")
def health():
    try:
        from predict import get_model_info
        model_info = get_model_info()
    except Exception:
        model_info = {"loaded": False, "model": "demo-engine"}
    return {"status": "healthy", "service": settings.app_name, "ai_model": model_info}
