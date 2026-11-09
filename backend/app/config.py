from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    app_name: str = "Radiology AI Assistant"
    api_prefix: str = "/api/v1"
    secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'radiology.db'}"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    heatmap_dir: Path = BASE_DIR / "data" / "heatmaps"
    report_dir: Path = BASE_DIR / "data" / "reports"

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    job_status_ttl_seconds: int = 86400
    orthanc_url: str = "http://orthanc:8042"
    orthanc_username: str = "orthanc"
    orthanc_password: str = "orthanc"
    encrypt_uploads: bool = True

    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_name: str = "densenet121-res224-all"
    confidence_review_threshold: float = 0.75

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Allow CORS override from env (Docker)
import os
_cors = os.getenv("CORS_ORIGINS")
if _cors:
    import json
    try:
        settings.cors_origins = json.loads(_cors)
    except Exception:
        pass

for directory in (settings.upload_dir, settings.heatmap_dir, settings.report_dir,
                  BASE_DIR / "data"):
    directory.mkdir(parents=True, exist_ok=True)
