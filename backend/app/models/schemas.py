from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str
    role: str = "doctor"


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    external_id: Optional[str] = None


class PatientResponse(BaseModel):
    id: int
    name: str
    age: Optional[int]
    gender: Optional[str]
    external_id: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class DifferentialItem(BaseModel):
    condition: str
    probability: float


class FindingItem(BaseModel):
    condition: str
    probability: float
    status: str = "possible"  # detected, possible
    category: str = "Other"
    urgency: str = "medium"


class PredictionResponse(BaseModel):
    id: int
    study_id: int
    primary_label: str
    confidence: float
    severity: Optional[str]
    uncertainty_level: Optional[str]
    differential_diagnosis: list[DifferentialItem] = []
    all_findings: list[FindingItem] = []
    heatmap_url: Optional[str] = None
    model_version: Optional[str]
    modality_validation: Optional[dict] = None
    anatomy_detection: Optional[dict] = None
    volumetrics: Optional[dict] = None
    calibrated: Optional[bool] = None
    in_review_band: Optional[bool] = None
    status: Optional[str] = None
    all_scores: Optional[dict[str, float]] = None
    disclaimer: Optional[str] = None
    model_metrics: Optional[dict] = None
    modality_warning: Optional[str] = None
    created_at: datetime


class ReportResponse(BaseModel):
    id: int
    study_id: int
    findings: str
    impression: str
    pdf_url: Optional[str] = None
    generated_by: str
    created_at: datetime


class StudyResponse(BaseModel):
    id: int
    patient_id: int
    modality: str
    file_type: str
    original_filename: str
    image_url: str
    created_at: datetime
    prediction: Optional[PredictionResponse] = None
    report: Optional[ReportResponse] = None


class UploadResponse(BaseModel):
    study_id: int
    patient_id: int
    message: str
    task_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    study: StudyResponse
    prediction: PredictionResponse
    report: ReportResponse


class HistoryItem(BaseModel):
    study_id: int
    patient_name: str
    primary_label: str
    confidence: float
    created_at: datetime


class MetricsResponse(BaseModel):
    total_studies: int
    total_patients: int
    pneumonia_cases: int
    normal_cases: int
    avg_confidence: float
    needs_review_count: int


class TimelinePoint(BaseModel):
    study_id: int
    date: str
    label: str
    confidence: float
    severity: Optional[str] = None


class AuditLogItem(BaseModel):
    id: int
    user: str
    role: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[int]
    details: Optional[str]
    created_at: str


class GradcamResponse(BaseModel):
    study_id: int
    heatmap_url: str


class OrthancStatus(BaseModel):
    connected: bool
    url: str
    message: str


class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    study_context: Optional[dict] = None


class ChatResponse(BaseModel):
    reply: str
