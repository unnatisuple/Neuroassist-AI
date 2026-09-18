"""
NeuroAssist AI v2 — Pydantic Schemas
All request/response models for the API.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ========================================
# Enums
# ========================================

class DementiaStage(str, Enum):
    NON_DEMENTED = "Non-Demented"
    VERY_MILD = "Very Mild Demented"
    MILD = "Mild Demented"
    MODERATE = "Moderate Demented"
    # Severe is explicitly NOT supported by current training data
    SEVERE_UNSUPPORTED = "Severe Demented (Not Supported)"


class VerificationStatus(str, Enum):
    PENDING = "pending_verification"
    VERIFIED = "verified"
    REJECTED = "rejected"


class UserRole(str, Enum):
    DOCTOR = "doctor"
    ADMIN = "admin"


class ExplainabilityMethod(str, Enum):
    GRADCAM = "gradcam"
    GRADCAM_PLUS = "gradcam++"
    HIRESCAM = "hirescam"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GUIDED_BACKPROP = "guided_backprop"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReportLanguage(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    MARATHI = "mr"


# ========================================
# Auth Schemas
# ========================================

class DoctorLoginRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr



class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    doctor_id: str
    full_name: str
    role: UserRole
    verification_status: VerificationStatus


class DoctorProfile(BaseModel):
    id: str
    full_name: str
    email: str
    specialization: str
    medical_license_number: str
    institution: str
    phone: str
    role: UserRole
    verification_status: VerificationStatus
    created_at: datetime
    last_login: Optional[datetime] = None


# ========================================
# MRI / Prediction Schemas
# ========================================

class BrainRegionAttribution(BaseModel):
    region_name: str
    attribution_level: str  # "High attribution", "Moderate attribution", "Lower attribution"
    attribution_score: float
    clinical_note: Optional[str] = None


class BrainRegionAnalysis(BaseModel):
    regions: List[BrainRegionAttribution]
    is_estimated: bool = True
    methodology: str = (
        "Estimated image-space anatomical localization based on canonical 2D axial brain template mapping. "
        "Not a 3D clinical volumetric segmentation."
    )
    disclaimer: str = (
        "Brain-region attribution represents model attention/attribution and is not equivalent to "
        "a confirmed anatomical lesion or clinical diagnosis."
    )


class XAIVisualization(BaseModel):
    heatmap_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    description: Optional[str] = None


class ExplainabilityData(BaseModel):
    grad_cam: Optional[XAIVisualization] = None
    grad_cam_plus_plus: Optional[XAIVisualization] = None
    integrated_gradients: Optional[XAIVisualization] = None
    hirescam: Optional[XAIVisualization] = None
    brain_regions: Optional[BrainRegionAnalysis] = None


class PredictionResponse(BaseModel):
    prediction_id: str
    predicted_class: DementiaStage
    predicted_class_index: Optional[int] = None
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score")
    class_probabilities: Dict[str, float]
    model_version: str
    model_checksum: str
    inference_latency_ms: float
    timestamp: datetime
    trace_id: str
    mri_file_id: str
    xai_overlays: Optional[Dict[str, str]] = None
    explainability: Optional[ExplainabilityData] = None
    brain_regions: Optional[BrainRegionAnalysis] = None
    disclaimer: str = (
        "Investigational software. Not a substitute for clinical judgment. "
        "Not FDA/CE cleared. This prediction is for decision-support purposes only."
    )
    severe_stage_note: str = (
        "Note: 'Severe Dementia' classification is NOT supported by the current "
        "training data (4-class model: Non-Demented, Very Mild, Mild, Moderate). "
        "This gap is documented in the model card."
    )


class PredictionRecord(BaseModel):
    """Stored in MongoDB for audit trail."""
    prediction_id: str
    doctor_id: str
    patient_id: Optional[str] = None
    mri_file_id: str
    mri_file_path: str
    predicted_class: str
    predicted_class_index: Optional[int] = None
    confidence: float
    class_probabilities: Dict[str, float]
    model_version: str
    model_checksum: str
    inference_latency_ms: float
    trace_id: str
    created_at: datetime
    xai_overlays: Optional[Dict[str, str]] = None
    explainability: Optional[Dict[str, Any]] = None
    brain_regions: Optional[Dict[str, Any]] = None


# ========================================
# Explainability Schemas
# ========================================

class ExplainabilityRequest(BaseModel):
    prediction_id: str
    method: ExplainabilityMethod


class ExplainabilityResponse(BaseModel):
    prediction_id: str
    method: ExplainabilityMethod
    heatmap_base64: str  # Base64-encoded PNG of the heatmap overlay
    overlay_base64: str  # Base64-encoded PNG of heatmap on original scan
    model_version: str
    inference_latency_ms: float
    trace_id: str
    timestamp: datetime
    disclaimer: str = (
        "Investigational software. XAI visualizations highlight regions "
        "the model weighted most heavily — they do not constitute a diagnosis."
    )


class BrainComparisonResponse(BaseModel):
    """Healthy template vs uploaded scan comparison."""
    uploaded_scan_base64: str
    healthy_template_base64: str
    difference_map_base64: str
    description: str
    trace_id: str


# ========================================
# Risk Assessment Schemas
# ========================================

class RiskAssessmentRequest(BaseModel):
    patient_id: Optional[str] = None
    age: int = Field(..., ge=18, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    mmse_score: Optional[float] = Field(None, ge=0, le=30, description="Mini-Mental State Examination")
    moca_score: Optional[float] = Field(None, ge=0, le=30, description="Montreal Cognitive Assessment")
    systolic_bp: Optional[int] = Field(None, ge=60, le=250)
    diastolic_bp: Optional[int] = Field(None, ge=30, le=150)
    has_diabetes: bool = False
    cholesterol_level: Optional[str] = Field(None, pattern="^(normal|borderline|high)$")
    family_history_dementia: bool = False
    smoking_status: Optional[str] = Field(None, pattern="^(never|former|current)$")
    alcohol_consumption: Optional[str] = Field(None, pattern="^(none|moderate|heavy)$")
    physical_activity_level: Optional[str] = Field(None, pattern="^(sedentary|light|moderate|active)$")
    sleep_quality: Optional[str] = Field(None, pattern="^(poor|fair|good|excellent)$")
    bmi: Optional[float] = Field(None, ge=10, le=60)
    depression_history: bool = False
    education_years: Optional[int] = Field(None, ge=0, le=30)


class RiskAssessmentResponse(BaseModel):
    assessment_id: str
    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_level: RiskLevel
    risk_factors: List[Dict[str, Any]]  # [{factor, value, contribution, explanation}]
    protective_factors: List[Dict[str, Any]]
    methodology: str = (
        "Rule-Based Risk Index — This score is computed using a transparent, "
        "clinician-adjustable rule-based algorithm based on published clinical "
        "risk factor weightings. It is NOT an ML-based prediction."
    )
    trace_id: str
    timestamp: datetime
    disclaimer: str = (
        "Investigational software. Risk scores are for clinical decision support only."
    )


# ========================================
# Report Schemas
# ========================================

class ReportGenerateRequest(BaseModel):
    prediction_id: str
    risk_assessment_id: Optional[str] = None
    patient_id: Optional[str] = None
    language: ReportLanguage = ReportLanguage.ENGLISH
    additional_clinical_notes: str = ""


class ReportResponse(BaseModel):
    report_id: str
    prediction_id: str
    doctor_id: str
    patient_id: Optional[str] = None
    language: ReportLanguage
    summary: str
    detailed_findings: str
    xai_interpretation: str
    risk_summary: Optional[str] = None
    recommendations: Dict[str, Any]
    rag_citations: List[Dict[str, str]]  # [{source, excerpt, url}]
    generated_at: datetime
    model_version: str
    trace_id: str
    disclaimer: str = (
        "Investigational software. Not a substitute for clinical judgment. "
        "Not FDA/CE cleared."
    )


# ========================================
# Chatbot Schemas
# ========================================

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    language: ReportLanguage = ReportLanguage.ENGLISH
    conversation: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    answer: Optional[str] = None
    is_on_topic: bool
    citations: List[Dict[str, str]] = []
    usage: Optional[Dict[str, Any]] = None
    trace_id: str
    timestamp: datetime


# ========================================
# Patient (as clinical record) Schemas
# ========================================

class PatientCreateRequest(BaseModel):
    """Patient record created by a doctor — NOT a patient login."""
    full_name: str = Field(..., min_length=2, max_length=100)
    age: int = Field(..., ge=0, le=150)
    gender: str = Field(..., pattern="^(male|female|other)$")
    medical_record_number: str = Field(default="", max_length=50)
    contact_info: str = Field(default="", max_length=200)
    clinical_notes: str = Field(default="", max_length=5000)


class PatientRecord(BaseModel):
    id: str
    doctor_id: str
    full_name: str
    age: int
    gender: str
    medical_record_number: str
    contact_info: str
    clinical_notes: str
    created_at: datetime
    updated_at: datetime
    prediction_ids: List[str] = []
    report_ids: List[str] = []
    risk_assessment_ids: List[str] = []


class PatientHistoryResponse(BaseModel):
    patient: PatientRecord
    predictions: List[PredictionResponse] = []
    reports: List[ReportResponse] = []
    risk_assessments: List[RiskAssessmentResponse] = []


# ========================================
# Admin Schemas
# ========================================

class DoctorVerificationRequest(BaseModel):
    doctor_id: str
    action: str = Field(..., pattern="^(approve|reject)$")
    admin_notes: str = ""


class AuditLogEntry(BaseModel):
    id: str
    doctor_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any] = {}
    ip_address: Optional[str] = None
    trace_id: str
    timestamp: datetime


class AuditLogResponse(BaseModel):
    entries: List[AuditLogEntry]
    total: int
    page: int
    page_size: int


# ========================================
# Recommendation Schemas
# ========================================

class RecommendationResponse(BaseModel):
    stage: DementiaStage
    specialists: List[Dict[str, str]]  # [{name, specialty, rationale}]
    exercises: List[Dict[str, str]]  # [{name, description, frequency}]
    education_resources: List[Dict[str, str]]  # [{title, url, description}]
    caregiver_guidance: List[str]
    disclaimer: str = (
        "Recommendations are rule-based mappings from clinical guidelines "
        "(WHO, Alzheimer's Association). They do not constitute personalized medical advice."
    )


# ========================================
# Common / Utility Schemas
# ========================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    model_loaded: bool
    mongodb_connected: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    error: str
    detail: str
    trace_id: str
    timestamp: datetime
