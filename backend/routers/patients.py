"""
NeuroAssist AI v2 — Patient Record Router
Patients are clinical records managed by doctors — NOT user accounts.
There is NO patient login or patient-facing functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
from backend.schemas import (
    PatientCreateRequest, PatientRecord, PatientHistoryResponse,
    PredictionResponse, RiskAssessmentResponse,
)
from backend.dependencies import require_verified_doctor, generate_trace_id
from backend.db.mongodb import (
    get_patients_collection, get_predictions_collection,
    get_reports_collection, get_risk_assessments_collection,
)
import uuid

router = APIRouter(prefix="/api/patients", tags=["Patient Records"])


@router.post("/", response_model=PatientRecord, status_code=status.HTTP_201_CREATED)
async def create_patient_record(
    req: PatientCreateRequest,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Create a patient clinical record.
    This is a record managed by the doctor — NOT a patient account.
    Patients do not have login access to this platform.
    """
    patient_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    patient_doc = {
        "_id": patient_id,
        "doctor_id": doctor["_id"],
        "full_name": req.full_name,
        "age": req.age,
        "gender": req.gender,
        "medical_record_number": req.medical_record_number,
        "contact_info": req.contact_info,
        "clinical_notes": req.clinical_notes,
        "created_at": now,
        "updated_at": now,
        "prediction_ids": [],
        "report_ids": [],
        "risk_assessment_ids": [],
    }

    patients = get_patients_collection()
    await patients.insert_one(patient_doc)

    return PatientRecord(id=patient_id, **{k: v for k, v in patient_doc.items() if k != "_id"})


@router.get("/", response_model=list)
async def list_patients(
    doctor: dict = Depends(require_verified_doctor),
):
    """List all patient records for the current doctor."""
    patients = get_patients_collection()
    cursor = patients.find({"doctor_id": doctor["_id"]}).sort("updated_at", -1).limit(200)

    result = []
    async for p in cursor:
        result.append({
            "id": p["_id"],
            "full_name": p["full_name"],
            "age": p["age"],
            "gender": p["gender"],
            "medical_record_number": p.get("medical_record_number", ""),
            "created_at": p["created_at"],
            "updated_at": p["updated_at"],
            "num_predictions": len(p.get("prediction_ids", [])),
            "num_reports": len(p.get("report_ids", [])),
        })

    return result


@router.get("/{patient_id}")
async def get_patient(
    patient_id: str,
    doctor: dict = Depends(require_verified_doctor),
):
    """Get a single patient record."""
    patients = get_patients_collection()
    patient = await patients.find_one({"_id": patient_id, "doctor_id": doctor["_id"]})

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient record {patient_id} not found.",
        )

    return PatientRecord(id=patient["_id"], **{k: v for k, v in patient.items() if k != "_id"})


@router.get("/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Get longitudinal patient history: all predictions, reports, and risk assessments.
    Used for the trend charts (stage/risk over time) in the dashboard.
    """
    trace_id = generate_trace_id()
    patients = get_patients_collection()
    patient = await patients.find_one({"_id": patient_id, "doctor_id": doctor["_id"]})

    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient record {patient_id} not found. Trace ID: {trace_id}",
        )

    # Fetch related predictions
    predictions_coll = get_predictions_collection()
    predictions = []
    async for pred in predictions_coll.find(
        {"patient_id": patient_id, "doctor_id": doctor["_id"]}
    ).sort("created_at", -1):
        predictions.append({
            "prediction_id": pred["_id"],
            "predicted_class": pred["predicted_class"],
            "confidence": pred["confidence"],
            "model_version": pred["model_version"],
            "created_at": pred["created_at"],
        })

    # Fetch related reports
    reports_coll = get_reports_collection()
    reports = []
    async for report in reports_coll.find(
        {"patient_id": patient_id, "doctor_id": doctor["_id"]}
    ).sort("created_at", -1):
        reports.append({
            "report_id": report["_id"],
            "summary": report.get("summary", "")[:200],
            "language": report["language"],
            "created_at": report["created_at"],
        })

    # Fetch related risk assessments
    risk_coll = get_risk_assessments_collection()
    risk_assessments = []
    async for ra in risk_coll.find(
        {"patient_id": patient_id, "doctor_id": doctor["_id"]}
    ).sort("created_at", -1):
        risk_assessments.append({
            "assessment_id": ra["_id"],
            "risk_score": ra["risk_score"],
            "risk_level": ra["risk_level"],
            "created_at": ra["created_at"],
        })

    return {
        "patient": PatientRecord(id=patient["_id"], **{k: v for k, v in patient.items() if k != "_id"}),
        "predictions": predictions,
        "reports": reports,
        "risk_assessments": risk_assessments,
    }


@router.put("/{patient_id}")
async def update_patient(
    patient_id: str,
    req: PatientCreateRequest,
    doctor: dict = Depends(require_verified_doctor),
):
    """Update a patient record."""
    patients = get_patients_collection()
    result = await patients.update_one(
        {"_id": patient_id, "doctor_id": doctor["_id"]},
        {"$set": {
            "full_name": req.full_name,
            "age": req.age,
            "gender": req.gender,
            "medical_record_number": req.medical_record_number,
            "contact_info": req.contact_info,
            "clinical_notes": req.clinical_notes,
            "updated_at": datetime.now(timezone.utc),
        }},
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient record {patient_id} not found.",
        )

    return {"message": "Patient record updated successfully."}
