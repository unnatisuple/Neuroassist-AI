"""
NeuroAssist AI v2 — Admin Router
Doctor verification, audit log viewing.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
from backend.schemas import DoctorVerificationRequest, VerificationStatus
from backend.dependencies import require_admin, generate_trace_id
from backend.db.mongodb import get_doctors_collection, get_audit_log_collection
from loguru import logger

router = APIRouter(prefix="/api/admin", tags=["Administration"])


@router.get("/doctors/pending")
async def list_pending_doctors(admin: dict = Depends(require_admin)):
    """List all doctors awaiting verification."""
    doctors = get_doctors_collection()
    cursor = doctors.find(
        {"verification_status": VerificationStatus.PENDING},
        {"password_hash": 0},  # Never expose password hashes
    ).sort("created_at", -1)

    result = []
    async for doc in cursor:
        result.append({
            "id": doc["_id"],
            "full_name": doc["full_name"],
            "email": doc["email"],
            "specialization": doc["specialization"],
            "medical_license_number": doc["medical_license_number"],
            "institution": doc.get("institution", ""),
            "created_at": doc["created_at"],
        })

    return {"pending_doctors": result, "count": len(result)}


@router.post("/doctors/verify")
async def verify_doctor(
    req: DoctorVerificationRequest,
    admin: dict = Depends(require_admin),
):
    """Approve or reject a doctor's registration."""
    trace_id = generate_trace_id()
    doctors = get_doctors_collection()

    doctor = await doctors.find_one({"_id": req.doctor_id})
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Doctor {req.doctor_id} not found. Trace ID: {trace_id}",
        )

    new_status = (
        VerificationStatus.VERIFIED if req.action == "approve"
        else VerificationStatus.REJECTED
    )

    await doctors.update_one(
        {"_id": req.doctor_id},
        {"$set": {
            "verification_status": new_status,
            "verified_by": admin["_id"],
            "verified_at": datetime.now(timezone.utc),
            "admin_notes": req.admin_notes,
        }},
    )

    logger.info(
        f"[{trace_id}] Admin {admin['_id']} {req.action}d doctor {req.doctor_id}"
    )

    return {
        "message": f"Doctor {req.action}d successfully.",
        "doctor_id": req.doctor_id,
        "new_status": new_status,
        "trace_id": trace_id,
    }


@router.get("/audit-log")
async def get_audit_log(
    page: int = 1,
    page_size: int = 50,
    doctor_id: str = None,
    resource_type: str = None,
    admin: dict = Depends(require_admin),
):
    """
    Query the audit log. Every prediction, report view, and clinical data
    access is logged here for regulatory compliance.
    """
    audit_log = get_audit_log_collection()

    query = {}
    if doctor_id:
        query["doctor_id"] = doctor_id
    if resource_type:
        query["resource_type"] = resource_type

    total = await audit_log.count_documents(query)
    skip = (page - 1) * page_size

    cursor = audit_log.find(query).sort("timestamp", -1).skip(skip).limit(page_size)

    entries = []
    async for entry in cursor:
        entries.append({
            "id": str(entry["_id"]),
            "doctor_id": entry["doctor_id"],
            "action": entry["action"],
            "resource_type": entry["resource_type"],
            "resource_path": entry.get("resource_path", ""),
            "ip_address": entry.get("ip_address", ""),
            "trace_id": entry.get("trace_id", ""),
            "timestamp": entry["timestamp"],
        })

    return {
        "entries": entries,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/stats")
async def admin_stats(admin: dict = Depends(require_admin)):
    """Get platform statistics for the admin dashboard."""
    doctors = get_doctors_collection()

    total_doctors = await doctors.count_documents({})
    pending = await doctors.count_documents({"verification_status": VerificationStatus.PENDING})
    verified = await doctors.count_documents({"verification_status": VerificationStatus.VERIFIED})

    from backend.db.mongodb import get_predictions_collection, get_reports_collection, get_patients_collection
    total_predictions = await get_predictions_collection().count_documents({})
    total_reports = await get_reports_collection().count_documents({})
    total_patients = await get_patients_collection().count_documents({})

    return {
        "doctors": {"total": total_doctors, "pending": pending, "verified": verified},
        "predictions": total_predictions,
        "reports": total_reports,
        "patients": total_patients,
    }
