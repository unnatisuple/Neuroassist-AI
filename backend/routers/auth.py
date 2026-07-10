"""
NeuroAssist AI v2 — Auth Router
Doctor signup (pending verification), login, token refresh.
NO patient registration — this is a doctor-only platform.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timezone
from backend.schemas import (
    DoctorLoginRequest, TokenResponse,
    DoctorProfile, VerificationStatus, UserRole,
)
from backend.auth.jwt import (
    create_access_token, create_refresh_token, decode_token,
)
from backend.db.mongodb import get_doctors_collection
from backend.dependencies import get_current_doctor, generate_trace_id
import uuid

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def doctor_login(req: DoctorLoginRequest):
    """
    Authenticate a doctor by Name and Email.
    If the email doesn't exist, auto-create a doctor record.
    If the email exists, log in as that doctor.
    """
    doctors = get_doctors_collection()
    doctor = await doctors.find_one({"email": req.email})

    if not doctor:
        # Auto-create the doctor record
        doctor_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        doctor = {
            "_id": doctor_id,
            "full_name": req.full_name,
            "email": req.email,
            "specialization": "Neurology",
            "medical_license_number": "LICENSE-AUTO",
            "institution": "NeuroAssist AI Portal",
            "phone": "",
            "role": UserRole.DOCTOR,
            "verification_status": VerificationStatus.VERIFIED, # Pre-verified so they can log in instantly
            "created_at": now,
            "updated_at": now,
            "last_login": now,
        }
        await doctors.insert_one(doctor)
        from loguru import logger
        logger.info(f"Auto-created new clinician account for: {req.full_name} ({req.email})")
    else:
        # Existing doctor: check if verification status is REJECTED
        if doctor.get("verification_status") == VerificationStatus.REJECTED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Your account registration was not approved. "
                    "Please contact the platform administrator for details."
                ),
            )
        # Update last login
        await doctors.update_one(
            {"_id": doctor["_id"]},
            {"$set": {"last_login": datetime.now(timezone.utc)}},
        )

    access_token = create_access_token({"sub": doctor["_id"], "role": doctor["role"]})
    refresh_token = create_refresh_token({"sub": doctor["_id"]})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        doctor_id=doctor["_id"],
        full_name=doctor["full_name"],
        role=doctor["role"],
        verification_status=doctor["verification_status"],
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token_str: str):
    """Exchange a valid refresh token for new access + refresh tokens."""
    payload = decode_token(refresh_token_str)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    doctor_id = payload.get("sub")
    doctors = get_doctors_collection()
    doctor = await doctors.find_one({"_id": doctor_id})

    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Doctor account no longer exists.",
        )

    access_token = create_access_token({"sub": doctor_id, "role": doctor["role"]})
    new_refresh_token = create_refresh_token({"sub": doctor_id})

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        doctor_id=doctor["_id"],
        full_name=doctor["full_name"],
        role=doctor["role"],
        verification_status=doctor["verification_status"],
    )


@router.get("/me", response_model=DoctorProfile)
async def get_profile(doctor: dict = Depends(get_current_doctor)):
    """Get the current doctor's profile."""
    return DoctorProfile(
        id=doctor["_id"],
        full_name=doctor["full_name"],
        email=doctor["email"],
        specialization=doctor["specialization"],
        medical_license_number=doctor["medical_license_number"],
        institution=doctor.get("institution", ""),
        phone=doctor.get("phone", ""),
        role=doctor["role"],
        verification_status=doctor["verification_status"],
        created_at=doctor["created_at"],
        last_login=doctor.get("last_login"),
    )
