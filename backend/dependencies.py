"""
NeuroAssist AI v2 — FastAPI Dependencies
Auth dependencies, DB dependencies, and common utilities.
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
from backend.auth.jwt import decode_token
from backend.db.mongodb import get_doctors_collection
from backend.schemas import VerificationStatus, UserRole
import uuid

from typing import Dict, Any, Optional

security = HTTPBearer(auto_error=False)


async def get_current_doctor(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    Extract and verify the current authenticated doctor from JWT.
    Supports either standard Authorization header or a ?token query parameter.
    """
    token = None
    if credentials:
        token = credentials.credentials
    else:
        token = request.query_params.get("token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or expired authentication token.",
        )

    payload = decode_token(token)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
        )

    doctor_id = payload.get("sub")
    if not doctor_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload — missing subject.",
        )

    doctors = get_doctors_collection()
    doctor = await doctors.find_one({"_id": doctor_id})
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Doctor account not found.",
        )

    return doctor


async def require_verified_doctor(
    doctor: Dict[str, Any] = Depends(get_current_doctor),
) -> Dict[str, Any]:
    """Require the doctor to be verified (approved by admin)."""
    if doctor.get("verification_status") != VerificationStatus.VERIFIED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Your account is pending verification. "
                "An administrator must approve your medical license before you can access clinical features. "
                f"Current status: {doctor.get('verification_status', 'unknown')}."
            ),
        )
    return doctor


async def get_optional_doctor(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Extract authenticated doctor if token provided, else None."""
    try:
        return await get_current_doctor(request, credentials)
    except Exception:
        return None


async def require_admin(
    doctor: Dict[str, Any] = Depends(get_current_doctor),
) -> Dict[str, Any]:
    """Require admin role."""
    if doctor.get("role") != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires administrator privileges.",
        )
    return doctor


def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking and support."""
    return f"NA-{uuid.uuid4().hex[:12].upper()}"
