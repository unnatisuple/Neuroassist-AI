"""
NeuroAssist AI v2 — Audit Logging Middleware
Logs every clinical action (prediction view, report download, etc.) for compliance.
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from backend.db.mongodb import get_audit_log_collection
from backend.dependencies import generate_trace_id
from datetime import datetime, timezone
from loguru import logger

# Routes that should be audit-logged (clinical data access)
AUDITED_PATHS = [
    "/api/mri/predict",
    "/api/mri/explain",
    "/api/mri/upload",
    "/api/risk/assess",
    "/api/report/generate",
    "/api/report/",
    "/api/patients/",
    "/api/chatbot/message",
    "/api/admin/",
]


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs clinical data access for regulatory compliance.
    Every prediction, report view, and data access is recorded.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Only audit relevant paths with successful responses
        path = request.url.path
        should_audit = any(path.startswith(p) for p in AUDITED_PATHS)

        if should_audit and response.status_code < 400:
            try:
                trace_id = getattr(request.state, "trace_id", generate_trace_id())
                doctor_id = "unknown"

                # Extract doctor_id from JWT if present
                auth_header = request.headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    from backend.auth.jwt import decode_token
                    payload = decode_token(auth_header.split(" ")[1])
                    if payload:
                        doctor_id = payload.get("sub", "unknown")

                audit_entry = {
                    "doctor_id": doctor_id,
                    "action": request.method,
                    "resource_type": _classify_resource(path),
                    "resource_path": path,
                    "ip_address": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", ""),
                    "status_code": response.status_code,
                    "trace_id": trace_id,
                    "timestamp": datetime.now(timezone.utc),
                }

                # Fire-and-forget audit log write
                audit_collection = get_audit_log_collection()
                await audit_collection.insert_one(audit_entry)

            except Exception as e:
                # Audit logging should never crash the request
                logger.warning(f"Audit logging failed (non-fatal): {e}")

        return response


def _classify_resource(path: str) -> str:
    """Map URL path to a resource type for the audit log."""
    if "/mri/" in path:
        return "mri_scan"
    elif "/predict" in path:
        return "prediction"
    elif "/explain" in path:
        return "explainability"
    elif "/risk/" in path:
        return "risk_assessment"
    elif "/report/" in path:
        return "clinical_report"
    elif "/patients/" in path:
        return "patient_record"
    elif "/chatbot/" in path:
        return "chatbot_session"
    elif "/admin/" in path:
        return "admin_action"
    return "unknown"
