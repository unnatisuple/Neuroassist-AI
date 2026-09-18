"""
NeuroAssist AI v2 — FastAPI Application Entry Point

DOCTOR-ONLY PLATFORM. No patient-facing routes exist anywhere in this application.
Every route requires authenticated clinician access.
LLM PROVIDER: Groq API (Configurable via GROQ_API_KEY & GROQ_MODEL).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add("logs/neuroassist.log", rotation="10 MB", retention="30 days", level="DEBUG")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    from backend.config import settings

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Decision Support Mode: {settings.decision_support_mode_only}")

    # Verify Groq LLM Configuration
    from backend.services.ai_service import is_groq_configured
    if is_groq_configured():
        logger.info(f"[AI] Provider: Groq configured successfully at startup (Model: {settings.groq_model}).")
    else:
        logger.warning(
            "[AI] GROQ_API_KEY is not configured or holds a placeholder. "
            "Set GROQ_API_KEY in .env.local to enable live Groq responses."
        )

    # Connect to MongoDB
    from backend.db.mongodb import connect_to_mongo, close_mongo_connection
    try:
        await connect_to_mongo()
    except Exception as e:
        logger.error(f"MongoDB connection failed at startup: {e}")
        logger.warning("Application will start but database features will be unavailable.")

    # Load ML model (if checkpoint exists)
    from backend.routers.mri import load_model
    model_loaded = load_model(settings.best_model_path)
    if not model_loaded:
        logger.warning(
            "ML model not loaded — predictions will return 503 until a trained "
            "model checkpoint is available. NO mock results will be returned."
        )

    # Load RAG index (if available)
    from backend.services.rag_service import load_rag_index
    rag_loaded = load_rag_index()
    if not rag_loaded:
        logger.warning("RAG index not loaded — report generation will work without guideline citations.")

    # Seed admin account if none exists
    await _seed_admin_if_needed()

    logger.info("Application startup complete.")

    yield

    # Shutdown
    await close_mongo_connection()
    logger.info("Application shutdown complete.")


async def _seed_admin_if_needed():
    """Create a default admin account if no admin exists."""
    try:
        from backend.db.mongodb import get_doctors_collection
        from backend.auth.jwt import hash_password
        from backend.schemas import UserRole, VerificationStatus
        from datetime import datetime, timezone
        import uuid

        doctors = get_doctors_collection()
        admin_exists = await doctors.find_one({"role": UserRole.ADMIN})

        if not admin_exists:
            admin_id = str(uuid.uuid4())
            await doctors.insert_one({
                "_id": admin_id,
                "full_name": "System Administrator",
                "email": "admin@neuroassist.ai",
                "password_hash": hash_password("admin123!CHANGE_ME"),
                "specialization": "System Administration",
                "medical_license_number": "ADMIN-000",
                "institution": "NeuroAssist AI",
                "phone": "",
                "role": UserRole.ADMIN,
                "verification_status": VerificationStatus.VERIFIED,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })
            logger.info(
                f"Default admin account created: admin@neuroassist.ai "
                f"(CHANGE THE PASSWORD IMMEDIATELY)"
            )
    except Exception as e:
        logger.warning(f"Could not seed admin account: {e}")


# ---- Create FastAPI app ----
app = FastAPI(
    title="NeuroAssist AI v2",
    description=(
        "Doctor-Only Explainable Clinical Decision Support System for Alzheimer's Disease. "
        "Investigational software — not FDA/CE cleared."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---- CORS ----
from backend.config import settings

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Middleware ----
from backend.middleware.error_handler import GlobalErrorHandlerMiddleware
from backend.middleware.audit import AuditLoggingMiddleware

app.add_middleware(GlobalErrorHandlerMiddleware)
app.add_middleware(AuditLoggingMiddleware)

# ---- Routers ----
from backend.routers.auth import router as auth_router
from backend.routers.mri import router as mri_router
from backend.routers.risk import router as risk_router
from backend.routers.report import router as report_router
from backend.routers.chatbot import router as chatbot_router
from backend.routers.patients import router as patients_router
from backend.routers.admin import router as admin_router
from backend.routers.health import router as health_router

app.include_router(auth_router)
app.include_router(mri_router)
app.include_router(risk_router)
app.include_router(report_router)
app.include_router(chatbot_router)
app.include_router(patients_router)
app.include_router(admin_router)
app.include_router(health_router)

# Direct /api/chat and /chat routes
from backend.routers.chatbot import chat_message
from backend.dependencies import get_optional_doctor
from backend.schemas import ChatRequest, ChatResponse
from typing import Optional
from fastapi import Depends

@app.post("/api/chat", response_model=ChatResponse, tags=["Medical Assistant Chatbot"])
@app.post("/chat", response_model=ChatResponse, tags=["Medical Assistant Chatbot"])
async def direct_chat_endpoint(req: ChatRequest, doctor: Optional[dict] = Depends(get_optional_doctor)):
    """Direct chat endpoint matching standard /api/chat routing."""
    return await chat_message(req, doctor)


@app.get("/")
async def root():
    return {
        "name": "NeuroAssist AI v2",
        "description": "Doctor-Only Clinical Decision Support System for Alzheimer's Disease",
        "version": "2.0.0",
        "disclaimer": (
            "Investigational software. Not a substitute for clinical judgment. "
            "Not FDA/CE cleared."
        ),
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
