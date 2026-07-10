"""
NeuroAssist AI v2 — Health Check Router
Dedicated routes for DB connectivity and Gemini API validation.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
import sys
from backend.config import settings
from backend.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """General health check endpoint."""
    # Check MongoDB
    mongo_ok = False
    try:
        from backend.db.mongodb import get_database
        db = get_database()
        await db.command("ping")
        mongo_ok = True
    except Exception:
        pass

    # Check model
    model_loaded = False
    try:
        from backend.routers.mri import _model
        model_loaded = _model is not None
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if mongo_ok else "degraded",
        version=settings.app_version,
        model_loaded=model_loaded,
        mongodb_connected=mongo_ok,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/health/db")
async def health_db():
    """Perform a real ping check against MongoDB."""
    try:
        from backend.db.mongodb import get_database
        db = get_database()
        await db.command("ping")
        return {"status": "OK", "detail": "MongoDB connected successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB connection failed: {type(e).__name__}: {str(e)}"
        )


@router.get("/health/gemini")
async def health_gemini():
    """Verify Gemini API connectivity with a trivial prompt."""
    if not settings.gemini_api_key or settings.gemini_api_key.startswith("REPLACE_"):
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured or holds a placeholder in .env"
        )
    
    try:
        import google.generativeai as genai
        # Lifespan configures it, but config here ensures clean test client isolation
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        
        # Request a single token response
        response = model.generate_content("ping", generation_config={"max_output_tokens": 5})
        if response and response.text:
            return {"status": "OK", "model": settings.gemini_model}
        raise Exception("Gemini returned empty or invalid text response.")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini API check failed: {type(e).__name__}: {str(e)}"
        )
