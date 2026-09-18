"""
NeuroAssist AI v2 — Health Check Router
Dedicated routes for DB connectivity and Groq API validation.
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
        import asyncio
        from backend.db.mongodb import _use_fallback, get_database
        if not _use_fallback:
            db = get_database()
            await asyncio.wait_for(db.command("ping"), timeout=1.0)
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


@router.get("/health/groq")
@router.get("/health/gemini")
async def health_groq():
    """Verify Groq API configuration and connectivity."""
    from backend.services.ai_service import is_groq_configured, get_groq_client, GroqConfigurationError

    if not is_groq_configured():
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured or holds a placeholder in .env.local"
        )

    try:
        from backend.services.ai_service import chat_completion
        res = chat_completion(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            trace_id="health-groq",
        )
        return {"status": "OK", "provider": "Groq", "model": res.get("model", settings.groq_model)}
    except GroqConfigurationError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Groq API check failed: {type(e).__name__}: {str(e)}"
        )
