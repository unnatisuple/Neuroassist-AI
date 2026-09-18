"""
NeuroAssist AI v2 — Report Generation Router
RAG-grounded clinical reports via Groq. Multilingual PDF export.
Every field populated from real computed data — no placeholder text.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from datetime import datetime, timezone
from backend.schemas import (
    ReportGenerateRequest, ReportResponse, ReportLanguage,
)
from backend.dependencies import require_verified_doctor, generate_trace_id
from backend.db.mongodb import (
    get_reports_collection, get_predictions_collection,
    get_risk_assessments_collection,
)
from backend.config import settings
from loguru import logger
import uuid
import os

router = APIRouter(prefix="/api/report", tags=["Clinical Reports"])


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    req: ReportGenerateRequest,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Generate a RAG-grounded clinical report using Gemini.
    All content is derived from real prediction data, risk assessment, and
    retrieved clinical guideline excerpts — never placeholder text.
    """
    trace_id = generate_trace_id()

    # Fetch the prediction
    predictions = get_predictions_collection()
    prediction = await predictions.find_one({"_id": req.prediction_id, "doctor_id": doctor["_id"]})

    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {req.prediction_id} not found. Trace ID: {trace_id}",
        )

    # Fetch risk assessment if provided
    risk_data = None
    if req.risk_assessment_id:
        assessments = get_risk_assessments_collection()
        risk_data = await assessments.find_one({"_id": req.risk_assessment_id})

    # ---- RAG Retrieval ----
    rag_citations = []
    rag_context = ""
    try:
        from backend.services.rag_service import retrieve_relevant_guidelines
        stage = prediction["predicted_class"]
        retrieved = retrieve_relevant_guidelines(stage)
        rag_context = "\n\n".join([chunk["text"] for chunk in retrieved])
        rag_citations = [{"source": c["source"], "excerpt": c["text"][:200], "url": c.get("url", "")}
                         for c in retrieved]
    except Exception as e:
        logger.warning(f"[{trace_id}] RAG retrieval failed (non-fatal): {e}")
        rag_context = "Clinical guideline retrieval unavailable. Report generated from model output only."

    # ---- Groq Report Generation ----
    try:
        from backend.services.ai_service import (
            generate_clinical_report_text,
            GroqConfigurationError,
            GroqAuthError,
            GroqRateLimitError,
            GroqServiceError,
        )

        report_text = generate_clinical_report_text(
            prediction=prediction,
            risk_data=risk_data,
            rag_context=rag_context,
            additional_notes=req.additional_clinical_notes,
            language=req.language.value,
            trace_id=trace_id,
        )

        # Parse sections from the generated text
        sections = _parse_report_sections(report_text)

    except GroqConfigurationError as e:
        logger.error(f"[{trace_id}] Groq configuration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Clinical report configuration error: {e}. Trace ID: {trace_id}",
        )
    except GroqAuthError as e:
        logger.error(f"[{trace_id}] Groq authentication failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Clinical report configuration error: Invalid Groq API credentials. Trace ID: {trace_id}",
        )
    except GroqRateLimitError as e:
        logger.warning(f"[{trace_id}] Groq rate limit reached: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Report generation rate limit reached. Please wait a moment and try again. Trace ID: {trace_id}",
        )
    except Exception as e:
        logger.exception(f"[{trace_id}] Report generation failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Clinical report generation is temporarily unavailable. Trace ID: {trace_id}",
        )

    # ---- Build recommendations from the deterministic engine ----
    from backend.services.recommendation_service import get_recommendations
    recommendations = get_recommendations(prediction["predicted_class"])

    # Save report
    report_id = str(uuid.uuid4())
    report_doc = {
        "_id": report_id,
        "prediction_id": req.prediction_id,
        "doctor_id": doctor["_id"],
        "patient_id": req.patient_id,
        "language": req.language.value,
        "summary": sections.get("summary", ""),
        "detailed_findings": sections.get("detailed_findings", ""),
        "xai_interpretation": sections.get("xai_interpretation", ""),
        "risk_summary": sections.get("risk_summary", ""),
        "recommendations": recommendations,
        "rag_citations": rag_citations,
        "full_text": report_text,
        "model_version": prediction["model_version"],
        "trace_id": trace_id,
        "created_at": datetime.now(timezone.utc),
    }

    reports = get_reports_collection()
    await reports.insert_one(report_doc)

    return ReportResponse(
        report_id=report_id,
        prediction_id=req.prediction_id,
        doctor_id=doctor["_id"],
        patient_id=req.patient_id,
        language=req.language,
        summary=sections.get("summary", ""),
        detailed_findings=sections.get("detailed_findings", ""),
        xai_interpretation=sections.get("xai_interpretation", ""),
        risk_summary=sections.get("risk_summary", ""),
        recommendations=recommendations,
        rag_citations=rag_citations,
        generated_at=datetime.now(timezone.utc),
        model_version=prediction["model_version"],
        trace_id=trace_id,
    )


@router.get("/{report_id}/download")
async def download_report_pdf(
    report_id: str,
    doctor: dict = Depends(require_verified_doctor),
):
    """Download a generated report as a PDF."""
    trace_id = generate_trace_id()

    reports = get_reports_collection()
    report = await reports.find_one({"_id": report_id, "doctor_id": doctor["_id"]})

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found. Trace ID: {trace_id}",
        )

    try:
        from backend.services.pdf_service import generate_pdf
        predictions = get_predictions_collection()
        prediction = await predictions.find_one({"_id": report["prediction_id"]})
        pdf_path = generate_pdf(report, doctor, prediction)

        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=f"NeuroAssist_Report_{report_id[:8]}.pdf",
        )
    except Exception as e:
        logger.error(f"[{trace_id}] PDF generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF generation failed: {type(e).__name__}. Trace ID: {trace_id}",
        )


@router.get("/list")
async def list_reports(
    patient_id: str = None,
    doctor: dict = Depends(require_verified_doctor),
):
    """List all reports for the current doctor, optionally filtered by patient."""
    reports = get_reports_collection()
    query = {"doctor_id": doctor["_id"]}
    if patient_id:
        query["patient_id"] = patient_id

    cursor = reports.find(query).sort("created_at", -1).limit(100)
    result = []
    async for report in cursor:
        result.append({
            "report_id": report["_id"],
            "prediction_id": report["prediction_id"],
            "patient_id": report.get("patient_id"),
            "language": report["language"],
            "summary": report.get("summary", "")[:200],
            "created_at": report["created_at"],
        })

    return {"reports": result}


def _parse_report_sections(text: str) -> dict:
    """Parse the clinical report markdown into sections."""
    sections = {
        "summary": "",
        "detailed_findings": "",
        "xai_interpretation": "",
        "risk_summary": "",
        "recommendations": "",
    }

    current_section = None
    lines = text.split("\n")

    for line in lines:
        lower = line.lower().strip()
        if "## summary" in lower:
            current_section = "summary"
        elif "## detailed findings" in lower or "## detailed" in lower:
            current_section = "detailed_findings"
        elif "## xai" in lower or "## explainab" in lower:
            current_section = "xai_interpretation"
        elif "## risk" in lower:
            current_section = "risk_summary"
        elif "## recommend" in lower:
            current_section = "recommendations"
        elif current_section:
            sections[current_section] += line + "\n"

    # Trim whitespace
    return {k: v.strip() for k, v in sections.items()}
