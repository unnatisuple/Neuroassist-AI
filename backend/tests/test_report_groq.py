"""
NeuroAssist AI v2 — Tests for Groq-powered Report Generation and AI Service
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from backend.main import app
from backend.auth.jwt import create_access_token
from backend.services.ai_service import (
    chat_completion,
    generate_clinical_report_text,
    GroqConfigurationError,
    GroqAuthError,
)

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def auth_headers():
    doctor_id = "2eb31ef2-6d17-4f9b-ac09-b06aab87072f"
    token = create_access_token({"sub": doctor_id, "role": "verified_doctor"})
    return {"Authorization": f"Bearer {token}"}


def test_ai_service_missing_key():
    """Verify GroqConfigurationError is raised when API key is placeholder or missing."""
    with patch("backend.config.settings.groq_api_key", "REPLACE_WITH_YOUR_KEY"), \
         patch("backend.config.settings.gemini_api_key", ""):
        with pytest.raises(GroqConfigurationError):
            chat_completion([{"role": "user", "content": "hello"}])


def test_generate_clinical_report_mock_groq():
    """Verify report generation constructs proper markdown sections with Groq."""
    mock_report = (
        "## Summary\nPatient exhibits neurostructural patterns consistent with Mild Demented stage.\n\n"
        "## Detailed Findings\nHippocampal volume reduction observed.\n\n"
        "## XAI Interpretation Guide\nGrad-CAM highlights temporal lobes.\n\n"
        "## Risk Summary\nModerate risk score.\n\n"
        "## Recommendations\nFollow-up in 6 months per WHO guidelines.\n"
    )

    with patch("backend.services.ai_service.is_groq_configured", return_value=True), \
         patch("backend.services.ai_service.chat_completion") as mock_chat:
        mock_chat.return_value = {"content": mock_report, "usage": {"total_tokens": 120}}

        prediction = {
            "predicted_class": "Mild Demented",
            "confidence": 0.89,
            "class_probabilities": {"Mild Demented": 0.89, "Non-Demented": 0.11},
            "model_version": "lever_b_resnet18",
        }

        report_text = generate_clinical_report_text(
            prediction=prediction,
            risk_data={"risk_score": 65, "risk_level": "Moderate", "risk_factors": [{"factor": "Hypertension"}]},
            rag_context="NICE NG97: Recommended Donepezil for mild dementia.",
            language="en",
        )

        assert "## Summary" in report_text
        assert "## Detailed Findings" in report_text
        assert "## XAI Interpretation Guide" in report_text
        assert "## Risk Summary" in report_text
        assert "## Recommendations" in report_text


def test_report_endpoint_auth_error_handling(client, auth_headers):
    """Verify that an auth error during report generation produces a clean 503 error without Google internal hostnames."""
    from backend.db.mongodb import get_predictions_collection

    pred_id = "pred-test-groq-001"
    predictions = get_predictions_collection()

    dummy_pred = {
        "_id": pred_id,
        "doctor_id": "2eb31ef2-6d17-4f9b-ac09-b06aab87072f",
        "predicted_class": "Mild Demented",
        "confidence": 0.92,
        "class_probabilities": {"Mild Demented": 0.92},
        "model_version": "lever_b_resnet18",
    }

    mock_preds = MagicMock()
    mock_preds.find_one = AsyncMock(return_value=dummy_pred)

    with patch("backend.routers.report.get_predictions_collection", return_value=mock_preds), \
         patch("backend.services.ai_service.generate_clinical_report_text") as mock_gen:
        mock_gen.side_effect = GroqAuthError("Invalid Groq API credentials")

        response = client.post(
            "/api/report/generate",
            json={"prediction_id": pred_id, "language": "en"},
            headers=auth_headers,
        )

        assert response.status_code == 503
        data = response.json()
        assert "clinical report configuration error" in data["detail"].lower()
        assert "trace id" in data["detail"].lower()
        # Verify no Google internal hostnames in detail
        assert "googleapis.com" not in data["detail"]
        assert "generativelanguage" not in data["detail"]
