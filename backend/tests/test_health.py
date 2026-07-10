import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_db_endpoint():
    """Test the /health/db endpoint behaves correctly."""
    response = client.get("/health/db")
    # In test client, MongoDB can be connected or mock fallback
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "OK"

def test_health_gemini_endpoint():
    """Test the /health/gemini endpoint behaves correctly under dummy key."""
    # Under dummy key during test mode, it might fail or succeed depending on API config
    # We want to verify it doesn't crash the server and returns a valid JSON error/success
    response = client.get("/health/gemini")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "OK"


def test_chatbot_service_error_handling():
    """Verify that a Gemini API auth/credentials failure returns HTTP 503 Service Unavailable, not an off-topic redirect."""
    from unittest.mock import patch
    from google.auth.exceptions import DefaultCredentialsError
    from backend.auth.jwt import create_access_token

    doctor_id = "2eb31ef2-6d17-4f9b-ac09-b06aab87072f"
    token = create_access_token({"sub": doctor_id, "role": "verified_doctor"})
    headers = {"Authorization": f"Bearer {token}"}

    with patch("google.generativeai.GenerativeModel") as mock_model:
        mock_model.side_effect = DefaultCredentialsError("Mock DefaultCredentialsError")
        
        response = client.post(
            "/api/chatbot/message",
            json={"message": "What are the stages of Alzheimer's?", "language": "en"},
            headers=headers,
        )
        
        assert response.status_code == 503
        data = response.json()
        assert "medical assistant configuration error" in data["detail"].lower()
        assert "trace id" in data["detail"].lower()

