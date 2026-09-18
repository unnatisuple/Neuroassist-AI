import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_db_endpoint(client):
    """Test the /health/db endpoint behaves correctly."""
    response = client.get("/health/db")
    # In test client, MongoDB can be connected or mock fallback
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "OK"

def test_health_gemini_endpoint(client):
    """Test the /health/gemini backward compatibility endpoint."""
    response = client.get("/health/gemini")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "OK"


def test_health_groq_endpoint(client):
    """Test the /health/groq endpoint behaves correctly."""
    response = client.get("/health/groq")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "OK"
        assert data["provider"] == "Groq"


def test_chatbot_service_error_handling(client):
    """Verify that a Groq API auth/credentials failure returns HTTP 503 Service Unavailable, not an off-topic redirect."""
    from unittest.mock import patch
    from backend.services.ai_service import GroqAuthError
    from backend.auth.jwt import create_access_token

    doctor_id = "2eb31ef2-6d17-4f9b-ac09-b06aab87072f"
    token = create_access_token({"sub": doctor_id, "role": "verified_doctor"})
    headers = {"Authorization": f"Bearer {token}"}

    with patch("backend.routers.chatbot.chat_completion") as mock_completion:
        mock_completion.side_effect = GroqAuthError("Invalid Groq API key")

        response = client.post(
            "/api/chatbot/message",
            json={"message": "What are the stages of Alzheimer's?", "language": "en"},
            headers=headers,
        )

        assert response.status_code == 503
        data = response.json()
        assert "medical assistant configuration error" in data["detail"].lower()
        assert "trace id" in data["detail"].lower()


def test_direct_api_chat_endpoint(client):
    """Verify that direct /api/chat works with conversation context."""
    from unittest.mock import patch

    with patch("backend.routers.chatbot.chat_completion") as mock_completion:
        mock_completion.return_value = {
            "content": "Alzheimer's disease progression typically involves preclinical, mild cognitive impairment, mild dementia, moderate dementia, and severe dementia stages.",
            "usage": {"total_tokens": 42},
        }

        response = client.post(
            "/api/chat",
            json={
                "message": "What are the stages of Alzheimer's?",
                "conversation": [],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "stages" in data["reply"].lower() or "stages" in data["answer"].lower()
        assert data["is_on_topic"] is True
        assert "usage" in data
        assert "trace_id" in data

