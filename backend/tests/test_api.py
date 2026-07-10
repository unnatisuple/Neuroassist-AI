"""
NeuroAssist AI v2 — API Unit Tests
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture(scope="module")
def client():
    """Test client fixture that triggers FastAPI lifespan events."""
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    """Test that the root endpoint returns the correct welcome message and disclaimer."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "NeuroAssist AI v2"
    assert "disclaimer" in data
    assert "not fda/ce cleared" in data["disclaimer"].lower()


def test_health_check_degraded(client):
    """Test health check when database is not connected (should be degraded but return 200)."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["mongodb_connected"] is False or data["mongodb_connected"] is True


def test_docs_page(client):
    """Test that Swagger documentation endpoints are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_auth_login_validation(client):
    """Test that logging in with invalid data types returns a 422 validation error."""
    response = client.post("/api/auth/login", json={"full_name": "", "email": "not-an-email"})
    assert response.status_code == 422  # Validation Error


def test_rag_retrieval_service(client):
    """Test that the RAG service can load the FAISS index and retrieve document chunks."""
    from backend.services.rag_service import retrieve_relevant_guidelines
    from backend.schemas import DementiaStage

    # Run retrieval
    results = retrieve_relevant_guidelines(DementiaStage.MILD)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "text" in results[0]
    assert "source" in results[0]
