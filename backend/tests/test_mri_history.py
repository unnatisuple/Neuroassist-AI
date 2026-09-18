import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.auth.jwt import create_access_token

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Generate auth headers for Dr. Sarah Jenkins (pre-existing mock doctor)."""
    # The default mock doctor ID used by the system
    doctor_id = "2eb31ef2-6d17-4f9b-ac09-b06aab87072f"
    token = create_access_token({"sub": doctor_id, "role": "verified_doctor"})
    return {"Authorization": f"Bearer {token}"}


def test_list_predictions_unauthorized(client):
    """Test that listing predictions without authorization fails with 401."""
    response = client.get("/api/mri/list")
    assert response.status_code in (401, 403)


def test_list_predictions_authorized(client, auth_headers):
    """Test that listing predictions succeeds for an authorized doctor."""
    response = client.get("/api/mri/list", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_get_mri_file_not_found(client, auth_headers):
    """Test that requesting a non-existent prediction ID returns 404."""
    response = client.get("/api/mri/file/non-existent-uuid", headers=auth_headers)
    assert response.status_code == 404


def test_mri_upload_predict_xai_integration(client, auth_headers):
    """E2E integration test: upload image, run predict, check side-by-side overlays, check list & stream file."""
    import os
    sample_candidates = [
        "ml/data/raw/Non_Demented/OAS1_9901_MR1_mpr-1_101.jpg",
        "ml/data/preprocessed/MildDemented/1 (10).jpg",
        "ml/data/preprocessed/NonDemented/26 (100).jpg",
    ]
    sample_path = next((p for p in sample_candidates if os.path.exists(p)), None)
    if sample_path is None:
        import glob
        found = glob.glob("ml/data/preprocessed/**/*.jpg", recursive=True)
        if found:
            sample_path = found[0]
    assert sample_path is not None and os.path.exists(sample_path), "Sample MRI slice not found at expected path."

    # 1. Upload scan
    with open(sample_path, "rb") as img_file:
        files = {"file": (os.path.basename(sample_path), img_file, "image/jpeg")}
        upload_resp = client.post("/api/mri/upload", files=files, headers=auth_headers)
    
    assert upload_resp.status_code == 201
    file_id = upload_resp.json()["file_id"]

    # 2. Predict scan (this triggers pre-generation of overlays & brain regions)
    predict_resp = client.post(f"/api/mri/predict?file_id={file_id}", headers=auth_headers)
    assert predict_resp.status_code == 200
    pred_data = predict_resp.json()
    prediction_id = pred_data["prediction_id"]

    # Verify overlays are pre-computed and returned
    assert "xai_overlays" in pred_data
    overlays = pred_data["xai_overlays"]
    assert "gradcam_overlay" in overlays
    assert "gradcam_plus_plus_overlay" in overlays
    assert "integrated_gradients_overlay" in overlays
    assert "hirescam_overlay" in overlays
    assert len(overlays["gradcam_overlay"]) > 0
    assert len(overlays["gradcam_plus_plus_overlay"]) > 0
    assert len(overlays["integrated_gradients_overlay"]) > 0
    assert len(overlays["hirescam_overlay"]) > 0

    # Verify brain region analysis is generated
    assert "brain_regions" in pred_data
    assert pred_data["brain_regions"] is not None
    assert len(pred_data["brain_regions"]["regions"]) == 8

    # 3. List scans
    list_resp = client.get("/api/mri/list", headers=auth_headers)
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert "predictions" in list_data
    scans_list = list_data["predictions"]
    
    # Check that our scan is in the list
    matching_scan = [s for s in scans_list if s["prediction_id"] == prediction_id]
    assert len(matching_scan) == 1
    assert matching_scan[0]["xai_overlays"]["gradcam_overlay"] == overlays["gradcam_overlay"]

    # 4. Stream/Decrypt file
    file_resp = client.get(f"/api/mri/file/{prediction_id}", headers=auth_headers)
    assert file_resp.status_code == 200
    assert file_resp.headers["content-type"] in ("image/png", "image/jpeg")
    assert len(file_resp.content) > 0

