"""
NeuroAssist AI v2 — MRI Upload, Prediction & Explainability Router
Every prediction is validated: model artifact must be loaded, inference must actually run,
model checksum + version + latency are logged with every response.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status
from datetime import datetime, timezone
from typing import Optional
from backend.schemas import (
    PredictionResponse, PredictionRecord, ExplainabilityMethod,
    ExplainabilityResponse, BrainComparisonResponse, DementiaStage,
)
from backend.dependencies import require_verified_doctor, generate_trace_id
from backend.db.mongodb import get_predictions_collection
from loguru import logger
import uuid
import os
import time
import hashlib

router = APIRouter(prefix="/api/mri", tags=["MRI Analysis"])

# ---- Model state (loaded at startup) ----
_model = None
_model_version = "not_loaded"
_model_checksum = "not_loaded"

CLASS_NAMES = [
    DementiaStage.MILD,
    DementiaStage.MODERATE,
    DementiaStage.NON_DEMENTED,
    DementiaStage.VERY_MILD,
]


def _get_model():
    """Get the loaded model or raise a specific error — never return mock results."""
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not yet loaded. The AI model has not been trained or the checkpoint "
                "file is missing. Please train the model first (see /ml/training/) or ensure "
                "the model checkpoint exists at the configured path. "
                "No mock or placeholder predictions will be returned."
            ),
        )
    return _model


def load_model(model_path: str) -> bool:
    """
    Load the trained model from a checkpoint file.
    Called at application startup. Returns False if model file doesn't exist.
    """
    global _model, _model_version, _model_checksum

    if not os.path.exists(model_path):
        logger.warning(f"Model checkpoint not found at {model_path}. Predictions will be unavailable.")
        return False

    try:
        import torch
        # Compute checksum for auditability
        with open(model_path, "rb") as f:
            file_hash = hashlib.sha256()
            for chunk in iter(lambda: f.read(8192), b""):
                file_hash.update(chunk)
            _model_checksum = file_hash.hexdigest()[:16]

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            _model_version = checkpoint.get("version", "unknown")
            # Architecture must be reconstructed from checkpoint metadata
            arch_name = checkpoint.get("architecture", "unknown")
            num_classes = checkpoint.get("num_classes", 4)

            from backend.services.model_service import build_model
            _model = build_model(arch_name, num_classes)
            _model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Direct model save
            _model = checkpoint
            _model_version = "v1_direct_load"

        _model.eval()
        logger.info(f"Model loaded: version={_model_version}, checksum={_model_checksum}")
        return True

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        _model = None
        return False


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_mri(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Upload a brain MRI scan for analysis.
    Accepts: JPEG, PNG (2D slices). DICOM support planned.
    Files are stored locally (or in configured cloud storage) — never in MongoDB.
    """
    trace_id = generate_trace_id()

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file type: {file.content_type}. "
                f"Accepted formats: JPEG, PNG, TIFF. "
                f"DICOM support is planned for a future release. "
                f"Trace ID: {trace_id}"
            ),
        )

    # Validate file size (max 50MB)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max 50MB). Trace ID: {trace_id}",
        )

    # Save file
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "scan.png")[1] or ".png"
    save_dir = os.path.join("uploads", "mri", doctor["_id"])
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{file_id}{ext}")

    from backend.services.encryption_service import encrypt_data
    encrypted_contents = encrypt_data(contents)
    with open(file_path, "wb") as f:
        f.write(encrypted_contents)

    logger.info(f"[{trace_id}] MRI uploaded: file_id={file_id}, doctor={doctor['_id']}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_path": file_path,
        "size_bytes": len(contents),
        "patient_id": patient_id,
        "trace_id": trace_id,
        "message": "MRI scan uploaded successfully. Use /api/mri/predict to run analysis.",
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_dementia_stage(
    file_id: str,
    patient_id: Optional[str] = None,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Run dementia stage prediction on an uploaded MRI scan.

    Returns: 4-class prediction (Non-Demented, Very Mild, Mild, Moderate)
    with calibrated confidence, model version, checksum, and inference latency.

    Note: 'Severe Dementia' is NOT supported by the current training data.
    This gap is disclosed in the response.
    """
    trace_id = generate_trace_id()
    model = _get_model()  # Raises 503 if not loaded — never returns mocks

    # Find the uploaded file
    upload_dir = os.path.join("uploads", "mri", doctor["_id"])
    matching_files = [
        f for f in os.listdir(upload_dir) if f.startswith(file_id)
    ] if os.path.exists(upload_dir) else []

    if not matching_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No uploaded MRI found with file_id={file_id}. Trace ID: {trace_id}",
        )

    file_path = os.path.join(upload_dir, matching_files[0])

    # ---- REAL INFERENCE (no mocks) ----
    try:
        import torch
        from PIL import Image
        from torchvision import transforms

        start_time = time.perf_counter()

        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        from backend.services.encryption_service import decrypted_temp_file
        xai_overlays = {}
        with decrypted_temp_file(file_path) as temp_path:
            img = Image.open(temp_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)

            # Inference
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

            inference_latency = (time.perf_counter() - start_time) * 1000  # ms

            # Build result
            class_probs = {cls.value: float(probabilities[i]) for i, cls in enumerate(CLASS_NAMES)}
            predicted_idx = int(torch.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            # Pre-generate explainability heatmaps (non-fatal)
            try:
                from backend.services.explainability_service import generate_explanation
                for method in ("gradcam", "gradcam++", "hirescam"):
                    try:
                        res = generate_explanation(model, temp_path, method, CLASS_NAMES)
                        key_prefix = method.replace("++", "_plus_plus")
                        xai_overlays[f"{key_prefix}_heatmap"] = res["heatmap_base64"]
                        xai_overlays[f"{key_prefix}_overlay"] = res["overlay_base64"]
                    except Exception as ex:
                        logger.warning(f"Failed to pre-compute {method} overlay: {ex}")
            except Exception as e:
                logger.warning(f"Failed to pre-compute explainability overlays: {e}")

    except Exception as e:
        logger.error(f"[{trace_id}] Inference failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Model inference failed: {type(e).__name__}. "
                f"This may indicate a corrupted image file or model incompatibility. "
                f"Trace ID: {trace_id}"
            ),
        )

    # Save prediction record for audit
    prediction_id = str(uuid.uuid4())
    prediction_record = {
        "_id": prediction_id,
        "doctor_id": doctor["_id"],
        "patient_id": patient_id,
        "mri_file_id": file_id,
        "mri_file_path": file_path,
        "predicted_class": predicted_class.value,
        "confidence": confidence,
        "class_probabilities": class_probs,
        "model_version": _model_version,
        "model_checksum": _model_checksum,
        "inference_latency_ms": round(inference_latency, 2),
        "trace_id": trace_id,
        "created_at": datetime.now(timezone.utc),
        "xai_overlays": xai_overlays,
    }

    predictions = get_predictions_collection()
    await predictions.insert_one(prediction_record)

    logger.info(
        f"[{trace_id}] Prediction: class={predicted_class.value}, "
        f"confidence={confidence:.4f}, latency={inference_latency:.1f}ms, "
        f"model={_model_version}/{_model_checksum}"
    )

    return PredictionResponse(
        prediction_id=prediction_id,
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        class_probabilities=class_probs,
        model_version=_model_version,
        model_checksum=_model_checksum,
        inference_latency_ms=round(inference_latency, 2),
        timestamp=datetime.now(timezone.utc),
        trace_id=trace_id,
        mri_file_id=file_id,
        xai_overlays=xai_overlays,
    )


@router.post("/explain/{method}", response_model=ExplainabilityResponse)
async def explain_prediction(
    method: ExplainabilityMethod,
    prediction_id: str,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Generate XAI visualization for a previous prediction.
    Every heatmap is generated LIVE from the actual scan and actual model activations.
    Never a stock/sample heatmap image.
    """
    trace_id = generate_trace_id()
    model = _get_model()

    # Fetch the prediction record
    predictions = get_predictions_collection()
    pred = await predictions.find_one({"_id": prediction_id, "doctor_id": doctor["_id"]})

    if not pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found. Trace ID: {trace_id}",
        )

    file_path = pred["mri_file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original MRI file no longer exists at {file_path}. Trace ID: {trace_id}",
        )

    try:
        start_time = time.perf_counter()
        from backend.services.explainability_service import generate_explanation
        from backend.services.encryption_service import decrypted_temp_file
        with decrypted_temp_file(file_path) as temp_path:
            result = generate_explanation(model, temp_path, method, CLASS_NAMES)
        latency = (time.perf_counter() - start_time) * 1000

        return ExplainabilityResponse(
            prediction_id=prediction_id,
            method=method,
            heatmap_base64=result["heatmap_base64"],
            overlay_base64=result["overlay_base64"],
            model_version=_model_version,
            inference_latency_ms=round(latency, 2),
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc),
        )

    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Explainability library not installed: {e}. "
                f"Please install pytorch-grad-cam and/or captum. Trace ID: {trace_id}"
            ),
        )
    except Exception as e:
        logger.error(f"[{trace_id}] Explainability failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"XAI generation failed: {type(e).__name__}. Trace ID: {trace_id}",
        )


@router.get("/compare/{prediction_id}", response_model=BrainComparisonResponse)
async def compare_with_healthy(
    prediction_id: str,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Compare uploaded scan against a class-averaged healthy brain template
    computed from real training data (not a stock image).
    """
    trace_id = generate_trace_id()

    predictions = get_predictions_collection()
    pred = await predictions.find_one({"_id": prediction_id, "doctor_id": doctor["_id"]})

    if not pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found. Trace ID: {trace_id}",
        )

    try:
        from backend.services.explainability_service import generate_healthy_comparison
        from backend.services.encryption_service import decrypted_temp_file
        with decrypted_temp_file(pred["mri_file_path"]) as temp_path:
            result = generate_healthy_comparison(temp_path)
        result["trace_id"] = trace_id
        return BrainComparisonResponse(**result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Healthy brain template not yet generated. Run the training pipeline "
                "first to compute class-averaged templates from real data. "
                f"Trace ID: {trace_id}"
            ),
        )


@router.get("/file/{prediction_id}")
async def get_mri_file(
    prediction_id: str,
    doctor: dict = Depends(require_verified_doctor),
):
    """Retrieve and decrypt the original uploaded MRI scan image file."""
    predictions = get_predictions_collection()
    pred = await predictions.find_one({"_id": prediction_id, "doctor_id": doctor["_id"]})
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    file_path = pred["mri_file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="MRI file no longer exists.")

    from backend.services.encryption_service import decrypt_data
    with open(file_path, "rb") as f:
        encrypted_data = f.read()
    decrypted_data = decrypt_data(encrypted_data)

    import io
    from fastapi.responses import StreamingResponse
    # Serve decrypted image file as PNG/JPEG stream
    ext = os.path.splitext(file_path)[1].lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return StreamingResponse(io.BytesIO(decrypted_data), media_type=media_type)


@router.get("/list")
async def list_predictions(
    patient_id: Optional[str] = None,
    doctor: dict = Depends(require_verified_doctor),
):
    """List all predictions/scans for the current doctor, checking if report exists."""
    predictions = get_predictions_collection()
    query = {"doctor_id": doctor["_id"]}
    if patient_id:
        query["patient_id"] = patient_id

    from backend.db.mongodb import get_reports_collection
    reports = get_reports_collection()

    cursor = predictions.find(query).sort("created_at", -1).limit(100)
    result = []
    async for doc in cursor:
        report = await reports.find_one({"prediction_id": doc["_id"]})
        result.append({
            "prediction_id": doc["_id"],
            "mri_file_id": doc.get("mri_file_id"),
            "predicted_class": doc.get("predicted_class"),
            "confidence": doc.get("confidence"),
            "class_probabilities": doc.get("class_probabilities"),
            "model_version": doc.get("model_version"),
            "created_at": doc.get("created_at"),
            "patient_id": doc.get("patient_id"),
            "report_id": report["_id"] if report else None,
            "xai_overlays": doc.get("xai_overlays", {})
        })

    return {"predictions": result}
