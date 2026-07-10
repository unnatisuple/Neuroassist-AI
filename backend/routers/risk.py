"""
NeuroAssist AI v2 — Clinical Risk Assessment Router
Transparent, rule-based risk index. Explicitly labeled as rule-based, NOT ML-based.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
from backend.schemas import (
    RiskAssessmentRequest, RiskAssessmentResponse, RiskLevel,
)
from backend.dependencies import require_verified_doctor, generate_trace_id
from backend.db.mongodb import get_risk_assessments_collection
import uuid

router = APIRouter(prefix="/api/risk", tags=["Clinical Risk Assessment"])


def _compute_rule_based_risk(data: RiskAssessmentRequest) -> dict:
    """
    Transparent, clinician-adjustable rule-based risk scoring.

    THIS IS NOT AN ML MODEL. It uses published clinical risk factor weightings
    from peer-reviewed literature. Each factor's contribution is visible and
    explainable to the clinician.

    Sources:
    - Livingston et al. (2020). "Dementia prevention, intervention, and care."
      The Lancet Commissions. DOI: 10.1016/S0140-6736(20)30367-6
    - Norton et al. (2014). "Potential for primary prevention of Alzheimer's disease."
      The Lancet Neurology.
    """
    risk_score = 0.0
    risk_factors = []
    protective_factors = []

    # Age (strongest non-modifiable risk factor)
    if data.age >= 85:
        risk_score += 25
        risk_factors.append({"factor": "Age", "value": data.age, "contribution": 25,
                             "explanation": "Age 85+ is the strongest risk factor for dementia."})
    elif data.age >= 75:
        risk_score += 18
        risk_factors.append({"factor": "Age", "value": data.age, "contribution": 18,
                             "explanation": "Dementia risk increases significantly after age 75."})
    elif data.age >= 65:
        risk_score += 10
        risk_factors.append({"factor": "Age", "value": data.age, "contribution": 10,
                             "explanation": "Age 65-74 carries moderate baseline risk."})
    elif data.age >= 50:
        risk_score += 3
        risk_factors.append({"factor": "Age", "value": data.age, "contribution": 3,
                             "explanation": "Age 50-64 carries low baseline risk."})

    # Cognitive scores (MMSE / MoCA)
    if data.mmse_score is not None:
        if data.mmse_score < 18:
            risk_score += 20
            risk_factors.append({"factor": "MMSE Score", "value": data.mmse_score, "contribution": 20,
                                 "explanation": "MMSE < 18 indicates significant cognitive impairment."})
        elif data.mmse_score < 24:
            risk_score += 12
            risk_factors.append({"factor": "MMSE Score", "value": data.mmse_score, "contribution": 12,
                                 "explanation": "MMSE 18-23 indicates mild cognitive impairment."})
        elif data.mmse_score >= 27:
            protective_factors.append({"factor": "MMSE Score", "value": data.mmse_score,
                                       "contribution": -3, "explanation": "MMSE ≥ 27 suggests normal cognition."})
            risk_score -= 3

    if data.moca_score is not None:
        if data.moca_score < 18:
            risk_score += 18
            risk_factors.append({"factor": "MoCA Score", "value": data.moca_score, "contribution": 18,
                                 "explanation": "MoCA < 18 indicates significant cognitive impairment."})
        elif data.moca_score < 26:
            risk_score += 10
            risk_factors.append({"factor": "MoCA Score", "value": data.moca_score, "contribution": 10,
                                 "explanation": "MoCA < 26 suggests mild cognitive impairment."})

    # Family history
    if data.family_history_dementia:
        risk_score += 10
        risk_factors.append({"factor": "Family History", "value": True, "contribution": 10,
                             "explanation": "First-degree relative with dementia increases risk ~1.7x."})

    # Cardiovascular risk factors
    if data.has_diabetes:
        risk_score += 7
        risk_factors.append({"factor": "Diabetes", "value": True, "contribution": 7,
                             "explanation": "Type 2 diabetes increases dementia risk ~1.5x (Livingston 2020)."})

    if data.systolic_bp and data.systolic_bp >= 140:
        risk_score += 5
        risk_factors.append({"factor": "Hypertension", "value": f"{data.systolic_bp}/{data.diastolic_bp}",
                             "contribution": 5, "explanation": "Midlife hypertension is a modifiable risk factor."})

    if data.cholesterol_level == "high":
        risk_score += 4
        risk_factors.append({"factor": "Cholesterol", "value": "high", "contribution": 4,
                             "explanation": "High cholesterol is associated with increased dementia risk."})

    # Lifestyle factors
    if data.smoking_status == "current":
        risk_score += 8
        risk_factors.append({"factor": "Smoking", "value": "current", "contribution": 8,
                             "explanation": "Current smoking increases dementia risk ~1.6x."})
    elif data.smoking_status == "former":
        risk_score += 3
        risk_factors.append({"factor": "Smoking (Former)", "value": "former", "contribution": 3,
                             "explanation": "Former smoking carries residual risk."})

    if data.alcohol_consumption == "heavy":
        risk_score += 7
        risk_factors.append({"factor": "Alcohol", "value": "heavy", "contribution": 7,
                             "explanation": "Heavy alcohol use is a significant dementia risk factor."})

    if data.physical_activity_level == "sedentary":
        risk_score += 6
        risk_factors.append({"factor": "Physical Inactivity", "value": "sedentary", "contribution": 6,
                             "explanation": "Physical inactivity increases dementia risk ~1.4x."})
    elif data.physical_activity_level == "active":
        protective_factors.append({"factor": "Physical Activity", "value": "active",
                                   "contribution": -5, "explanation": "Regular exercise is protective."})
        risk_score -= 5

    if data.sleep_quality == "poor":
        risk_score += 5
        risk_factors.append({"factor": "Poor Sleep", "value": "poor", "contribution": 5,
                             "explanation": "Poor sleep quality is associated with increased amyloid burden."})

    if data.depression_history:
        risk_score += 6
        risk_factors.append({"factor": "Depression History", "value": True, "contribution": 6,
                             "explanation": "Depression history increases dementia risk ~1.9x."})

    if data.bmi and data.bmi >= 30:
        risk_score += 4
        risk_factors.append({"factor": "Obesity", "value": data.bmi, "contribution": 4,
                             "explanation": "Midlife obesity (BMI ≥ 30) increases dementia risk."})

    if data.education_years is not None and data.education_years < 12:
        risk_score += 5
        risk_factors.append({"factor": "Low Education", "value": f"{data.education_years} years",
                             "contribution": 5, "explanation": "Less than 12 years of education increases risk."})
    elif data.education_years and data.education_years >= 16:
        protective_factors.append({"factor": "Higher Education", "value": f"{data.education_years} years",
                                   "contribution": -4, "explanation": "Higher education provides cognitive reserve."})
        risk_score -= 4

    # Clamp score
    risk_score = max(0, min(100, risk_score))

    # Determine level
    if risk_score >= 60:
        level = RiskLevel.VERY_HIGH
    elif risk_score >= 40:
        level = RiskLevel.HIGH
    elif risk_score >= 20:
        level = RiskLevel.MODERATE
    else:
        level = RiskLevel.LOW

    return {
        "risk_score": round(risk_score, 1),
        "risk_level": level,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
    }


@router.post("/assess", response_model=RiskAssessmentResponse)
async def assess_risk(
    req: RiskAssessmentRequest,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Compute a clinical risk assessment score.

    IMPORTANT: This is a transparent, RULE-BASED risk index — NOT an ML prediction.
    Each factor's contribution is visible and adjustable. The methodology is documented
    and based on published literature (Livingston et al. 2020, Norton et al. 2014).
    """
    trace_id = generate_trace_id()

    result = _compute_rule_based_risk(req)

    assessment_id = str(uuid.uuid4())
    record = {
        "_id": assessment_id,
        "doctor_id": doctor["_id"],
        "patient_id": req.patient_id,
        "input_data": req.model_dump(),
        **result,
        "trace_id": trace_id,
        "created_at": datetime.now(timezone.utc),
    }

    assessments = get_risk_assessments_collection()
    await assessments.insert_one(record)

    return RiskAssessmentResponse(
        assessment_id=assessment_id,
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc),
        **result,
    )
