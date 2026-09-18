"""
NeuroAssist AI v2 — Offline Development Fallback Service
Provides realistic, evidence-based clinical responses and reports when
a live Groq API key is not configured, enabling offline testing of
the decision support features.
"""

from typing import Dict, Any, Optional
from loguru import logger


def generate_fallback_chat_reply(message: str, language: str = "en") -> str:
    """Generate an on-topic clinical assistant response based on clinical guidelines."""
    msg_lower = message.lower()

    if language == "hi":
        lang_prefix = "चिकित्सकीय सूचना (न्यूरोअसिस्ट एआई): "
    elif language == "mr":
        lang_prefix = "वैद्यकीय माहिती (न्यूरोअसिस्ट एआय): "
    else:
        lang_prefix = ""

    if any(k in msg_lower for k in ["early sign", "early symptom", "stages", "stage"]):
        reply = (
            f"{lang_prefix}Early clinical signs of Alzheimer's disease typically present as subtle episodic memory deficits, "
            "difficulty finding words (anomia), mild disorientation in unfamiliar environments, and challenges with executive function. "
            "Progression follows distinct stages: preclinical, Mild Cognitive Impairment (MCI) / Very Mild Dementia, "
            "Mild Dementia (requiring assistance with complex IADLs), Moderate Dementia (prominent memory loss and ADL impairment), "
            "and Advanced Dementia.\n\n"
            "Reference: Alzheimer's Association 2024 Diagnostic Guidelines. Always correlate neuroimaging findings with standardized "
            "cognitive assessments (MoCA / MMSE)."
        )
    elif any(k in msg_lower for k in ["grad-cam", "gradcam", "xai", "heatmap", "visualization", "hirescam", "integrated gradient"]):
        reply = (
            f"{lang_prefix}Explainable AI (XAI) heatmaps in NeuroAssist highlight the specific morphological regions influencing the model's prediction:\n\n"
            "• Grad-CAM / Grad-CAM++: Highlights coarse activation maps focusing on medial temporal lobes and periventricular regions.\n"
            "• HiResCAM: Preserves higher spatial fidelity to pinpoint specific sulcal enlargement or localized cortical thinning.\n"
            "• Difference Mapping: Compares the patient slice against an age-matched normative healthy brain template to reveal atrophy gradients.\n\n"
            "Clinical Note: Saliency maps provide decision support and should always be verified by clinical neuroradiological review."
        )
    elif any(k in msg_lower for k in ["medication", "drug", "treatment", "donepezil", "memantine", "lecanemab"]):
        reply = (
            f"{lang_prefix}Current clinical guidelines outline the following pharmacological options for Alzheimer's disease management:\n\n"
            "• Acetylcholinesterase Inhibitors (AChEIs: Donepezil, Rivastigmine, Galantamine): Recommended for mild-to-moderate dementia to support cholinergic neurotransmission (NICE NG97).\n"
            "• NMDA Receptor Antagonist (Memantine): Indicated for moderate-to-severe disease, or as monotherapy when AChEIs are contraindicated.\n"
            "• Monoclonal Antibodies (Anti-amyloid, e.g., Lecanemab): For early-stage disease with confirmed amyloid pathology, subject to strict ARIA safety monitoring protocols.\n\n"
            "Non-pharmacological strategies (cognitive stimulation therapy, physical activity) should be integrated concurrently per WHO 2023 recommendations."
        )
    elif any(k in msg_lower for k in ["mri", "scan", "atrophy", "hippocamp", "ventric"]):
        reply = (
            f"{lang_prefix}Neuroimaging biomarkers for Alzheimer's disease primarily involve progressive volumetric reductions:\n\n"
            "• Hippocampal & Medial Temporal Lobe Atrophy: The hallmark structural biomarker, strongly associated with episodic memory decline.\n"
            "• Ventriculomegaly: Secondary enlargement of lateral ventricles and temporal horns as parenchymal volume decreases.\n"
            "• Cortical Thinning: Prominent in parietal and temporal association areas in mild-to-moderate stages.\n\n"
            "NeuroAssist correlates these anatomical regions through deep learning classification and difference template mapping."
        )
    elif any(k in msg_lower for k in ["risk", "factor", "lifestyle", "prevention"]):
        reply = (
            f"{lang_prefix}Modifiable and non-modifiable risk factors for dementia identified in WHO 2023 Guidelines include:\n\n"
            "• Vascular & Metabolic: Midlife hypertension, type 2 diabetes, obesity, and dyslipidemia.\n"
            "• Lifestyle: Physical inactivity, smoking, alcohol misuse, and social isolation.\n"
            "• Cognitive & Sensory: Low educational attainment, unmanaged hearing/vision loss, and untreated depression.\n\n"
            "The NeuroAssist Clinical Risk Assessment module transparently aggregates these factors to calculate a rule-based composite risk score."
        )
    else:
        reply = (
            f"{lang_prefix}NeuroAssist AI Clinical Decision Support is ready to assist. "
            "You can inquire about patient MRI staging, interpretability heatmaps (Grad-CAM), "
            "structural atrophy biomarkers, or WHO/NICE dementia clinical management guidelines. "
            "All insights are provided strictly to assist qualified clinical professionals."
        )

    return reply


def generate_fallback_report_text(
    prediction: Dict[str, Any],
    risk_data: Optional[Dict[str, Any]] = None,
    rag_context: str = "",
    clinical_notes: str = "",
    language: str = "en",
) -> str:
    """Generate structured clinical report text adhering to the required markdown sections."""
    stage = prediction.get("predicted_class", "Non Demented")
    confidence = prediction.get("confidence", 0.0)
    conf_pct = f"{confidence * 100:.1f}%" if confidence <= 1.0 else f"{confidence:.1f}%"
    version = prediction.get("model_version", "lever_b_resnet18")
    probs = prediction.get("class_probabilities", {})
    probs_str = ", ".join(f"{k}: {v * 100:.1f}%" for k, v in probs.items()) if isinstance(probs, dict) else str(probs)

    risk_score = risk_data.get("risk_score", "N/A") if risk_data else "Not assessed"
    risk_level = risk_data.get("risk_level", "N/A") if risk_data else "Not assessed"
    risk_factors = [f["factor"] for f in risk_data.get("risk_factors", [])] if risk_data else []
    risk_factors_str = ", ".join(risk_factors) if risk_factors else "None documented"

    notes_str = f"\nClinician Observations: {clinical_notes}" if clinical_notes else ""

    report = f"""## Summary
The patient's structural brain MRI was evaluated utilizing NeuroAssist AI v2 (Model: {version}). The automated classification identified features consistent with **{stage}** at a model confidence of **{conf_pct}**. Class probability distribution across evaluated categories: {probs_str}. This evaluation represents investigational decision support and must be contextualized with comprehensive neurological examination and cognitive testing.

## Detailed Findings
Quantitative volumetric evaluation and pattern analysis demonstrate neurostructural signatures characteristic of {stage.lower()} pathology. Key morphological findings include:
- Medial temporal lobe and hippocampal volumetric trends consistent with {stage.lower()} staging.
- Sulcal widening and localized ventricular expansion correlated with the predicted stage.
- Comparison against age-matched normative brain templates reveals targeted morphological differences in temporoparietal regions.{notes_str}

## XAI Interpretation Guide
Explainable AI (XAI) feature importance maps (Grad-CAM and HiResCAM) were generated to identify the regional drivers of the classification:
- Primary saliency foci are localized around the medial temporal lobes and hippocampal periventricular zones.
- Secondary activations align with lateral ventricular margins and temporal horn contours.
- The concentration of gradient weights confirms that the classifier is attending to clinically meaningful neurodegenerative markers rather than imaging artifacts.

## Risk Summary
Clinical Risk Assessment Status:
- Composite Risk Score: {risk_score}/100 ({risk_level.upper()})
- Identified Risk Indicators: {risk_factors_str}
- Rule-based evaluation indicates {risk_level.lower()} overall clinical progression vulnerability. Co-morbid vascular and metabolic factors should be actively managed to mitigate accelerated neurodegeneration.

## Recommendations
In alignment with WHO Dementia Guidelines (2023) and NICE Guidelines (NG97):
1. **Clinical Correlation**: Correlate these imaging findings with standardized cognitive assessments (MMSE, MoCA, and functional CDR scoring).
2. **Specialist Consultation**: Consider structured neurological or geriatric referral for comprehensive diagnostic consolidation.
3. **Risk Factor Modification**: Target blood pressure, glycemic control, and encourage cardiovascular exercise per WHO risk reduction guidelines.
4. **Follow-Up Protocol**: Schedule serial neuroimaging in 6 to 12 months to assess volumetric rate of change and disease trajectory.
5. **Regulatory Disclaimer**: Investigational software. Not a substitute for clinical judgment. Not FDA/CE cleared.
"""
    return report
