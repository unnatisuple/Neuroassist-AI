"""
NeuroAssist AI v2 — AI Service & Groq Provider Abstraction
Unified LLM interface for clinical chatbot decision support and report generation.
Provider: Groq (via official groq Python SDK).
Strictly separates clinical prediction CNNs from language generation.
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import groq
from backend.config import settings
from backend.services.fallback_service import (
    generate_fallback_chat_reply,
    generate_fallback_report_text,
)


class GroqConfigurationError(Exception):
    """Raised when GROQ_API_KEY is missing, empty, or a placeholder."""
    pass


class GroqAuthError(Exception):
    """Raised when Groq API key is rejected by the Groq API (401)."""
    pass


class GroqRateLimitError(Exception):
    """Raised when Groq API rate limit is reached (429)."""
    pass


class GroqServiceError(Exception):
    """Raised when Groq API encounters an unexpected error or connection failure."""
    pass


def is_groq_configured() -> bool:
    """Check whether a valid-looking Groq API key is configured."""
    key = settings.effective_groq_api_key
    return bool(key and not key.startswith("REPLACE_") and not key.startswith("YOUR_"))


def get_groq_client() -> groq.Groq:
    """
    Initialize and return a Groq client instance.
    Raises GroqConfigurationError if GROQ_API_KEY is not configured.
    """
    key = settings.effective_groq_api_key
    if not key or key.startswith("REPLACE_") or key.startswith("YOUR_"):
        raise GroqConfigurationError(
            "GROQ_API_KEY is not configured or holds a placeholder. "
            "Please configure your GROQ_API_KEY in .env.local"
        )
    return groq.Groq(api_key=key)


def chat_completion(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    trace_id: str = "trace-na",
) -> Dict[str, Any]:
    """
    Execute a chat completion request via Groq.
    Returns dict: {"content": str, "usage": dict}
    """
    target_model = model or settings.groq_model or "llama-3.3-70b-versatile"

    client = get_groq_client()

    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        formatted_messages.append({"role": role, "content": msg.get("content", "")})

    FALLBACK_MODELS = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
    ]

    # Try target_model first, followed by fallbacks if 404 model not found
    candidate_models = [target_model] + [m for m in FALLBACK_MODELS if m != target_model]

    last_error = None
    for candidate in candidate_models:
        try:
            logger.info(f"[{trace_id}] [AI] Provider: Groq | Model: {candidate} | Executing completion request")
            completion = client.chat.completions.create(
                model=candidate,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = completion.choices[0].message.content or ""
            usage = {}
            if completion.usage:
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }

            logger.info(f"[{trace_id}] [AI] Provider: Groq | Model: {candidate} | Chat request completed successfully")
            return {"content": content, "usage": usage, "model": candidate}

        except groq.NotFoundError as e:
            logger.warning(f"[{trace_id}] [AI] Model '{candidate}' not available on this Groq account (404). Trying next available model...")
            last_error = e
            continue
        except groq.AuthenticationError as e:
            logger.error(f"[{trace_id}] [AI] Groq authentication error: {e}")
            raise GroqAuthError("Invalid Groq API key. Please verify GROQ_API_KEY configuration.") from e
        except groq.RateLimitError as e:
            logger.warning(f"[{trace_id}] [AI] Groq rate limit reached: {e}")
            raise GroqRateLimitError("Groq API rate limit reached. Please wait a moment and try again.") from e
        except (groq.APIConnectionError, groq.APITimeoutError) as e:
            logger.error(f"[{trace_id}] [AI] Groq connection/timeout error: {e}")
            raise GroqServiceError("Unable to reach Groq API. Please check network connectivity.") from e
        except groq.GroqError as e:
            logger.error(f"[{trace_id}] [AI] Groq API error: {e}")
            raise GroqServiceError(f"Groq API error: {e}") from e
        except Exception as e:
            logger.exception(f"[{trace_id}] [AI] Unexpected error during Groq completion: {e}")
            raise GroqServiceError(f"Unexpected AI error: {e}") from e

    if last_error:
        raise GroqServiceError(f"None of the attempted Groq models were available on this account: {last_error}") from last_error


def generate_clinical_report_text(
    prediction: Dict[str, Any],
    risk_data: Optional[Dict[str, Any]] = None,
    rag_context: str = "",
    additional_notes: str = "",
    language: str = "en",
    trace_id: str = "trace-na",
) -> str:
    """
    Generate a structured, evidence-grounded clinical decision-support report using Groq.
    Adheres strictly to supplied MRI prediction, risk factors, and RAG clinical guidelines.
    Never hallucinates unprovided patient data or diagnoses.
    """
    # If Groq is not configured (e.g. offline dev/testing), fall back gracefully
    if not is_groq_configured():
        logger.warning(f"[{trace_id}] GROQ_API_KEY not set. Using offline clinical fallback report generator.")
        return generate_fallback_report_text(
            prediction=prediction,
            risk_data=risk_data,
            rag_context=rag_context,
            clinical_notes=additional_notes,
            language=language,
        )

    target_model = settings.groq_model or "llama-3.3-70b-versatile"
    logger.info(f"[{trace_id}] [AI] Provider: Groq | Model: {target_model} | Report generation request received")

    # Format risk data summary
    if risk_data:
        risk_score = risk_data.get("risk_score", "N/A")
        risk_level = risk_data.get("risk_level", "N/A")
        risk_factors = [f.get("factor", "") for f in risk_data.get("risk_factors", []) if isinstance(f, dict)]
        risk_factors_str = ", ".join(risk_factors) if risk_factors else "None documented"
        risk_text = f"Risk Score: {risk_score}/100\nRisk Level: {risk_level}\nIdentified Factors: {risk_factors_str}"
    else:
        risk_text = "Risk Assessment: Not available in the provided data."

    # Language instruction
    lang_directive = ""
    if language == "hi":
        lang_directive = "\n\nCRITICAL LANGUAGE INSTRUCTION: Write the entire clinical report narrative in Hindi (हिन्दी)."
    elif language == "mr":
        lang_directive = "\n\nCRITICAL LANGUAGE INSTRUCTION: Write the entire clinical report narrative in Marathi (मराठी)."

    system_instruction = (
        "You are NeuroAssist AI, a clinical decision-support report generator for neurologists and clinicians.\n"
        "Generate a factual, objective, evidence-based report grounded strictly in the provided prediction data, "
        "risk indicators, and clinical guideline excerpts.\n\n"
        "STRICT CLINICAL RULES:\n"
        "1. Decision Support Only: Do NOT state definitive diagnoses. Always frame results as decision-support findings.\n"
        "2. Zero Hallucination: Do NOT invent patient history, laboratory values, medications, symptoms, or imaging findings "
        "not supplied in the prompt. If information is missing, explicitly state: 'Not available in the provided data.'\n"
        "3. Guideline Citations: Cite provided guidelines (WHO 2023, Alzheimer's Association 2024, NICE NG97) where relevant.\n"
        "4. Model Limitations: If stage is 'Moderate Demented', note that 'Severe Dementia' classification is not supported "
        "by the current 4-class model.\n"
        "5. Mandatory Disclaimer: Always include: 'Investigational software. Not a substitute for clinical judgment.'\n"
        "6. Required Output Structure: You MUST use these exact markdown headers:\n"
        "## Summary\n"
        "## Detailed Findings\n"
        "## XAI Interpretation Guide\n"
        "## Risk Summary\n"
        "## Recommendations\n"
        "7. Explainability & Brain Regions: In the '## XAI Interpretation Guide', discuss the provided Grad-CAM, "
        "Grad-CAM++, Integrated Gradients, and estimated brain region attributions. Note explicitly that brain-region "
        "attribution represents model attention and is not equivalent to a confirmed anatomical lesion or clinical diagnosis.\n"
    )

    # Format Explainability (XAI) and Brain Region findings
    xai_parts = []
    overlays = prediction.get("xai_overlays") or {}
    explainability = prediction.get("explainability") or {}

    available_methods = []
    if overlays.get("gradcam_overlay") or "grad_cam" in explainability:
        available_methods.append("Grad-CAM (coarse convolutional feature localization)")
    if overlays.get("gradcam_plus_plus_overlay") or "grad_cam_plus_plus" in explainability:
        available_methods.append("Grad-CAM++ (higher-order gradient-weighted localization)")
    if overlays.get("integrated_gradients_overlay") or "integrated_gradients" in explainability:
        available_methods.append("Integrated Gradients (path-integral pixel attribution from zero baseline)")

    if available_methods:
        xai_parts.append("Available XAI Visualizations:\n" + "\n".join(f"  • {m}" for m in available_methods))
    else:
        xai_parts.append("Available XAI Visualizations: None available for this scan.")

    br = prediction.get("brain_regions") or explainability.get("brain_regions")
    if br and isinstance(br, dict) and br.get("regions"):
        region_lines = []
        for reg in br["regions"]:
            region_lines.append(
                f"  • {reg.get('region_name')}: {reg.get('attribution_level')} (score: {reg.get('attribution_score', 0):.3f})"
            )
        xai_parts.append(
            "Estimated 2D Axial Brain Region Attributions (Image-Space Localization):\n"
            + "\n".join(region_lines)
            + f"\n  Limitation Note: {br.get('disclaimer', 'Brain-region attribution represents model attention and is not equivalent to a confirmed anatomical lesion.')}"
        )
    else:
        xai_parts.append("Brain Region Localization: Not available for this scan format.")

    xai_text = "\n\n".join(xai_parts)

    prompt = f"""Generate a structured clinical decision-support report for the following patient MRI analysis:

PATIENT & IMAGING PREDICTION DATA:
- Predicted Category: {prediction.get('predicted_class', 'Unknown')}
- Calibrated Confidence: {prediction.get('confidence', 0.0):.1%}
- Class Probabilities: {prediction.get('class_probabilities', {})}
- Classification Model Version: {prediction.get('model_version', 'lever_b_resnet18')}

EXPLAINABLE AI & BRAIN REGION FINDINGS:
{xai_text}

CLINICAL RISK ASSESSMENT DATA:
{risk_text}

ADDITIONAL CLINICAL NOTES:
{additional_notes if additional_notes else 'None provided.'}

RETRIEVED CLINICAL GUIDELINES CONTEXT:
{rag_context if rag_context else 'No external guideline documents retrieved. Base report strictly on model outputs and standard clinical prudence.'}
{lang_directive}
"""

    response = chat_completion(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system_instruction,
        model=target_model,
        temperature=0.1,
        max_tokens=2000,
        trace_id=trace_id,
    )

    logger.info(f"[{trace_id}] [AI] Provider: Groq | Model: {target_model} | Report generation completed successfully")
    return response["content"]
