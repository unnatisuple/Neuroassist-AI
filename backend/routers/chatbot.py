"""
NeuroAssist AI v2 — Chatbot Router (Gemini-powered, scope-enforced)
Only answers Alzheimer's/dementia/MRI/XAI/platform questions.
Declines off-topic queries with a professional redirect, not an apology loop.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
from backend.schemas import ChatRequest, ChatResponse, ChatMessage
from backend.dependencies import require_verified_doctor, generate_trace_id
from backend.db.mongodb import get_chat_sessions_collection
from backend.config import settings
from loguru import logger
import uuid
import re

router = APIRouter(prefix="/api/chatbot", tags=["Medical Assistant Chatbot"])

# ========================================
# INTENT CLASSIFIER / GUARDRAIL
# Runs BEFORE the Gemini call — deterministic off-topic detection.
# ========================================

ON_TOPIC_KEYWORDS = [
    r"alzheimer", r"dement", r"cognitive", r"memory", r"brain", r"mri", r"scan",
    r"neurolog", r"neurodegen", r"amyloid", r"tau\b", r"plaque",
    r"hippocampus", r"cortex", r"atrophy", r"ventricle", r"csf\b",
    r"mmse\b", r"moca\b", r"cognitive test", r"mental status",
    r"gradcam", r"grad-cam", r"heatmap", r"explain", r"xai\b", r"saliency",
    r"integrated gradient", r"shap\b", r"lime\b", r"captum",
    r"prediction", r"confidence", r"model", r"classification", r"stage",
    r"risk", r"assessment", r"factor",
    r"report", r"pdf\b", r"download", r"generate",
    r"mild", r"moderate", r"severe", r"non-demented", r"demented",
    r"caregiver", r"care", r"treatment", r"medication", r"drug", r"therapy",
    r"donepezil", r"memantine", r"aricept", r"namenda", r"lecanemab", r"aducanumab",
    r"exercise", r"diet", r"sleep", r"lifestyle", r"prevention",
    r"specialist", r"referral", r"recommendation",
    r"clinical", r"patient", r"history", r"diagnosis", r"prognosis",
    r"biomarker", r"pet\b", r"spect\b", r"eeg\b", r"lumbar",
    r"neuroassist", r"platform", r"upload", r"feature", r"how to",
    r"medical", r"health", r"disease", r"symptom", r"sign\b",
    r"vascular", r"lewy", r"frontotemporal", r"parkinson",
]

OFF_TOPIC_PATTERNS = [
    r"\b(joke[s]?|funny|laugh|humor|entertain|comedy|comedian)\b",
    r"\b(politic[s]?|election[s]?|vote[sd]?|president[s]?|congress|parliament|government|senate)\b",
    r"\b(weather|forecast|temperature|rain|snow|cloudy|sunny|windy)\b",
    r"\b(recipe[s]?|cook(ing)?|food|restaurant[s]?|baking|dinner|lunch|breakfast|cuisine|chef)\b",
    r"\b(movie[s]?|film[s]?|tv show[s]?|netflix|show[s]?|series|game[s]?|sport[s]?|score[s]?|football|basketball|soccer|baseball|hockey|cricket|tennis|olympic[s]?)\b",
    r"\b(stock[s]?|crypto|bitcoin|invest|trading|finance|market[s]?|money|dollar[s]?|portfolio[s]?)\b",
    r"\b(code[sd]?|coding|python|javascript|typescript|react|debug|programming|html|css|developer|software|git|github)\b",
    r"\b(poem[s]?|poetry|write me a|compose|story|stories|essay[s]?|song[s]?|lyric[s]?|novel[s]?|fiction)\b",
    r"\b(who are you|what are you|your name|are you human|creator|who made you|who built you)\b",
    r"\b(hello|hi|hey|sup|what'?s up|good morning|good afternoon|good evening)\s*$",
    r"\b(cat[s]?|dog[s]?|animal[s]?|capital[s]?|country|countries|history|earth|science|math|physics|chemistry|space|galaxy|moon|sun|star[s]?|planet[s]?|world|fact[s]?)\b",
    r"\b(meaning of life|philosophy|existential|love|relationship[s]?|friend[s]?|advice|opinion[s]?)\b",
    r"\b(homework|assignment[s]?|grade[s]?|school|teacher[s]?|class[es]?|university|college|study|student[s]?)\b",
]


def is_on_topic(message: str) -> bool:
    """
    Lightweight intent classifier: check if the message is about
    Alzheimer's, dementia, MRI, XAI, or platform features.
    Returns True if on-topic, False if off-topic.
    """
    msg_lower = message.lower().strip()

    # 1. Check for off-topic patterns first
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            # But allow if it also has medical/platform keywords (e.g., "write me a report about dementia")
            has_medical = False
            for kw in ON_TOPIC_KEYWORDS:
                if re.search(kw, msg_lower, re.IGNORECASE):
                    has_medical = True
                    break
            if has_medical:
                return True
            return False

    # 2. Short greetings/ambiguous messages without substance
    if len(msg_lower.split()) <= 3:
        # Check if it has any medical/platform keyword
        for kw in ON_TOPIC_KEYWORDS:
            if re.search(kw, msg_lower, re.IGNORECASE):
                return True
        return False

    # 3. For longer messages, check if there's any on-topic medical or platform keyword
    has_medical = False
    for kw in ON_TOPIC_KEYWORDS:
        if re.search(kw, msg_lower, re.IGNORECASE):
            has_medical = True
            break

    if has_medical:
        return True

    # Standard short conversational continuations are allowed
    conversational_followups = [
        r"\b(why|how|explain|what|who|where|when|which|yes|no|ok|okay|thanks|thank you)\b"
    ]
    for pattern in conversational_followups:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return True

    return False


REFUSAL_MESSAGE = (
    "I'm the NeuroAssist AI medical assistant, specialized in Alzheimer's disease, "
    "dementia, brain MRI interpretation, and this platform's clinical features. "
    "I can help you with:\n\n"
    "• Understanding dementia stages and progression\n"
    "• Interpreting MRI scan results and XAI visualizations\n"
    "• Clinical risk factors and assessment\n"
    "• Treatment options and clinical guidelines\n"
    "• Using this platform's features\n\n"
    "Please ask a question related to these topics."
)

SYSTEM_PROMPT = """You are the NeuroAssist AI Medical Assistant — a specialized clinical decision support chatbot embedded in a doctor-only platform for Alzheimer's disease analysis.

SCOPE — You ONLY answer questions about:
1. Alzheimer's disease (pathology, stages, biomarkers, diagnosis, treatment, prognosis)
2. Dementia (all types: vascular, Lewy body, frontotemporal, mixed)
3. MRI/neuroimaging interpretation concepts (brain anatomy, atrophy patterns, volumetrics)
4. This platform's features (uploading scans, running predictions, XAI heatmaps, risk assessment, reports)
5. Explainable AI outputs (Grad-CAM, Integrated Gradients — what they show and how to interpret them)
6. Clinical risk assessment and risk factors
7. Medical terminology related to neurology and cognitive disorders
8. Report interpretation and clinical workflow guidance

RULES:
- Be concise, professional, and evidence-based.
- Cite clinical guidelines (WHO, Alzheimer's Association, NICE) when relevant.
- NEVER provide a definitive diagnosis — always frame outputs as "decision support."
- If asked about topics outside your scope, respond: "That's outside my area of expertise. I specialize in Alzheimer's disease, dementia, and neuroimaging. How can I help you with those topics?"
- NEVER apologize repeatedly or say "I'm sorry, I can't help with that" in a loop.
- Keep responses focused and actionable.
- You may respond in English, Hindi, or Marathi as requested.

IMPORTANT DISCLAIMER TO INCLUDE WHEN DISCUSSING CLINICAL DECISIONS:
"This is investigational software for decision support only. All clinical decisions should be made by the treating physician."
"""


@router.post("/message", response_model=ChatResponse)
async def chat_message(
    req: ChatRequest,
    doctor: dict = Depends(require_verified_doctor),
):
    """
    Send a message to the Gemini-powered medical assistant.
    Off-topic messages are rejected deterministically by the guardrail,
    not by hoping the LLM follows its system prompt.
    """
    trace_id = generate_trace_id()

    # ---- GUARDRAIL: deterministic off-topic detection ----
    if not is_on_topic(req.message):
        # Log the refusal
        logger.info(f"[{trace_id}] Chatbot guardrail: off-topic message from doctor={doctor['_id']}")

        return ChatResponse(
            session_id=req.session_id or str(uuid.uuid4()),
            reply=REFUSAL_MESSAGE,
            is_on_topic=False,
            trace_id=trace_id,
            timestamp=datetime.now(timezone.utc),
        )

    # ---- ON-TOPIC: call Gemini ----
    session_id = req.session_id or str(uuid.uuid4())
    sessions = get_chat_sessions_collection()

    # Load conversation history
    session = await sessions.find_one({"_id": session_id, "doctor_id": doctor["_id"]})
    history = session.get("messages", []) if session else []

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)

        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=SYSTEM_PROMPT,
        )

        # Build conversation for Gemini
        gemini_history = []
        for msg in history[-10:]:  # Last 10 messages for context
            gemini_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]],
            })

        chat = model.start_chat(history=gemini_history)

        # Add language instruction if not English
        user_message = req.message
        if req.language.value != "en":
            lang_name = {"hi": "Hindi", "mr": "Marathi"}.get(req.language.value, "English")
            user_message = f"[Respond in {lang_name}] {req.message}"

        response = chat.send_message(user_message)
        reply = response.text

    except Exception as e:
        logger.error(f"[{trace_id}] Gemini API call failed: {type(e).__name__}: {e}")

        error_str = str(e).lower()
        is_auth_error = (
            "api key" in error_str or 
            "authentication" in error_str or 
            "credential" in error_str or 
            "google_application_credentials" in error_str or 
            "defaultcredentialserror" in error_str
        )

        if is_auth_error:
            logger.error(
                f"[{trace_id}] Gemini credentials/auth failure: {type(e).__name__}. "
                f"Configured API key length = {len(settings.gemini_api_key) if settings.gemini_api_key else 0}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Medical assistant configuration error — contact your administrator. Trace ID: {trace_id}",
            )
        elif "quota" in error_str or "rate" in error_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "Gemini API rate limit reached. Please wait a moment and try again. "
                    f"Trace ID: {trace_id}"
                ),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Gemini API error: {type(e).__name__}. "
                    f"The medical assistant is temporarily unavailable. "
                    f"Trace ID: {trace_id}"
                ),
            )

    # Save conversation to MongoDB
    new_messages = [
        {"role": "user", "content": req.message, "timestamp": datetime.now(timezone.utc).isoformat()},
        {"role": "assistant", "content": reply, "timestamp": datetime.now(timezone.utc).isoformat()},
    ]

    if session:
        await sessions.update_one(
            {"_id": session_id},
            {"$push": {"messages": {"$each": new_messages}},
             "$set": {"updated_at": datetime.now(timezone.utc)}},
        )
    else:
        await sessions.insert_one({
            "_id": session_id,
            "doctor_id": doctor["_id"],
            "messages": new_messages,
            "language": req.language.value,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        is_on_topic=True,
        trace_id=trace_id,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/sessions")
async def list_chat_sessions(
    doctor: dict = Depends(require_verified_doctor),
):
    """List all chat sessions for the current doctor."""
    sessions = get_chat_sessions_collection()
    cursor = sessions.find(
        {"doctor_id": doctor["_id"]},
        {"messages": {"$slice": -1}, "created_at": 1, "updated_at": 1},
    ).sort("updated_at", -1).limit(50)

    result = []
    async for session in cursor:
        result.append({
            "session_id": session["_id"],
            "last_message": session.get("messages", [{}])[-1].get("content", "")[:100] if session.get("messages") else "",
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
        })

    return {"sessions": result}
