"""
NeuroAssist AI v2 — Recommendation Service
Deterministic stage→recommendations mappings from clinical guidelines.
These are rule-based by design (not ML) and sourced from WHO/Alzheimer's Association.
"""


def get_recommendations(predicted_class: str) -> dict:
    """
    Get stage-appropriate recommendations.
    Rule-based mapping — explicitly NOT ML-based.
    Content sourced from WHO Dementia Action Plan and Alzheimer's Association guidelines.
    """
    stage_map = {
        "Non-Demented": _non_demented_recs(),
        "Very Mild Demented": _very_mild_recs(),
        "Mild Demented": _mild_recs(),
        "Moderate Demented": _moderate_recs(),
    }

    return stage_map.get(predicted_class, _non_demented_recs())


def _non_demented_recs() -> dict:
    return {
        "stage": "Non-Demented",
        "specialists": [
            {"name": "Primary Care Physician", "specialty": "General Medicine",
             "rationale": "Routine cognitive health monitoring"},
        ],
        "exercises": [
            {"name": "Aerobic Exercise", "description": "Brisk walking, swimming, or cycling",
             "frequency": "150 min/week (WHO guideline)"},
            {"name": "Cognitive Training", "description": "Puzzles, reading, learning new skills",
             "frequency": "Daily, 20-30 minutes"},
            {"name": "Social Engagement", "description": "Regular social activities and group participation",
             "frequency": "Multiple times per week"},
        ],
        "education_resources": [
            {"title": "Alzheimer's Association: 10 Ways to Love Your Brain",
             "url": "https://www.alz.org/help-support/brain_health/10-ways-to-love-your-brain",
             "description": "Evidence-based lifestyle recommendations for brain health."},
            {"title": "WHO Risk Reduction of Cognitive Decline",
             "url": "https://www.who.int/publications/i/item/risk-reduction-of-cognitive-decline-and-dementia",
             "description": "WHO evidence-based guidelines for dementia risk reduction."},
        ],
        "caregiver_guidance": [
            "No caregiver support needed at this stage.",
            "Encourage healthy lifestyle habits for prevention.",
            "Schedule regular cognitive screening (annually after age 65).",
        ],
        "methodology_note": (
            "These recommendations are rule-based mappings from WHO and Alzheimer's Association "
            "clinical guidelines. They are not personalized ML predictions."
        ),
    }


def _very_mild_recs() -> dict:
    return {
        "stage": "Very Mild Demented",
        "specialists": [
            {"name": "Neurologist", "specialty": "Neurology",
             "rationale": "Comprehensive cognitive evaluation and baseline assessment"},
            {"name": "Neuropsychologist", "specialty": "Neuropsychology",
             "rationale": "Detailed cognitive testing (MMSE, MoCA, full neuropsych battery)"},
        ],
        "exercises": [
            {"name": "Structured Aerobic Exercise", "description": "Supervised walking or swimming programs",
             "frequency": "150 min/week, moderate intensity"},
            {"name": "Balance & Strength Training", "description": "Yoga, tai chi, resistance exercises",
             "frequency": "2-3 times per week"},
            {"name": "Cognitive Stimulation Therapy (CST)", "description": "Group-based themed activities",
             "frequency": "Twice per week, in structured sessions"},
            {"name": "Music Therapy", "description": "Listening to or playing familiar music",
             "frequency": "Daily, 30 minutes"},
        ],
        "education_resources": [
            {"title": "Alzheimer's Association: Mild Cognitive Impairment",
             "url": "https://www.alz.org/alzheimers-dementia/what-is-dementia/related_conditions/mild-cognitive-impairment",
             "description": "Understanding early cognitive changes and what to expect."},
            {"title": "NICE Guideline: Dementia Assessment and Diagnosis",
             "url": "https://www.nice.org.uk/guidance/ng97",
             "description": "Evidence-based pathway for dementia assessment."},
        ],
        "caregiver_guidance": [
            "Begin discussing future care preferences and legal planning (advance directives).",
            "Monitor for driving safety and medication management challenges.",
            "Introduce memory aids (calendars, reminders, labeled items).",
            "Consider joining a caregiver support group early.",
        ],
    }


def _mild_recs() -> dict:
    return {
        "stage": "Mild Demented",
        "specialists": [
            {"name": "Neurologist", "specialty": "Neurology",
             "rationale": "Ongoing disease monitoring and medication management"},
            {"name": "Geriatric Psychiatrist", "specialty": "Geriatric Psychiatry",
             "rationale": "Management of behavioral and psychological symptoms"},
            {"name": "Occupational Therapist", "specialty": "Occupational Therapy",
             "rationale": "Strategies for maintaining daily living activities"},
            {"name": "Speech-Language Pathologist", "specialty": "Speech Therapy",
             "rationale": "Communication strategies and swallowing assessment"},
        ],
        "exercises": [
            {"name": "Supervised Walking", "description": "Guided walks in safe environments",
             "frequency": "30 min daily"},
            {"name": "Chair-Based Exercises", "description": "Seated stretching and light resistance",
             "frequency": "3-4 times per week"},
            {"name": "Reminiscence Therapy", "description": "Discussing past experiences with photos/objects",
             "frequency": "2-3 times per week"},
            {"name": "Art or Music Therapy", "description": "Creative expression activities",
             "frequency": "Weekly sessions"},
        ],
        "education_resources": [
            {"title": "Alzheimer's Association: Stages of Alzheimer's",
             "url": "https://www.alz.org/alzheimers-dementia/stages",
             "description": "Understanding disease progression and planning ahead."},
            {"title": "WHO iSupport Programme",
             "url": "https://www.who.int/publications/i/item/9789241515863",
             "description": "WHO-endorsed caregiver training and support program."},
        ],
        "caregiver_guidance": [
            "Establish consistent daily routines — predictability reduces agitation.",
            "Ensure home safety (remove tripping hazards, install grab bars, use door alarms).",
            "Supervise medication administration.",
            "Plan for respite care to prevent caregiver burnout.",
            "Consider in-home care assistance for ADLs (bathing, dressing).",
        ],
    }


def _moderate_recs() -> dict:
    return {
        "stage": "Moderate Demented",
        "specialists": [
            {"name": "Neurologist", "specialty": "Neurology",
             "rationale": "Disease progression monitoring and treatment adjustment"},
            {"name": "Geriatrician", "specialty": "Geriatric Medicine",
             "rationale": "Comprehensive management of comorbidities"},
            {"name": "Palliative Care Specialist", "specialty": "Palliative Medicine",
             "rationale": "Quality of life optimization and symptom management"},
            {"name": "Social Worker", "specialty": "Medical Social Work",
             "rationale": "Care coordination, community resources, and legal/financial planning"},
        ],
        "exercises": [
            {"name": "Gentle Movement", "description": "Assisted range-of-motion exercises",
             "frequency": "Daily, with caregiver assistance"},
            {"name": "Sensory Stimulation", "description": "Tactile activities, aromatherapy, garden visits",
             "frequency": "Multiple times daily"},
            {"name": "Music Listening", "description": "Familiar music from the person's past",
             "frequency": "Daily, as tolerated"},
        ],
        "education_resources": [
            {"title": "Alzheimer's Association: Late-Stage Caregiving",
             "url": "https://www.alz.org/help-support/caregiving/stages-behaviors",
             "description": "Guidance for moderate-to-late stage caregiving challenges."},
            {"title": "National Institute on Aging: Caring for a Person with Alzheimer's Disease",
             "url": "https://www.nia.nih.gov/health/caregiving/caring-person-alzheimers-disease",
             "description": "Comprehensive federal resource for dementia caregiving."},
        ],
        "caregiver_guidance": [
            "24-hour supervision is likely needed — evaluate residential care options.",
            "Use simple, one-step instructions for communication.",
            "Monitor nutritional intake and hydration closely.",
            "Plan for end-of-life care preferences (if not already done).",
            "Prioritize caregiver mental health — burnout is common at this stage.",
            "Consider professional in-home or memory care facility placement.",
        ],
    }
