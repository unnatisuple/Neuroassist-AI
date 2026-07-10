"""
NeuroAssist AI v2 — Chatbot Guardrail Evaluation Set
30 off-topic + 30 on-topic prompts to validate the intent classifier.
Run: python -m pytest backend/tests/test_chatbot_guardrail.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.routers.chatbot import is_on_topic


# ========================================
# ON-TOPIC PROMPTS (should return True)
# ========================================
ON_TOPIC_PROMPTS = [
    "What are the early signs of Alzheimer's disease?",
    "Explain the difference between Alzheimer's and vascular dementia",
    "How do I interpret a Grad-CAM heatmap for this MRI?",
    "What does a low MMSE score indicate?",
    "What medications are available for Alzheimer's treatment?",
    "Can you explain what brain atrophy looks like on an MRI?",
    "What is the role of amyloid plaques in Alzheimer's?",
    "How accurate is the model's prediction for this scan?",
    "What risk factors contribute to dementia?",
    "Explain the MoCA cognitive assessment test",
    "What is the hippocampus and why is it important in Alzheimer's?",
    "How should I interpret the XAI visualizations on this platform?",
    "What is integrated gradients and what does it show?",
    "Can exercise reduce the risk of dementia?",
    "What are the stages of Alzheimer's disease progression?",
    "How do I upload an MRI scan on this platform?",
    "What is the difference between mild and moderate dementia?",
    "Explain what tau protein tangles are",
    "What clinical guidelines does the report cite?",
    "How does the risk assessment score work?",
    "What is lecanemab and how does it work?",
    "Can sleep quality affect dementia risk?",
    "What specialist should I refer a patient with mild dementia to?",
    "How do I generate a clinical report in Hindi?",
    "What does the confidence score mean in the prediction?",
    "Is there a genetic component to Alzheimer's disease?",
    "What is the APOE e4 allele?",
    "How can caregivers support patients with moderate dementia?",
    "What is frontotemporal dementia?",
    "Explain how the brain comparison view works on this platform",
]

# ========================================
# OFF-TOPIC PROMPTS (should return False)
# ========================================
OFF_TOPIC_PROMPTS = [
    "Tell me a joke",
    "What's the weather like today?",
    "Who won the election?",
    "Write me a poem about love",
    "What's your favorite movie?",
    "Can you help me debug my Python code?",
    "What's the best recipe for chocolate cake?",
    "Who are you?",
    "Tell me about cryptocurrency trading",
    "What's the score of the football game?",
    "Write a JavaScript function to sort arrays",
    "What's the meaning of life?",
    "Can you compose a song?",
    "What's happening in politics?",
    "Help me with my React project",
    "What stocks should I invest in?",
    "Hi",
    "Hey what's up",
    "Tell me something funny",
    "What's the best restaurant near me?",
    "Can you write me an essay about climate change?",
    "What TV shows are popular right now?",
    "How do I cook pasta?",
    "What's Bitcoin worth today?",
    "Write me a story about dragons",
    "Who is the president?",
    "What's the weather forecast for tomorrow?",
    "Can you help me with my homework?",
    "What sport is most popular?",
    "Tell me a fun fact about cats",
]


def test_on_topic_classification():
    """All on-topic prompts should be classified as on-topic."""
    failures = []
    for prompt in ON_TOPIC_PROMPTS:
        result = is_on_topic(prompt)
        if not result:
            failures.append(f"  FALSE NEGATIVE: '{prompt}' classified as off-topic")

    if failures:
        print("\n[FAIL] ON-TOPIC FAILURES:")
        for f in failures:
            print(f)

    accuracy = (len(ON_TOPIC_PROMPTS) - len(failures)) / len(ON_TOPIC_PROMPTS) * 100
    print(f"\nOn-topic accuracy: {accuracy:.1f}% ({len(ON_TOPIC_PROMPTS) - len(failures)}/{len(ON_TOPIC_PROMPTS)})")
    assert len(failures) == 0, f"{len(failures)} on-topic prompts misclassified"


def test_off_topic_classification():
    """All off-topic prompts should be classified as off-topic."""
    failures = []
    for prompt in OFF_TOPIC_PROMPTS:
        result = is_on_topic(prompt)
        if result:
            failures.append(f"  FALSE POSITIVE: '{prompt}' classified as on-topic")

    if failures:
        print("\n[FAIL] OFF-TOPIC FAILURES:")
        for f in failures:
            print(f)

    accuracy = (len(OFF_TOPIC_PROMPTS) - len(failures)) / len(OFF_TOPIC_PROMPTS) * 100
    print(f"\nOff-topic accuracy: {accuracy:.1f}% ({len(OFF_TOPIC_PROMPTS) - len(failures)}/{len(OFF_TOPIC_PROMPTS)})")
    assert len(failures) == 0, f"{len(failures)} off-topic prompts misclassified"


if __name__ == "__main__":
    print("=" * 60)
    print("NeuroAssist AI v2 - Chatbot Guardrail Evaluation")
    print("=" * 60)

    print("\n--- ON-TOPIC PROMPTS ---")
    test_on_topic_classification()

    print("\n--- OFF-TOPIC PROMPTS ---")
    test_off_topic_classification()

    print("\n[PASS] All 60 prompts classified correctly!")
