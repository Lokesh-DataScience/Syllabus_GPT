import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load API key
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in your .env file")

client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# HYDE DOCUMENT GENERATION
# ============================================================
def generate_hyde_document(topic: str) -> str:
    """
    HYDE = Hypothetical Document Embedding
    Generates a synthetic explanation of a topic that looks like
    textbook material, improving retrieval quality.
    """

    system_prompt = (
        "You are an academic assistant. "
        "Given a topic, generate a short hypothetical explanation as if from a textbook. "
        "Keep it factual, structured, and instructional."
    )

    user_prompt = f"""
Generate a hypothetical academic explanation for the topic:

TOPIC:
{topic}

Write 1–2 paragraphs that resemble real study material.
Do NOT mention that this is hypothetical.
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",   # ✅ A real, current Groq model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )

    message = response.choices[0].message
    return message["content"] if isinstance(message, dict) else message.content


# ============================================================
# SYLLABUS → TOPIC LIST PARSER
# ============================================================
def parse_syllabus_into_topics(syllabus_text: str) -> list:
    """
    Converts raw syllabus text into a structured topic list.
    Always attempts JSON parsing; falls back gracefully.
    """

    system_prompt = (
        "You extract topics from syllabus text. "
        "Return ONLY a JSON list of clean topic names. No extra text."
    )

    user_prompt = f"""
Extract the list of topics from this syllabus:

{syllabus_text}

Output MUST be valid JSON list, example:
["Topic 1", "Topic 2", "Topic 3"]
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",   # Same stable model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0
    )

    raw = response.choices[0].message.content

    # Try converting to JSON
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except:
        pass

    # Fallback: return line split
    return [
        line.strip("-• ").strip()
        for line in raw.split("\n")
        if len(line.strip()) > 2
    ]
