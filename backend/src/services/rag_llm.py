import os
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq

from src.services.hyde_llm import generate_hyde_document
from src.services.vector_store import retrieve_relevant_context

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in the environment (.env)")

client = Groq(api_key=GROQ_API_KEY)


def _call_groq_chat(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """
    Small helper to call Groq LLM.
    """
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    # Groq returns choices[0].message.content
    msg = resp.choices[0].message
    # depending on SDK version this can be dict-like or object with .content
    content = msg["content"] if isinstance(msg, dict) else msg.content
    return content


def generate_notes(topic: str, context_chunks: List[str]) -> str:
    """
    Low-level: generate structured notes from topic + context chunks.
    This is used by the higher-level RAG pipeline.
    """
    context = "\n\n".join(context_chunks)

    system_prompt = (
        "You are an expert engineering mentor. "
        "You generate exam-focused, clear and well-structured notes for university students."
    )

    user_prompt = f"""
Generate detailed, exam-ready study notes for the topic:

TOPIC:
{topic}

USING ONLY the following context (books, notes, PYQs):

CONTEXT:
{context}

Requirements:
- Write in simple, clear English.
- Structure the notes with headings and bullet points.
- Include:
    1. Short introduction
    2. Explanation of concepts
    3. Important formulas / definitions (if any)
    4. Examples or explanations for common questions
    5. Key points / summary
- Focus on what is important for university exams.
- Do NOT mention that you used context.
"""

    return _call_groq_chat(system_prompt, user_prompt, temperature=0.15)


def generate_notes_with_rag(
    syllabus_text: str,
    subject: Optional[str] = None,
    use_pyq: bool = False,
    top_k: int = 12,
) -> str:
    """
    FULL PIPELINE:

    1. Take user's syllabus text (topic / unit).
    2. Use HyDE to generate a hypothetical doc for better retrieval.
    3. Retrieve top_k chunks from ChromaDB (books or PYQs).
    4. Generate final structured notes using Groq LLM.
    """

    # 1) HyDE – generate hypothetical doc based on the syllabus text
    hyde_doc = generate_hyde_document(syllabus_text)

    # 2) Retrieve context (BOOK or PYQ) from Chroma
    combined_context = retrieve_relevant_context(
        syllabus_text=hyde_doc,
        subject=subject,
        use_pyq=use_pyq,
        top_k=top_k,
    )

    if not combined_context.strip():
        # Fallback: if retrieval returns nothing, still answer from model
        return generate_notes(syllabus_text, [ "No context found in KB, answer from general knowledge." ])

    # 3) Use LLM to generate final notes
    notes = generate_notes(
        topic=syllabus_text,
        context_chunks=[combined_context],
    )

    return notes
