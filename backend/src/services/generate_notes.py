import os
from dotenv import load_dotenv
from groq import Groq

from src.routes.parse_topics import parse_syllabus_into_topics
from src.services.hyde_llm import generate_hyde_document
from src.services.vector_store import retrieve_relevant_context

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env")

client = Groq(api_key=GROQ_API_KEY)


# -----------------------------------------------------------
# Final Notes Generator
# -----------------------------------------------------------

def generate_final_notes(topics: list, rag_context: str) -> str:
    """
    Generates final structured markdown notes using topics + retrieved context.
    """

    prompt = f"""
You are a university-level notes generator.
Create extremely clean, well-structured Markdown notes.

Follow this format strictly:

# {topics[0] if topics else "Generated Notes"}

## 📌 Topics Covered
- {chr(10).join(topics)}

---

## 📘 Detailed Notes
Use the context below to generate accurate explanations.

### Context:
{rag_context}

---

Now write the notes:

- Use proper headings
- Use bullet points
- Keep explanations clear and concise
- Add small examples where helpful
- Do NOT include the context directly
- Make it look like a handwritten guide for exam preparation
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # recommended stable model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
    )

    return response.choices[0].message.content
