import os
import re
from typing import List, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

# Import your existing services
from src.services.hyde_llm import generate_hyde_document
from src.services.vector_store import retrieve_relevant_context

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")

client = Groq(api_key=GROQ_API_KEY)

# Using a larger model if available (70b) is better for formatting compliance, 
# otherwise sticking to 8b-instant.
MODEL_NAME = "llama-3.3-70b-versatile" 
# Fallback to "llama-3.1-8b-instant" if rate limits are an issue.

# -------------------------------------------------
# 1. Syllabus Parsing Utilities
# -------------------------------------------------
def split_syllabus_into_units(syllabus_text: str) -> List[Dict[str, str]]:
    """
    Splits a syllabus text into individual units based on UNIT markers.
    Handles 'UNIT-1', 'UNIT I', 'Unit 1', etc.
    """
    text = syllabus_text.replace("\r", " ").strip()
    # Robust regex for unit headers
    pattern = r"(?i)(UNIT[\s\-]*(?:[IVX]+|\d+))[:\s]*"
    
    parts = re.split(pattern, text)
    
    # If split didn't work (no clear markers), return whole text as one unit
    if len(parts) < 2:
        return [{"unit_title": "UNIT-I", "unit_text": text}]

    units: List[Dict[str, str]] = []
    
    # re.split keeps the delimiter (the unit title) in the list
    # The list usually starts with empty string if the text starts with UNIT
    start_index = 1 if parts[0].strip() == "" else 0
    
    for i in range(start_index, len(parts) - 1, 2):
        unit_title = parts[i].strip().replace(":", "")
        unit_text = parts[i+1].strip()
        if unit_text:
            units.append({"unit_title": unit_title, "unit_text": unit_text})

    return units


def extract_subtopics(unit_text: str) -> List[str]:
    """
    Extracts individual subtopics for the LLM to focus on.
    """
    # Clean text
    text = re.sub(r"\s+", " ", unit_text)
    # Split by common delimiters used in syllabus
    raw_parts = re.split(r"[.,;]|\sand\s|\sor\s|\n", text)
    # Filter out empty or too short strings
    subtopics = [p.strip(" -–—:•") for p in raw_parts if len(p.strip()) > 3]
    return list(set(subtopics)) # Deduplicate


def _truncate_context(text: str, max_chars: int = 6000) -> str:
    """
    Truncates context to ensure we don't hit token limits while keeping key info.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[context truncated]..."


# -------------------------------------------------
# 2. Core Note Generation Logic
# -------------------------------------------------
def generate_unit_notes(
    unit_title: str,
    unit_text: str,
    subject: Optional[str],
    use_pyq: bool,
    top_k: int,
) -> str:
    """
    Generates detailed, textbook-style notes for a single unit.
    """
    subtopics = extract_subtopics(unit_text)
    
    # 1. Semantic Search Prep (HyDE)
    hyde_seed = f"Explain the concepts of {unit_title} in {subject or 'Data Science'}: {unit_text}"
    hyde_doc = generate_hyde_document(hyde_seed)

    # 2. Retrieve Context (RAG)
    # Concepts
    book_context = retrieve_relevant_context(
        syllabus_text=hyde_doc,
        subject=subject,
        use_pyq=False,
        top_k=min(top_k, 25),
    )
    # Previous Year Questions (if enabled)
    pyq_context = ""
    if use_pyq:
        pyq_raw = retrieve_relevant_context(
            syllabus_text=hyde_doc,
            subject=subject,
            use_pyq=True,
            top_k=5,
        )
        pyq_context = f"\nRELEVANT PAST EXAM QUESTIONS:\n{pyq_raw}\n"

    # Truncate to fit context window
    book_context = _truncate_context(book_context, 5000)
    
    # 3. Construct the Prompt
    subtopic_list_str = "\n".join([f"- {s}" for s in subtopics])

    system_prompt = """You are an expert academic author and university professor. 
You create high-quality, comprehensive study notes that look like they come from a premium textbook.

**YOUR GOAL:** Convert the provided syllabus syllabus into detailed, structured, and visually scannable notes.

**FORMATTING RULES:**
1. **Hierarchy:** Use Markdown `#` for Unit, `##` for Main Topics, `###` for Sub-sections.
2. **Visuals:** You MUST use ASCII diagrams or Mermaid syntax for processes (e.g., `Input -> Processing -> Output`).
3. **Math:** Use LaTeX formatting for all formulas (e.g., `$y = mx + c$`).
4. **Tables:** Use Markdown tables to compare concepts (e.g., Supervised vs Unsupervised Learning).
5. **No Fluff:** Avoid robotic transitions. Write directly and professionally.

**TONE:** Educational, insightful, and clear. Similar to 'Head First' series or standard University notes.
"""

    user_prompt = f"""
**CONTEXT:**
Subject: {subject or "General"}
Unit: {unit_title}
Syllabus Topics: 
{unit_text}

**RETRIEVED KNOWLEDGE BASE (Source Material):**
{book_context}

{pyq_context}

---

**TASK:** Write comprehensive study notes for **{unit_title}**. 

**REQUIRED STRUCTURE (Follow this exactly):**

# {unit_title} - [Topic Name]

> **Unit Overview:** Write a 3-4 sentence summary of what this unit covers and why it matters in the real world.

---

## [Topic 1 Name]

**1. Definition**
[Provide a clear, formal definition in a blockquote]

**2. Conceptual Explanation**
[2-3 paragraphs explaining the concept in depth. Explain "Why" we need it, not just "What" it is.]

**3. Key Characteristics/Features**
[Bulleted list of properties]

**4. Process/Workflow (IF APPLICABLE)**
[If this is a process/algorithm, describe steps AND provide a visual representation]
*Example format:*
[Step 1] --> [Step 2] --> [Decision] --> [Outcome]

**5. Real-World Case Study**
[Create a specific scenario Example". detailed explanation of how the concept applies here.]

**6. Applications**
[List of industries/areas where this is used]

---
*(Repeat the structure above for every major topic in the syllabus: {subtopic_list_str})*
---

## Key Differences & Comparisons
[Create 1 or 2 comparison tables for confusing topics in this unit, e.g., Linear vs Logistic, or Data Warehouse vs Database]

| Feature | Concept A | Concept B |
| :--- | :--- | :--- |
| ... | ... | ... |

---

## Chapter Summary & Revision
* **Key Takeaway 1:** ...
* **Key Takeaway 2:** ...
* **Important Formulae:** [List any math equations used]

## Practice Questions (Based on Exam Patterns)
1. [Conceptual Question]
2. [Application Question]
3. [Problem Solving/Scenario Question]

"""

    # 4. Call LLM
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3, # Low temp for factual accuracy
            max_tokens=6000, # Allow long output
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"# Error Generating Notes for {unit_title}\n\nTechnical error: {str(e)}"


# -------------------------------------------------
# 3. Final Orchestrator
# -------------------------------------------------
def generate_final_notes(
    syllabus_text: str,
    subject: Optional[str] = None,
    use_pyq: bool = False,
    top_k: int = 40,
) -> str:
    """
    Main entry point to generate the full subject notes.
    """
    # 1. Parse Syllabus
    units = split_syllabus_into_units(syllabus_text)
    if not units:
        # Fallback if regex fails completely
        units = [{"unit_title": "Complete Syllabus", "unit_text": syllabus_text}]

    # 2. Generate content for each unit
    all_unit_content: List[str] = []
    
    # Progress indication (for console logs)
    print(f"Found {len(units)} units. Generating notes...")

    for unit in units:
        print(f"Processing {unit['unit_title']}...")
        unit_content = generate_unit_notes(
            unit_title=unit["unit_title"],
            unit_text=unit["unit_text"],
            subject=subject,
            use_pyq=use_pyq,
            top_k=top_k,
        )
        all_unit_content.append(unit_content)

    # 3. Assemble Final Document
    subject_header = subject.upper() if subject else "SUBJECT NOTES"
    
    final_markdown = f"""
# {subject_header}
**Comprehensive Study Notes & Exam Preparation**

---

## Table of Contents
"""
    # Dynamic TOC
    for unit in units:
        final_markdown += f"- [{unit['unit_title']}](#{unit['unit_title'].lower().replace(' ', '-').replace(':', '')})\n"
    
    final_markdown += "\n---\n"
    
    # Append all unit contents
    final_markdown += "\n\n".join(all_unit_content)
    
    final_markdown += f"""
\n
---
**End of Notes**
*Generated by SyllabusGPT | {subject_header}*
"""

    return final_markdown