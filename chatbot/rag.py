from __future__ import annotations

from dataclasses import dataclass

from chatbot.gemini_client import generate_answer
from utils.vector_store import SearchResult, search


LANGUAGE_LABELS = {
    "English": "English",
    "Hindi": "Hindi written in clear Devanagari where appropriate",
    "Nepali": "Nepali written in clear Devanagari where appropriate",
}


@dataclass
class TutorAnswer:
    answer: str
    sources: list[SearchResult]


def format_context(results: list[SearchResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        meta = result.metadata
        source = meta.get("source", "Uploaded material")
        page = meta.get("page", "?")
        subject = meta.get("subject", "General")
        chapter = meta.get("chapter", "Unclassified")
        blocks.append(
            f"[Source {index}] {source}, page {page}, subject: {subject}, chapter: {chapter}\n{result.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    context: str,
    *,
    language: str,
    subject: str,
    chapter: str,
    chat_history: list[dict],
) -> str:
    recent_history = "\n".join(
        f"{message['role'].title()}: {message['content'][:700]}"
        for message in chat_history[-6:]
    )
    language_instruction = LANGUAGE_LABELS.get(language, "English")

    return f"""
You are NEET AI Tutor, an expert NEET teacher for Biology, Physics, and Chemistry.
Answer using the retrieved previous-year NEET context first. Prioritize NCERT-based explanations.

Student question:
{question}

Selected subject: {subject}
Selected chapter: {chapter}
Response language: {language_instruction}

Recent chat history:
{recent_history or "No previous chat."}

Retrieved NEET material:
{context or "No relevant uploaded material was found."}

Teaching rules:
1. Explain step by step like a patient NEET teacher.
2. Include concept explanation, formulas used, and shortcuts/tricks where applicable.
3. Explain why the other options are wrong when options are present.
4. Mention similar previous-year NEET questions or recommend related practice questions if the retrieved context contains them.
5. Show final answer clearly.
6. If the context is weak or uncertain, say so honestly and separate what is inferred from what is sourced.
7. Keep the explanation exam-focused, simple, and not overly short.
8. For Biology, anchor explanations in NCERT language where possible.
9. For Physics and Chemistry numerical questions, show formula, substitution, units, and final result.

Return the answer with these headings:
Answer
Step-by-Step Solution
Concept & NCERT Link
Formulas / Key Facts
Why Other Options Are Wrong
Shortcut or Exam Tip
Similar PYQ References
Confidence
""".strip()


def answer_question(
    question: str,
    *,
    subject: str = "All",
    chapter: str = "All",
    language: str = "English",
    chat_history: list[dict] | None = None,
    k: int = 6,
) -> TutorAnswer:
    results = search(question, subject=subject, chapter=chapter, k=k)
    context = format_context(results)
    prompt = build_prompt(
        question,
        context,
        language=language,
        subject=subject,
        chapter=chapter,
        chat_history=chat_history or [],
    )
    answer = generate_answer(prompt)
    return TutorAnswer(answer=answer, sources=results)
