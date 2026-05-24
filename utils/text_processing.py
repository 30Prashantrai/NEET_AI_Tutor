from __future__ import annotations

import re
from dataclasses import dataclass


SUBJECTS = ("Physics", "Chemistry", "Biology")


@dataclass(frozen=True)
class TextChunk:
    text: str
    metadata: dict


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_subject(text: str, fallback: str = "General") -> str:
    lower = text.lower()
    biology_terms = ("botany", "zoology", "biology", "genetics", "ecology", "plant", "animal", "cell", "human physiology")
    physics_terms = ("physics", "force", "motion", "current", "magnetic", "optics", "thermodynamics", "wave", "kinematics")
    chemistry_terms = ("chemistry", "mole", "organic", "inorganic", "equilibrium", "bond", "reaction", "periodic", "compound")

    scores = {
        "Biology": sum(term in lower for term in biology_terms),
        "Physics": sum(term in lower for term in physics_terms),
        "Chemistry": sum(term in lower for term in chemistry_terms),
    }
    subject, score = max(scores.items(), key=lambda item: item[1])
    return subject if score else fallback


def infer_chapter(text: str) -> str:
    patterns = [
        r"chapter\s*[:\-]\s*([A-Za-z0-9 ,&\-()]+)",
        r"unit\s*[:\-]\s*([A-Za-z0-9 ,&\-()]+)",
        r"topic\s*[:\-]\s*([A-Za-z0-9 ,&\-()]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()[:80]
    return "Unclassified"


def extract_question_metadata(text: str) -> dict:
    question_match = re.search(r"(?:^|\n)\s*(?:Q\.?\s*)?(\d{1,3})[\.\)]\s+", text, flags=re.IGNORECASE)
    answer_match = re.search(
        r"(?:answer|ans|correct option)\s*[:\-]?\s*([A-Da-d]|\d+|[^\n]{1,80})",
        text,
        flags=re.IGNORECASE,
    )
    explanation_match = re.search(
        r"(?:explanation|solution)\s*[:\-]\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    options = re.findall(r"(?:^|\n)\s*([A-Da-d])[\.\)]\s*([^\n]+)", text)
    if not options:
        options = re.findall(r"(?:^|\n)\s*\(([A-Da-d])\)\s*([^\n]+)", text)

    return {
        "question_number": question_match.group(1) if question_match else "",
        "options": " | ".join(f"{label.upper()}. {value.strip()}" for label, value in options)[:900],
        "answer": answer_match.group(1).strip() if answer_match else "",
        "has_explanation": bool(explanation_match),
    }


def split_question_blocks(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    pattern = r"(?m)(?=^\s*(?:Q\.?\s*)?\d{1,3}[\.\)]\s+)"
    blocks = [block.strip() for block in re.split(pattern, cleaned) if block.strip()]
    usable_blocks = [block for block in blocks if len(block.split()) >= 12]
    return usable_blocks if len(usable_blocks) >= 2 else []


def chunk_neet_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 180,
    base_metadata: dict | None = None,
) -> list[TextChunk]:
    question_blocks = split_question_blocks(text)
    if not question_blocks:
        return chunk_text(text, chunk_size=chunk_size, overlap=overlap, base_metadata=base_metadata)

    chunks: list[TextChunk] = []
    base = base_metadata or {}
    for block in question_blocks:
        metadata = {
            **base,
            **extract_question_metadata(block),
            "subject": base.get("subject") or infer_subject(block),
            "chapter": base.get("chapter") or infer_chapter(block),
            "content_type": "question",
        }
        if len(block.split()) <= chunk_size:
            chunks.append(TextChunk(text=block, metadata=metadata))
            continue
        chunks.extend(
            chunk_text(
                block,
                chunk_size=chunk_size,
                overlap=overlap,
                base_metadata=metadata,
            )
        )
    return chunks


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 180,
    base_metadata: dict | None = None,
) -> list[TextChunk]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    chunks: list[TextChunk] = []
    start = 0
    base = base_metadata or {}

    while start < len(words):
        end = min(start + chunk_size, len(words))
        body = " ".join(words[start:end])
        metadata = {
            **base,
            "subject": base.get("subject") or infer_subject(body),
            "chapter": base.get("chapter") or infer_chapter(body),
        }
        chunks.append(TextChunk(text=body, metadata=metadata))
        if end == len(words):
            break
        start = max(0, end - overlap)

    return chunks
