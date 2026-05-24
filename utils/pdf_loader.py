from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import fitz
import pdfplumber

from utils.text_processing import TextChunk, chunk_neet_text, clean_text, infer_chapter, infer_subject


PDF_DIR = Path("pdfs")


def save_uploaded_pdf(uploaded_file) -> Path:
    PDF_DIR.mkdir(exist_ok=True)
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    target = PDF_DIR / f"{uuid4().hex[:8]}_{safe_name}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    pages: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(
                    {
                        "text": text,
                        "page": page_index,
                        "source": pdf_path.name,
                    }
                )
    if pages:
        return pages

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if text:
                pages.append(
                    {
                        "text": text,
                        "page": page_index,
                        "source": pdf_path.name,
                    }
                )
    return pages


def pdf_to_chunks(pdf_path: Path, subject: str | None = None, chapter: str | None = None) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page in extract_pdf_pages(pdf_path):
        page_subject = subject if subject and subject != "Auto detect" else infer_subject(page["text"])
        page_chapter = chapter.strip() if chapter else infer_chapter(page["text"])
        chunks.extend(
            chunk_neet_text(
                page["text"],
                base_metadata={
                    "source": page["source"],
                    "page": page["page"],
                    "subject": page_subject,
                    "chapter": page_chapter,
                },
            )
        )
    return chunks
