from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from utils.text_processing import TextChunk


CHROMA_DIR = Path("embeddings/chroma")
LOCAL_EMBEDDING_CACHE_DIR = Path("embeddings/chroma_onnx_cache")
COLLECTION_NAME = "neet_previous_year_questions"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _embed_texts(texts: list[str]) -> list[list[float]]:
    from google.genai import types

    from chatbot.gemini_client import get_gemini_client

    client = get_gemini_client()
    prepared = [f"task: semantic retrieval | query: {text}" for text in texts]
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=prepared,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )
    return [embedding.values for embedding in result.embeddings]


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return _embed_texts(list(input))


def embedding_provider() -> str:
    import os

    return os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()


def get_embedding_function() -> EmbeddingFunction:
    provider = embedding_provider()
    if provider == "gemini":
        return GeminiEmbeddingFunction()

    try:
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise RuntimeError(
            "Local embedding mode needs ChromaDB embedding functions."
        ) from exc

    if provider == "sentence_transformers":
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=SENTENCE_TRANSFORMER_MODEL
        )

    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    LOCAL_EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = LOCAL_EMBEDDING_CACHE_DIR.resolve()
    return embedding_functions.DefaultEmbeddingFunction()


def embed_query(text: str) -> list[float]:
    return _embed_texts([text])[0]


@dataclass
class SearchResult:
    text: str
    metadata: dict
    distance: float | None = None


@lru_cache(maxsize=1)
def get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    provider = embedding_provider()
    collection_name = f"{COLLECTION_NAME}_{provider}"
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[TextChunk]) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    ids = [uuid4().hex for _ in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def build_where(subject: str | None = None, chapter: str | None = None) -> dict | None:
    filters = []
    if subject and subject != "All":
        filters.append({"subject": subject})
    if chapter and chapter != "All":
        filters.append({"chapter": chapter})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def search(query: str, *, subject: str = "All", chapter: str = "All", k: int = 6) -> list[SearchResult]:
    collection = get_collection()
    if not query.strip() or collection.count() == 0:
        return []

    query_kwargs = {
        "n_results": k,
        "where": build_where(subject, chapter),
        "include": ["documents", "metadatas", "distances"],
    }
    if embedding_provider() == "gemini":
        query_kwargs["query_embeddings"] = [embed_query(query)]
    else:
        query_kwargs["query_texts"] = [query]

    results = collection.query(**query_kwargs)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return [
        SearchResult(text=document, metadata=metadata or {}, distance=distance)
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def collection_stats() -> dict:
    collection = get_collection()
    count = collection.count()
    chapters: set[str] = set()
    subjects: set[str] = set()

    if count:
        data = collection.get(include=["metadatas"], limit=min(count, 5000))
        for metadata in data.get("metadatas", []):
            if metadata.get("chapter"):
                chapters.add(metadata["chapter"])
            if metadata.get("subject"):
                subjects.add(metadata["subject"])

    return {
        "chunks": count,
        "chapters": sorted(chapters),
        "subjects": sorted(subjects),
    }
