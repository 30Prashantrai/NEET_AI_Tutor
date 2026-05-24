from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


DATA_DIR = Path("data")
BOOKMARKS_FILE = DATA_DIR / "bookmarks.json"
FAVORITES_FILE = DATA_DIR / "favorite_chats.json"
PERFORMANCE_FILE = DATA_DIR / "performance.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, data) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def add_bookmark(question: str, answer: str, metadata: dict | None = None) -> None:
    bookmarks = _read_json(BOOKMARKS_FILE, [])
    bookmarks.append(
        {
            "question": question,
            "answer": answer,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }
    )
    _write_json(BOOKMARKS_FILE, bookmarks)


def add_favorite_chat(messages: list[dict]) -> None:
    favorites = _read_json(FAVORITES_FILE, [])
    favorites.append(
        {
            "messages": messages,
            "created_at": datetime.utcnow().isoformat(),
        }
    )
    _write_json(FAVORITES_FILE, favorites)


def list_bookmarks() -> list[dict]:
    return _read_json(BOOKMARKS_FILE, [])


def list_favorites() -> list[dict]:
    return _read_json(FAVORITES_FILE, [])


def record_quiz_result(score: int, total: int) -> None:
    history = _read_json(PERFORMANCE_FILE, [])
    history.append(
        {
            "score": score,
            "total": total,
            "percent": round((score / total) * 100, 2) if total else 0,
            "created_at": datetime.utcnow().isoformat(),
        }
    )
    _write_json(PERFORMANCE_FILE, history)


def performance_history() -> list[dict]:
    return _read_json(PERFORMANCE_FILE, [])
