"""Session persistence — daily by default, named override."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from agent.config import get_settings


def session_path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return get_settings().sessions_dir / f"{safe}.json"


def default_session_name() -> str:
    return datetime.date.today().isoformat()


def load_session(name: str) -> list[dict]:
    path = session_path(name)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def save_session(name: str, messages: list[dict]) -> Path:
    sessions_dir = get_settings().sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = session_path(name)
    path.write_text(
        json.dumps(messages, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def reset_session(name: str) -> bool:
    """Delete the named session file. Returns True if a file was removed."""
    path = session_path(name)
    if path.exists():
        path.unlink()
        return True
    return False
