"""Session persistence — daily by default, named override."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from agent.config import get_settings
from agent.logging_setup import get_logger

log = get_logger(__name__)


def session_path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return get_settings().sessions_dir / f"{safe}.json"


def default_session_name() -> str:
    return datetime.date.today().isoformat()


def _valid_message(m: object) -> bool:
    """A message must be a dict with a non-empty string role."""
    return isinstance(m, dict) and isinstance(m.get("role"), str) and bool(m["role"])


def load_session(name: str) -> list[dict]:
    """Load a session from disk. Returns [] if the file is missing, corrupt,
    or doesn't contain a list of well-formed message dicts."""
    path = session_path(name)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Session %s unreadable (%s); starting fresh.", path.name, e)
        return []
    if not isinstance(data, list):
        return []
    return [m for m in data if _valid_message(m)]


def save_session(name: str, messages: list[dict]) -> Path:
    """Atomically write `messages` to the session file. Survives mid-write crashes."""
    sessions_dir = get_settings().sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = session_path(name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(messages, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def reset_session(name: str) -> bool:
    """Delete the named session file. Returns True if a file was removed."""
    path = session_path(name)
    if path.exists():
        path.unlink()
        return True
    return False
