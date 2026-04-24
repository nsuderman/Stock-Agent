"""Tests for session persistence."""

from __future__ import annotations

import json
from pathlib import Path

from agent.session import (
    default_session_name,
    load_session,
    reset_session,
    save_session,
    session_path,
)


def test_default_session_name_is_today(tmp_memory: Path):
    import datetime

    assert default_session_name() == datetime.date.today().isoformat()


def test_session_path_sanitizes_name(tmp_memory: Path):
    path = session_path("weird name / with slashes!")
    assert "/" not in path.name
    assert " " not in path.name
    # Allowed chars: alnum, -, _
    assert path.suffix == ".json"


def test_load_session_missing_returns_empty(tmp_memory: Path):
    assert load_session("nonexistent") == []


def test_save_load_roundtrip(tmp_memory: Path):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    path = save_session("test1", messages)
    assert path.exists()
    loaded = load_session("test1")
    assert loaded == messages


def test_save_creates_sessions_dir(tmp_memory: Path, tmp_path: Path):
    sessions_dir = tmp_memory.parent / "sessions"
    import shutil

    shutil.rmtree(sessions_dir, ignore_errors=True)
    save_session("auto", [{"role": "user", "content": "x"}])
    assert sessions_dir.exists()


def test_reset_session_removes_file(tmp_memory: Path):
    save_session("zap", [{"role": "user", "content": "x"}])
    assert load_session("zap") != []
    removed = reset_session("zap")
    assert removed is True
    assert load_session("zap") == []


def test_reset_missing_session_returns_false(tmp_memory: Path):
    assert reset_session("never-existed") is False


def test_load_ignores_non_list_json(tmp_memory: Path):
    """If the file is valid JSON but not a list, we treat it as empty (forward compat)."""
    path = session_path("broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"messages": []}), encoding="utf-8")
    assert load_session("broken") == []


def test_save_handles_unicode(tmp_memory: Path):
    messages = [{"role": "user", "content": "héllo — ☕"}]
    save_session("unicode", messages)
    loaded = load_session("unicode")
    assert loaded == messages


def test_load_ignores_corrupt_json(tmp_memory: Path):
    """A truncated/corrupt file should return []; the caller shouldn't see a crash."""
    path = session_path("corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[{"role": "user", "content": "hi"', encoding="utf-8")
    assert load_session("corrupt") == []


def test_load_drops_malformed_messages(tmp_memory: Path):
    """Non-dict entries and dicts missing a role are filtered out, not crashy."""
    path = session_path("mixed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                None,
                {"role": "user", "content": "ok"},
                {"content": "no role"},
                {"role": "", "content": "empty role"},
                "string entry",
                {"role": "assistant", "content": "also ok"},
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_session("mixed")
    assert loaded == [
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "also ok"},
    ]


def test_save_is_atomic(tmp_memory: Path, monkeypatch):
    """If the write fails mid-way, the original file must remain intact."""
    save_session("atomic", [{"role": "user", "content": "original"}])
    original = load_session("atomic")

    # Force os.replace to fail, simulating a crash between tmp write and rename.
    import agent.session as sess

    def _boom(*args, **kwargs):
        raise OSError("simulated crash")

    import contextlib

    monkeypatch.setattr(sess.os, "replace", _boom)
    with contextlib.suppress(OSError):
        save_session("atomic", [{"role": "user", "content": "new"}])

    # Original file is untouched.
    assert load_session("atomic") == original
    # No stray .tmp left for the caller? (It's acceptable to leave one since a
    # future successful write will overwrite it; assert the main path is intact.)
    assert session_path("atomic").exists()
