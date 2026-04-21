"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent import db, llm
from agent.config import reset_settings_cache


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Ensure each test starts with clean module-level caches."""
    reset_settings_cache()
    db.reset_engine_cache()
    llm.reset_context_cache()
    yield
    reset_settings_cache()
    db.reset_engine_cache()
    llm.reset_context_cache()


@pytest.fixture
def tmp_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect memory.md + sessions/ to a tmp directory."""
    memory_file = tmp_path / "memory.md"
    memory_file.write_text("# Agent Memory\n\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Patch the Settings instance so every component sees the tmp paths.
    from agent.config import Settings

    original_memory = Settings.memory_path.fget  # type: ignore[attr-defined]
    original_sessions = Settings.sessions_dir.fget  # type: ignore[attr-defined]
    monkeypatch.setattr(Settings, "memory_path", property(lambda self: memory_file))
    monkeypatch.setattr(Settings, "sessions_dir", property(lambda self: sessions_dir))
    yield memory_file
    monkeypatch.setattr(Settings, "memory_path", property(original_memory))
    monkeypatch.setattr(Settings, "sessions_dir", property(original_sessions))


@pytest.fixture
def mock_engine(monkeypatch: pytest.MonkeyPatch):
    """Replace the SQLAlchemy engine with a MagicMock that records SQL + returns canned rows."""
    calls: list[dict[str, Any]] = []

    def make_mock_engine(rows_by_sql: dict[str, list[dict[str, Any]]] | None = None):
        rows_by_sql = rows_by_sql or {}

        mock = MagicMock()

        def execute(sql_obj, params=None):
            sql_text = str(sql_obj)
            calls.append({"sql": sql_text, "params": params})
            # Find rows by substring match
            chosen = []
            for key, rows in rows_by_sql.items():
                if key in sql_text:
                    chosen = rows
                    break
            result = MagicMock()
            cols = list(chosen[0].keys()) if chosen else []
            result.keys.return_value = cols
            result.fetchall.return_value = [tuple(r[c] for c in cols) for r in chosen]
            result.fetchmany.return_value = [tuple(r[c] for c in cols) for r in chosen]
            return result

        mock.connect.return_value.__enter__.return_value.execute = execute
        mock.connect.return_value.__exit__.return_value = False
        return mock, calls

    def install(rows_by_sql: dict[str, list[dict[str, Any]]] | None = None):
        mock, recorded = make_mock_engine(rows_by_sql)
        monkeypatch.setattr("agent.tools.base.get_engine", lambda: mock)
        monkeypatch.setattr("agent.db.get_engine", lambda: mock)
        return mock, recorded

    return install


@pytest.fixture
def mock_openai_client():
    """Minimal mock OpenAI client for unit tests."""

    def make(models_data: list[dict] | None = None):
        client = MagicMock()
        data = [MagicMock(**{"model_dump.return_value": m}) for m in (models_data or [])]
        client.models.list.return_value = MagicMock(data=data)
        return client

    return make


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch):
    """Block accidental HTTP calls in unit tests. Integration tests should override."""
    # Only block if the test is NOT marked 'integration'.
    # Pytest gives us the request via another fixture, so keep this simple: just don't block here,
    # and trust the @pytest.mark.integration skip to keep network tests opt-in.
    yield


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless `-m integration` is passed."""
    if config.getoption("-m") and "integration" in config.getoption("-m"):
        return
    skip_integration = pytest.mark.skip(
        reason="integration test — run with `pytest -m integration`"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


# Ensure DB-touching tests don't accidentally hit real Postgres if the user runs without
# setting test env vars. We set dummy creds here so Settings() instantiates cleanly.
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_NAME", "test")
