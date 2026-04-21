"""Tests for the Pydantic Settings module."""

from __future__ import annotations

import pytest

from agent.config import Settings, get_settings, reset_settings_cache


def test_defaults_load(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_USER", "u")
    monkeypatch.setenv("DB_PASSWORD", "p")
    reset_settings_cache()
    s = get_settings()
    assert s.db_schema == "stock"
    assert s.backtest_schema == "stock"
    assert s.compact_at == 0.75
    assert s.max_response_tokens == 4096


def test_database_url_composed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_USER", "alice")
    monkeypatch.setenv("DB_PASSWORD", "hunter2")
    monkeypatch.setenv("DB_HOST", "example.com")
    monkeypatch.setenv("DB_NAME", "mydb")
    reset_settings_cache()
    s = get_settings()
    assert s.database_url == "postgresql://alice:hunter2@example.com/mydb"


def test_compact_at_range_enforced():
    with pytest.raises(ValueError):
        Settings(compact_at=1.5)
    with pytest.raises(ValueError):
        Settings(compact_at=0.0)


def test_context_window_lower_bound():
    with pytest.raises(ValueError):
        Settings(local_context_window=100)


def test_singleton_cached():
    reset_settings_cache()
    a = get_settings()
    b = get_settings()
    assert a is b


def test_reset_clears_cache():
    a = get_settings()
    reset_settings_cache()
    b = get_settings()
    assert a is not b
