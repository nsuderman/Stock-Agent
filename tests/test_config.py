"""Tests for the Pydantic Settings module."""

from __future__ import annotations

import pytest

from agent.config import Settings, get_settings, reset_settings_cache


def test_defaults_load():
    # Bypass the repo's .env (which overrides several defaults) so we're actually
    # exercising the class-level defaults.
    s = Settings(_env_file=None, db_user="u", db_password="p")
    assert s.db_schema == "stock"
    assert s.backtest_schema == "stock"
    assert s.compact_at == 0.75
    assert s.max_response_tokens == 4096
    assert s.max_iterations == 12


def test_max_iterations_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_ITERATIONS", "20")
    reset_settings_cache()
    assert get_settings().max_iterations == 20


def test_max_iterations_range_enforced():
    with pytest.raises(ValueError):
        Settings(max_iterations=0)
    with pytest.raises(ValueError):
        Settings(max_iterations=101)


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


class TestConfigure:
    """Library-mode programmatic overrides. Explicit kwargs must beat env vars."""

    def test_overrides_beat_env(self, monkeypatch: pytest.MonkeyPatch):
        from agent.config import configure

        # Host app has a conflicting env (Quantara case: DB_SCHEMA=dev would
        # otherwise cascade into the agent).
        monkeypatch.setenv("DB_SCHEMA", "dev")
        reset_settings_cache()

        configure(db_schema="stock", backtest_schema="dev")

        s = get_settings()
        assert s.db_schema == "stock"
        assert s.backtest_schema == "dev"

    def test_unspecified_fields_still_read_env(self, monkeypatch: pytest.MonkeyPatch):
        from agent.config import configure

        monkeypatch.setenv("DB_USER", "alice")
        monkeypatch.setenv("DB_PASSWORD", "hunter2")
        monkeypatch.setenv("DB_HOST", "example.com")
        monkeypatch.setenv("DB_NAME", "mydb")
        reset_settings_cache()

        configure(db_schema="custom_market")

        s = get_settings()
        assert s.db_schema == "custom_market"
        # Unspecified: read from env.
        assert s.db_user == "alice"
        assert s.db_host == "example.com"

    def test_configure_replaces_cached_singleton(self):
        from agent.config import configure

        reset_settings_cache()
        first = get_settings()
        configure(db_schema="other")
        second = get_settings()
        assert first is not second
        assert second.db_schema == "other"
