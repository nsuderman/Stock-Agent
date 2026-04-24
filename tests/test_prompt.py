"""Tests for system prompt assembly."""

from __future__ import annotations

import datetime
from pathlib import Path

from agent.prompt import build_system_prompt


def test_introduces_as_finn(tmp_memory: Path):
    prompt = build_system_prompt()
    assert "Finn" in prompt


def test_injects_today_date(tmp_memory: Path):
    prompt = build_system_prompt()
    today = datetime.date.today().isoformat()
    assert today in prompt


def test_injects_memory_contents(tmp_memory: Path):
    tmp_memory.write_text(
        "# Agent Memory\n\n- [2026-01-01] I only trade long positions.\n",
        encoding="utf-8",
    )
    prompt = build_system_prompt()
    assert "I only trade long positions" in prompt


def test_empty_memory_rendered(tmp_memory: Path):
    tmp_memory.write_text("# Agent Memory\n\n", encoding="utf-8")
    prompt = build_system_prompt()
    # Memory section still present (even if empty); contract is intact.
    assert "Saved Memory" in prompt


def test_references_stock_schema_not_dev(tmp_memory: Path):
    prompt = build_system_prompt()
    assert "stock.backtest_results" in prompt
    assert "dev.backtest_results" not in prompt


def test_warns_about_postgres_round(tmp_memory: Path):
    prompt = build_system_prompt()
    assert "ROUND" in prompt
    assert "numeric" in prompt


def test_mentions_json_not_jsonb(tmp_memory: Path):
    prompt = build_system_prompt()
    assert "json_array_elements" in prompt


class TestSchemaTemplating:
    def test_default_schemas_render_as_stock(self, tmp_memory: Path):
        """With defaults (stock / stock), rendered prompt looks identical to the legacy text."""
        prompt = build_system_prompt()
        assert "stock.analytics" in prompt
        assert "stock.backtest_results" in prompt
        assert "{db_schema}" not in prompt
        assert "{backtest_schema}" not in prompt

    def test_configure_with_split_schemas(self, tmp_memory: Path):
        """When configure() pins different schemas (Quantara case), the prompt reflects them."""
        from agent.config import configure

        configure(db_schema="stock", backtest_schema="dev")
        try:
            prompt = build_system_prompt()
            # Market tables still reference stock.
            assert "stock.analytics" in prompt
            assert "stock.symbols_info" in prompt
            # Backtest tables now reference dev.
            assert "dev.backtest_results" in prompt
            assert "dev.strategies" in prompt
            # The opposite combinations must NOT appear.
            assert "stock.backtest_results" not in prompt
            assert "dev.analytics" not in prompt
            # And no unrendered placeholders leak.
            assert "{db_schema}" not in prompt
            assert "{backtest_schema}" not in prompt
        finally:
            # Reset for other tests — tmp_memory fixture isolates but belt-and-braces.
            from agent.config import reset_settings_cache

            reset_settings_cache()

    def test_render_schemas_is_called_for_memory_path_override_too(self, tmp_memory: Path):
        """The `memory_path=...` back-compat branch still renders schemas."""
        from agent.config import configure

        configure(db_schema="stock", backtest_schema="dev")
        try:
            prompt = build_system_prompt(memory_path=tmp_memory)
            assert "dev.backtest_results" in prompt
        finally:
            from agent.config import reset_settings_cache

            reset_settings_cache()
