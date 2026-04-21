"""Tests for system prompt assembly."""

from __future__ import annotations

import datetime
from pathlib import Path

from agent.prompt import build_system_prompt


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
