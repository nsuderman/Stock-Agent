"""Tests for the `remember` tool."""

from __future__ import annotations

import datetime
from pathlib import Path

from agent.tools import invoke_tool


def test_remember_appends_to_memory_file(tmp_memory: Path):
    result = invoke_tool("remember", {"fact": "Test fact alpha"})
    assert result["saved"] == "Test fact alpha"
    assert result["date"] == datetime.date.today().isoformat()
    assert "Test fact alpha" in tmp_memory.read_text()


def test_remember_multiple_facts(tmp_memory: Path):
    invoke_tool("remember", {"fact": "first"})
    invoke_tool("remember", {"fact": "second"})
    content = tmp_memory.read_text()
    assert "first" in content
    assert "second" in content
    assert content.count("[2026-") >= 2 or content.count(str(datetime.date.today().year)) >= 2


def test_remember_empty_fact_rejected(tmp_memory: Path):
    result = invoke_tool("remember", {"fact": ""})
    assert "error" in result


def test_remember_creates_memory_file_if_missing(tmp_memory: Path):
    tmp_memory.unlink()
    assert not tmp_memory.exists()
    invoke_tool("remember", {"fact": "bootstrap"})
    assert tmp_memory.exists()
    assert "# Agent Memory" in tmp_memory.read_text()
    assert "bootstrap" in tmp_memory.read_text()


def test_remember_preserves_existing_content(tmp_memory: Path):
    tmp_memory.write_text("# Agent Memory\n\n- [2026-01-01] old fact\n", encoding="utf-8")
    invoke_tool("remember", {"fact": "new fact"})
    content = tmp_memory.read_text()
    assert "old fact" in content
    assert "new fact" in content
