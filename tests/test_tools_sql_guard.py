"""Tests for the SQL write-keyword guard (defense-in-depth against LLM-generated writes)."""

from __future__ import annotations

import pytest

from agent.tools.base import contains_write_keyword


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO foo VALUES (1)",
        "insert into foo values (1)",
        "UPDATE foo SET x=1",
        "DELETE FROM foo",
        "DROP TABLE foo",
        "TRUNCATE foo",
        "ALTER TABLE foo ADD col",
        "CREATE TABLE foo (id int)",
        "GRANT SELECT ON foo TO bar",
        "REVOKE ALL ON foo FROM bar",
        "COPY foo FROM '/x'",
        "MERGE INTO foo USING bar",
        "SELECT * FROM foo; DROP TABLE bar;",
    ],
)
def test_write_keywords_detected(query: str):
    assert contains_write_keyword(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM foo",
        "select count(*) from bar",
        "SELECT updater FROM foo",  # 'updater' not 'UPDATE' as a word
        "WITH cte AS (SELECT 1) SELECT * FROM cte",
        "SELECT * FROM stock.analytics WHERE symbol = 'AAPL'",
    ],
)
def test_benign_queries_allowed(query: str):
    assert contains_write_keyword(query) is False


def test_run_sql_rejects_writes():
    from agent.tools import invoke_tool

    result = invoke_tool("run_sql", {"query": "DELETE FROM stock.analytics"})
    assert "error" in result
    assert "read-only" in result["error"].lower() or "select" in result["error"].lower()
