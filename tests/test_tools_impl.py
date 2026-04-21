"""Tool implementation tests — mock `fetch` + `inspect` rather than the DB engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.tools import invoke_tool


@pytest.fixture
def mock_fetch(monkeypatch: pytest.MonkeyPatch):
    """Patch agent.tools.base.fetch to return canned row lists."""
    calls: list[dict] = []
    canned: dict[str, list[dict]] = {}

    def fake_fetch(sql, params=None, limit=None):
        calls.append({"sql": sql, "params": params, "limit": limit})
        for match, rows in canned.items():
            if match in sql:
                return rows
        return []

    monkeypatch.setattr("agent.tools.base.fetch", fake_fetch)
    # Tool modules imported `fetch` at import time — patch those too.
    for mod in (
        "agent.tools.market",
        "agent.tools.backtest",
        "agent.tools.db_meta",
        "agent.tools.sql",
    ):
        monkeypatch.setattr(f"{mod}.fetch", fake_fetch, raising=False)

    return {"calls": calls, "canned": canned}


@pytest.fixture
def mock_inspect(monkeypatch: pytest.MonkeyPatch):
    """Patch sqlalchemy.inspect so tools can read column metadata without a DB."""

    def fake_inspect(engine):
        inspector = MagicMock()
        inspector.get_columns.return_value = [
            {"name": "symbol", "type": "VARCHAR"},
            {"name": "date", "type": "DATE"},
            {"name": "open", "type": "FLOAT"},
            {"name": "high", "type": "FLOAT"},
            {"name": "low", "type": "FLOAT"},
            {"name": "close", "type": "FLOAT"},
            {"name": "volume", "type": "BIGINT"},
            {"name": "rsi", "type": "FLOAT"},
            {"name": "sma_50", "type": "FLOAT"},
            {"name": "sma_200", "type": "FLOAT"},
            {"name": "atr", "type": "FLOAT"},
            {"name": "returns_day", "type": "FLOAT"},
        ]
        return inspector

    for mod in ("agent.tools.db_meta", "agent.tools.market"):
        monkeypatch.setattr(f"{mod}.inspect", fake_inspect, raising=False)
    return fake_inspect


class TestDbMetaTools:
    def test_list_analytics_columns(self, mock_inspect):
        result = invoke_tool("list_analytics_columns", {})
        assert result["table"] == "analytics"
        assert len(result["columns"]) == 12
        names = {c["name"] for c in result["columns"]}
        assert "rsi" in names

    def test_describe_table_success(self, mock_inspect):
        result = invoke_tool("describe_table", {"schema": "stock", "table": "analytics"})
        assert result["table"] == "analytics"
        assert len(result["columns"]) > 0

    def test_describe_table_empty_returns_error(self, monkeypatch):
        def fake_inspect(engine):
            i = MagicMock()
            i.get_columns.return_value = []
            return i

        monkeypatch.setattr("agent.tools.db_meta.inspect", fake_inspect)
        result = invoke_tool("describe_table", {"schema": "stock", "table": "ghost"})
        assert "error" in result
        assert "ghost" in result["error"]

    def test_sample_rows_blocks_write_keyword(self):
        result = invoke_tool("sample_rows", {"schema": "stock", "table": "DELETE", "limit": 1})
        assert "error" in result

    def test_sample_rows_returns_rows(self, mock_fetch):
        mock_fetch["canned"]["LIMIT"] = [{"a": 1, "b": 2}]
        result = invoke_tool("sample_rows", {"schema": "stock", "table": "foo", "limit": 1})
        assert result["count"] == 1
        assert result["rows"][0] == {"a": 1, "b": 2}


class TestMarketTools:
    def test_get_price_history_filters_invalid_columns(self, mock_fetch, mock_inspect):
        mock_fetch["canned"]["analytics"] = [{"date": "2024-01-01", "close": 100}]
        result = invoke_tool(
            "get_price_history",
            {
                "symbol": "aapl",
                "start": "2024-01-01",
                "end": "2024-12-31",
                "columns": ["close", "nonexistent_col", "open"],
            },
        )
        assert result["symbol"] == "AAPL"  # uppercased
        sql = mock_fetch["calls"][0]["sql"]
        assert "nonexistent_col" not in sql
        assert "close" in sql

    def test_get_price_history_default_columns(self, mock_fetch, mock_inspect):
        mock_fetch["canned"]["analytics"] = []
        invoke_tool(
            "get_price_history",
            {"symbol": "MSFT", "start": "2024-01-01", "end": "2024-06-30"},
        )
        sql = mock_fetch["calls"][0]["sql"]
        assert "close" in sql
        assert "rsi" in sql

    def test_get_fundamentals_found(self, mock_fetch):
        mock_fetch["canned"]["symbols_info"] = [{"symbol": "AAPL", "market_cap": 3e12}]
        result = invoke_tool("get_fundamentals", {"symbol": "aapl"})
        assert result["symbol"] == "AAPL"
        assert result["market_cap"] == 3e12

    def test_get_fundamentals_not_found(self, mock_fetch):
        mock_fetch["canned"]["symbols_info"] = []
        result = invoke_tool("get_fundamentals", {"symbol": "ZZZZ"})
        assert "error" in result

    def test_get_market_regime_single_day(self, mock_fetch):
        mock_fetch["canned"]["market_exposure"] = [
            {"date": "2024-06-01", "exposure_tier": "Long 75%"}
        ]
        result = invoke_tool("get_market_regime", {"start_date": "2024-06-01"})
        assert result["count"] == 1
        assert "date = :d" in mock_fetch["calls"][0]["sql"]

    def test_get_market_regime_range(self, mock_fetch):
        mock_fetch["canned"]["market_exposure"] = [{}, {}, {}]
        result = invoke_tool(
            "get_market_regime",
            {"start_date": "2024-01-01", "end_date": "2024-01-03"},
        )
        assert result["count"] == 3
        assert "BETWEEN" in mock_fetch["calls"][0]["sql"]

    def test_get_breakouts_invokes_function(self, mock_fetch):
        mock_fetch["canned"]["get_live_breakouts"] = [
            {"symbol": "EVO", "match_count": 42, "win_rate": 0.62}
        ]
        result = invoke_tool("get_breakouts", {"target_date": "2026-04-21"})
        assert result["count"] == 1
        assert "CAST(:d AS date)" in mock_fetch["calls"][0]["sql"]

    def test_screen_symbols_blocks_write(self):
        result = invoke_tool(
            "screen_symbols",
            {"where_clause": "DROP TABLE foo", "order_by": "a.symbol"},
        )
        assert "error" in result


class TestBacktestTools:
    def test_list_backtests_no_filter(self, mock_fetch):
        mock_fetch["canned"]["backtest_results"] = [
            {"id": 1, "strategy_name": "s1"},
            {"id": 2, "strategy_name": "s2"},
        ]
        result = invoke_tool("list_backtests", {"limit": 10})
        assert result["count"] == 2
        sql = mock_fetch["calls"][0]["sql"]
        assert "ILIKE" not in sql

    def test_list_backtests_with_filter(self, mock_fetch):
        mock_fetch["canned"]["backtest_results"] = [{"id": 1, "strategy_name": "DTW x"}]
        result = invoke_tool("list_backtests", {"strategy_name": "DTW"})
        assert result["count"] == 1
        assert "ILIKE" in mock_fetch["calls"][0]["sql"]
        assert mock_fetch["calls"][0]["params"]["name"] == "%DTW%"

    def test_get_backtest_detail_missing(self, mock_fetch):
        mock_fetch["canned"]["backtest_results"] = []
        result = invoke_tool("get_backtest_detail", {"backtest_id": 999})
        assert "error" in result

    def test_get_backtest_detail_downsamples_equity_curve(self, mock_fetch):
        curve = [{"date": f"2024-{i:03d}", "value": 100 + i} for i in range(500)]
        mock_fetch["canned"]["backtest_results"] = [{"id": 1, "equity_curve": curve, "trades": []}]
        result = invoke_tool(
            "get_backtest_detail",
            {"backtest_id": 1, "include": ["equity_curve"]},
        )
        assert len(result["equity_curve"]) <= 250

    def test_get_backtest_detail_caps_trades(self, mock_fetch):
        trades = [{"symbol": "AAPL", "type": "BUY"} for _ in range(250)]
        mock_fetch["canned"]["backtest_results"] = [
            {"id": 1, "equity_curve": None, "trades": trades}
        ]
        result = invoke_tool("get_backtest_detail", {"backtest_id": 1, "include": ["trades"]})
        assert len(result["trades"]) == 100
        assert result["trades_truncated"] is True
        assert result["total_trades"] == 250

    def test_get_recent_backtest_holdings(self, mock_fetch):
        # Match-order matters: the main query also contains "COUNT(*)" inside a FILTER
        # clause, so we need the more-specific "position_balance" match before it.
        mock_fetch["canned"]["FROM position_balance"] = [
            {"symbol": "EGBN", "n_backtests": 28, "n_strategies": 9, "strategies": ["a"]}
        ]
        mock_fetch["canned"]["SELECT COUNT(*) AS n"] = [{"n": 43}]
        result = invoke_tool("get_recent_backtest_holdings", {"days_back": 7, "min_backtests": 1})
        assert result["total_backtests_in_window"] == 43
        assert result["count"] == 1
        assert result["rows"][0]["symbol"] == "EGBN"

    def test_list_strategies(self, mock_fetch):
        mock_fetch["canned"]["FROM "] = [{"id": 1, "name": "X"}]
        result = invoke_tool("list_strategies", {})
        assert result["count"] == 1


class TestRunSql:
    def test_run_sql_success(self, mock_fetch):
        mock_fetch["canned"]["SELECT"] = [{"a": 1}]
        result = invoke_tool("run_sql", {"query": "SELECT 1 AS a", "limit": 10})
        assert result["count"] == 1
        assert result["truncated"] is False

    def test_run_sql_truncation_flag(self, mock_fetch):
        mock_fetch["canned"]["SELECT"] = [{"a": i} for i in range(10)]
        result = invoke_tool("run_sql", {"query": "SELECT x", "limit": 10})
        assert result["truncated"] is True
