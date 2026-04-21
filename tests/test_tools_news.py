"""Tests for the Yahoo Finance news tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.tools import invoke_tool
from agent.tools.news import _extract


class TestExtract:
    """The normalizer handles both nested (>=0.2.40) and flat yfinance shapes."""

    def test_nested_content_shape(self):
        item = {
            "id": "abc",
            "content": {
                "title": "Apple beats earnings",
                "provider": {"displayName": "Reuters"},
                "pubDate": "2026-04-21T20:00:00Z",
                "clickThroughUrl": {"url": "https://example.com/news/1"},
                "summary": "Apple's Q2 revenue topped estimates...",
            },
        }
        out = _extract(item)
        assert out["title"] == "Apple beats earnings"
        assert out["publisher"] == "Reuters"
        assert out["published_at"] == "2026-04-21T20:00:00Z"
        assert out["link"] == "https://example.com/news/1"
        assert "revenue" in out["summary"]

    def test_flat_legacy_shape(self):
        item = {
            "title": "MSFT hits all-time high",
            "publisher": "Bloomberg",
            "providerPublishTime": 1704067200,
            "link": "https://example.com/msft",
            "summary": "Microsoft shares closed at a record...",
        }
        out = _extract(item)
        assert out["title"] == "MSFT hits all-time high"
        assert out["publisher"] == "Bloomberg"
        assert out["link"] == "https://example.com/msft"

    def test_missing_fields_tolerated(self):
        out = _extract({"content": {"title": "Only a title"}})
        assert out["title"] == "Only a title"
        assert out["publisher"] is None
        assert out["link"] is None


@pytest.fixture
def mock_yfinance(monkeypatch: pytest.MonkeyPatch):
    """Patch yfinance.Ticker.news with a controllable list."""

    articles: list[dict] = []
    fake_yf = MagicMock()

    def fake_ticker(symbol):
        t = MagicMock()
        t.news = articles
        return t

    fake_yf.Ticker = fake_ticker
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    return articles


class TestGetStockNews:
    def test_success(self, mock_yfinance):
        mock_yfinance.extend(
            [
                {"content": {"title": "A1", "provider": {"displayName": "Reuters"}}},
                {"content": {"title": "A2", "provider": {"displayName": "Bloomberg"}}},
                {"content": {"title": "A3", "provider": {"displayName": "WSJ"}}},
            ]
        )
        result = invoke_tool("get_stock_news", {"symbol": "aapl", "limit": 2})
        assert result["symbol"] == "AAPL"
        assert result["count"] == 2
        assert [a["title"] for a in result["articles"]] == ["A1", "A2"]

    def test_empty_news(self, mock_yfinance):
        result = invoke_tool("get_stock_news", {"symbol": "ZZZZZZ"})
        assert result["count"] == 0
        assert result["articles"] == []

    def test_yfinance_error_surfaces_as_tool_error(self, monkeypatch):
        fake_yf = MagicMock()
        fake_yf.Ticker.side_effect = RuntimeError("yahoo timeout")
        monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
        result = invoke_tool("get_stock_news", {"symbol": "AAPL"})
        assert "error" in result
        assert "yahoo timeout" in result["error"]

    def test_limit_validation_rejects_out_of_range(self):
        result = invoke_tool("get_stock_news", {"symbol": "AAPL", "limit": 999})
        assert "error" in result
        assert "Invalid arguments" in result["error"]

    def test_symbol_required(self):
        result = invoke_tool("get_stock_news", {})
        assert "error" in result
