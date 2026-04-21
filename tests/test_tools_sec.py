"""Tests for the SEC EDGAR tools (recent filings, insider transactions).

These tests mock the `edgar` module so we don't hit SEC in unit tests.
"""

from __future__ import annotations

import datetime
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from agent.tools import invoke_tool


def _fake_filing(form: str, date: str, accession: str) -> MagicMock:
    f = MagicMock()
    f.form = form
    f.filing_date = datetime.date.fromisoformat(date)
    f.accession_no = accession
    f.filing_url = f"https://www.sec.gov/Archives/edgar/data/x/{accession}/doc.htm"
    f.homepage_url = f"https://www.sec.gov/Archives/edgar/data/x/{accession}-index.html"
    return f


def _fake_filings_list(filings: list):
    """Mimic EntityFilings: supports len() and [:n] slicing."""
    mock = MagicMock()
    mock.__getitem__.side_effect = lambda key: filings[key]
    mock.__len__.return_value = len(filings)
    return mock


@pytest.fixture
def mock_edgar(monkeypatch: pytest.MonkeyPatch):
    """Install a fake `edgar` module and return a control dict."""

    fake_edgar = MagicMock()
    # Avoid the real network call when _configure_edgar runs.
    fake_edgar.set_identity = MagicMock()

    company_mock = MagicMock()
    company_mock.name = "Test Co"
    company_mock.cik = 12345
    fake_edgar.Company = MagicMock(return_value=company_mock)

    monkeypatch.setitem(sys.modules, "edgar", fake_edgar)

    # Reset the module-level cached config flag so _configure_edgar re-runs.
    from agent.tools import sec as sec_mod

    monkeypatch.setattr(sec_mod, "_edgar_configured", False)
    return {"edgar": fake_edgar, "company": company_mock}


class TestGetRecentFilings:
    def test_success(self, mock_edgar):
        mock_edgar["company"].get_filings.return_value = _fake_filings_list(
            [
                _fake_filing("10-K", "2025-10-31", "0000320193-25-000079"),
                _fake_filing("10-K", "2024-11-01", "0000320193-24-000123"),
                _fake_filing("10-K", "2023-11-03", "0000320193-23-000106"),
            ]
        )
        result = invoke_tool(
            "get_recent_filings",
            {"symbol": "aapl", "form_type": "10-K", "limit": 2},
        )
        assert result["symbol"] == "AAPL"
        assert result["company"] == "Test Co"
        assert result["count"] == 2
        assert result["filings"][0]["form"] == "10-K"
        assert result["filings"][0]["filing_date"] == "2025-10-31"
        assert "sec.gov" in result["filings"][0]["filing_url"]

    def test_form_type_passed_through(self, mock_edgar):
        mock_edgar["company"].get_filings.return_value = _fake_filings_list([])
        invoke_tool("get_recent_filings", {"symbol": "MSFT", "form_type": "8-K"})
        # Verify form= kwarg was used
        call_kwargs = mock_edgar["company"].get_filings.call_args.kwargs
        assert call_kwargs["form"] == "8-K"

    def test_no_form_type_omits_filter(self, mock_edgar):
        mock_edgar["company"].get_filings.return_value = _fake_filings_list([])
        invoke_tool("get_recent_filings", {"symbol": "MSFT"})
        # Called without form= (or with form=None if get_filings was invoked without args)
        call_args = mock_edgar["company"].get_filings.call_args
        assert not call_args.kwargs or call_args.kwargs.get("form") is None

    def test_unknown_ticker_returns_error(self, mock_edgar):
        mock_edgar["edgar"].Company.side_effect = Exception("ticker not found")
        result = invoke_tool("get_recent_filings", {"symbol": "ZZZZZ", "form_type": "10-K"})
        assert "error" in result
        assert "ZZZZZ" in result["error"]

    def test_limit_validation(self):
        result = invoke_tool("get_recent_filings", {"symbol": "AAPL", "limit": 9999})
        assert "error" in result
        assert "Invalid arguments" in result["error"]


def _fake_form4(insider: str, rows: list[dict]) -> SimpleNamespace:
    nd = SimpleNamespace(
        has_transactions=bool(rows),
        transactions=SimpleNamespace(data=pd.DataFrame(rows)) if rows else None,
    )
    return SimpleNamespace(insider_name=insider, non_derivative_table=nd)


class TestGetInsiderTransactions:
    def test_success_with_trades(self, mock_edgar):
        f1 = _fake_filing("4", "2026-04-17", "0001140361-26-015421")
        f1.obj = lambda: _fake_form4(
            "Jane Doe",
            [
                {
                    "Security": "Common Stock",
                    "Date": datetime.date(2026, 4, 15),
                    "Shares": 1000,
                    "Remaining": 5000,
                    "Price": 180.5,
                    "AcquiredDisposed": "A",
                    "DirectIndirect": "D",
                    "NatureOfOwnership": None,
                    "form": "4",
                    "Code": "P",
                    "EquitySwap": False,
                    "footnotes": None,
                    "TransactionType": "Purchase",
                }
            ],
        )
        f2 = _fake_filing("4", "2026-04-16", "0001140361-26-015420")
        f2.obj = lambda: _fake_form4(
            "John Smith",
            [
                {
                    "Security": "Common Stock",
                    "Date": datetime.date(2026, 4, 14),
                    "Shares": 500,
                    "Remaining": 3000,
                    "Price": 179.0,
                    "AcquiredDisposed": "D",
                    "DirectIndirect": "D",
                    "NatureOfOwnership": None,
                    "form": "4",
                    "Code": "S",
                    "EquitySwap": False,
                    "footnotes": None,
                    "TransactionType": "Sale",
                }
            ],
        )
        mock_edgar["company"].get_filings.return_value = _fake_filings_list([f1, f2])

        result = invoke_tool("get_insider_transactions", {"symbol": "AAPL", "limit": 5})
        assert result["symbol"] == "AAPL"
        assert result["filings_examined"] == 2
        assert result["trade_count"] == 2
        jd = next(t for t in result["trades"] if t["insider"] == "Jane Doe")
        assert jd["shares"] == 1000
        assert jd["price"] == 180.5
        assert jd["acquired_disposed"] == "A"
        assert jd["transaction_type"] == "Purchase"

    def test_empty_transactions_skipped(self, mock_edgar):
        f = _fake_filing("4", "2026-04-10", "acc")
        f.obj = lambda: _fake_form4("Empty Insider", [])  # no rows
        mock_edgar["company"].get_filings.return_value = _fake_filings_list([f])
        result = invoke_tool("get_insider_transactions", {"symbol": "AAPL"})
        assert result["filings_examined"] == 1
        assert result["trade_count"] == 0

    def test_parse_error_per_filing_skipped(self, mock_edgar):
        bad = _fake_filing("4", "2026-04-10", "bad")

        def _raise():
            raise ValueError("unparseable XBRL")

        bad.obj = _raise
        good = _fake_filing("4", "2026-04-09", "good")
        good.obj = lambda: _fake_form4(
            "OK Insider",
            [
                {
                    "Security": "Common Stock",
                    "Date": datetime.date(2026, 4, 9),
                    "Shares": 100,
                    "Remaining": 200,
                    "Price": 50.0,
                    "AcquiredDisposed": "D",
                    "DirectIndirect": "D",
                    "NatureOfOwnership": None,
                    "form": "4",
                    "Code": "S",
                    "EquitySwap": False,
                    "footnotes": None,
                    "TransactionType": "Sale",
                }
            ],
        )
        mock_edgar["company"].get_filings.return_value = _fake_filings_list([bad, good])
        result = invoke_tool("get_insider_transactions", {"symbol": "AAPL"})
        assert result["filings_examined"] == 2
        assert result["trade_count"] == 1
        assert result["trades"][0]["insider"] == "OK Insider"

    def test_symbol_required(self):
        result = invoke_tool("get_insider_transactions", {})
        assert "error" in result

    def test_limit_validation(self):
        result = invoke_tool("get_insider_transactions", {"symbol": "AAPL", "limit": 999})
        assert "error" in result
        assert "Invalid arguments" in result["error"]
