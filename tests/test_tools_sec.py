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


# ---------------------------------------------------------------------------
# 13F tools
# ---------------------------------------------------------------------------


def _fake_13f_filing(
    company: str,
    cik: int,
    accession: str,
    filing_date: str,
    report_date: str,
) -> MagicMock:
    f = MagicMock()
    f.company = company
    f.cik = cik
    f.accession_no = accession
    f.filing_date = datetime.date.fromisoformat(filing_date)
    f.report_date = datetime.date.fromisoformat(report_date)
    return f


def _fake_thirteenf(
    manager: str,
    total_value: int,
    total_holdings: int,
    report_period: str,
    filing_date: str,
    accession: str,
    holdings_rows: list[dict],
) -> SimpleNamespace:
    df = pd.DataFrame(holdings_rows)
    return SimpleNamespace(
        management_company_name=manager,
        report_period=report_period,
        filing_date=filing_date,
        accession_number=accession,
        total_value=total_value,
        total_holdings=total_holdings,
        holdings=df,
        compare_holdings=lambda display_limit=200: None,  # overridden per test
    )


@pytest.fixture
def mock_edgar_13f(monkeypatch: pytest.MonkeyPatch):
    """Like mock_edgar, but also stubs `edgar.get_filings` and `find_company`."""
    fake_edgar = MagicMock()
    fake_edgar.set_identity = MagicMock()

    company_mock = MagicMock()
    company_mock.name = "Test Manager LLC"
    company_mock.cik = 1067983
    fake_edgar.Company = MagicMock(return_value=company_mock)
    fake_edgar.get_filings = MagicMock()

    fake_search_mod = MagicMock()
    fake_search_mod.find_company = MagicMock()

    monkeypatch.setitem(sys.modules, "edgar", fake_edgar)
    monkeypatch.setitem(sys.modules, "edgar.entity.search", fake_search_mod)

    from agent.tools import sec as sec_mod

    monkeypatch.setattr(sec_mod, "_edgar_configured", False)
    return {"edgar": fake_edgar, "company": company_mock, "search": fake_search_mod}


class TestGet13FFilings:
    def test_defaults_to_current_quarter(self, mock_edgar_13f):
        mock_edgar_13f["edgar"].get_filings.return_value = _fake_filings_list(
            [
                _fake_13f_filing(
                    "BERKSHIRE HATHAWAY INC",
                    1067983,
                    "0001067983-26-000010",
                    "2026-02-14",
                    "2025-12-31",
                ),
                _fake_13f_filing(
                    "SCION ASSET MANAGEMENT, LLC",
                    1649339,
                    "0001649339-26-000002",
                    "2026-02-13",
                    "2025-12-31",
                ),
            ]
        )
        result = invoke_tool("get_13f_filings", {"limit": 5})
        assert result["count"] == 2
        assert result["year"] is not None and result["quarter"] in {1, 2, 3, 4}
        assert result["filings"][0]["manager"] == "BERKSHIRE HATHAWAY INC"
        assert result["filings"][0]["report_period"] == "2025-12-31"
        # get_filings must have been called with year + quarter + form
        kwargs = mock_edgar_13f["edgar"].get_filings.call_args.kwargs
        assert kwargs["form"] == "13F-HR"
        assert kwargs["year"] is not None
        assert kwargs["quarter"] is not None

    def test_explicit_year_quarter(self, mock_edgar_13f):
        mock_edgar_13f["edgar"].get_filings.return_value = _fake_filings_list([])
        invoke_tool("get_13f_filings", {"year": 2025, "quarter": 3})
        kwargs = mock_edgar_13f["edgar"].get_filings.call_args.kwargs
        assert kwargs == {"year": 2025, "quarter": 3, "form": "13F-HR"}

    def test_empty_quarter_returns_note(self, mock_edgar_13f):
        mock_edgar_13f["edgar"].get_filings.return_value = _fake_filings_list([])
        result = invoke_tool("get_13f_filings", {"year": 2099, "quarter": 1})
        assert result["count"] == 0
        assert "note" in result

    def test_quarter_validation(self):
        result = invoke_tool("get_13f_filings", {"year": 2025, "quarter": 5})
        assert "error" in result
        assert "Invalid arguments" in result["error"]


class TestGet13FHoldings:
    def _wire_filings(self, mock_edgar_13f, filings: list):
        # Company(...).get_filings(form="13F-HR") → filings
        mock_edgar_13f["company"].get_filings.return_value = _fake_filings_list(filings)

    def test_latest_filing_happy_path(self, mock_edgar_13f):
        f = _fake_13f_filing(
            "BERKSHIRE HATHAWAY INC", 1067983, "0001067983-26-000010", "2026-02-14", "2025-12-31"
        )
        tf = _fake_thirteenf(
            manager="BERKSHIRE HATHAWAY INC",
            total_value=300_000_000_000,
            total_holdings=40,
            report_period="2025-12-31",
            filing_date="2026-02-14",
            accession="0001067983-26-000010",
            holdings_rows=[
                {
                    "Issuer": "APPLE INC",
                    "Ticker": "AAPL",
                    "Cusip": "037833100",
                    "SharesPrnAmount": 300_000_000,
                    "Value": 60_000_000_000,
                    "Type": "Shares",
                    "PutCall": "",
                },
                {
                    "Issuer": "COCA COLA CO",
                    "Ticker": "KO",
                    "Cusip": "191216100",
                    "SharesPrnAmount": 400_000_000,
                    "Value": 25_000_000_000,
                    "Type": "Shares",
                    "PutCall": "",
                },
            ],
        )
        f.obj = lambda: tf
        self._wire_filings(mock_edgar_13f, [f])

        result = invoke_tool("get_13f_holdings", {"manager": "BRK-A", "top_n": 2})
        assert result["manager"] == "BERKSHIRE HATHAWAY INC"
        assert result["total_value"] == 300_000_000_000
        assert result["total_holdings"] == 40
        assert [h["ticker"] for h in result["holdings"]] == ["AAPL", "KO"]
        assert result["holdings"][0]["value"] == 60_000_000_000

    def test_period_selection_picks_matching_quarter(self, mock_edgar_13f):
        # Two filings: one for Q4 2025 (latest), one for Q3 2025. Requesting
        # Q3 should skip the latest and parse the second.
        f_q4 = _fake_13f_filing("Mgr", 999, "acc-q4", "2026-02-14", "2025-12-31")
        f_q3 = _fake_13f_filing("Mgr", 999, "acc-q3", "2025-11-14", "2025-09-30")

        tf_q3 = _fake_thirteenf(
            "Mgr",
            10,
            1,
            "2025-09-30",
            "2025-11-14",
            "acc-q3",
            [
                {
                    "Issuer": "X",
                    "Ticker": "X",
                    "Cusip": "0",
                    "SharesPrnAmount": 1,
                    "Value": 10,
                    "Type": "Shares",
                    "PutCall": "",
                }
            ],
        )
        f_q3.obj = lambda: tf_q3
        f_q4.obj = lambda: (_ for _ in ()).throw(AssertionError("should not parse Q4"))
        self._wire_filings(mock_edgar_13f, [f_q4, f_q3])

        result = invoke_tool(
            "get_13f_holdings",
            {"manager": "1067983", "year": 2025, "quarter": 3},
        )
        assert result["accession_no"] == "acc-q3"
        assert result["report_period"] == "2025-09-30"

    def test_no_filings_returns_error(self, mock_edgar_13f):
        self._wire_filings(mock_edgar_13f, [])
        result = invoke_tool("get_13f_holdings", {"manager": "BRK-A"})
        assert "error" in result
        assert "No parseable" in result["error"]

    def test_name_fallback_via_find_company(self, mock_edgar_13f):
        # Company(...) raises for the name → code falls back to find_company.
        mock_edgar_13f["edgar"].Company.side_effect = Exception("ticker not found")

        resolved = MagicMock()
        resolved.name = "Scion Asset Management, LLC"
        resolved.cik = 1649339

        tf = _fake_thirteenf(
            "Scion Asset Management, LLC",
            100,
            1,
            "2025-12-31",
            "2026-02-14",
            "acc",
            [
                {
                    "Issuer": "GME",
                    "Ticker": "GME",
                    "Cusip": "36467W109",
                    "SharesPrnAmount": 5000,
                    "Value": 100,
                    "Type": "Shares",
                    "PutCall": "PUT",
                }
            ],
        )
        f = _fake_13f_filing("Scion", 1649339, "acc", "2026-02-14", "2025-12-31")
        f.obj = lambda: tf
        resolved.get_filings.return_value = _fake_filings_list([f])

        search_results = MagicMock()
        search_results.empty = False
        search_results.__getitem__.side_effect = lambda i: resolved
        mock_edgar_13f["search"].find_company.return_value = search_results

        result = invoke_tool("get_13f_holdings", {"manager": "Scion Asset"})
        assert result["manager"] == "Scion Asset Management, LLC"
        assert result["holdings"][0]["put_call"] == "PUT"

    def test_invalid_sort(self, mock_edgar_13f):
        self._wire_filings(mock_edgar_13f, [])
        result = invoke_tool("get_13f_holdings", {"manager": "BRK-A", "sort": "bogus"})
        assert "error" in result
        assert "Invalid sort" in result["error"]


class TestGet13FChanges:
    def _wire(self, mock_edgar_13f, comparison_df: pd.DataFrame):
        f = _fake_13f_filing("Mgr", 999, "acc", "2026-02-14", "2025-12-31")
        comparison = SimpleNamespace(
            data=comparison_df,
            current_period="2025-12-31",
            previous_period="2025-09-30",
            manager_name="Test Manager LLC",
        )
        tf = _fake_thirteenf("Test Manager LLC", 100, 5, "2025-12-31", "2026-02-14", "acc", [])
        tf.compare_holdings = lambda display_limit=200: comparison
        f.obj = lambda: tf
        mock_edgar_13f["company"].get_filings.return_value = _fake_filings_list([f])

    def _comparison_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # Big new position
                {
                    "Issuer": "NVDA",
                    "Ticker": "NVDA",
                    "Cusip": "n",
                    "Status": "NEW",
                    "Shares": 1000,
                    "PrevShares": float("nan"),
                    "ShareChange": float("nan"),
                    "ShareChangePct": float("nan"),
                    "Value": 50_000_000,
                    "PrevValue": float("nan"),
                    "ValueChange": 50_000_000,
                    "ValueChangePct": float("nan"),
                },
                # Modest increase
                {
                    "Issuer": "AAPL",
                    "Ticker": "AAPL",
                    "Cusip": "a",
                    "Status": "INCREASED",
                    "Shares": 200,
                    "PrevShares": 100,
                    "ShareChange": 100,
                    "ShareChangePct": 100.0,
                    "Value": 20_000_000,
                    "PrevValue": 10_000_000,
                    "ValueChange": 10_000_000,
                    "ValueChangePct": 100.0,
                },
                # Big trim
                {
                    "Issuer": "XOM",
                    "Ticker": "XOM",
                    "Cusip": "x",
                    "Status": "DECREASED",
                    "Shares": 50,
                    "PrevShares": 500,
                    "ShareChange": -450,
                    "ShareChangePct": -90.0,
                    "Value": 5_000_000,
                    "PrevValue": 40_000_000,
                    "ValueChange": -35_000_000,
                    "ValueChangePct": -87.5,
                },
                # Closed
                {
                    "Issuer": "GE",
                    "Ticker": "GE",
                    "Cusip": "g",
                    "Status": "CLOSED",
                    "Shares": float("nan"),
                    "PrevShares": 100,
                    "ShareChange": float("nan"),
                    "ShareChangePct": float("nan"),
                    "Value": float("nan"),
                    "PrevValue": 8_000_000,
                    "ValueChange": float("nan"),
                    "ValueChangePct": float("nan"),
                },
            ]
        )

    def test_sort_increased_orders_by_value_change_desc(self, mock_edgar_13f):
        self._wire(mock_edgar_13f, self._comparison_df())
        result = invoke_tool("get_13f_changes", {"manager": "BRK-A", "sort": "increased"})
        tickers = [c["ticker"] for c in result["changes"]]
        assert tickers == ["NVDA", "AAPL"]
        assert result["current_period"] == "2025-12-31"
        assert result["previous_period"] == "2025-09-30"

    def test_sort_decreased_orders_by_value_change_asc(self, mock_edgar_13f):
        self._wire(mock_edgar_13f, self._comparison_df())
        result = invoke_tool("get_13f_changes", {"manager": "BRK-A", "sort": "decreased"})
        tickers = [c["ticker"] for c in result["changes"]]
        # XOM's -$35M is more negative than CLOSED GE (NaN ValueChange sorts last).
        assert tickers[0] == "XOM"

    def test_sort_new_filters_new_only(self, mock_edgar_13f):
        self._wire(mock_edgar_13f, self._comparison_df())
        result = invoke_tool("get_13f_changes", {"manager": "BRK-A", "sort": "new"})
        assert [c["ticker"] for c in result["changes"]] == ["NVDA"]
        assert result["changes"][0]["status"] == "NEW"

    def test_sort_closed_filters_closed_only(self, mock_edgar_13f):
        self._wire(mock_edgar_13f, self._comparison_df())
        result = invoke_tool("get_13f_changes", {"manager": "BRK-A", "sort": "closed"})
        assert [c["ticker"] for c in result["changes"]] == ["GE"]

    def test_sort_absolute_change_ranks_by_magnitude(self, mock_edgar_13f):
        self._wire(mock_edgar_13f, self._comparison_df())
        result = invoke_tool(
            "get_13f_changes", {"manager": "BRK-A", "sort": "absolute_change", "top_n": 3}
        )
        tickers = [c["ticker"] for c in result["changes"]]
        # |-35M| > |50M|? No: 50 > 35. Order: NVDA (50), XOM (35), AAPL (10)
        assert tickers == ["NVDA", "XOM", "AAPL"]

    def test_invalid_sort(self, mock_edgar_13f):
        result = invoke_tool("get_13f_changes", {"manager": "BRK-A", "sort": "bogus"})
        assert "error" in result
        assert "Invalid sort" in result["error"]

    def test_no_previous_quarter(self, mock_edgar_13f):
        f = _fake_13f_filing("Mgr", 999, "acc", "2026-02-14", "2025-12-31")
        tf = _fake_thirteenf("Mgr", 100, 1, "2025-12-31", "2026-02-14", "acc", [])
        tf.compare_holdings = lambda display_limit=200: None
        f.obj = lambda: tf
        mock_edgar_13f["company"].get_filings.return_value = _fake_filings_list([f])
        result = invoke_tool("get_13f_changes", {"manager": "BRK-A"})
        assert "error" in result
        assert "Previous-quarter" in result["error"]
