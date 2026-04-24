"""Tests for the matplotlib chart-rendering tool."""

from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import pytest

from agent.tools import charts as charts_mod
from agent.tools import invoke_tool
from agent.tools import macro as macro_mod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tmp_charts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(charts_mod, "CHARTS_DIR", tmp_path / "charts")
    yield tmp_path / "charts"


@pytest.fixture(autouse=True)
def _set_fred_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    macro_mod._reset_cache()
    yield
    macro_mod._reset_cache()


def _ohlcv(dates: list[date], closes: list[float]) -> dict[str, list]:
    """Build a minimal OHLCV dict where OHL ≈ close and volume is constant."""
    return {
        "dates": dates,
        "open": list(closes),
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": list(closes),
        "volume": [1_000_000.0] * len(closes),
    }


@pytest.fixture
def mock_symbol_ohlcv(monkeypatch: pytest.MonkeyPatch):
    """Patch _fetch_symbol_ohlcv with a dict keyed by upper-case ticker."""
    data: dict[str, dict[str, list]] = {}

    def fake(symbol, start, end):
        return data.get(symbol.upper())

    monkeypatch.setattr(charts_mod, "_fetch_symbol_ohlcv", fake)
    return data


@pytest.fixture
def mock_fred(monkeypatch: pytest.MonkeyPatch):
    data: dict[str, list[tuple[date, float]]] = {}

    def fake(series_id, start, end):
        return data.get(series_id, [])

    monkeypatch.setattr(charts_mod, "_fetch_fred_series", fake)
    return data


@pytest.fixture
def mock_backtest(monkeypatch: pytest.MonkeyPatch):
    data: dict[int, list[tuple[date, float]]] = {}

    def fake(backtest_id, start, end):
        return data.get(backtest_id, [])

    monkeypatch.setattr(charts_mod, "_fetch_backtest_equity", fake)
    return data


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_normalize_rebases_to_100(self):
        out = charts_mod._normalize([200.0, 210.0, 220.0])
        assert out[0] == 100.0
        assert out[1] == pytest.approx(105.0)

    def test_slugify(self):
        assert charts_mod._slugify("SPY vs VIX") == "spy-vs-vix"
        assert charts_mod._slugify("") == "chart"

    def test_align_inner_joins(self):
        d1 = date(2026, 1, 1)
        d2 = date(2026, 1, 2)
        d3 = date(2026, 1, 3)
        a = [(d1, 1.0), (d2, 2.0), (d3, 3.0)]
        b = [(d2, 20.0), (d3, 30.0)]
        xs, aligned = charts_mod._align_on_common_dates([("A", a), ("B", b)])
        assert xs == [d2, d3]
        assert aligned["A"] == [2.0, 3.0]

    def test_sma_has_leading_nulls(self):
        s = charts_mod._sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        assert s[0] is None
        assert s[1] is None
        assert s[2] == pytest.approx(2.0)
        assert s[4] == pytest.approx(4.0)

    def test_rsi_bounded_0_to_100(self):
        values = [100 + i + (i % 3) for i in range(50)]
        out = charts_mod._rsi(values)
        non_null = [v for v in out if v is not None]
        assert all(0 <= v <= 100 for v in non_null)

    def test_bollinger_width_shrinks_on_flat(self):
        flat = [10.0] * 30
        upper, mid, lower = charts_mod._bollinger(flat, period=20)
        assert upper[-1] == pytest.approx(10.0)
        assert lower[-1] == pytest.approx(10.0)
        assert mid[-1] == pytest.approx(10.0)

    def test_atr_nonneg(self):
        closes = [100.0 + i * 0.5 for i in range(40)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        out = charts_mod._atr(highs, lows, closes)
        non_null = [v for v in out if v is not None]
        assert all(v >= 0 for v in non_null)

    def test_macd_shapes_match(self):
        vals = [100 + (i % 7) for i in range(60)]
        macd, signal, hist = charts_mod._macd(vals)
        assert len(macd) == len(signal) == len(hist) == len(vals)


# ---------------------------------------------------------------------------
# Tool integration tests
# ---------------------------------------------------------------------------


class TestBasicModes:
    def _dates(self, n: int = 60) -> list[date]:
        return [date(2026, 1, 1) + pd_offset(i) for i in range(n)]

    def test_requires_at_least_one_series(self):
        result = invoke_tool(
            "plot_comparison",
            {"symbols": [], "fred_series": [], "start": "2026-01-01", "end": "2026-01-31"},
        )
        assert "error" in result

    def test_normalized_mode(self, mock_symbol_ohlcv, mock_fred, _tmp_charts_dir: Path):
        d = self._dates(30)
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0 + i for i in range(30)])
        mock_fred["VIXCLS"] = [(dd, 18.0 + i * 0.1) for i, dd in enumerate(d)]

        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["spy"],
                "fred_series": ["VIXCLS"],
                "start": "2026-01-01",
                "end": "2026-03-31",
            },
        )
        assert "error" not in result
        assert result["mode"] == "normalized"
        assert Path(result["path"]).exists()
        assert Path(result["path"]).stat().st_size > 1000

    def test_absolute_mode(self, mock_fred, _tmp_charts_dir: Path):
        d = self._dates(5)
        mock_fred["DGS2"] = [(dd, 3.5 + i * 0.01) for i, dd in enumerate(d)]
        mock_fred["DGS10"] = [(dd, 4.2 + i * 0.01) for i, dd in enumerate(d)]
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": [],
                "fred_series": ["DGS2", "DGS10"],
                "start": "2026-01-01",
                "end": "2026-03-31",
                "mode": "absolute",
            },
        )
        assert "error" not in result

    def test_dual_axis_rejects_wrong_count(self, mock_symbol_ohlcv, mock_fred):
        d = self._dates(5)
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0] * 5)
        mock_symbol_ohlcv["QQQ"] = _ohlcv(d, [450.0] * 5)
        mock_fred["VIXCLS"] = [(dd, 18.0) for dd in d]
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY", "QQQ"],
                "fred_series": ["VIXCLS"],
                "start": "2026-01-01",
                "end": "2026-03-31",
                "mode": "dual_axis",
            },
        )
        assert "error" in result
        assert "exactly 2" in result["error"]


class TestCandlestick:
    def test_requires_single_symbol(self, mock_symbol_ohlcv):
        d = [date(2026, 1, i) for i in range(1, 6)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0] * 5)
        mock_symbol_ohlcv["QQQ"] = _ohlcv(d, [450.0] * 5)
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY", "QQQ"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "chart_type": "candlestick",
                "mode": "absolute",
            },
        )
        assert "error" in result
        assert "exactly 1 symbol" in result["error"]

    def test_incompatible_with_normalized(self, mock_symbol_ohlcv):
        d = [date(2026, 1, i) for i in range(1, 6)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0] * 5)
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "chart_type": "candlestick",
            },
        )
        assert "error" in result
        assert "normalized" in result["error"]

    def test_renders(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        d = [date(2026, 1, i) for i in range(1, 11)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0 + i for i in range(10)])
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "chart_type": "candlestick",
                "mode": "absolute",
            },
        )
        assert "error" not in result
        assert result["chart_type"] == "candlestick"
        assert Path(result["path"]).exists()


class TestOverlays:
    def test_moving_averages_render(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(60)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0 + i * 0.5 for i in range(60)])
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-03-31",
                "mode": "absolute",
                "moving_averages": [10, 30],
            },
        )
        assert "error" not in result
        assert result["overlays"]["moving_averages"] == [10, 30]
        assert Path(result["path"]).exists()

    def test_bollinger(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(40)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0 + (i % 5) for i in range(40)])
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-03-31",
                "mode": "absolute",
                "bollinger": True,
            },
        )
        assert "error" not in result
        assert result["overlays"]["bollinger"] is True

    def test_horizontal_lines(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(10)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0] * 10)
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "horizontal_lines": [495.0, 510.0],
            },
        )
        assert "error" not in result
        assert result["overlays"]["horizontal_lines"] == [495.0, 510.0]

    def test_log_scale(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(10)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0 + i for i in range(10)])
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "log_scale": True,
            },
        )
        assert "error" not in result
        assert result["log_scale"] is True


class TestIndicators:
    def _make_symbol(self, n: int = 60) -> tuple[list[date], dict[str, list]]:
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(n)]
        closes = [500.0 + i * 0.5 + (i % 7) for i in range(n)]
        return d, _ohlcv(d, closes)

    def test_rsi_subplot(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        _d, data = self._make_symbol()
        mock_symbol_ohlcv["SPY"] = data
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-03-31",
                "mode": "absolute",
                "indicators": ["rsi"],
            },
        )
        assert "error" not in result
        assert result["indicators"] == ["rsi"]
        assert Path(result["path"]).exists()

    def test_multiple_indicators(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        _d, data = self._make_symbol()
        mock_symbol_ohlcv["SPY"] = data
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-03-31",
                "mode": "absolute",
                "indicators": ["rsi", "macd", "volume", "atr"],
            },
        )
        assert "error" not in result
        assert result["indicators"] == ["rsi", "macd", "volume", "atr"]

    def test_indicators_require_symbol(self, mock_fred):
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(20)]
        mock_fred["DGS10"] = [(dd, 4.0) for dd in d]
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": [],
                "fred_series": ["DGS10"],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "indicators": ["rsi"],
            },
        )
        assert "error" in result
        assert "symbol" in result["error"].lower()


class TestEvents:
    def _setup_spy(self, mock_symbol_ohlcv, n: int = 30) -> list[date]:
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(n)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0 + i for i in range(n)])
        return d

    def test_vertical_line_event(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        self._setup_spy(mock_symbol_ohlcv)
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "events": [{"date": "2026-01-15", "label": "Earnings", "style": "line"}],
            },
        )
        assert "error" not in result
        assert len(result["events_drawn"]) == 1
        ev = result["events_drawn"][0]
        assert ev["style"] == "line"
        assert ev["date"] == "2026-01-15"
        assert Path(result["path"]).exists()

    def test_marker_looks_up_close_when_price_omitted(
        self, mock_symbol_ohlcv, _tmp_charts_dir: Path
    ):
        self._setup_spy(mock_symbol_ohlcv)
        # 2026-01-15 is index 14 — close = 500 + 14 = 514.0
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "events": [
                    {"date": "2026-01-15", "style": "marker", "label": "Beat", "color": "green"}
                ],
            },
        )
        assert "error" not in result
        assert len(result["events_drawn"]) == 1
        ev = result["events_drawn"][0]
        assert ev["style"] == "marker"
        assert ev["symbol"] == "SPY"
        assert ev["price"] == pytest.approx(514.0)

    def test_marker_uses_provided_price(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        self._setup_spy(mock_symbol_ohlcv)
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "events": [
                    {
                        "date": "2026-01-10",
                        "style": "marker",
                        "price": 999.0,
                        "symbol": "SPY",
                    }
                ],
            },
        )
        assert "error" not in result
        assert result["events_drawn"][0]["price"] == pytest.approx(999.0)

    def test_marker_normalized_mode_uses_anchor_base(
        self, mock_symbol_ohlcv, _tmp_charts_dir: Path
    ):
        # In normalized mode the chart Y-axis is index-100; _draw_events must
        # still report the raw price in events_drawn (normalization is a
        # rendering concern, not a data concern).
        self._setup_spy(mock_symbol_ohlcv)
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "events": [{"date": "2026-01-15", "style": "marker"}],
            },
        )
        assert "error" not in result
        assert result["events_drawn"][0]["price"] == pytest.approx(514.0)

    def test_marker_falls_back_to_nearest_prior_close(
        self, mock_symbol_ohlcv, _tmp_charts_dir: Path
    ):
        # Trading days only on 1, 2, 5, 6 (weekend gap). Event on the 4th should
        # snap to the 2nd's close.
        dates = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(dates, [100.0, 101.0, 103.0, 104.0])
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "events": [{"date": "2026-01-04", "style": "marker"}],
            },
        )
        assert "error" not in result
        assert result["events_drawn"][0]["price"] == pytest.approx(101.0)

    def test_marker_skipped_when_before_history(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        self._setup_spy(mock_symbol_ohlcv)
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "events": [{"date": "2025-12-01", "style": "marker"}],
            },
        )
        assert "error" not in result
        assert result["events_drawn"] == []

    def test_bad_date_is_skipped(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        self._setup_spy(mock_symbol_ohlcv)
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "events": [
                    {"date": "not-a-date", "style": "line"},
                    {"date": "2026-01-10", "style": "line", "label": "ok"},
                ],
            },
        )
        assert "error" not in result
        assert len(result["events_drawn"]) == 1
        assert result["events_drawn"][0]["date"] == "2026-01-10"

    def test_marker_without_any_symbol_is_skipped(self, mock_fred, _tmp_charts_dir: Path):
        # FRED-only chart, marker has nowhere to anchor → skipped.
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(10)]
        mock_fred["DGS10"] = [(dd, 4.0 + i * 0.01) for i, dd in enumerate(d)]
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": [],
                "fred_series": ["DGS10"],
                "start": "2026-01-01",
                "end": "2026-01-31",
                "mode": "absolute",
                "events": [
                    {"date": "2026-01-05", "style": "line", "label": "FOMC"},
                    {"date": "2026-01-05", "style": "marker", "label": "no anchor"},
                ],
            },
        )
        assert "error" not in result
        styles = [e["style"] for e in result["events_drawn"]]
        assert styles == ["line"]

    def test_events_change_fingerprint(self, mock_symbol_ohlcv, _tmp_charts_dir: Path):
        self._setup_spy(mock_symbol_ohlcv)
        base_args = {
            "symbols": ["SPY"],
            "fred_series": [],
            "start": "2026-01-01",
            "end": "2026-01-31",
            "mode": "absolute",
        }
        a = invoke_tool("plot_comparison", base_args)
        b = invoke_tool(
            "plot_comparison",
            {**base_args, "events": [{"date": "2026-01-15", "style": "line"}]},
        )
        assert a["filename"] != b["filename"]


class TestBacktest:
    def test_backtest_overlay(self, mock_symbol_ohlcv, mock_backtest, _tmp_charts_dir: Path):
        d = [date(2026, 1, 1) + pd_offset(i) for i in range(20)]
        mock_symbol_ohlcv["SPY"] = _ohlcv(d, [500.0 + i for i in range(20)])
        mock_backtest[42] = [(dd, 10000.0 + i * 50) for i, dd in enumerate(d)]
        result = invoke_tool(
            "plot_comparison",
            {
                "symbols": ["SPY"],
                "fred_series": [],
                "backtest_ids": [42],
                "start": "2026-01-01",
                "end": "2026-03-31",
            },
        )
        assert "error" not in result
        assert result["backtest_ids"] == [42]
        assert Path(result["path"]).exists()


class TestCleanup:
    def test_deletes_stale_pngs(self, _tmp_charts_dir: Path):
        _tmp_charts_dir.mkdir(parents=True)
        fresh = _tmp_charts_dir / "fresh.png"
        stale = _tmp_charts_dir / "stale.png"
        fresh.write_bytes(b"fake")
        stale.write_bytes(b"fake")
        past = time.time() - (charts_mod.CHARTS_RETENTION_DAYS + 1) * 86400
        os.utime(stale, (past, past))
        deleted = charts_mod._cleanup_stale_charts()
        assert deleted == 1
        assert fresh.exists()
        assert not stale.exists()


class TestFetchHelpers:
    def test_fetch_symbol_ohlcv(self, monkeypatch: pytest.MonkeyPatch):
        def fake_fetch(sql, params):
            return [
                {
                    "date": "2026-01-05",
                    "open": 100,
                    "high": 102,
                    "low": 99,
                    "close": 101,
                    "volume": 1000,
                },
                {
                    "date": "2026-01-06",
                    "open": 101,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": None,
                },  # dropped
                {
                    "date": "2026-01-07",
                    "open": 102,
                    "high": 104,
                    "low": 101,
                    "close": 103,
                    "volume": 2000,
                },
            ]

        monkeypatch.setattr(charts_mod, "fetch", fake_fetch)
        out = charts_mod._fetch_symbol_ohlcv("SPY", "2026-01-01", "2026-01-31")
        assert out is not None
        assert out["dates"] == [date(2026, 1, 5), date(2026, 1, 7)]
        assert out["close"] == [101.0, 103.0]
        assert out["volume"] == [1000.0, 2000.0]

    def test_fetch_backtest_equity_parses_json(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            charts_mod,
            "fetch",
            lambda sql, params: [
                {
                    "equity_curve": [
                        {"date": "2026-01-05", "value": 10000.0},
                        {"date": "2026-01-06", "value": 10100.0},
                        {"date": "2026-03-01", "value": 10500.0},  # out of range
                    ]
                }
            ],
        )
        out = charts_mod._fetch_backtest_equity(1, "2026-01-01", "2026-01-31")
        assert len(out) == 2
        assert out[0] == (date(2026, 1, 5), 10000.0)


# Helper: pandas timedelta offset as a function so tests read naturally
def pd_offset(i: int):
    from datetime import timedelta

    return timedelta(days=i)
