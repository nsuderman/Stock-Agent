"""Tests for the FRED-backed macro tools."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agent.tools import invoke_tool
from agent.tools import macro as macro_mod


@pytest.fixture(autouse=True)
def _set_fred_key(monkeypatch: pytest.MonkeyPatch):
    """Give Settings a FRED key so tool entry point doesn't short-circuit."""
    monkeypatch.setenv("FRED_API_KEY", "test-key-for-mocks")
    yield


@pytest.fixture(autouse=True)
def _clear_cache():
    macro_mod._reset_cache()
    yield
    macro_mod._reset_cache()


def _mock_resp(observations: list[dict[str, Any]]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"observations": observations}
    return resp


class TestAsFloat:
    def test_dot_is_missing(self):
        assert macro_mod._as_float(".") is None

    def test_none_is_missing(self):
        assert macro_mod._as_float(None) is None

    def test_number_string_parses(self):
        assert macro_mod._as_float("4.25") == 4.25

    def test_bad_string_is_none(self):
        assert macro_mod._as_float("not a number") is None


class TestGetFredSeries:
    def test_success_returns_observations(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[dict[str, Any]] = []

        def fake_get(url, params=None, timeout=None):
            calls.append({"url": url, "params": params})
            return _mock_resp(
                [
                    {"date": "2026-04-23", "value": "4.25"},
                    {"date": "2026-04-22", "value": "4.22"},
                    {"date": "2026-04-21", "value": "."},  # missing
                ]
            )

        monkeypatch.setattr(httpx, "get", fake_get)

        result = invoke_tool("get_fred_series", {"series_id": "DGS10", "limit": 3})
        assert result["series_id"] == "DGS10"
        assert result["count"] == 3
        assert result["observations"][0] == {"date": "2026-04-23", "value": 4.25}
        assert result["observations"][2] == {"date": "2026-04-21", "value": None}
        assert calls[0]["params"]["series_id"] == "DGS10"
        assert calls[0]["params"]["api_key"] == "test-key-for-mocks"
        assert calls[0]["params"]["sort_order"] == "desc"

    def test_start_end_flow_through(self, monkeypatch: pytest.MonkeyPatch):
        calls: list[dict[str, Any]] = []

        def fake_get(url, params=None, timeout=None):
            calls.append({"params": params})
            return _mock_resp([])

        monkeypatch.setattr(httpx, "get", fake_get)

        invoke_tool(
            "get_fred_series",
            {"series_id": "DGS2", "start": "2026-01-01", "end": "2026-04-01"},
        )
        assert calls[0]["params"]["observation_start"] == "2026-01-01"
        assert calls[0]["params"]["observation_end"] == "2026-04-01"

    def test_missing_key_returns_error(self):
        # `configure(...)` overrides .env so the real FRED_API_KEY doesn't leak in.
        from agent.config import configure

        configure(fred_api_key=None)

        result = invoke_tool("get_fred_series", {"series_id": "DGS10"})
        assert "error" in result
        assert "FRED_API_KEY" in result["error"]

    def test_http_error_returns_tool_error(self, monkeypatch: pytest.MonkeyPatch):
        def boom(url, params=None, timeout=None):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "get", boom)

        result = invoke_tool("get_fred_series", {"series_id": "DGS10"})
        assert "error" in result
        assert "FRED request failed" in result["error"]

    def test_invalid_limit_rejected(self):
        result = invoke_tool("get_fred_series", {"series_id": "DGS10", "limit": 99999})
        assert "error" in result
        assert "Invalid arguments" in result["error"]

    def test_series_id_required(self):
        result = invoke_tool("get_fred_series", {})
        assert "error" in result


class TestCaching:
    def test_second_call_same_key_hits_cache(self, monkeypatch: pytest.MonkeyPatch):
        call_count = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            call_count["n"] += 1
            return _mock_resp([{"date": "2026-04-23", "value": "4.25"}])

        monkeypatch.setattr(httpx, "get", fake_get)

        invoke_tool("get_fred_series", {"series_id": "DGS10", "limit": 30})
        invoke_tool("get_fred_series", {"series_id": "DGS10", "limit": 30})
        assert call_count["n"] == 1

    def test_different_limit_is_separate_cache_key(self, monkeypatch: pytest.MonkeyPatch):
        call_count = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            call_count["n"] += 1
            return _mock_resp([])

        monkeypatch.setattr(httpx, "get", fake_get)

        invoke_tool("get_fred_series", {"series_id": "DGS10", "limit": 30})
        invoke_tool("get_fred_series", {"series_id": "DGS10", "limit": 60})
        assert call_count["n"] == 2

    def test_cache_expires_after_ttl(self, monkeypatch: pytest.MonkeyPatch):
        call_count = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            call_count["n"] += 1
            return _mock_resp([])

        monkeypatch.setattr(httpx, "get", fake_get)

        # First call: miss
        macro_mod._fetch_observations("DGS10", limit=30)
        # Simulate time past TTL by clearing + recalling
        monkeypatch.setattr(macro_mod, "_CACHE_TTL_SECONDS", 0)
        macro_mod._fetch_observations("DGS10", limit=30)
        assert call_count["n"] == 2


class TestObservationLookback:
    def test_finds_first_obs_at_or_before_cutoff(self):
        from datetime import date, timedelta

        today = date.today()
        obs = [
            {"date": today.isoformat(), "value": 1.0},
            {"date": (today - timedelta(days=3)).isoformat(), "value": 2.0},
            {"date": (today - timedelta(days=10)).isoformat(), "value": 3.0},
            {"date": (today - timedelta(days=35)).isoformat(), "value": 4.0},
        ]
        wk = macro_mod._observation_at_lookback(obs, 7)
        assert wk == {"date": (today - timedelta(days=10)).isoformat(), "value": 3.0}
        mo = macro_mod._observation_at_lookback(obs, 30)
        assert mo == {"date": (today - timedelta(days=35)).isoformat(), "value": 4.0}

    def test_returns_none_when_no_obs_old_enough(self):
        from datetime import date

        obs = [{"date": date.today().isoformat(), "value": 1.0}]
        assert macro_mod._observation_at_lookback(obs, 7) is None

    def test_empty_obs(self):
        assert macro_mod._observation_at_lookback([], 7) is None


class TestMacroSnapshot:
    def test_bundles_all_series(self, monkeypatch: pytest.MonkeyPatch):
        from datetime import date, timedelta

        today = date.today()

        def fake_get(url, params=None, timeout=None):
            # Simple history with one obs today + one a month ago.
            return _mock_resp(
                [
                    {"date": today.isoformat(), "value": "5.0"},
                    {"date": (today - timedelta(days=35)).isoformat(), "value": "4.0"},
                ]
            )

        monkeypatch.setattr(httpx, "get", fake_get)

        result = invoke_tool("get_macro_snapshot", {})
        assert "error" not in result
        assert result["count"] == len(macro_mod._SNAPSHOT_SERIES)
        names = {ind["series_id"] for ind in result["indicators"]}
        assert "DGS10" in names
        assert "VIXCLS" in names
        assert "T10Y2Y" in names
        # Verify deltas computed against the 1m lookback.
        any_ind = result["indicators"][0]
        assert any_ind["latest_value"] == 5.0
        assert any_ind["change_1m"] == 1.0
        assert any_ind["pct_change_1m"] == 25.0

    def test_missing_key_returns_error(self):
        from agent.config import configure

        configure(fred_api_key=None)

        result = invoke_tool("get_macro_snapshot", {})
        assert "error" in result
        assert "FRED_API_KEY" in result["error"]

    def test_per_series_error_does_not_kill_snapshot(self, monkeypatch: pytest.MonkeyPatch):
        """If one series 500s, the rest still return."""
        call_count = {"n": 0}

        def fake_get(url, params=None, timeout=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.HTTPError("boom")
            return _mock_resp([{"date": "2026-04-23", "value": "1.0"}])

        monkeypatch.setattr(httpx, "get", fake_get)

        result = invoke_tool("get_macro_snapshot", {})
        assert result["count"] == len(macro_mod._SNAPSHOT_SERIES) - 1
        assert "errors" in result
        assert len(result["errors"]) == 1

    def test_series_with_only_dots_surfaces_as_error(self, monkeypatch: pytest.MonkeyPatch):
        def fake_get(url, params=None, timeout=None):
            return _mock_resp([{"date": "2026-04-23", "value": "."}])

        monkeypatch.setattr(httpx, "get", fake_get)

        result = invoke_tool("get_macro_snapshot", {})
        assert result["count"] == 0
        assert len(result["errors"]) == len(macro_mod._SNAPSHOT_SERIES)


class TestChangeHelpers:
    def test_change(self):
        assert macro_mod._change(5.0, 4.0) == 1.0
        assert macro_mod._change(None, 4.0) is None
        assert macro_mod._change(5.0, None) is None

    def test_pct_change(self):
        assert macro_mod._pct_change(110.0, 100.0) == 10.0
        assert macro_mod._pct_change(5.0, 0.0) is None
        assert macro_mod._pct_change(None, 100.0) is None
