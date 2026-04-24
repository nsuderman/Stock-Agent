"""Macro & market-regime data tools backed by FRED (St. Louis Fed).

Pulls treasury yields, VIX, credit spreads, dollar index, oil, and CPI so the
agent has a regime-aware view when making buy/hold/sell calls. A curated
`get_macro_snapshot()` covers the standard dashboard; `get_fred_series()` is
the escape hatch for anything else.

Observations are cached per `(series_id, start, end, limit)` for 4 hours —
FRED daily series release once per business day, so longer TTLs don't miss
much. The cache is per-process, so it only matters in long-lived library
contexts (e.g. FastAPI); the CLI is short-lived and won't benefit much.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.logging_setup import get_logger
from agent.tools.base import tool

log = get_logger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

_CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours
_cache: dict[tuple[str, str, str, int], tuple[float, list[dict[str, Any]]]] = {}


def _reset_cache() -> None:
    """Testing hook — drop the observation cache."""
    _cache.clear()


def _fetch_observations(
    series_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch observations from FRED (newest first). Cached per key for 4h."""
    key = (series_id, start or "", end or "", limit)
    hit = _cache.get(key)
    if hit is not None:
        fetched_at, data = hit
        if time.time() - fetched_at < _CACHE_TTL_SECONDS:
            return data

    api_key = get_settings().fred_api_key
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key at fredaccount.stlouisfed.org/apikeys."
        )

    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc",
    }
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end

    resp = httpx.get(FRED_BASE, params=params, timeout=15.0)
    resp.raise_for_status()
    data = resp.json().get("observations", []) or []
    _cache[key] = (time.time(), data)
    return data


def _as_float(raw: Any) -> float | None:
    """FRED encodes missing observations as '.'."""
    if raw is None or raw == ".":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class FredSeriesArgs(BaseModel):
    series_id: str = Field(
        ...,
        description=(
            "FRED series identifier. Common IDs: DGS2 (2Y yield), DGS10 (10Y), "
            "T10Y2Y (2s10s spread), DFII10 (10Y real yield), DFF (Fed funds), "
            "VIXCLS (VIX), BAMLH0A0HYM2 (HY credit spread), CPIAUCSL (CPI), "
            "UNRATE (unemployment), DCOILWTICO (WTI oil), DTWEXBGS (trade-weighted USD)."
        ),
    )
    start: str | None = Field(
        default=None,
        description="ISO date (YYYY-MM-DD). Omit for most-recent observations.",
    )
    end: str | None = Field(
        default=None,
        description="ISO date (YYYY-MM-DD). Omit for most-recent observations.",
    )
    limit: int = Field(
        default=30,
        ge=1,
        le=500,
        description="Max observations to return (newest first).",
    )


@tool(
    description=(
        "Fetch a FRED (Federal Reserve Economic Data) time series by ID. Generic "
        "escape hatch — prefer `get_macro_snapshot` for the standard regime "
        "dashboard. Returns observations as [{date, value}], newest first. "
        "Common series: DGS2/DGS10 (treasury yields), T10Y2Y (2s10s spread), "
        "DFII10 (10Y real yield), DFF (Fed funds), VIXCLS (VIX), BAMLH0A0HYM2 "
        "(HY OAS), CPIAUCSL (CPI), UNRATE (unemployment), DCOILWTICO (oil), "
        "DTWEXBGS (trade-weighted dollar)."
    )
)
def get_fred_series(args: FredSeriesArgs) -> dict[str, Any]:
    try:
        obs = _fetch_observations(
            args.series_id,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
    except RuntimeError as e:
        return {"error": str(e)}
    except httpx.HTTPError as e:
        log.warning("FRED request failed for %s: %s", args.series_id, e)
        return {"error": f"FRED request failed: {e}"}
    except Exception as e:
        log.exception("FRED fetch failed for %s", args.series_id)
        return {"error": f"{type(e).__name__}: {e}"}

    rows = [{"date": o.get("date"), "value": _as_float(o.get("value"))} for o in obs]
    return {
        "series_id": args.series_id,
        "count": len(rows),
        "observations": rows,
    }


# Curated dashboard: (series_id, friendly_name, one-line meaning).
_SNAPSHOT_SERIES: list[tuple[str, str, str]] = [
    ("DGS2", "2Y Treasury yield", "Front-end rate; tracks Fed rate expectations"),
    ("DGS10", "10Y Treasury yield", "Long rate; DCF denominator for equities"),
    ("T10Y2Y", "2s10s spread", "Inverted = classic recession signal (12-18mo lead historically)"),
    ("DFII10", "10Y real yield (TIPS)", "Inflation-adjusted; what equity multiples actually track"),
    ("DFF", "Effective Fed Funds rate", "Policy rate"),
    ("VIXCLS", "VIX", "Implied S&P vol; >20 elevated, >30 stress"),
    (
        "BAMLH0A0HYM2",
        "HY credit spread (OAS)",
        "Risk appetite; credit usually leads equities at turns",
    ),
    ("DTWEXBGS", "Trade-weighted USD (broad)", "Strong USD = headwind for multinationals + EM"),
    ("DCOILWTICO", "WTI crude oil ($/bbl)", "Feeds CPI + drives energy vs transports rotation"),
    ("CPIAUCSL", "CPI (all items, SA)", "Headline inflation — monthly, look at YoY"),
]


def _change(latest: float | None, prior: float | None) -> float | None:
    if latest is None or prior is None:
        return None
    return round(latest - prior, 6)


def _pct_change(latest: float | None, prior: float | None) -> float | None:
    if latest is None or prior is None or prior == 0:
        return None
    return round((latest / prior - 1) * 100, 4)


def _observation_at_lookback(obs: list[dict[str, Any]], days_back: int) -> dict[str, Any] | None:
    """obs is newest-first; return the first obs dated <= (today - days_back)."""
    if not obs:
        return None
    cutoff = date.today().toordinal() - days_back
    for o in obs:
        d = o.get("date")
        if not d:
            continue
        try:
            if date.fromisoformat(d).toordinal() <= cutoff:
                return o
        except ValueError:
            continue
    return None


@tool(
    description=(
        "Macro regime snapshot — latest value plus 1-week and 1-month deltas for the "
        "10 key macro indicators: 2Y/10Y/10Y-real yields, 2s10s spread, Fed funds, "
        "VIX, HY credit spread, trade-weighted USD, WTI oil, CPI. Use this when "
        "answering buy/hold/sell questions to anchor the call in the current "
        "rate/risk/inflation backdrop. One call returns the whole dashboard — "
        "prefer this over ten separate `get_fred_series` calls."
    )
)
def get_macro_snapshot() -> dict[str, Any]:
    if not get_settings().fred_api_key:
        return {
            "error": "FRED_API_KEY not set. Get a free key at fredaccount.stlouisfed.org/apikeys.",
        }

    indicators: list[dict[str, Any]] = []
    errors: list[str] = []

    for sid, name, meaning in _SNAPSHOT_SERIES:
        try:
            obs = _fetch_observations(sid, limit=40)
        except Exception as e:
            errors.append(f"{sid}: {e}")
            continue

        valid = [
            {"date": o.get("date"), "value": _as_float(o.get("value"))}
            for o in obs
            if _as_float(o.get("value")) is not None and o.get("date")
        ]
        if not valid:
            errors.append(f"{sid}: no valid observations")
            continue

        latest = valid[0]
        wk = _observation_at_lookback(valid, 7)
        mo = _observation_at_lookback(valid, 30)

        indicators.append(
            {
                "series_id": sid,
                "name": name,
                "meaning": meaning,
                "latest_date": latest["date"],
                "latest_value": latest["value"],
                "change_1w": _change(latest["value"], wk["value"]) if wk else None,
                "change_1m": _change(latest["value"], mo["value"]) if mo else None,
                "pct_change_1w": _pct_change(latest["value"], wk["value"]) if wk else None,
                "pct_change_1m": _pct_change(latest["value"], mo["value"]) if mo else None,
            }
        )

    result: dict[str, Any] = {
        "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "count": len(indicators),
        "indicators": indicators,
    }
    if errors:
        result["errors"] = errors
    return result
