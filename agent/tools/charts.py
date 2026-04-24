"""Chart-rendering tool — PNG comparison charts of tickers, FRED series, and backtests.

Renders via matplotlib's non-interactive Agg backend so it works headless in
CLI and FastAPI contexts. Output PNGs land in `charts/` at repo root; the tool
returns the absolute path and relative path (the latter is what a FastAPI
host should expose via static-file mount).

## Render modes (`mode`)
- `normalized` (default) — rebase each main-pane series to 100 at the start
  date, single Y-axis. Best for relative-performance with mixed scales.
- `absolute` — single Y-axis, raw values. Same-scale series only.
- `dual_axis` — two Y-axes with native scales. Exactly 2 series.

## Overlays (same pane as main chart, apply to symbols)
- `moving_averages` — list of SMA periods (e.g. `[50, 200]`).
- `ema_periods` — list of EMA periods.
- `bollinger` — 20-period / 2σ Bollinger Bands.
- `horizontal_lines` — constant Y levels (support / resistance / alerts).

## Indicator subplots (below main, applied to the FIRST symbol only)
- `rsi`, `macd`, `volume`, `atr` — one stacked subplot per entry. GridSpec
  heights: main=3, each subplot=1. Indicator subplots share the main X-axis.

## Chart type
- `line` (default) or `candlestick` (exactly 1 symbol, 0 FRED series, not
  compatible with `mode="normalized"`).

## Backtest overlays
- `backtest_ids` — plot `backtest_results.equity_curve` alongside symbols.

## Events (vertical lines + point markers)
- `events` — list of date-anchored annotations (earnings dates, revenue
  beats/misses, FOMC meetings, news items, anything). Each event is either
  a dashed vertical line (`style="line"`) or a dot on a symbol's price
  (`style="marker"` — y-value defaults to the anchor symbol's close on that
  date). In `mode="normalized"`, marker y-values are rebased to the 100-index.

## Auto-cleanup
PNGs in `charts/` older than `CHARTS_RETENTION_DAYS` are deleted at the start
of every call — directory stays bounded without cron.

## Branding
Quantara logo watermark bottom-right + purple grid/title accents.
"""

from __future__ import annotations

import bisect
import hashlib
import re
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Literal

import matplotlib

# Must set backend before any pyplot import; Agg is headless-safe.
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from pydantic import BaseModel, Field

from agent.config import REPO_ROOT, get_settings
from agent.logging_setup import get_logger
from agent.tools.base import fetch, tool
from agent.tools.macro import _fetch_observations

log = get_logger(__name__)

CHARTS_DIR = REPO_ROOT / "charts"
CHARTS_RETENTION_DAYS = 7

QUANTARA_PURPLE = "#4A1A5E"
QUANTARA_PURPLE_LIGHT = "#8B4FA8"
DUAL_AXIS_LEFT_COLOR = "#2E86AB"
CANDLE_UP_COLOR = "#2E7D32"
CANDLE_DOWN_COLOR = "#C62828"

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "quantara_logo.png"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
ChartMode = Literal["normalized", "absolute", "dual_axis"]
ChartType = Literal["line", "candlestick"]
Indicator = Literal["rsi", "macd", "volume", "atr"]
EventStyle = Literal["line", "marker"]

EVENT_LINE_DEFAULT_COLOR = "#666666"
EVENT_MARKER_DEFAULT_COLOR = QUANTARA_PURPLE


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "chart"


def _cleanup_stale_charts() -> int:
    """Delete PNGs older than CHARTS_RETENTION_DAYS. Returns count removed."""
    if not CHARTS_DIR.exists():
        return 0
    cutoff = time.time() - CHARTS_RETENTION_DAYS * 86400
    deleted = 0
    for p in CHARTS_DIR.glob("*.png"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                deleted += 1
        except OSError:
            continue
    return deleted


def _nan_to_none(values: Any) -> list[float | None]:
    return [None if pd.isna(v) else float(v) for v in values]


# ---------------------------------------------------------------------------
# Indicator math (pandas-backed)
# ---------------------------------------------------------------------------


def _sma(values: list[float], period: int) -> list[float | None]:
    return _nan_to_none(pd.Series(values).rolling(period).mean())


def _ema(values: list[float], period: int) -> list[float | None]:
    return _nan_to_none(pd.Series(values).ewm(span=period, adjust=False).mean())


def _rsi(values: list[float], period: int = 14) -> list[float | None]:
    s = pd.Series(values)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return _nan_to_none(rsi)


def _macd(
    values: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    s = pd.Series(values)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return _nan_to_none(macd_line), _nan_to_none(signal_line), _nan_to_none(histogram)


def _bollinger(
    values: list[float], period: int = 20, std: float = 2.0
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    s = pd.Series(values)
    mean = s.rolling(period).mean()
    stdev = s.rolling(period).std()
    upper = mean + std * stdev
    lower = mean - std * stdev
    return _nan_to_none(upper), _nan_to_none(mean), _nan_to_none(lower)


def _atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[float | None]:
    h = pd.Series(highs)
    low = pd.Series(lows)
    c = pd.Series(closes)
    prev_c = c.shift(1)
    tr = pd.concat([h - low, (h - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return _nan_to_none(atr)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_symbol_ohlcv(symbol: str, start: str, end: str) -> dict[str, list] | None:
    """Full OHLCV for a ticker. Returns None if no rows.

    Result shape:
        {"dates": [date, ...], "open": [float, ...], "high": [...],
         "low": [...], "close": [...], "volume": [...]}
    """
    schema = get_settings().db_schema
    rows = fetch(
        f"SELECT date, open, high, low, close, volume FROM {schema}.analytics "
        "WHERE symbol = :sym AND date BETWEEN :start AND :end ORDER BY date",
        {"sym": symbol.upper(), "start": start, "end": end},
    )
    if not rows:
        return None
    dates: list[date] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    for r in rows:
        c = r.get("close")
        d = r.get("date")
        if c is None or d is None:
            continue
        try:
            dd = date.fromisoformat(d) if isinstance(d, str) else d
        except ValueError:
            continue
        dates.append(dd)
        o = r.get("open")
        hi = r.get("high")
        lo = r.get("low")
        vol = r.get("volume")
        opens.append(float(o) if o is not None else float(c))
        highs.append(float(hi) if hi is not None else float(c))
        lows.append(float(lo) if lo is not None else float(c))
        closes.append(float(c))
        volumes.append(float(vol) if vol is not None else 0.0)
    if not dates:
        return None
    return {
        "dates": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


def _fetch_fred_series(series_id: str, start: str, end: str) -> list[tuple[date, float]]:
    obs = _fetch_observations(series_id, start=start, end=end, limit=500)
    out: list[tuple[date, float]] = []
    for o in obs:
        raw = o.get("value")
        if raw is None or raw == ".":
            continue
        try:
            v = float(raw)
            dd = date.fromisoformat(o["date"])
        except (TypeError, ValueError):
            continue
        out.append((dd, v))
    out.sort(key=lambda x: x[0])
    return out


def _fetch_backtest_equity(backtest_id: int, start: str, end: str) -> list[tuple[date, float]]:
    """Parse `backtest_results.equity_curve` JSON and filter to date range.

    Expected shape: [{"date": "YYYY-MM-DD", "value": 10000.0}, ...]
    """
    backtest_schema = get_settings().backtest_schema
    rows = fetch(
        f"SELECT equity_curve FROM {backtest_schema}.backtest_results WHERE id = :id",
        {"id": backtest_id},
    )
    if not rows:
        return []
    ec = rows[0].get("equity_curve")
    if not isinstance(ec, list):
        return []
    try:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
    except ValueError:
        return []
    out: list[tuple[date, float]] = []
    for point in ec:
        if not isinstance(point, dict):
            continue
        d_raw = point.get("date")
        v = point.get("value", point.get("equity"))
        if d_raw is None or v is None:
            continue
        try:
            d = date.fromisoformat(d_raw) if isinstance(d_raw, str) else d_raw
        except ValueError:
            continue
        if start_d <= d <= end_d:
            out.append((d, float(v)))
    out.sort(key=lambda x: x[0])
    return out


def _align_on_common_dates(
    named_series: list[tuple[str, list[tuple[date, float]]]],
) -> tuple[list[date], dict[str, list[float]]]:
    """Inner-join named series on their common date set."""
    if not named_series:
        return [], {}
    per_series_maps: list[tuple[str, dict[date, float]]] = [
        (name, dict(vals)) for name, vals in named_series
    ]
    common: set[date] | None = None
    for _, m in per_series_maps:
        keys = set(m.keys())
        common = keys if common is None else (common & keys)
    dates_sorted = sorted(common or set())
    aligned: dict[str, list[float]] = {
        name: [m[d] for d in dates_sorted] for name, m in per_series_maps
    }
    return dates_sorted, aligned


def _normalize(values: list[float]) -> list[float]:
    if not values or values[0] == 0:
        return values
    base = values[0]
    return [v / base * 100 for v in values]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _draw_candlesticks(
    ax: plt.Axes,
    dates_sorted: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> None:
    """Draw OHLC candlesticks on `ax`. Positions by mdates so X-axis stays date-typed."""
    body_width = 0.6  # in days
    for d, o, h, low, c in zip(dates_sorted, opens, highs, lows, closes, strict=False):
        x = mdates.date2num(d)
        color = CANDLE_UP_COLOR if c >= o else CANDLE_DOWN_COLOR
        ax.plot([x, x], [low, h], color=color, linewidth=0.8, zorder=2)
        ax.add_patch(
            Rectangle(
                (x - body_width / 2, min(o, c)),
                body_width,
                abs(c - o) or body_width * 0.02,  # show a line for doji
                facecolor=color,
                edgecolor=color,
                alpha=0.85,
                zorder=3,
            )
        )


def _draw_rsi(ax: plt.Axes, xs: list[date], closes: list[float]) -> None:
    ax.plot(xs, _rsi(closes), color=QUANTARA_PURPLE, linewidth=1.2)  # type: ignore[arg-type]
    ax.axhline(70, color="#C62828", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axhline(30, color="#2E7D32", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI(14)", fontsize=9)


def _draw_macd(ax: plt.Axes, xs: list[date], closes: list[float]) -> None:
    macd_line, signal_line, hist = _macd(closes)
    ax.plot(xs, macd_line, color=QUANTARA_PURPLE, linewidth=1.2, label="MACD")  # type: ignore[arg-type]
    ax.plot(xs, signal_line, color=DUAL_AXIS_LEFT_COLOR, linewidth=1.0, label="Signal")  # type: ignore[arg-type]
    colors = [CANDLE_UP_COLOR if (h is not None and h >= 0) else CANDLE_DOWN_COLOR for h in hist]
    ax.bar(
        xs,  # type: ignore[arg-type]
        [h if h is not None else 0 for h in hist],
        width=0.8,
        color=colors,
        alpha=0.45,
        label="Histogram",
    )
    ax.axhline(0, color="#999999", linewidth=0.5)
    ax.set_ylabel("MACD(12,26,9)", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)


def _draw_volume(
    ax: plt.Axes, xs: list[date], opens: list[float], closes: list[float], vols: list[float]
) -> None:
    colors = [
        CANDLE_UP_COLOR if c >= o else CANDLE_DOWN_COLOR
        for o, c in zip(opens, closes, strict=False)
    ]
    ax.bar(xs, vols, width=0.8, color=colors, alpha=0.6)  # type: ignore[arg-type]
    ax.set_ylabel("Volume", fontsize=9)


def _draw_atr(
    ax: plt.Axes, xs: list[date], highs: list[float], lows: list[float], closes: list[float]
) -> None:
    ax.plot(xs, _atr(highs, lows, closes), color=QUANTARA_PURPLE, linewidth=1.2)  # type: ignore[arg-type]
    ax.set_ylabel("ATR(14)", fontsize=9)


def _nearest_close_on_or_before(sd: dict[str, list], target: date) -> tuple[date, float] | None:
    """Close on `target`, else the most recent trading day before it. None if target < first."""
    dates: list[date] = sd["dates"]
    if not dates or target < dates[0]:
        return None
    idx = bisect.bisect_right(dates, target) - 1
    if idx < 0:
        return None
    return dates[idx], float(sd["close"][idx])


def _draw_events(
    main_ax: plt.Axes,
    events: list[ChartEvent],
    symbol_data: dict[str, dict[str, list]],
    mode: ChartMode,
) -> list[dict[str, Any]]:
    """Draw vertical lines and point markers for arbitrary calendar events.

    Markers anchor to a symbol's close on (or just before) the event date so
    the y-coordinate sits on the plotted line. In normalized mode the y value
    is rebased using the anchor symbol's start-of-range close so events land
    on the same 100-indexed axis.

    Returns a serializable summary of which events actually rendered — the
    tool's return payload surfaces this so the agent can see what was drawn.
    """
    drawn: list[dict[str, Any]] = []
    if not events:
        return drawn

    first_symbol_name = next(iter(symbol_data), None)

    for ev in events:
        try:
            event_date = date.fromisoformat(ev.date)
        except ValueError:
            log.debug("Skipping event with bad date: %s", ev.date)
            continue

        if ev.style == "line":
            color = ev.color or EVENT_LINE_DEFAULT_COLOR
            main_ax.axvline(
                event_date,  # type: ignore[arg-type]
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=1.5,
            )
            if ev.label:
                main_ax.annotate(
                    ev.label,
                    xy=(event_date, 1.0),  # type: ignore[arg-type]
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    rotation=90,
                    ha="left",
                    va="top",
                    alpha=0.9,
                )
            drawn.append({"date": ev.date, "style": "line", "label": ev.label})
            continue

        # Marker: needs an anchor symbol to compute a y value.
        anchor_name = (ev.symbol or first_symbol_name or "").upper()
        sd = symbol_data.get(anchor_name)
        if sd is None:
            log.debug(
                "Skipping marker event %s — no anchor symbol data (%s)",
                ev.date,
                anchor_name or "<none>",
            )
            continue

        if ev.price is not None:
            y_raw = float(ev.price)
            used_date = event_date
        else:
            lookup = _nearest_close_on_or_before(sd, event_date)
            if lookup is None:
                log.debug(
                    "Skipping marker event %s — date before %s price history",
                    ev.date,
                    anchor_name,
                )
                continue
            used_date, y_raw = lookup

        if mode == "normalized":
            base = sd["close"][0] if sd["close"] else 0.0
            y_plot = (y_raw / base * 100) if base else y_raw
        else:
            y_plot = y_raw

        color = ev.color or EVENT_MARKER_DEFAULT_COLOR
        main_ax.scatter(
            [used_date],  # type: ignore[arg-type]
            [y_plot],
            color=color,
            s=70,
            zorder=5,
            edgecolor="white",
            linewidth=1.0,
            marker="o",
        )
        if ev.label:
            main_ax.annotate(
                ev.label,
                xy=(used_date, y_plot),  # type: ignore[arg-type]
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=color,
                alpha=0.95,
            )
        drawn.append(
            {
                "date": ev.date,
                "style": "marker",
                "label": ev.label,
                "symbol": anchor_name,
                "price": y_raw,
            }
        )
    return drawn


def _apply_branding(fig: plt.Figure, main_ax: plt.Axes, subplot_axes: list[plt.Axes]) -> None:
    """Quantara styling on every axes + watermark at bottom-right."""
    for ax in [main_ax, *subplot_axes]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(QUANTARA_PURPLE_LIGHT)
        ax.spines["bottom"].set_color(QUANTARA_PURPLE_LIGHT)
        ax.grid(True, alpha=0.2, color=QUANTARA_PURPLE, linestyle="--", linewidth=0.6)
        ax.tick_params(colors="#555555", which="both")
    main_ax.title.set_color(QUANTARA_PURPLE)

    if not LOGO_PATH.exists():
        return
    try:
        logo = mpimg.imread(str(LOGO_PATH))
    except Exception as e:
        log.debug("Could not load brand watermark: %s", e)
        return

    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.16))
    aspect = logo.shape[0] / logo.shape[1]
    width = 0.13
    height = width * aspect * (fig.get_figwidth() / fig.get_figheight())
    logo_ax = fig.add_axes((1.0 - width - 0.015, 0.005, width, height))
    logo_ax.imshow(logo, alpha=0.55)
    logo_ax.axis("off")


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


class ChartEvent(BaseModel):
    """One calendar event to annotate on the chart.

    Use for earnings dates, FOMC meetings, revenue beats/misses, news items, or
    any other date-specific callout. Two styles:

    - `line`: a dashed vertical line spanning the main chart — good for
      "something happened on this day" with no specific price.
    - `marker`: a filled dot anchored to a symbol's price line — good for
      "$X closed at Y after the beat" callouts. Needs either `price` or a
      symbol whose close can be looked up on that date.
    """

    date: str = Field(..., description="Event date YYYY-MM-DD.")
    label: str | None = Field(
        default=None,
        description="Short text label (<30 chars works best). Omitted = no annotation, just the mark.",
    )
    style: EventStyle = Field(
        default="line",
        description="'line' (vertical line) or 'marker' (point on a symbol's price).",
    )
    symbol: str | None = Field(
        default=None,
        description=(
            "For markers: which symbol's price to anchor to. Defaults to the first symbol "
            "on the chart. Ignored for 'line' style."
        ),
    )
    price: float | None = Field(
        default=None,
        description=(
            "For markers: explicit y-value. Omit to use the anchor symbol's close on the "
            "event date (or the last trading day before it). Ignored for 'line' style."
        ),
    )
    color: str | None = Field(
        default=None,
        description=(
            "Matplotlib color (hex like '#2E7D32' or name like 'red'). Defaults: gray for "
            "lines, Quantara purple for markers. Use green for beats / positive, red for "
            "misses / negative when relevant."
        ),
    )


class PlotComparisonArgs(BaseModel):
    symbols: list[str] = Field(
        default_factory=list,
        description="Tickers from {db_schema}.analytics.",
    )
    fred_series: list[str] = Field(
        default_factory=list,
        description="FRED series IDs (e.g. 'VIXCLS', 'DGS10').",
    )
    backtest_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Backtest result IDs ({backtest_schema}.backtest_results.id) to overlay "
            "their equity_curve. Great for 'my strategy vs SPY'."
        ),
    )
    start: str = Field(..., description="Start date YYYY-MM-DD.")
    end: str = Field(..., description="End date YYYY-MM-DD.")
    mode: ChartMode = Field(
        default="normalized",
        description=(
            "'normalized' (default): rebase to 100 at start, single axis. 'absolute': "
            "raw values, single axis — same-scale series only. 'dual_axis': two Y-axes, "
            "native scales — exactly 2 series."
        ),
    )
    chart_type: ChartType = Field(
        default="line",
        description=(
            "'line' (default) or 'candlestick'. Candlestick requires exactly 1 symbol, "
            "0 fred_series, 0 backtest_ids, and cannot be combined with mode='normalized'."
        ),
    )
    title: str | None = Field(default=None, description="Chart title (auto if omitted).")
    moving_averages: list[int] = Field(
        default_factory=list,
        description="SMA periods to overlay on each symbol's price, e.g. [50, 200].",
    )
    ema_periods: list[int] = Field(
        default_factory=list,
        description="EMA periods to overlay on each symbol's price, e.g. [20].",
    )
    bollinger: bool = Field(
        default=False,
        description="If True, overlay 20-period / 2σ Bollinger Bands on the first symbol.",
    )
    horizontal_lines: list[float] = Field(
        default_factory=list,
        description="Constant Y values to draw as dashed horizontal lines (support/resistance/alerts).",
    )
    indicators: list[Indicator] = Field(
        default_factory=list,
        description=(
            "Stacked indicator subplots (below main chart) for the FIRST symbol. "
            "Order defines top-to-bottom layout. Options: 'rsi' (14-period with 30/70 "
            "reference lines), 'macd' (12/26/9 with signal + histogram), 'volume' "
            "(bars, color-coded up/down), 'atr' (14-period true range)."
        ),
    )
    log_scale: bool = Field(
        default=False,
        description="Log Y-axis on main chart. Only sensible with mode='absolute' and positive values.",
    )
    events: list[ChartEvent] = Field(
        default_factory=list,
        description=(
            "Calendar events to annotate on the main chart — earnings dates, revenue "
            "beats/misses, FOMC meetings, news items, insider filings, anything "
            "date-specific. Each event is either a vertical line (style='line') or a "
            "point marker anchored to a symbol's price (style='marker')."
        ),
    )


@tool(
    description=(
        "Render a PNG chart of stock tickers, FRED macro series, and/or backtest "
        "equity curves. Save to `charts/` for the user to download. Supports "
        "overlays (SMA/EMA moving averages, Bollinger Bands, horizontal support/"
        "resistance lines), indicator subplots (RSI, MACD, volume, ATR — computed "
        "on the first symbol), candlestick mode, log scale, three mixing modes "
        "(normalized / absolute / dual_axis), and date-anchored `events` for "
        "earnings dates, revenue beats/misses, news items, or any other calendar "
        "annotation (vertical lines or point markers on a symbol's price). Call "
        "whenever the user asks to plot, chart, visualize, or compare price or "
        "indicator history. Always include the returned path in your answer so "
        "the user can open the image. Charts older than 7 days are auto-cleaned."
    )
)
def plot_comparison(args: PlotComparisonArgs) -> dict[str, Any]:
    if not args.symbols and not args.fred_series and not args.backtest_ids:
        return {"error": "Provide at least one of symbols, fred_series, or backtest_ids."}

    # Fetch symbols (full OHLCV — needed for candlestick + indicators).
    symbol_data: dict[str, dict[str, list]] = {}
    for sym in args.symbols:
        try:
            d = _fetch_symbol_ohlcv(sym, args.start, args.end)
        except Exception as e:
            log.warning("Could not fetch %s: %s", sym, e)
            return {"error": f"Failed fetching {sym}: {type(e).__name__}: {e}"}
        if d is None:
            return {"error": f"No price data for {sym} in {args.start}..{args.end}."}
        symbol_data[sym.upper()] = d

    # Fetch FRED series.
    fred_data: dict[str, list[tuple[date, float]]] = {}
    for sid in args.fred_series:
        try:
            d2 = _fetch_fred_series(sid, args.start, args.end)
        except Exception as e:
            log.warning("Could not fetch FRED %s: %s", sid, e)
            return {"error": f"Failed fetching FRED {sid}: {type(e).__name__}: {e}"}
        if not d2:
            return {"error": f"No FRED data for {sid} in {args.start}..{args.end}."}
        fred_data[sid] = d2

    # Fetch backtest equity curves.
    backtest_data: dict[str, list[tuple[date, float]]] = {}
    for bid in args.backtest_ids:
        try:
            eq = _fetch_backtest_equity(bid, args.start, args.end)
        except Exception as e:
            log.warning("Could not fetch backtest %d: %s", bid, e)
            return {"error": f"Failed fetching backtest {bid}: {type(e).__name__}: {e}"}
        if not eq:
            return {"error": f"No equity_curve for backtest {bid} in {args.start}..{args.end}."}
        backtest_data[f"Backtest #{bid}"] = eq

    # Constraint checks.
    total_series = len(symbol_data) + len(fred_data) + len(backtest_data)
    if args.mode == "dual_axis" and total_series != 2:
        return {
            "error": (
                f"mode='dual_axis' requires exactly 2 series; got {total_series}. "
                "Use 'normalized' for more."
            )
        }
    if args.chart_type == "candlestick":
        if len(symbol_data) != 1 or fred_data or backtest_data:
            return {
                "error": (
                    "chart_type='candlestick' requires exactly 1 symbol and no "
                    "FRED series or backtests."
                )
            }
        if args.mode == "normalized":
            return {
                "error": (
                    "chart_type='candlestick' is incompatible with mode='normalized'. "
                    "Use mode='absolute'."
                )
            }

    # Build named_series for date alignment (use close for symbols).
    named_series: list[tuple[str, list[tuple[date, float]]]] = []
    for name, sd in symbol_data.items():
        named_series.append((name, list(zip(sd["dates"], sd["close"], strict=False))))
    for name, fs in fred_data.items():
        named_series.append((name, fs))
    for name, bd in backtest_data.items():
        named_series.append((name, bd))

    dates_sorted, aligned = _align_on_common_dates(named_series)
    if not dates_sorted:
        return {"error": "No overlapping dates across the requested series."}

    # Figure layout: main + N indicator subplots.
    main_height = 5.5
    subplot_height = 1.7
    fig_h = main_height + len(args.indicators) * subplot_height
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = _cleanup_stale_charts()

    fig = plt.figure(figsize=(11, fig_h))
    total_rows = 1 + len(args.indicators)
    height_ratios = [3] + [1] * len(args.indicators)
    gs = fig.add_gridspec(total_rows, 1, height_ratios=height_ratios, hspace=0.08)
    main_ax = fig.add_subplot(gs[0])
    subplot_axes: list[plt.Axes] = [
        fig.add_subplot(gs[i + 1], sharex=main_ax) for i in range(len(args.indicators))
    ]

    # --- Main chart ---
    if args.chart_type == "candlestick":
        first_name, sd = next(iter(symbol_data.items()))
        # Inner-joined dates means we use the symbol's own dates subset.
        d_set = set(dates_sorted)
        idx = [i for i, d in enumerate(sd["dates"]) if d in d_set]
        _draw_candlesticks(
            main_ax,
            [sd["dates"][i] for i in idx],
            [sd["open"][i] for i in idx],
            [sd["high"][i] for i in idx],
            [sd["low"][i] for i in idx],
            [sd["close"][i] for i in idx],
        )
        main_ax.set_ylabel(f"{first_name} (candlestick)")
    elif args.mode == "dual_axis":
        names = list(aligned.keys())
        name_a, name_b = names[0], names[1]
        main_ax.plot(
            dates_sorted,  # type: ignore[arg-type]
            aligned[name_a],
            label=name_a,
            color=DUAL_AXIS_LEFT_COLOR,
            linewidth=1.6,
        )
        main_ax.set_ylabel(name_a, color=DUAL_AXIS_LEFT_COLOR)
        main_ax.tick_params(axis="y", colors=DUAL_AXIS_LEFT_COLOR)
        ax2 = main_ax.twinx()
        ax2.plot(
            dates_sorted,  # type: ignore[arg-type]
            aligned[name_b],
            label=name_b,
            color=QUANTARA_PURPLE,
            linewidth=1.6,
        )
        ax2.set_ylabel(name_b, color=QUANTARA_PURPLE)
        ax2.tick_params(axis="y", colors=QUANTARA_PURPLE)
        ax2.spines["top"].set_visible(False)
        la, labels_a = main_ax.get_legend_handles_labels()
        lb, labels_b = ax2.get_legend_handles_labels()
        main_ax.legend(la + lb, labels_a + labels_b, loc="best", fontsize=9)
    else:
        if args.mode == "normalized":
            plotted = {name: _normalize(vals) for name, vals in aligned.items()}
            main_ax.set_ylabel("Index (start = 100)")
        else:  # absolute
            plotted = dict(aligned)
            main_ax.set_ylabel("Value")
        for name, values in plotted.items():
            main_ax.plot(dates_sorted, values, label=name, linewidth=1.6)  # type: ignore[arg-type]
        main_ax.legend(loc="best", fontsize=9)

    def _rescaled(vals: list[float | None], base: float) -> Any:
        # Return an ndarray (mypy-friendly for matplotlib.plot) with NaN
        # sentinels where values are missing.
        import math

        if args.mode == "normalized" and base:
            return [(v / base * 100) if v is not None else math.nan for v in vals]
        return [v if v is not None else math.nan for v in vals]

    # --- Overlays (applied only if we have symbol_data) ---
    if symbol_data and args.chart_type == "line":
        for sym_name, sd in symbol_data.items():
            closes = sd["close"]
            base = closes[0] if args.mode == "normalized" and closes else 1.0
            for p in args.moving_averages:
                main_ax.plot(
                    sd["dates"],
                    _rescaled(_sma(closes, p), base),
                    label=f"{sym_name} SMA({p})",
                    linewidth=0.9,
                    alpha=0.75,
                    linestyle="--",
                )
            for p in args.ema_periods:
                main_ax.plot(
                    sd["dates"],
                    _rescaled(_ema(closes, p), base),
                    label=f"{sym_name} EMA({p})",
                    linewidth=0.9,
                    alpha=0.75,
                    linestyle=":",
                )
    if args.bollinger and symbol_data:
        first_name, sd = next(iter(symbol_data.items()))
        closes = sd["close"]
        base = closes[0] if args.mode == "normalized" and closes else 1.0
        upper, mid, lower = _bollinger(closes)
        xs = sd["dates"]
        main_ax.plot(
            xs,
            _rescaled(upper, base),
            color=QUANTARA_PURPLE_LIGHT,
            linewidth=0.8,
            alpha=0.7,
            label=f"{first_name} BB upper",
        )
        main_ax.plot(
            xs,
            _rescaled(lower, base),
            color=QUANTARA_PURPLE_LIGHT,
            linewidth=0.8,
            alpha=0.7,
            label=f"{first_name} BB lower",
        )
        main_ax.plot(
            xs,
            _rescaled(mid, base),
            color=QUANTARA_PURPLE,
            linewidth=0.6,
            alpha=0.5,
            linestyle="-.",
            label=f"{first_name} BB mid",
        )
    for y in args.horizontal_lines:
        main_ax.axhline(y, color="#666666", linestyle=":", linewidth=0.8, alpha=0.6)

    events_drawn = _draw_events(main_ax, args.events, symbol_data, args.mode)

    if args.log_scale:
        main_ax.set_yscale("log")

    # Legend may now be bigger; redraw once to keep it tidy.
    handles, labels = main_ax.get_legend_handles_labels()
    if handles:
        main_ax.legend(handles, labels, loc="best", fontsize=8)

    # --- Indicator subplots (first symbol only) ---
    if args.indicators:
        if not symbol_data:
            plt.close(fig)
            return {"error": "Indicator subplots require at least one symbol."}
        first_name, sd = next(iter(symbol_data.items()))
        xs = sd["dates"]
        for ax_i, ind in zip(subplot_axes, args.indicators, strict=False):
            if ind == "rsi":
                _draw_rsi(ax_i, xs, sd["close"])
            elif ind == "macd":
                _draw_macd(ax_i, xs, sd["close"])
            elif ind == "volume":
                _draw_volume(ax_i, xs, sd["open"], sd["close"], sd["volume"])
            elif ind == "atr":
                _draw_atr(ax_i, xs, sd["high"], sd["low"], sd["close"])

        # Hide x-tick labels on all but the last subplot to reduce clutter.
        for ax_i in subplot_axes[:-1]:
            plt.setp(ax_i.get_xticklabels(), visible=False)

    # Title + x-axis formatting.
    title = args.title or (" vs ".join(aligned.keys()))
    subtitle_map = {
        "normalized": "rebased to 100 at start",
        "absolute": "absolute values",
        "dual_axis": "dual axis, native scales",
    }
    subtitle = subtitle_map[args.mode]
    if args.chart_type == "candlestick":
        subtitle = "candlestick"
    main_ax.set_title(f"{title}\n({subtitle}, {args.start} → {args.end})", fontsize=12)

    bottom_ax = subplot_axes[-1] if subplot_axes else main_ax
    bottom_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    # `tight_layout` warns when twinx or custom patches are present; the chart
    # still renders correctly, so silence the cosmetic warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*not compatible with tight_layout.*",
        )
        fig.tight_layout()
    _apply_branding(fig, main_ax, subplot_axes)

    # Save.
    slug = _slugify(title)
    key_parts = [
        args.start,
        args.end,
        "-".join(aligned.keys()),
        args.mode,
        args.chart_type,
        ",".join(str(p) for p in args.moving_averages),
        ",".join(str(p) for p in args.ema_periods),
        str(args.bollinger),
        ",".join(args.indicators),
        ",".join(str(x) for x in args.horizontal_lines),
        str(args.log_scale),
        ";".join(
            f"{e.date}|{e.style}|{e.symbol or ''}|{e.price or ''}|{e.label or ''}|{e.color or ''}"
            for e in args.events
        ),
    ]
    fp = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:8]
    filename = f"{date.today().isoformat()}_{slug}_{fp}.png"
    abs_path = CHARTS_DIR / filename
    fig.savefig(abs_path, dpi=110)
    plt.close(fig)

    return {
        "path": str(abs_path),
        "relative_path": f"charts/{filename}",
        "filename": filename,
        "symbols": list(symbol_data.keys()),
        "fred_series": list(fred_data.keys()),
        "backtest_ids": list(args.backtest_ids),
        "date_range": {"start": args.start, "end": args.end},
        "mode": args.mode,
        "chart_type": args.chart_type,
        "overlays": {
            "moving_averages": list(args.moving_averages),
            "ema_periods": list(args.ema_periods),
            "bollinger": args.bollinger,
            "horizontal_lines": list(args.horizontal_lines),
        },
        "indicators": list(args.indicators),
        "log_scale": args.log_scale,
        "events_drawn": events_drawn,
        "rows_plotted": len(dates_sorted),
        "title": title,
        "charts_cleaned": cleaned,
    }
