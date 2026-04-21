"""Stock/market data tools backed by the `stock` schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import inspect

from agent.config import get_settings
from agent.db import get_engine
from agent.tools.base import contains_write_keyword, fetch, tool


class PriceHistoryArgs(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g. 'AAPL'.")
    start: str = Field(..., description="Start date YYYY-MM-DD.")
    end: str = Field(..., description="End date YYYY-MM-DD.")
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Optional analytics columns. Defaults to OHLCV + RSI/SMAs/ATR/returns. "
            "Call list_analytics_columns first if unsure."
        ),
    )


class FundamentalsArgs(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g. 'AAPL'.")


class MarketRegimeArgs(BaseModel):
    start_date: str = Field(..., description="YYYY-MM-DD.")
    end_date: str | None = Field(default=None, description="Optional YYYY-MM-DD for a range.")


class BreakoutsArgs(BaseModel):
    target_date: str = Field(..., description="YYYY-MM-DD.")
    min_win_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    min_match_count: int = Field(default=3, ge=1)
    window_lookback: int = Field(default=10, ge=1)
    min_gain_threshold: float = Field(default=0.10, ge=0.0)


class ScreenSymbolsArgs(BaseModel):
    where_clause: str = Field(..., description="SQL fragment without WHERE keyword.")
    order_by: str = Field(
        default="a.rs_value DESC NULLS LAST",
        description="SQL fragment without ORDER BY keyword.",
    )
    limit: int = Field(default=25, ge=1, le=100)


_DEFAULT_PRICE_COLS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi",
    "sma_50",
    "sma_200",
    "atr",
    "returns_day",
]


@tool(
    description="Fetch OHLCV + indicator columns for one symbol over a date range from stock.analytics."
)
def get_price_history(args: PriceHistoryArgs) -> dict[str, Any]:
    schema = get_settings().db_schema
    valid = {c["name"] for c in inspect(get_engine()).get_columns("analytics", schema=schema)}
    picked = [c for c in (args.columns or _DEFAULT_PRICE_COLS) if c in valid]
    if "date" not in picked:
        picked.insert(0, "date")
    col_list = ", ".join(picked)
    sql = f"""
        SELECT {col_list}
        FROM {schema}.analytics
        WHERE symbol = :sym AND date BETWEEN :start AND :end
        ORDER BY date
    """
    rows = fetch(sql, {"sym": args.symbol.upper(), "start": args.start, "end": args.end})
    return {"symbol": args.symbol.upper(), "count": len(rows), "rows": rows}


@tool(
    description="Fetch the symbols_info row for one symbol (market cap, PE, margins, analyst targets, etc.)."
)
def get_fundamentals(args: FundamentalsArgs) -> dict[str, Any]:
    schema = get_settings().db_schema
    rows = fetch(
        f"SELECT * FROM {schema}.symbols_info WHERE symbol = :sym",
        {"sym": args.symbol.upper()},
    )
    return rows[0] if rows else {"error": f"No fundamentals for {args.symbol}"}


@tool(
    description=(
        "Query market_exposure view for regime classification. Pass just start_date for "
        "a single day, or start_date + end_date for a range."
    )
)
def get_market_regime(args: MarketRegimeArgs) -> dict[str, Any]:
    schema = get_settings().db_schema
    if args.end_date:
        sql = f"""
            SELECT date, close, ema_10, pct_diff, exposure_tier, bar_rank
            FROM {schema}.market_exposure
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """
        rows = fetch(sql, {"start": args.start_date, "end": args.end_date})
    else:
        sql = f"""
            SELECT date, close, ema_10, pct_diff, exposure_tier, bar_rank
            FROM {schema}.market_exposure
            WHERE date = :d
        """
        rows = fetch(sql, {"d": args.start_date})
    return {"count": len(rows), "rows": rows}


@tool(
    description="Get DTW pattern breakout signals as of a target date via stock.get_live_breakouts()."
)
def get_breakouts(args: BreakoutsArgs) -> dict[str, Any]:
    schema = get_settings().db_schema
    sql = f"""
        SELECT *
        FROM {schema}.get_live_breakouts(
            CAST(:d AS date), :lookback, :gain, :matches, true, :wr
        )
        ORDER BY match_count DESC, win_rate DESC
    """
    rows = fetch(
        sql,
        {
            "d": args.target_date,
            "lookback": args.window_lookback,
            "gain": args.min_gain_threshold,
            "matches": args.min_match_count,
            "wr": args.min_win_rate,
        },
        limit=100,
    )
    return {"target_date": args.target_date, "count": len(rows), "rows": rows}


@tool(
    description=(
        "Rank/filter the universe by the latest row per symbol joined with symbols_info. "
        "`where_clause` and `order_by` are raw SQL fragments (no WHERE/ORDER BY keyword). "
        "Aliases: `a`=analytics, `s`=symbols_info. Example where: "
        "`a.close > a.sma_200 AND s.market_cap > 1e9 AND s.quote_type = 'EQUITY'`. "
        "Example order_by: `a.suder_momentum DESC NULLS LAST`."
    )
)
def screen_symbols(args: ScreenSymbolsArgs) -> dict[str, Any]:
    schema = get_settings().db_schema
    if contains_write_keyword(args.where_clause) or contains_write_keyword(args.order_by):
        return {"error": "Write keywords not allowed in where/order clauses."}
    sql = f"""
        WITH latest AS (
            SELECT DISTINCT ON (symbol) *
            FROM {schema}.analytics
            ORDER BY symbol, date DESC
        )
        SELECT a.symbol, a.date, a.close, a.sma_50, a.sma_200,
               a.rsi, a.rs_value, a.rs_grade, a.suder_momentum,
               s.name, s.sector, s.industry, s.market_cap,
               s.trailing_pe, s.quote_type
        FROM latest a
        JOIN {schema}.symbols_info s ON s.symbol = a.symbol
        WHERE {args.where_clause}
        ORDER BY {args.order_by}
        LIMIT :lim
    """
    rows = fetch(sql, {"lim": args.limit})
    return {"count": len(rows), "rows": rows}
