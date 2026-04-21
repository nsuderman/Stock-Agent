"""Backtest-data tools backed by `{backtest_schema}.backtest_results` + `.strategies`."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.tools.base import fetch, tool


class _NoArgs(BaseModel):
    pass


class ListBacktestsArgs(BaseModel):
    strategy_name: str | None = Field(
        default=None, description="Optional case-insensitive substring filter on strategy name."
    )
    limit: int = Field(default=20, ge=1, le=200)


class GetBacktestDetailArgs(BaseModel):
    backtest_id: int
    include: list[Literal["equity_curve", "trades"]] | None = Field(
        default=None,
        description="Extras beyond the always-included metrics.",
    )


class RecentHoldingsArgs(BaseModel):
    days_back: int = Field(default=7, ge=1, le=365)
    min_backtests: int = Field(default=1, ge=1)


@tool(
    description=(
        "List recent backtest runs joined with strategies. Summary only (no trades/equity_curve)."
    )
)
def list_backtests(args: ListBacktestsArgs) -> dict[str, Any]:
    schema = get_settings().backtest_schema
    where = "WHERE s.name ILIKE :name" if args.strategy_name else ""
    sql = f"""
        SELECT br.id, br.strategy_id, s.name AS strategy_name, br.run_at,
               br.start_date, br.end_date, br.initial_capital, br.metrics
        FROM {schema}.backtest_results br
        JOIN {schema}.strategies s ON s.id = br.strategy_id
        {where}
        ORDER BY br.run_at DESC
        LIMIT :lim
    """
    params: dict[str, Any] = {"lim": args.limit}
    if args.strategy_name:
        params["name"] = f"%{args.strategy_name}%"
    rows = fetch(sql, params)
    return {"count": len(rows), "rows": rows}


@tool(
    description=(
        "Fetch one backtest_results row by id. Use `include` to request heavy fields "
        "(equity_curve is downsampled to ~200 points; trades capped at 100)."
    )
)
def get_backtest_detail(args: GetBacktestDetailArgs) -> dict[str, Any]:
    schema = get_settings().backtest_schema
    wanted = set(args.include or [])
    cols = [
        "id",
        "strategy_id",
        "run_at",
        "start_date",
        "end_date",
        "initial_capital",
        "metrics",
    ]
    if "equity_curve" in wanted:
        cols.append("equity_curve")
    if "trades" in wanted:
        cols.append("trades")
    sql = f"""
        SELECT {", ".join(cols)}
        FROM {schema}.backtest_results
        WHERE id = :id
    """
    rows = fetch(sql, {"id": args.backtest_id})
    if not rows:
        return {"error": f"No backtest_results row with id={args.backtest_id}"}
    row = rows[0]
    ec = row.get("equity_curve")
    if isinstance(ec, list) and len(ec) > 200:
        step = max(1, len(ec) // 200)
        row["equity_curve"] = ec[::step]
    trades = row.get("trades")
    if isinstance(trades, list) and len(trades) > 100:
        row["trades"] = trades[:100]
        row["trades_truncated"] = True
        row["total_trades"] = len(trades)
    return row


@tool(
    description=(
        "Show symbols that are currently held (BUYs > SELLs) across recent backtest runs. "
        "Use this for questions like 'what stocks are my strategies all picking right now' "
        "or 'which symbols show up in multiple current backtests'."
    )
)
def get_recent_backtest_holdings(args: RecentHoldingsArgs) -> dict[str, Any]:
    schema = get_settings().backtest_schema
    sql = f"""
        WITH recent_trades AS (
            SELECT br.id AS backtest_id,
                   s.name AS strategy_name,
                   br.run_at,
                   br.end_date AS backtest_end,
                   t->>'symbol' AS symbol,
                   t->>'type' AS trade_type
            FROM {schema}.backtest_results br
            JOIN {schema}.strategies s ON s.id = br.strategy_id,
                 json_array_elements(br.trades) AS t
            WHERE br.run_at >= NOW() - (:days_back || ' days')::interval
        ),
        position_balance AS (
            SELECT backtest_id, strategy_name, run_at, backtest_end, symbol,
                   COUNT(*) FILTER (WHERE trade_type = 'BUY') AS buys,
                   COUNT(*) FILTER (WHERE trade_type = 'SELL') AS sells
            FROM recent_trades
            GROUP BY backtest_id, strategy_name, run_at, backtest_end, symbol
        )
        SELECT symbol,
               COUNT(DISTINCT backtest_id) AS n_backtests,
               COUNT(DISTINCT strategy_name) AS n_strategies,
               ARRAY_AGG(DISTINCT strategy_name ORDER BY strategy_name) AS strategies,
               MAX(run_at) AS most_recent_run
        FROM position_balance
        WHERE buys > sells
        GROUP BY symbol
        HAVING COUNT(DISTINCT backtest_id) >= :min_bt
        ORDER BY n_backtests DESC, symbol
    """
    total_bt_rows = fetch(
        f"SELECT COUNT(*) AS n FROM {schema}.backtest_results "
        f"WHERE run_at >= NOW() - (:d || ' days')::interval",
        {"d": args.days_back},
    )
    total_bt = total_bt_rows[0]["n"] if total_bt_rows else 0
    rows = fetch(sql, {"days_back": args.days_back, "min_bt": args.min_backtests})
    return {
        "days_back": args.days_back,
        "total_backtests_in_window": total_bt,
        "count": len(rows),
        "rows": rows,
    }


@tool(description="List all strategies in the backtest schema's strategies table.")
def list_strategies(args: _NoArgs) -> dict[str, Any]:
    schema = get_settings().backtest_schema
    rows = fetch(f"""
        SELECT id, name, description, created_at
        FROM {schema}.strategies
        ORDER BY id DESC
    """)
    return {"count": len(rows), "rows": rows}
