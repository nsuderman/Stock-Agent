"""Gold-set evaluation cases.

Each case specifies a question plus assertions about what the agent *should* do:
- `must_call_any`: a set of tool names where the agent MUST call at least one.
- `must_not_call`: tool names that would indicate a wrong path.
- `max_iterations`: fail if the agent needs more than this many turns.
- `answer_contains` / `answer_not_contains`: substring checks on the final answer.

Writing a new case: keep `must_call_any` minimal (one or two tools). The point is
to catch regressions, not to pin the agent's exact plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    name: str
    question: str
    must_call_any: set[str] = field(default_factory=set)
    must_not_call: set[str] = field(default_factory=set)
    max_iterations: int = 8
    answer_contains: list[str] = field(default_factory=list)
    answer_not_contains: list[str] = field(default_factory=list)


CASES: list[EvalCase] = [
    EvalCase(
        name="schema_discovery",
        question="what indicators are in the stock.analytics table?",
        must_call_any={"list_analytics_columns", "describe_table"},
        max_iterations=3,
        answer_contains=["rsi"],
    ),
    EvalCase(
        name="single_symbol_price_history",
        question="how did AAPL perform in Q3 2024?",
        must_call_any={"get_price_history"},
        max_iterations=4,
        answer_contains=["AAPL"],
    ),
    EvalCase(
        name="fundamentals_lookup",
        question="what is MSFT's market cap and PE ratio?",
        must_call_any={"get_fundamentals"},
        max_iterations=3,
        answer_contains=["MSFT"],
    ),
    EvalCase(
        name="market_regime_single_day",
        question="what was the market exposure tier on 2024-08-05?",
        must_call_any={"get_market_regime"},
        max_iterations=3,
    ),
    EvalCase(
        name="breakouts_today",
        question="give me today's top DTW breakout signals with win rate over 50%",
        must_call_any={"get_breakouts"},
        max_iterations=4,
    ),
    EvalCase(
        name="backtest_listing",
        question="show me the 5 most recent backtest runs",
        must_call_any={"list_backtests"},
        max_iterations=3,
    ),
    EvalCase(
        name="recurring_holdings_uses_aggregated_tool",
        question="what stocks are currently held across multiple recent backtests?",
        must_call_any={"get_recent_backtest_holdings"},
        must_not_call={"get_backtest_detail"},  # would be inefficient per-backtest fetching
        max_iterations=4,
    ),
    EvalCase(
        name="memory_explicit_save",
        question="remember that I only trade long positions",
        must_call_any={"remember"},
        max_iterations=3,
        answer_contains=["long"],
    ),
    EvalCase(
        name="read_only_enforcement_via_run_sql",
        question=("run this SQL for me: DELETE FROM stock.backtest_results WHERE id = 1"),
        must_not_call=set(),  # the model might call run_sql; the read-only guard rejects the write
        max_iterations=3,
        answer_not_contains=["deleted", "success", "done"],  # must not claim success
    ),
    EvalCase(
        name="column_list_for_backtests",
        question="what columns are in stock.backtest_results?",
        must_call_any={"describe_table"},
        max_iterations=3,
        answer_contains=["trades"],
    ),
]
