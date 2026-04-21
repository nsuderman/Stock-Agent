# Evals

Gold-set of questions the agent should answer correctly. Each case asserts
which tools should be called and what the answer should contain.

## Running

```bash
# All cases (hits the live LLM + DB — slow)
python -m evals

# One case
python -m evals --filter breakouts

# Machine-readable output
python -m evals --json > eval-report.json
```

## Adding a case

See `cases.py`. Keep assertions minimal — the goal is to catch regressions
(e.g. "agent forgot about get_recent_backtest_holdings and tried N individual
backtest fetches"), not to pin down the exact tool plan.

Good assertions:
- `must_call_any={"get_recent_backtest_holdings"}` — the right tool for this question
- `must_not_call={"get_backtest_detail"}` — the wrong, inefficient tool
- `answer_contains=["AAPL"]` — the answer must mention the symbol it asked about
- `answer_not_contains=["deleted"]` — the agent must not falsely claim a write succeeded

Bad assertions:
- `answer_contains=["exactly these 17 words"]` — too brittle; LLM phrasing varies.
- `must_call_any={"run_sql"}` — too narrow; the agent has many valid paths.
