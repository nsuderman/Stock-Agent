# CLAUDE.md — guidance for Claude Code on this project

A local-LLM ReAct agent that queries a PostgreSQL stock/backtest DB. Entry
point is `ask` (or `python -m agent`); core loop is `agent/loop.py`; tools
are under `agent/tools/`.

## Mental model

A single `run_agent(question)` invocation:

1. Builds a fresh system prompt (`agent/prompt.py:build_system_prompt()`) that
   injects today's date + current `memory.md` contents.
2. Prepends it to any `prior_messages` (session resume) and appends the user
   question.
3. Loops up to `max_iterations` (default 12):
   a. Compact if needed (Stage 1 tool-result trim, then Stage 2 LLM summary).
   b. Stream one LLM turn (`_stream_turn`), accumulating content deltas +
      tool_call deltas.
   c. If no tool_calls, return the content as the final answer.
   d. Otherwise dispatch each tool call via `agent.tools.invoke_tool`.
      Duplicate calls (md5 fingerprint of name + canonical args) return a
      synthetic "you just ran this" error without re-executing.
4. Returns `(answer, messages)`. `agent/cli.py:main` persists `messages` if a
   session is active.

## Sessions

- Default session name = today's ISO date (`2026-04-21`). `agent/cli.py`
  computes this if `--session` isn't passed.
- `--session <name>` overrides (long-running named projects).
- `--no-session` skips session load/save entirely.
- `memory.md` is global — independent of session, loaded fresh each
  invocation via `build_system_prompt()`.

## Layout

```
agent/agent/              the installable package
├── __init__.py
├── __main__.py           python -m agent
├── cli.py                CLI entry
├── loop.py               ReAct loop, streaming, dedup guard
├── compaction.py         stage 1/2 compaction + ThinkFilter
├── config.py             Pydantic Settings
├── db.py                 read-only PG engine
├── llm.py                OpenAI client + /v1/models probe
├── logging_setup.py
├── prompt.py
├── session.py
└── tools/
    ├── base.py           @tool decorator + ToolEntry + fetch helper
    ├── market.py         stock.analytics + symbols_info + regime + breakouts
    ├── backtest.py       backtest_results + strategies + holdings
    ├── db_meta.py        describe_table, sample_rows, list_analytics_columns
    ├── memory.py         remember()
    └── sql.py            run_sql escape hatch
tests/                    pytest unit tests
evals/                    gold-set regression harness
```

## Conventions and non-obvious decisions

- **Tools use Pydantic for args.** Define a `BaseModel` per tool and annotate
  the function's single parameter. `@tool(description=...)` generates the
  OpenAI schema from the model automatically. No hand-written JSON schemas.
- **LLM-visible arg names can differ from Python attribute names.** E.g.
  `describe_table` uses `db_schema` internally but exposes `schema` via
  Pydantic `Field(alias="schema")` + `ConfigDict(populate_by_name=True)`. This
  is required because `schema` is a BaseModel method and collides.
- **DB is read-only** at the session level via `default_transaction_read_only=on`
  in the PG connect options. `run_sql` also regex-checks for write keywords
  as defense-in-depth. Never relax either layer without user approval.
- **All stock + backtest data lives in the `stock` schema.** If you ever see
  `dev.*` references, they're stale.
- **`stock.backtest_results.trades` is `json`, not `jsonb`.** Use
  `json_array_elements(...)`. Documented in `agent/prompt.py`.
- **Trade shape** is `{date, symbol, type, price, quantity, value}` with
  `type ∈ {'BUY','SELL'}`. `get_recent_backtest_holdings` relies on this.

## Streaming + the `<think>` tag problem

qwen3.6-35b-a3b emits `<think>...</think>` blocks as part of its reasoning,
often empty. `agent/compaction.py:ThinkFilter` is a chunk-stateful stripper
that hides them during streaming AND the `THINK_RE` regex strips them from
stored message content before appending to history.

## Compaction

`compact_if_needed` runs before every LLM round. It queries
`get_context_window(client, model, local=...)`, computes
`budget = window * COMPACT_AT - MAX_RESPONSE_TOKENS`, and if the estimated
token count (char/4 heuristic) exceeds the budget, runs Stage 1 then Stage 2.

- **Stage 1**: walks messages from oldest to newest, replaces each `tool`
  role message's `content` with a one-line summary. Skips the system message
  and the last `COMPACT_KEEP_RECENT` messages.
- **Stage 2**: makes a single LLM call asking for a ≤250-word summary of the
  old block, replaces that block with one synthetic assistant message.

### Context-window probe

`get_context_window` resolution order:
1. Target model's own `--ctx-size` in `status.args`.
2. Standard fields (`context_length`, `max_model_len`, `n_ctx`, `context_window`).
3. **Alias match** — another model entry with the same `--model <path>` that
   has `--ctx-size` set. This is how `qwen3.6-35b-a3b` (no explicit flag)
   resolves to 262144 from its `-ud-iq4_xs` sibling.
4. `LOCAL/REMOTE_CONTEXT_WINDOW` env var.
5. Hardcoded default.

Results are cached per-process by `(local, model_id)`.

Tunables:

| Env | Default | Effect |
|---|---|---|
| `COMPACT_AT` | `0.75` | Trigger threshold (fraction of effective budget). |
| `COMPACT_KEEP_RECENT` | `4` | Messages at the tail always kept verbatim. |
| `MAX_RESPONSE_TOKENS` | `4096` | Reserved for reply; subtracted from window. |
| `LOCAL_CONTEXT_WINDOW` | `32768` | Fallback when probe fails (local). |
| `REMOTE_CONTEXT_WINDOW` | `128000` | Fallback when probe fails (`--remote`). |

## Duplicate-call guard

`_fingerprint(name, raw_args)` = md5 of `"{name}|{canonical_json_args}"`. A
`deque(maxlen=4)` tracks the last 4 executed tool calls. If a fingerprint
re-appears, the tool is NOT executed; the model gets a synthetic
`{"error": "Duplicate call: ..."}` telling it to try something different or
provide the final answer. This busted a real infinite loop the agent hit
before the guard existed.

## Adding a new tool

1. Create a Pydantic `BaseModel` subclass in the appropriate module under
   `agent/tools/` (see existing tools for examples).
2. Annotate the function signature: `def my_tool(args: MyArgs) -> dict[str, Any]`.
3. Decorate with `@tool(description="...")`. The tool is auto-registered.
4. Return a JSON-serializable dict. Catch domain errors and return
   `{"error": "..."}` rather than raising.
5. Add unit tests under `tests/` (use the `mock_fetch` / `mock_inspect`
   fixtures in `test_tools_impl.py` — do NOT hit the real DB in unit tests).
6. If the tool changes how the agent should think about a class of questions,
   update `_BASE` in `agent/prompt.py`.
7. Consider adding an eval case in `evals/cases.py`.

## Tool result payload limits

- `get_backtest_detail` downsamples `equity_curve` to ~200 points and caps
  `trades` at 100.
- `run_sql` caps rows at 500 by default; agent can override via `limit`.
- `MAX_TOOL_RESULT_CHARS = 8000` in `agent/loop.py` truncates any single tool
  result in the message sent to the LLM.

## User preferences I've observed

- Prefers concise output with concrete numbers/dates over prose.
- Wants streaming progress so runs aren't silent.
- Treats the database as sacred — read-only enforcement is a hard requirement.
- Iterative/exploratory style — confirm approach with short recommendations
  before building, then build in small verifiable chunks.

## Things to run after making changes

```bash
ruff check .          # lint
ruff format .         # format
mypy agent            # types
pytest                # unit tests (should be <1s)
pytest --cov=agent    # with coverage (target ≥85%)
python -m evals       # evals (slow — hits live LLM + DB)
```

Everything above runs in CI on every push. Keep it green.
