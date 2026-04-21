# Contributing

Thanks for your interest in contributing.

## Development setup

```bash
git clone git@github.com:nsuderman/Stock-Agent.git
cd Stock-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env  # then edit .env with real values
```

## Running checks locally

```bash
# Lint + format
ruff check .
ruff format .

# Types
mypy agent

# Tests (unit only — fast)
pytest

# Tests (include integration — hits the real DB + LLM)
pytest -m "integration"

# Tests with coverage
pytest --cov=agent --cov-report=term-missing
```

Pre-commit runs `ruff`, `mypy`, and file hygiene checks on every commit.

## Adding a tool

1. Define a Pydantic `BaseModel` subclass for arguments in the appropriate
   module under `agent/tools/` (see existing tools for examples).
2. Register the tool with the `@tool(...)` decorator — it auto-generates the
   OpenAI schema from the model.
3. Return a JSON-serializable dict. Catch domain errors and return
   `{"error": "..."}` rather than raising.
4. Add unit tests in `tests/tools/` (mock the SQLAlchemy engine; use real SQL
   only in integration tests).

## Style

- Ruff enforces formatting + a broad lint set (see `pyproject.toml`).
- Type hints are required on all public functions; `check_untyped_defs`
  is on in mypy.
- Docstrings: single-line unless the function has non-obvious behavior.
- No `print` — use `logging`. The agent configures root logging at startup.

## Commit messages

Conventional-Commits style preferred but not required:

```
feat(tools): add get_sector_performance
fix(loop): strip <think> tags across chunk boundaries
docs(readme): add compaction diagram
```

## Running the eval harness

```bash
python -m agent.evals
```

Evals live in `evals/cases.py`. Add new cases when you add a tool or change
behavior that would benefit from a regression check.
