"""Allow `python -m agent "..."` as an alternative to the `stock-agent` entry point."""

from agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
