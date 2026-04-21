"""CLI entry point: `ask "your question" [--session name]`"""

from __future__ import annotations

import argparse
import sys

from agent.loop import run_agent
from agent.session import default_session_name, load_session, reset_session, save_session


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ask",
        description="Ask the trading-data agent a question.",
    )
    parser.add_argument("question", nargs="+", help="Your question (any words).")
    parser.add_argument(
        "--session",
        default=None,
        help="Session name. Defaults to today's date (YYYY-MM-DD) for auto daily rollover.",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Don't load or save any session — true one-shot.",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Start the session fresh (delete prior history)."
    )
    parser.add_argument("--remote", action="store_true", help="Use remote LLM instead of local.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-tool trace output.")
    parser.add_argument("--max-iterations", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.no_session:
        session_name: str | None = None
    else:
        session_name = args.session or default_session_name()

    prior: list[dict] = []
    if session_name:
        if args.reset:
            reset_session(session_name)
            if not args.quiet:
                print(f"[reset session '{session_name}']")
        else:
            prior = load_session(session_name)
            if not args.quiet:
                if prior:
                    print(f"[resuming session '{session_name}' — {len(prior)} prior messages]")
                else:
                    print(f"[new session '{session_name}']")

    answer, messages = run_agent(
        " ".join(args.question),
        max_iterations=args.max_iterations,
        local=not args.remote,
        verbose=not args.quiet,
        prior_messages=prior,
    )

    if session_name:
        path = save_session(session_name, messages)
        if not args.quiet:
            print(f"[session saved → {path.relative_to(path.parent.parent)}]")

    print("\n=== ANSWER ===")
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
