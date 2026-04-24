"""CLI entry point: `stock-agent "your question" [--session name]`

With no question argument, drops into an interactive REPL (Finn).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Silence the libedit/readline "Cannot read termcap database" warning before
# ANY other import triggers terminfo initialization. We save the original TERM
# for ANSI detection and fall back to a terminfo entry that exists on every
# Unix (xterm) when the current one is missing, dumb, or not recognized.
_ORIGINAL_TERM = os.environ.get("TERM", "")
if _ORIGINAL_TERM in ("", "dumb") or not sys.stdout.isatty():
    os.environ["TERM"] = "xterm"

from agent.loop import run_agent  # noqa: E402
from agent.session import (  # noqa: E402
    default_session_name,
    load_session,
    reset_session,
    save_session,
)


def _ansi_supported() -> bool:
    """stdout is a TTY AND the original TERM looks like it renders ANSI escapes."""
    if not sys.stdout.isatty():
        return False
    if _ORIGINAL_TERM.lower() in ("", "dumb"):
        return False
    return not os.environ.get("NO_COLOR")


_ANSI = _ansi_supported()

# ANSI colors — empty strings when the terminal can't render them.
_G = "\033[92m" if _ANSI else ""
_D = "\033[90m" if _ANSI else ""
_C = "\033[96m" if _ANSI else ""
_Y = "\033[93m" if _ANSI else ""
_B = "\033[94m" if _ANSI else ""
_RESET = "\033[0m" if _ANSI else ""

_ANSI_SGR_RE = re.compile(r"(\x1b\[[0-9;]*m)")


def _readline_prompt(raw: str) -> str:
    r"""Wrap ANSI sequences with \001/\002 so readline computes prompt width correctly."""
    if not _ANSI:
        return raw
    return _ANSI_SGR_RE.sub(lambda m: f"\001{m.group(1)}\002", raw)


def print_banner(*, model: str, session: str | None) -> None:
    """Print the Finn banner + status line."""
    logo = (
        "\n"
        "    ███████╗██╗███╗   ██╗███╗   ██╗\n"
        "    ██╔════╝██║████╗  ██║████╗  ██║\n"
        "    █████╗  ██║██╔██╗ ██║██╔██╗ ██║\n"
        "    ██╔══╝  ██║██║╚██╗██║██║╚██╗██║\n"
        "    ██║     ██║██║ ╚████║██║ ╚████║\n"
        "    ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝\n"
    )
    print(f"{_G}{logo}{_RESET}")
    print(f"    {_D}STOCK AGENT · local-LLM quant research assistant{_RESET}")
    print(f"    {_D}------------------------------------------------{_RESET}")
    status = (
        f"    STATUS: {_G}ONLINE{_RESET}  |  "
        f"MODEL: {_C}{model}{_RESET}  |  "
        f"SESSION: {_Y}{session or 'none'}{_RESET}"
    )
    print(status)
    print(
        f"    {_D}Type your question. Slash commands: /exit, /reset, /session <name>, /help{_RESET}\n"
    )


def _print_help() -> None:
    print(
        f"""
  {_C}Slash commands{_RESET}
    {_G}/help{_RESET}              show this help
    {_G}/exit{_RESET}, {_G}/quit{_RESET}      end the session
    {_G}/reset{_RESET}             clear the current conversation context
    {_G}/session <name>{_RESET}    switch to a named session (saves the current first)
    {_G}/nosession{_RESET}         drop to ephemeral mode (no load/save)

  {_C}Tips{_RESET}
    - Ctrl-D or Ctrl-C also exits cleanly.
    - Memory (facts saved via the `remember` tool) persists across sessions.
"""
    )


def _interactive(
    *,
    session_name: str | None,
    local: bool,
    verbose: bool,
    debug: bool,
    max_iterations: int,
) -> int:
    """REPL mode: banner + multi-turn conversation loop."""
    # Readline gives us line editing + history for free on POSIX, but skip it on
    # dumb terminals where it prints noisy termcap warnings and can't do its job.
    if _ANSI:
        import contextlib

        with contextlib.suppress(ImportError):
            import readline  # noqa: F401

    from agent.llm import active_model

    print_banner(model=active_model(local=local), session=session_name)

    prior: list[dict] = []
    if session_name:
        prior = load_session(session_name)
        if prior:
            print(f"  {_D}[resumed {len(prior)} prior messages]{_RESET}\n")

    def _persist_on_exit() -> None:
        """Save whatever's accumulated if we're in a session."""
        if session_name and prior:
            try:
                save_session(session_name, prior)
            except OSError as e:
                print(f"  {_D}[session save failed: {e}]{_RESET}")

    prompt = _readline_prompt(f"{_B}>{_RESET} ")
    while True:
        try:
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            _persist_on_exit()
            print(f"\n{_D}bye.{_RESET}")
            return 0

        q = raw.strip()
        if not q:
            continue

        # Slash commands
        if q.startswith("/"):
            cmd, *rest = q.split(maxsplit=1)
            cmd = cmd.lower()
            arg = rest[0].strip() if rest else ""
            if cmd in ("/exit", "/quit"):
                _persist_on_exit()
                print(f"{_D}bye.{_RESET}")
                return 0
            if cmd == "/help":
                _print_help()
                continue
            if cmd == "/reset":
                prior = []
                if session_name:
                    reset_session(session_name)
                print(f"  {_D}[context cleared]{_RESET}\n")
                continue
            if cmd == "/session":
                if not arg:
                    print(f"  {_D}current session: {session_name or '(none)'}{_RESET}\n")
                    continue
                # Save current, switch, load
                if session_name:
                    save_session(session_name, prior)
                session_name = arg
                prior = load_session(session_name)
                msg = f"switched to '{session_name}'"
                if prior:
                    msg += f" — {len(prior)} prior messages"
                print(f"  {_D}[{msg}]{_RESET}\n")
                continue
            if cmd == "/nosession":
                if session_name:
                    save_session(session_name, prior)
                session_name = None
                print(f"  {_D}[ephemeral mode — no session will be saved]{_RESET}\n")
                continue
            print(f"  {_D}unknown command: {cmd} (try /help){_RESET}\n")
            continue

        try:
            answer, prior = run_agent(
                q,
                max_iterations=max_iterations,
                local=local,
                verbose=verbose,
                debug=debug,
                prior_messages=prior,
            )
        except KeyboardInterrupt:
            print(f"\n  {_D}[interrupted]{_RESET}\n")
            continue

        if session_name:
            save_session(session_name, prior)

        print(f"\n{answer}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock-agent",
        description="Ask the trading-data agent a question. Run with no question for interactive mode.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Your question. Omit to drop into the interactive REPL.",
    )
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full per-tool trace (args, result summary, and iteration headers).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Cap on ReAct loop iterations (default from MAX_ITERATIONS env var, or 12).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.no_session:
        session_name: str | None = None
    else:
        session_name = args.session or default_session_name()

    # No question → interactive mode.
    if not args.question:
        return _interactive(
            session_name=session_name,
            local=not args.remote,
            verbose=not args.quiet,
            debug=args.debug,
            max_iterations=args.max_iterations,
        )

    # One-shot mode (backwards compatible).
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
        debug=args.debug,
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
