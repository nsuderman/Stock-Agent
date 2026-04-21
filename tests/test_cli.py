"""Tests for the CLI argument parsing + banner."""

from __future__ import annotations

import datetime
import io
from contextlib import redirect_stdout

from agent.cli import _build_parser, print_banner


class TestParser:
    def test_parses_question(self):
        ns = _build_parser().parse_args(["hello", "world"])
        assert ns.question == ["hello", "world"]

    def test_no_question_allowed_for_interactive(self):
        """`stock-agent` with no args should parse — question becomes empty list."""
        ns = _build_parser().parse_args([])
        assert ns.question == []

    def test_debug_flag(self):
        ns = _build_parser().parse_args(["--debug", "hi"])
        assert ns.debug is True

    def test_debug_defaults_off(self):
        ns = _build_parser().parse_args(["hi"])
        assert ns.debug is False

    def test_session_defaults_to_none_in_args(self):
        ns = _build_parser().parse_args(["hi"])
        assert ns.session is None
        # CLI's main() then fills it in with today's date.

    def test_explicit_session(self):
        ns = _build_parser().parse_args(["--session", "research", "hi"])
        assert ns.session == "research"

    def test_no_session_flag(self):
        ns = _build_parser().parse_args(["--no-session", "hi"])
        assert ns.no_session is True

    def test_remote_flag(self):
        ns = _build_parser().parse_args(["--remote", "hi"])
        assert ns.remote is True

    def test_quiet_flag(self):
        ns = _build_parser().parse_args(["--quiet", "hi"])
        assert ns.quiet is True

    def test_max_iterations_default(self):
        ns = _build_parser().parse_args(["hi"])
        assert ns.max_iterations == 12

    def test_max_iterations_override(self):
        ns = _build_parser().parse_args(["--max-iterations", "20", "hi"])
        assert ns.max_iterations == 20


def test_session_resolution_today_by_default():
    """Confirm the expected session-resolution semantics live in cli.main."""
    today = datetime.date.today().isoformat()
    # This mirrors the logic inside cli.main. If that gets refactored, update here.
    from agent.session import default_session_name

    assert default_session_name() == today


class TestBanner:
    def test_banner_contains_brand_and_status(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_banner(model="qwen3.6-35b-a3b", session="2026-04-21")
        out = buf.getvalue()
        # ASCII art for FINN uses block glyphs
        assert "███" in out
        assert "STOCK AGENT" in out
        assert "ONLINE" in out
        assert "qwen3.6-35b-a3b" in out
        assert "2026-04-21" in out

    def test_banner_handles_no_session(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_banner(model="test-model", session=None)
        out = buf.getvalue()
        assert "none" in out
