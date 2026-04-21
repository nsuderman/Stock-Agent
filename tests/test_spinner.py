"""Tests for the Spinner class — no TTY expected, so most paths are no-ops."""

from __future__ import annotations

import sys
import time
from unittest.mock import patch

import pytest

from agent.loop import Spinner


def test_spinner_noop_when_not_tty():
    """Under pytest stdout is not a TTY; start()/stop() should be safe no-ops."""
    s = Spinner("testing")
    assert not s.running
    s.start()
    # isatty() is False under pytest → thread not started
    assert not s.running
    s.stop()  # still safe


def test_spinner_starts_when_tty(monkeypatch):
    """Force isatty() → True and confirm the thread starts + stops cleanly."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with patch.object(sys.stdout, "write"), patch.object(sys.stdout, "flush"):
        s = Spinner("testing", interval=0.01)
        s.start()
        assert s.running
        time.sleep(0.05)  # let the thread tick a few times
        s.stop()
        assert not s.running


def test_spinner_stop_idempotent(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with patch.object(sys.stdout, "write"), patch.object(sys.stdout, "flush"):
        s = Spinner("x", interval=0.01)
        s.start()
        s.stop()
        s.stop()  # second stop must not error


def test_spinner_context_manager(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with patch.object(sys.stdout, "write"), patch.object(sys.stdout, "flush"):
        with Spinner("inside", interval=0.01) as s:
            assert s.running
            time.sleep(0.03)
        assert not s.running


def test_spinner_double_start_is_noop(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    with patch.object(sys.stdout, "write"), patch.object(sys.stdout, "flush"):
        s = Spinner("x", interval=0.01)
        s.start()
        first_thread = s._thread
        s.start()  # second start should not replace the thread
        assert s._thread is first_thread
        s.stop()


class TestRender:
    """The bar renderer is pure — test without threading/TTY."""

    def test_render_length_matches_phase_count(self):
        s = Spinner()
        bars = s._render(0.0)
        assert len(bars) == len(Spinner.PHASES)

    def test_render_always_uses_visible_blocks(self):
        """Every bar should be in ▁..█; the space glyph is reserved for padding."""
        s = Spinner()
        for t in (0.0, 0.2, 0.45, 0.7, 1.1, 3.7):
            bars = s._render(t)
            for ch in bars:
                assert ch in Spinner.BLOCKS[1:], f"bar rendered empty cell at t={t}"

    @pytest.mark.parametrize("t", [0.0, 0.1, 0.5, 0.9, 1.5])
    def test_render_is_deterministic(self, t):
        """Same elapsed time → same bars."""
        s = Spinner()
        assert s._render(t) == s._render(t)

    def test_render_varies_over_time(self):
        """Animation actually animates — bars at t=0 differ from bars at t=0.3."""
        s = Spinner()
        assert s._render(0.0) != s._render(0.3)
