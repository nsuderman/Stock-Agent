"""Run eval cases against the live agent + LLM + DB and report results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from typing import Any

from agent.loop import run_agent
from evals.cases import CASES, EvalCase


def _tool_calls_from_messages(messages: list[dict]) -> list[tuple[str, str]]:
    """Extract (name, arguments) tuples for every tool_call in the conversation."""
    calls: list[tuple[str, str]] = []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                calls.append((fn.get("name", ""), fn.get("arguments", "")))
    return calls


def _evaluate(case: EvalCase, answer: str, messages: list[dict]) -> dict[str, Any]:
    tool_calls = _tool_calls_from_messages(messages)
    called = {name for name, _ in tool_calls}
    failures: list[str] = []

    if case.must_call_any and not (called & case.must_call_any):
        failures.append(
            f"no tool in must_call_any={sorted(case.must_call_any)} was called; got {sorted(called)}"
        )

    forbidden = called & case.must_not_call
    if forbidden:
        failures.append(f"forbidden tools called: {sorted(forbidden)}")

    # Tool-call count reflects ReAct iterations; each tool call is one round.
    n_rounds = len({m.get("tool_call_id") for m in messages if m.get("role") == "tool"})
    if n_rounds > case.max_iterations:
        failures.append(f"used {n_rounds} tool rounds; limit {case.max_iterations}")

    for needle in case.answer_contains:
        if needle.lower() not in (answer or "").lower():
            failures.append(f"answer missing required substring: {needle!r}")

    for forbidden_str in case.answer_not_contains:
        if forbidden_str.lower() in (answer or "").lower():
            failures.append(f"answer contains forbidden substring: {forbidden_str!r}")

    return {
        "passed": not failures,
        "failures": failures,
        "n_rounds": n_rounds,
        "tools_called": sorted(called),
    }


def run_one(case: EvalCase, *, local: bool = True, verbose: bool = False) -> dict[str, Any]:
    start = time.monotonic()
    try:
        answer, messages = run_agent(
            case.question,
            local=local,
            verbose=verbose,
            max_iterations=case.max_iterations + 4,  # leave headroom for tool dispatch
            prior_messages=None,
        )
        result = _evaluate(case, answer, messages)
        result["error"] = None
    except Exception as e:
        result = {"passed": False, "failures": [f"exception: {e}"], "error": str(e)}
    result["duration_s"] = round(time.monotonic() - start, 2)
    result["case"] = asdict(case)
    result["case"]["must_call_any"] = sorted(case.must_call_any)
    result["case"]["must_not_call"] = sorted(case.must_not_call)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run agent eval cases.")
    parser.add_argument("--remote", action="store_true", help="Use remote LLM.")
    parser.add_argument("--verbose", action="store_true", help="Show per-tool trace.")
    parser.add_argument("--filter", help="Substring filter on case name.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report instead of text.")
    args = parser.parse_args(argv)

    cases = [c for c in CASES if not args.filter or args.filter in c.name]
    if not cases:
        print(f"No cases matched filter {args.filter!r}", file=sys.stderr)
        return 2

    results = []
    for case in cases:
        if not args.json:
            print(f"▶  {case.name:40s} ", end="", flush=True)
        r = run_one(case, local=not args.remote, verbose=args.verbose)
        results.append(r)
        if not args.json:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{status:5s}  {r['duration_s']:5.1f}s  tools={r.get('tools_called', [])}")
            for f in r.get("failures", []):
                print(f"    ↳ {f}")

    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    avg = sum(r["duration_s"] for r in results) / max(n_total, 1)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": {"passed": n_pass, "total": n_total, "avg_s": round(avg, 2)},
                    "results": results,
                },
                indent=2,
            )
        )
    else:
        print()
        print(f"== {n_pass}/{n_total} passed   avg {avg:.1f}s ==")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
