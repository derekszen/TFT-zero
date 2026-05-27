"""Evaluate lightweight combat model sanity fixtures."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from typing import Any

from mini_tft.core.combat import board_strength
from mini_tft.core.set_data import load_set
from mini_tft.experiments.combat_fixtures import combat_fixtures


def run_combat_eval(*, benchmark_iters: int = 10_000) -> dict[str, Any]:
    """Score current combat implementation against fixed sanity fixtures."""

    data = load_set()
    fixtures = combat_fixtures()
    rows = []
    passed = 0
    for fixture in fixtures:
        better = board_strength(fixture.better_board, data).strength
        worse = board_strength(fixture.worse_board, data).strength
        margin = better - worse
        ok = margin >= fixture.min_margin
        passed += int(ok)
        rows.append(
            {
                "name": fixture.name,
                "status": "pass" if ok else "fail",
                "better_strength": round(better, 3),
                "worse_strength": round(worse, 3),
                "margin": round(margin, 3),
                "min_margin": fixture.min_margin,
                "rationale": fixture.rationale,
            }
        )

    started = time.perf_counter()
    calls = 0
    if benchmark_iters > 0:
        for _ in range(benchmark_iters):
            for fixture in fixtures:
                board_strength(fixture.better_board, data)
                board_strength(fixture.worse_board, data)
                calls += 2
    elapsed = time.perf_counter() - started

    return {
        "status": "pass" if passed == len(fixtures) else "fail",
        "passed": passed,
        "total": len(fixtures),
        "score": passed / len(fixtures) if fixtures else 0.0,
        "fixtures": rows,
        "benchmark": {
            "calls": calls,
            "elapsed_sec": round(elapsed, 4),
            "calls_per_sec": calls / elapsed if elapsed else 0.0,
        },
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MiniTFT Combat Model Evaluation",
        "",
        f"**Status:** `{report['status']}`",
        f"**Score:** `{report['passed']}/{report['total']}`",
        "",
        "## Fixture Results",
        "",
        "| Fixture | Status | Better | Worse | Margin | Required |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["fixtures"]:
        lines.append(
            f"| {row['name']} | `{row['status']}` | {row['better_strength']:.3f} | "
            f"{row['worse_strength']:.3f} | {row['margin']:.3f} | "
            f"{row['min_margin']:.3f} |"
        )

    benchmark = report["benchmark"]
    lines.extend(
        [
            "",
            "## Throughput",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Calls | {benchmark['calls']} |",
            f"| Seconds | {benchmark['elapsed_sec']:.4f} |",
            f"| Calls/sec | {benchmark['calls_per_sec']:.1f} |",
            "",
            "## Fixture Rationales",
            "",
        ]
    )
    for row in report["fixtures"]:
        lines.append(f"- `{row['name']}`: {row['rationale']}")
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate combat model sanity fixtures.")
    parser.add_argument("--benchmark-iters", type=int, default=10_000)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless all fixtures pass.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_combat_eval(benchmark_iters=args.benchmark_iters)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_markdown(report), end="")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
