"""Evaluate lightweight combat model sanity fixtures."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from mini_tft.core.combat import board_strength
from mini_tft.core.set_data import load_set
from mini_tft.experiments.combat_fixtures import combat_fixtures

DEFAULT_CATEGORY = "general"
DEFAULT_WEIGHT = 1.0


@dataclass
class _Aggregate:
    total: int = 0
    passed: int = 0
    weight: float = 0.0
    passed_weight: float = 0.0
    weighted_margin_score: float = 0.0
    weighted_surplus: float = 0.0
    weighted_normalized_surplus: float = 0.0

    def add(
        self,
        *,
        ok: bool,
        weight: float,
        margin_score: float,
        surplus: float,
        normalized_surplus: float,
    ) -> None:
        self.total += 1
        self.passed += int(ok)
        self.weight += weight
        self.passed_weight += weight if ok else 0.0
        self.weighted_margin_score += margin_score * weight
        self.weighted_surplus += surplus * weight
        self.weighted_normalized_surplus += normalized_surplus * weight

    def summary(self) -> dict[str, Any]:
        if self.weight <= 0:
            margin_score = 0.0
            mean_surplus = 0.0
            mean_normalized_surplus = 0.0
            weighted_pass_rate = 0.0
        else:
            margin_score = self.weighted_margin_score / self.weight
            mean_surplus = self.weighted_surplus / self.weight
            mean_normalized_surplus = self.weighted_normalized_surplus / self.weight
            weighted_pass_rate = self.passed_weight / self.weight

        return {
            "passed": self.passed,
            "total": self.total,
            "weight": round(self.weight, 3),
            "weighted_pass_rate": round(weighted_pass_rate, 6),
            "margin_score": round(margin_score, 6),
            "mean_surplus": round(mean_surplus, 6),
            "mean_normalized_surplus": round(mean_normalized_surplus, 6),
            "decision_score": round((weighted_pass_rate * 100.0) + margin_score, 6),
        }


def run_combat_eval(*, benchmark_iters: int = 10_000) -> dict[str, Any]:
    """Score current combat implementation against fixed sanity fixtures."""

    data = load_set()
    fixtures = combat_fixtures()
    rows = []
    passed = 0
    overall = _Aggregate()
    categories: dict[str, _Aggregate] = {}
    for fixture in fixtures:
        better = board_strength(fixture.better_board, data).strength
        worse = board_strength(fixture.worse_board, data).strength
        margin = better - worse
        ok = margin >= fixture.min_margin
        passed += int(ok)
        category = str(getattr(fixture, "category", DEFAULT_CATEGORY))
        weight = float(getattr(fixture, "weight", DEFAULT_WEIGHT))
        threshold = max(abs(fixture.min_margin), 1.0)
        surplus = margin - fixture.min_margin
        normalized_margin = margin / threshold
        normalized_surplus = surplus / threshold
        overall.add(
            ok=ok,
            weight=weight,
            margin_score=normalized_margin,
            surplus=surplus,
            normalized_surplus=normalized_surplus,
        )
        categories.setdefault(category, _Aggregate()).add(
            ok=ok,
            weight=weight,
            margin_score=normalized_margin,
            surplus=surplus,
            normalized_surplus=normalized_surplus,
        )
        rows.append(
            {
                "name": fixture.name,
                "category": category,
                "weight": weight,
                "status": "pass" if ok else "fail",
                "better_strength": round(better, 3),
                "worse_strength": round(worse, 3),
                "margin": round(margin, 3),
                "min_margin": fixture.min_margin,
                "surplus": round(surplus, 3),
                "normalized_surplus": round(normalized_surplus, 6),
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
    aggregate = overall.summary()
    category_summaries = {
        category: aggregate.summary()
        for category, aggregate in sorted(categories.items(), key=lambda item: item[0])
    }

    return {
        "status": "pass" if passed == len(fixtures) else "fail",
        "passed": passed,
        "total": len(fixtures),
        "score": passed / len(fixtures) if fixtures else 0.0,
        "margin_score": aggregate["margin_score"],
        "mean_surplus": aggregate["mean_surplus"],
        "mean_normalized_surplus": aggregate["mean_normalized_surplus"],
        "decision_score": aggregate["decision_score"],
        "category_summaries": category_summaries,
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
        f"**Decision score:** `{report['decision_score']:.3f}`",
        f"**Margin score:** `{report['margin_score']:.3f}`",
        f"**Mean normalized surplus:** `{report['mean_normalized_surplus']:.3f}`",
        "",
        "## Fixture Results",
        "",
        "| Fixture | Category | Weight | Status | Better | Worse | Margin | "
        "Required | Surplus | Norm Surplus |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["fixtures"]:
        lines.append(
            f"| {row['name']} | {row['category']} | {row['weight']:.2f} | "
            f"`{row['status']}` | {row['better_strength']:.3f} | "
            f"{row['worse_strength']:.3f} | {row['margin']:.3f} | "
            f"{row['min_margin']:.3f} | {row['surplus']:.3f} | "
            f"{row['normalized_surplus']:.3f} |"
        )

    benchmark = report["benchmark"]
    lines.extend(
        [
            "",
            "## Category Summary",
            "",
            "| Category | Score | Weight | Decision | Margin Score | Mean Surplus | "
            "Mean Norm Surplus |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, summary in report["category_summaries"].items():
        lines.append(
            f"| {category} | {summary['passed']}/{summary['total']} | "
            f"{summary['weight']:.2f} | {summary['decision_score']:.3f} | "
            f"{summary['margin_score']:.3f} | {summary['mean_surplus']:.3f} | "
            f"{summary['mean_normalized_surplus']:.3f} |"
        )

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
