"""Markdown reporting for MiniTFT autoresearch runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_decision(report: dict[str, Any]) -> str:
    lines = [
        f"# Autoresearch Decision: {report['objective']} / {report['phase']}",
        "",
        f"**Status:** `{report['status']}`",
        f"**Environment:** `{report['environment']['kind']}`",
        f"**Episodes:** `{report['episodes']}`",
        f"**Seed start:** `{report['seed_start']}`",
        f"**Elapsed:** `{report['elapsed_sec']:.3f}s`",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta | Required | Pass |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["comparisons"]:
        required = "yes" if row["required"] else "no"
        passed = "yes" if row["passed"] else "no"
        lines.append(
            f"| `{row['metric']}` | {row['baseline']:.4f} | {row['candidate']:.4f} | "
            f"{row['delta']:+.4f} | {required} | {passed} |"
        )

    if report["required_failures"]:
        lines.extend(["", "## Required Failures", ""])
        lines.extend(f"- `{name}`" for name in report["required_failures"])

    if report["risk_flags"]:
        lines.extend(["", "## Reward-Hack Risk Flags", ""])
        lines.extend(
            f"- `{flag['path']}` matched protected pattern `{flag['pattern']}`"
            for flag in report["risk_flags"]
        )

    lines.extend(
        [
            "",
            "## Policies",
            "",
            f"- Baseline: `{report['baseline']['policy']['name']}`",
            f"- Candidate: `{report['candidate']['policy']['name']}`",
            "",
            "## Artifacts",
            "",
            f"- Metrics: `{report['artifacts']['metrics']}`",
            f"- Decision: `{report['artifacts']['decision']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_decision(path: Path, report: dict[str, Any]) -> None:
    path.write_text(format_decision(report), encoding="utf-8")
