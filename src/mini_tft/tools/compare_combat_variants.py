"""Compare combat evaluation reports across git worktrees."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def compare_variants(
    variants: list[tuple[str, Path]],
    benchmark_iters: int,
) -> list[dict[str, Any]]:
    rows = []
    for name, path in variants:
        report = _run_combat_eval(path, benchmark_iters)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "commit": _git_commit(path),
                "status": report["status"],
                "passed": report["passed"],
                "total": report["total"],
                "calls_per_sec": report["benchmark"]["calls_per_sec"],
                "failures": [
                    fixture["name"]
                    for fixture in report["fixtures"]
                    if fixture["status"] != "pass"
                ],
            }
        )
    return rows


def format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# MiniTFT Combat Variant Comparison",
        "",
        "| Variant | Commit | Score | Calls/sec | Failing Fixtures | Path |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    ranked = sorted(
        rows,
        key=lambda item: (-item["passed"], -item["calls_per_sec"], item["name"]),
    )
    for row in ranked:
        failures = ", ".join(row["failures"]) if row["failures"] else "None"
        lines.append(
            f"| {row['name']} | `{row['commit']}` | {row['passed']}/{row['total']} | "
            f"{row['calls_per_sec']:.1f} | {failures} | `{row['path']}` |"
        )
    return "\n".join(lines) + "\n"


def _run_combat_eval(path: Path, benchmark_iters: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "mini_tft.tools.combat_eval",
            "--benchmark-iters",
            str(benchmark_iters),
            "--format",
            "json",
        ],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"combat_eval failed in {path} with code {result.returncode}:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _parse_variant(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, raw_path = value.split("=", 1)
    else:
        path = Path(value)
        name = path.name
        raw_path = value
    return name, Path(raw_path).expanduser().resolve()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare combat variants across worktrees.")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant as name=/path/to/worktree. May be repeated.",
    )
    parser.add_argument("--benchmark-iters", type=int, default=10_000)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args(list(argv) if argv is not None else None)

    variants = [_parse_variant(value) for value in args.variant]
    if not variants:
        variants = [("current", Path.cwd())]
    rows = compare_variants(variants, args.benchmark_iters)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(format_markdown(rows), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
