"""CLI entry point for running MiniTFT autoresearch objectives."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from mini_tft.autoresearch.evaluator import run_objective_phase, write_json
from mini_tft.autoresearch.reporting import write_decision
from mini_tft.autoresearch.schema import ObjectiveSpec, SeedSet, load_objective

DEFAULT_OBJECTIVE = Path("experiments/autoresearch/objectives/ppo_lobby_placement.toml")
DEFAULT_RUNS_DIR = Path("experiments/autoresearch/runs")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=Path, default=DEFAULT_OBJECTIVE)
    parser.add_argument("--phase", choices=["public", "holdout"], default="public")
    parser.add_argument("--auto-holdout", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--candidate-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--allow-protected-changes",
        action="store_true",
        help="Record protected dirty files but do not fail the run solely because of them.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path.cwd()
    objective = load_objective(args.objective, repo_root)
    if args.candidate_checkpoint is not None:
        objective = _with_candidate_checkpoint(objective, args.candidate_checkpoint.resolve())
    if args.allow_protected_changes:
        objective = replace(
            objective,
            safety=replace(objective.safety, fail_on_protected_changes=False),
        )

    reports: list[dict[str, Any]] = []
    public_report = _run_one_phase(
        objective,
        args.phase,
        repo_root,
        args.runs_dir,
        args.run_name,
        args.seed_start,
        args.episodes,
    )
    reports.append(public_report)

    if args.auto_holdout and args.phase == "public" and public_report["status"] == "pass":
        reports.append(
            _run_one_phase(
                objective,
                "holdout",
                repo_root,
                args.runs_dir,
                args.run_name,
                None,
                None,
            )
        )

    print(_compact_summary(reports))
    return 0 if all(report["status"] == "pass" for report in reports) else 1


def _run_one_phase(
    objective: ObjectiveSpec,
    phase: str,
    repo_root: Path,
    runs_dir: Path,
    run_name: str | None,
    seed_start: int | None,
    episodes: int | None,
) -> dict[str, Any]:
    seed_set = _seed_set(objective, phase, seed_start, episodes)
    run_dir = runs_dir / _run_dir_name(objective.name, seed_set.name, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    report = run_objective_phase(objective, seed_set, repo_root, run_dir)
    write_decision(run_dir / "decision.md", report)
    write_json(run_dir / "manifest.json", _manifest(report))
    return report


def _seed_set(
    objective: ObjectiveSpec,
    phase: str,
    seed_start: int | None,
    episodes: int | None,
) -> SeedSet:
    base = objective.seed_sets[phase]
    return SeedSet(
        name=base.name,
        start=base.start if seed_start is None else seed_start,
        episodes=base.episodes if episodes is None else episodes,
    )


def _with_candidate_checkpoint(objective: ObjectiveSpec, checkpoint: Path) -> ObjectiveSpec:
    return replace(
        objective,
        candidate=replace(
            objective.candidate,
            kind="checkpoint",
            name=checkpoint.stem,
            path=checkpoint,
        ),
    )


def _run_dir_name(objective_name: str, phase: str, run_name: str | None) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = run_name or "run"
    return f"{timestamp}_{objective_name}_{phase}_{suffix}"


def _manifest(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective": report["objective"],
        "environment": report["environment"],
        "phase": report["phase"],
        "status": report["status"],
        "git": report["git"],
        "artifacts": report["artifacts"],
        "risk_flags": report["risk_flags"],
    }


def _compact_summary(reports: list[dict[str, Any]]) -> str:
    lines = []
    for report in reports:
        lines.append(
            f"{report['objective']}:{report['phase']} status={report['status']} "
            f"episodes={report['episodes']} seed_start={report['seed_start']}"
        )
        for row in report["comparisons"]:
            lines.append(
                f"  {row['metric']}: baseline={row['baseline']:.4f} "
                f"candidate={row['candidate']:.4f} delta={row['delta']:+.4f} "
                f"pass={row['passed']}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
