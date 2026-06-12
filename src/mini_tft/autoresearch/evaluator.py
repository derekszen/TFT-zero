"""Evaluation and decision logic for MiniTFT autoresearch objectives."""

from __future__ import annotations

import fnmatch
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from mini_tft.autoresearch.schema import (
    EnvironmentSpec,
    MetricRule,
    ObjectiveSpec,
    PolicySpec,
    SeedSet,
)
from mini_tft.bots import FastLevelBot, RandomBot
from mini_tft.bots.base import BaseBot
from mini_tft.rl.evaluate_policy import evaluate_bot, evaluate_checkpoint
from mini_tft.tools.evaluate_lobby_policy import run_lobby_evaluation


def evaluate_policy_spec(
    policy: PolicySpec,
    seed_set: SeedSet,
    environment: EnvironmentSpec,
) -> dict[str, float]:
    if environment.kind == "single_player":
        return _evaluate_single_player(policy, seed_set.seeds())
    if environment.kind == "lobby":
        return _numeric_metrics(_evaluate_lobby(policy, seed_set, environment))
    raise ValueError(f"unsupported environment kind: {environment.kind}")


def run_objective_phase(
    objective: ObjectiveSpec,
    seed_set: SeedSet,
    repo_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    baseline_metrics = evaluate_policy_spec(objective.baseline, seed_set, objective.environment)
    candidate_metrics = evaluate_policy_spec(objective.candidate, seed_set, objective.environment)
    comparisons = compare_metrics(
        baseline_metrics,
        candidate_metrics,
        objective.metric_rules,
    )
    risk_flags = protected_change_flags(repo_root, objective.safety.protected_paths)
    required_failures = [
        row["metric"] for row in comparisons if row["required"] and not row["passed"]
    ]
    blocked_by_risk = objective.safety.fail_on_protected_changes and bool(risk_flags)
    status = "pass" if not required_failures and not blocked_by_risk else "fail"

    report: dict[str, Any] = {
        "status": status,
        "objective": objective.name,
        "description": objective.description,
        "environment": environment_payload(objective.environment),
        "phase": seed_set.name,
        "seed_start": seed_set.start,
        "episodes": seed_set.episodes,
        "baseline": {
            "policy": policy_payload(objective.baseline),
            "metrics": baseline_metrics,
        },
        "candidate": {
            "policy": policy_payload(objective.candidate),
            "metrics": candidate_metrics,
        },
        "comparisons": comparisons,
        "required_failures": required_failures,
        "risk_flags": risk_flags,
        "git": git_snapshot(repo_root),
        "elapsed_sec": time.perf_counter() - started,
        "artifacts": {
            "run_dir": str(run_dir),
            "metrics": str(run_dir / "metrics.json"),
            "decision": str(run_dir / "decision.md"),
        },
    }
    write_json(run_dir / "metrics.json", report)
    return report


def compare_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
    rules: tuple[MetricRule, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in rules:
        if rule.name not in baseline:
            raise KeyError(f"baseline missing metric {rule.name!r}")
        if rule.name not in candidate:
            raise KeyError(f"candidate missing metric {rule.name!r}")
        baseline_value = float(baseline[rule.name])
        candidate_value = float(candidate[rule.name])
        raw_delta = candidate_value - baseline_value
        improvement = raw_delta if rule.mode == "higher" else -raw_delta
        rows.append(
            {
                "metric": rule.name,
                "mode": rule.mode,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": raw_delta,
                "improvement": improvement,
                "min_delta": rule.min_delta,
                "required": rule.required,
                "passed": improvement >= rule.min_delta,
            }
        )
    return rows


def protected_change_flags(
    repo_root: Path,
    protected_paths: tuple[str, ...],
) -> list[dict[str, str]]:
    if not protected_paths:
        return []
    changed = changed_files(repo_root)
    flags: list[dict[str, str]] = []
    for path in changed:
        for pattern in protected_paths:
            if fnmatch.fnmatch(path, pattern):
                flags.append({"path": path, "pattern": pattern})
                break
    return flags


def changed_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    def git(args: list[str]) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "branch": git(["branch", "--show-current"]),
        "commit": git(["rev-parse", "HEAD"]),
        "dirty": bool(changed_files(repo_root)),
        "status_porcelain": git(["status", "--porcelain=v1"]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def policy_payload(policy: PolicySpec) -> dict[str, str | None]:
    return {
        "kind": policy.kind,
        "name": policy.name,
        "path": str(policy.path) if policy.path is not None else None,
    }


def environment_payload(environment: EnvironmentSpec) -> dict[str, int | str | None]:
    return {
        "kind": environment.kind,
        "opponent_policy": environment.opponent_policy,
        "player_count": environment.player_count,
        "max_actions_per_player": environment.max_actions_per_player,
    }


def _evaluate_single_player(policy: PolicySpec, seeds: range) -> dict[str, float]:
    if policy.kind == "bot":
        return evaluate_bot(_bot_by_name(policy.name), seeds)
    if policy.kind == "checkpoint":
        if policy.path is None:
            raise ValueError(f"checkpoint policy {policy.name!r} is missing path")
        if not policy.path.exists():
            raise FileNotFoundError(policy.path)
        return evaluate_checkpoint(policy.path, seeds)
    raise ValueError(f"unsupported policy kind: {policy.kind}")


def _evaluate_lobby(
    policy: PolicySpec,
    seed_set: SeedSet,
    environment: EnvironmentSpec,
) -> dict[str, Any]:
    if policy.kind == "bot":
        return run_lobby_evaluation(
            episodes=seed_set.episodes,
            seed=seed_set.start,
            hero_policy_name=policy.name,
            opponent_policy_name=environment.opponent_policy,
            player_count=environment.player_count,
            max_actions_per_player=environment.max_actions_per_player,
        )
    if policy.kind == "checkpoint":
        if policy.path is None:
            raise ValueError(f"checkpoint policy {policy.name!r} is missing path")
        if not policy.path.exists():
            raise FileNotFoundError(policy.path)
        return run_lobby_evaluation(
            episodes=seed_set.episodes,
            seed=seed_set.start,
            checkpoint=policy.path,
            opponent_policy_name=environment.opponent_policy,
            player_count=environment.player_count,
            max_actions_per_player=environment.max_actions_per_player,
        )
    raise ValueError(f"unsupported policy kind: {policy.kind}")


def _numeric_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in report.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _bot_by_name(name: str) -> BaseBot:
    normalized = name.lower().replace("-", "_")
    if normalized in {"fast_level", "fastlevel", "fastlevelbot"}:
        return FastLevelBot()
    if normalized in {"random", "randombot"}:
        return RandomBot()
    raise ValueError(f"unsupported bot policy: {name}")
