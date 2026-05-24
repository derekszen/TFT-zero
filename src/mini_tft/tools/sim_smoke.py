"""One-command simulator smoke report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mini_tft.bots import (
    EconBot,
    FastLevelBot,
    GreedyBoardBot,
    RandomBot,
    RerollBot,
    TraitCommitBot,
)
from mini_tft.bots.base import BaseBot
from mini_tft.core.actions import action_name
from mini_tft.core.config import EnvConfig
from mini_tft.rl.dataset import load_dataset
from mini_tft.rl.evaluate_policy import evaluate_bot
from mini_tft.rl.gym_env import MiniTFTGymEnv
from mini_tft.tools.generate_bot_dataset import generate_dataset


@dataclass(frozen=True)
class SmokeConfig:
    benchmark_episodes: int = 100
    dataset_episodes: int = 100
    eval_episodes: int = 10
    trace_steps: int = 8
    seed: int = 0
    dataset_out: Path | None = None
    run_checks: bool = True


def run_smoke(config: SmokeConfig) -> dict[str, Any]:
    """Run smoke checks and return a pipeline-friendly report object."""

    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "pass",
        "seed": config.seed,
        "config": {
            "benchmark_episodes": config.benchmark_episodes,
            "dataset_episodes": config.dataset_episodes,
            "eval_episodes": config.eval_episodes,
            "trace_steps": config.trace_steps,
            "run_checks": config.run_checks,
        },
        "checks": [],
        "benchmark": {},
        "dataset": {},
        "evaluation": {},
        "trace": [],
        "failures": [],
    }

    if config.run_checks:
        checks = [
            _run_command("pytest", [sys.executable, "-m", "pytest"]),
            _run_command("ruff", [sys.executable, "-m", "ruff", "check"]),
        ]
        report["checks"] = checks
        report["failures"].extend(check["name"] for check in checks if not check["ok"])

    benchmark = _run_benchmark(config.benchmark_episodes, config.seed)
    report["benchmark"] = benchmark
    if benchmark["steps"] <= 0:
        report["failures"].append("benchmark produced zero steps")

    dataset = _run_dataset(config)
    report["dataset"] = dataset
    if dataset["episodes"] != config.dataset_episodes:
        report["failures"].append("dataset episode count mismatch")
    if dataset["done_count"] != config.dataset_episodes:
        report["failures"].append("dataset done count mismatch")

    report["evaluation"] = _run_evaluation(config.eval_episodes)
    report["trace"] = _run_trace(config.seed, config.trace_steps)

    report["elapsed_sec"] = round(time.perf_counter() - started, 3)
    if report["failures"]:
        report["status"] = "fail"
    return report


def format_markdown(report: dict[str, Any]) -> str:
    """Format a report for docs, coding-agent debugging, and PR comments."""

    lines = [
        "# MiniTFT Simulator Smoke Report",
        "",
        f"**Status:** `{report['status']}`",
        f"**Seed:** `{report['seed']}`",
        f"**Elapsed:** `{report['elapsed_sec']:.3f}s`",
        "",
        "## Summary",
        "",
        "| Area | Result | Key Metrics |",
        "| --- | --- | --- |",
    ]
    checks_ok = all(check["ok"] for check in report["checks"]) if report["checks"] else True
    dataset = report["dataset"]
    benchmark = report["benchmark"]
    evaluation = report["evaluation"]
    lines.extend(
        [
            f"| Checks | `{_status(checks_ok)}` | {len(report['checks'])} command checks |",
            (
                f"| Benchmark | `{_status(benchmark['steps'] > 0)}` | "
                f"{benchmark['episodes']} episodes, {benchmark['steps']} steps, "
                f"{benchmark['steps_per_sec']:.1f} steps/sec |"
            ),
            (
                f"| Dataset | `{_status(dataset['done_count'] == dataset['episodes'])}` | "
                f"{dataset['episodes']} episodes, {dataset['transitions']} transitions, "
                f"{dataset['transitions_per_sec']:.1f} transitions/sec |"
            ),
            (
                f"| Evaluation | `{_status(bool(evaluation))}` | "
                f"{len(evaluation)} bot baselines |"
            ),
            (
                f"| Trace | `{_status(bool(report['trace']))}` | "
                f"{len(report['trace'])} rendered steps |"
            ),
            "",
            "## Command Checks",
            "",
        ]
    )
    if report["checks"]:
        lines.extend(["| Check | Status | Duration |", "| --- | --- | ---: |"])
        for check in report["checks"]:
            lines.append(
                f"| `{check['name']}` | `{_status(check['ok'])}` | "
                f"{check['elapsed_sec']:.3f}s |"
            )
        lines.append("")
    else:
        lines.extend(["Command checks were skipped.", ""])

    lines.extend(
        [
            "## Throughput",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Benchmark episodes | {benchmark['episodes']} |",
            f"| Benchmark steps | {benchmark['steps']} |",
            f"| Benchmark seconds | {benchmark['elapsed_sec']:.3f} |",
            f"| Benchmark steps/sec | {benchmark['steps_per_sec']:.1f} |",
            f"| Dataset episodes | {dataset['episodes']} |",
            f"| Dataset transitions | {dataset['transitions']} |",
            f"| Dataset seconds | {dataset['elapsed_sec']:.3f} |",
            f"| Dataset transitions/sec | {dataset['transitions_per_sec']:.1f} |",
            f"| Dataset file MB | {dataset['file_mb']:.3f} |",
            "",
            "## Dataset Shape",
            "",
            "| Array | Shape |",
            "| --- | --- |",
            f"| obs | `{tuple(dataset['obs_shape'])}` |",
            f"| masks | `{tuple(dataset['mask_shape'])}` |",
            f"| actions | `({dataset['transitions']},)` |",
            f"| dones | `{dataset['done_count']} done rows` |",
            "",
            "## Bot Evaluation",
            "",
            "| Bot | Final HP | Survival Rate | Survived Round | Board Strength |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for bot_name, metrics in evaluation.items():
        lines.append(
            f"| {bot_name} | {metrics['mean_final_hp']:.2f} | "
            f"{metrics['survival_rate']:.2f} | {metrics['mean_survived_round']:.2f} | "
            f"{metrics['mean_final_board_strength']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Trace Preview",
            "",
            "| Step | Action | Reward | Round | HP | Gold | Level | Board Strength | Done |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["trace"]:
        lines.append(
            f"| {row['step']} | `{row['action_name']}` | {row['reward']:.3f} | "
            f"{row['round']} | {row['hp']} | {row['gold']} | {row['level']} | "
            f"{row['board_strength']:.1f} | `{row['done']}` |"
        )

    failure_lines = _failure_output(report)
    if failure_lines:
        lines.extend(["", "## Failure Output", "", *failure_lines])

    lines.extend(
        [
            "",
            "## Debug Notes",
            "",
            "- `steps/sec` is bot-driven Python env throughput, not vectorized rollout throughput.",
            "- `transitions/sec` includes dataset generation and compressed `.npz` write time.",
            "- `trace` is a compact action/state preview for coding-agent debugging.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_command(name: str, cmd: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "name": name,
        "cmd": cmd,
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "stdout_tail": _tail(result.stdout),
        "stderr_tail": _tail(result.stderr),
    }


def _run_benchmark(episodes: int, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    bot = RandomBot()
    steps = 0
    for offset in range(episodes):
        episode_seed = seed + offset
        env = MiniTFTGymEnv(config=EnvConfig(seed=episode_seed))
        rng = np.random.default_rng(episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = bot.act(env, obs, rng)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
    elapsed = time.perf_counter() - started
    return {
        "episodes": episodes,
        "steps": steps,
        "elapsed_sec": round(elapsed, 3),
        "steps_per_sec": steps / elapsed if elapsed else 0.0,
    }


def _run_dataset(config: SmokeConfig) -> dict[str, Any]:
    output = config.dataset_out
    if output is None:
        output = Path(tempfile.gettempdir()) / "tft_zero_sim_smoke_dataset.npz"

    started = time.perf_counter()
    generate_dataset(config.dataset_episodes, output, config.seed)
    elapsed = time.perf_counter() - started
    dataset = load_dataset(output)
    first_returns = dataset.returns[dataset.step_idxs == 0]
    episodes = len(set(dataset.episode_ids.tolist()))
    transitions = len(dataset.actions)
    return {
        "path": str(output),
        "file_mb": output.stat().st_size / 1_000_000,
        "episodes": episodes,
        "transitions": transitions,
        "elapsed_sec": round(elapsed, 3),
        "transitions_per_sec": transitions / elapsed if elapsed else 0.0,
        "obs_shape": list(dataset.obs.shape),
        "mask_shape": list(dataset.masks.shape),
        "done_count": int(dataset.dones.sum()),
        "mean_return": float(first_returns.mean()) if len(first_returns) else 0.0,
    }


def _run_evaluation(episodes: int) -> dict[str, dict[str, float]]:
    seeds = range(1000, 1000 + episodes)
    bots: list[BaseBot] = [
        RandomBot(),
        GreedyBoardBot(),
        EconBot(),
        RerollBot(),
        FastLevelBot(),
        TraitCommitBot("ranger", "TraitCommitBot[ranger]"),
    ]
    return {bot.name: evaluate_bot(bot, seeds) for bot in bots}


def _run_trace(seed: int, max_steps: int) -> list[dict[str, Any]]:
    env = MiniTFTGymEnv(config=EnvConfig(seed=seed))
    bot = GreedyBoardBot()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    terminated = truncated = False
    for step in range(max_steps):
        if terminated or truncated:
            break
        action = bot.act(env, obs, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "step": step,
                "action": int(action),
                "action_name": action_name(int(action)),
                "reward": float(reward),
                "round": int(info["round"]),
                "hp": int(info["hp"]),
                "gold": int(info["gold"]),
                "level": int(info["level"]),
                "board_strength": float(info["board_strength"]),
                "done": bool(terminated or truncated),
            }
        )
    return rows


def _failure_output(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for check in report["checks"]:
        if check["ok"]:
            continue
        lines.extend(
            [
                f"### `{check['name']}`",
                "",
                f"Command: `{' '.join(check['cmd'])}`",
                f"Return code: `{check['returncode']}`",
                "",
            ]
        )
        if check["stdout_tail"]:
            lines.extend(["Stdout tail:", "", "```text", check["stdout_tail"], "```", ""])
        if check["stderr_tail"]:
            lines.extend(["Stderr tail:", "", "```text", check["stderr_tail"], "```", ""])
    if report["failures"]:
        lines.extend(["Failures:", ""])
        lines.extend(f"- {failure}" for failure in report["failures"])
    return lines


def _tail(text: str, lines: int = 40) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return "\n".join(stripped.splitlines()[-lines:])


def _status(ok: bool) -> str:
    return "pass" if ok else "fail"


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-episodes", type=int, default=100)
    parser.add_argument("--dataset-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--trace-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-out", type=Path, default=None)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args(argv)

    report = run_smoke(
        SmokeConfig(
            benchmark_episodes=args.benchmark_episodes,
            dataset_episodes=args.dataset_episodes,
            eval_episodes=args.eval_episodes,
            trace_steps=args.trace_steps,
            seed=args.seed,
            dataset_out=args.dataset_out,
            run_checks=not args.skip_checks,
        )
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    else:
        print(format_markdown(report), end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
