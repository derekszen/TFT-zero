"""Benchmark emulated versus native Puffer lobby PPO env backends."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import NUM_ACTIONS
from mini_tft.core.config import EnvConfig
from mini_tft.rl.train_puffer_ppo import PufferBackend, make_training_env_batch


def run_backend_benchmarks(
    *,
    out_dir: Path,
    seed: int,
    repeats: int,
    steps: int,
    num_envs: int,
    players: int,
    max_round: int,
    max_actions_per_round: int,
    max_actions_per_player: int | None,
    opponent_policy: str,
    allow_oracle_macro_actions: bool,
    smoke_only: bool,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "seed": seed,
        "repeats": repeats,
        "steps": steps,
        "num_envs": num_envs,
        "players": players,
        "max_round": max_round,
        "max_actions_per_round": max_actions_per_round,
        "max_actions_per_player": max_actions_per_player,
        "opponent_policy": opponent_policy,
        "allow_oracle_macro_actions": allow_oracle_macro_actions,
        "smoke_only": smoke_only,
    }
    runs: list[dict[str, Any]] = []
    for backend in ("emulated", "native"):
        for repeat in range(repeats):
            runs.append(
                benchmark_backend(
                    backend=cast(PufferBackend, backend),
                    seed=seed + repeat * 10_000,
                    repeat=repeat,
                    steps=steps,
                    num_envs=num_envs,
                    players=players,
                    max_round=max_round,
                    max_actions_per_round=max_actions_per_round,
                    max_actions_per_player=max_actions_per_player,
                    opponent_policy=opponent_policy,
                    allow_oracle_macro_actions=allow_oracle_macro_actions,
                )
            )
    report = {
        "kind": "puffer_backend_benchmark",
        "settings": settings,
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    metrics_path = out_dir / "metrics.json"
    decision_path = out_dir / "decision.md"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision_path.write_text(format_decision(report), encoding="utf-8")
    return report


def benchmark_backend(
    *,
    backend: PufferBackend,
    seed: int,
    repeat: int,
    steps: int,
    num_envs: int,
    players: int,
    max_round: int,
    max_actions_per_round: int,
    max_actions_per_player: int | None,
    opponent_policy: str,
    allow_oracle_macro_actions: bool,
) -> dict[str, Any]:
    config = EnvConfig(
        seed=seed,
        max_round=max_round,
        max_actions_per_round=max_actions_per_round,
    )
    env_batch = make_training_env_batch(
        env_kind="lobby",
        puffer_backend=backend,
        seed=seed,
        config=config,
        num_envs=num_envs,
        players=players,
        max_actions_per_player=max_actions_per_player,
        lobby_opponent_policy=opponent_policy,
        allow_oracle_macro_actions=allow_oracle_macro_actions,
    )
    rng = np.random.default_rng(seed)
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    next_reset_seeds = np.arange(seed + 10_000, seed + 10_000 + num_envs)
    try:
        obs, _infos = env_batch.reset(seed)
        started = time.perf_counter()
        for _ in range(steps):
            actions = sample_legal_actions(obs, rng=rng)
            next_obs, rewards, terminals, truncations, _step_infos = env_batch.step(actions)
            dones = np.logical_or(terminals, truncations)
            for env_index, done_value in enumerate(dones):
                episode_returns[env_index] += float(rewards[env_index])
                episode_lengths[env_index] += 1
                if bool(done_value):
                    completed_returns.append(float(episode_returns[env_index]))
                    completed_lengths.append(int(episode_lengths[env_index]))
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0
                    if not env_batch.auto_resets:
                        reset_obs, _ = env_batch.reset_one(
                            env_index,
                            seed=int(next_reset_seeds[env_index]),
                        )
                        next_obs[env_index] = reset_obs
                    next_reset_seeds[env_index] += num_envs
            obs = np.asarray(next_obs, dtype=np.float32)
        elapsed = time.perf_counter() - started
    finally:
        env_batch.close()

    total_agent_steps = steps * num_envs
    return {
        "backend": backend,
        "repeat": repeat,
        "seed": seed,
        "num_envs": num_envs,
        "steps": steps,
        "agent_steps": total_agent_steps,
        "wall_clock_seconds": elapsed,
        "steps_per_sec": total_agent_steps / elapsed if elapsed > 0 else None,
        "episodes_per_sec": len(completed_returns) / elapsed if elapsed > 0 else None,
        "completed_episodes": len(completed_returns),
        "mean_episode_return": (
            float(np.mean(completed_returns)) if completed_returns else None
        ),
        "mean_episode_length": (
            float(np.mean(completed_lengths)) if completed_lengths else None
        ),
        "device": "cpu",
    }


def sample_legal_actions(
    obs: NDArray[np.float32],
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    actions = np.zeros(obs.shape[0], dtype=np.int64)
    masks = obs[:, -NUM_ACTIONS:] > 0.5
    for env_index, mask in enumerate(masks):
        legal = np.flatnonzero(mask)
        if legal.size:
            actions[env_index] = int(rng.choice(legal))
    return actions


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for backend in ("emulated", "native"):
        backend_runs = [run for run in runs if run["backend"] == backend]
        summary[backend] = {
            "mean_steps_per_sec": _mean_metric(backend_runs, "steps_per_sec"),
            "mean_episodes_per_sec": _mean_metric(backend_runs, "episodes_per_sec"),
            "completed_episodes": int(sum(run["completed_episodes"] for run in backend_runs)),
            "mean_episode_return": _mean_metric(backend_runs, "mean_episode_return"),
            "mean_episode_length": _mean_metric(backend_runs, "mean_episode_length"),
        }
    native_sps = summary["native"]["mean_steps_per_sec"]
    emulated_sps = summary["emulated"]["mean_steps_per_sec"]
    if native_sps is None or emulated_sps is None:
        verdict: Literal["faster", "slower", "inconclusive"] = "inconclusive"
        speedup = None
    else:
        speedup = native_sps / emulated_sps if emulated_sps else None
        if speedup is None:
            verdict = "inconclusive"
        elif speedup >= 1.05:
            verdict = "faster"
        elif speedup <= 0.95:
            verdict = "slower"
        else:
            verdict = "inconclusive"
    summary["decision"] = {
        "native_vs_emulated": verdict,
        "native_speedup": speedup,
    }
    return summary


def _mean_metric(runs: list[dict[str, Any]], metric: str) -> float | None:
    values = [run[metric] for run in runs if run[metric] is not None]
    if not values:
        return None
    return float(np.mean(values))


def format_decision(report: dict[str, Any]) -> str:
    settings = report["settings"]
    summary = report["summary"]
    decision = summary["decision"]
    label = "smoke-only" if settings["smoke_only"] or settings["repeats"] < 2 else "repeated"
    rows = [
        "# Puffer Backend Benchmark Decision",
        "",
        f"Run label: `{label}`",
        f"Decision: native backend is `{decision['native_vs_emulated']}`.",
        f"Native speedup: `{decision['native_speedup']}`",
        "",
        "| Backend | Mean steps/sec | Mean episodes/sec | Completed episodes |",
        "| --- | ---: | ---: | ---: |",
    ]
    for backend in ("emulated", "native"):
        backend_summary = summary[backend]
        rows.append(
            "| "
            f"{backend} | "
            f"{backend_summary['mean_steps_per_sec']} | "
            f"{backend_summary['mean_episodes_per_sec']} | "
            f"{backend_summary['completed_episodes']} |"
        )
    rows.extend(
        [
            "",
            "Matched settings:",
            "",
            "```json",
            json.dumps(settings, indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(rows) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/puffer_native_vector"))
    parser.add_argument("--seed", type=int, default=32000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-round", type=int, default=36)
    parser.add_argument("--max-actions-per-round", type=int, default=20)
    parser.add_argument("--max-actions-per-player", type=int, default=None)
    parser.add_argument("--opponent-policy", default="mixed")
    parser.add_argument("--disallow-oracle-macro-actions", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Label the benchmark as smoke-only even when it writes comparable metrics.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_backend_benchmarks(
        out_dir=args.out_dir,
        seed=args.seed,
        repeats=args.repeats,
        steps=args.steps,
        num_envs=args.num_envs,
        players=args.players,
        max_round=args.max_round,
        max_actions_per_round=args.max_actions_per_round,
        max_actions_per_player=args.max_actions_per_player,
        opponent_policy=args.opponent_policy,
        allow_oracle_macro_actions=not args.disallow_oracle_macro_actions,
        smoke_only=args.smoke_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
