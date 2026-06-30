"""Run a fast simulator-backed MCTS smoke on the strategic lane."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from mini_tft.strategic.adapters.baselines import random_policy, tft_heuristic_policy
from mini_tft.strategic.adapters.mcts import StrategicMCTSConfig, StrategicMCTSPlanner
from mini_tft.strategic.core import (
    StrategicConfig,
    legal_action_mask,
    placement_proxy,
    reset,
    scenario_score,
    step,
)
from mini_tft.strategic.core.actions import action_name
from mini_tft.strategic.core.rules import run_episode


@dataclass(frozen=True)
class StrategicMCTSSmokeConfig:
    out_dir: Path
    backend: str = "python"
    episodes: int = 4
    seed: int = 0
    simulations: tuple[int, ...] = (8, 16)
    max_depth: int = 10
    rollout_steps: int = 6
    prior_mode: str = "uniform"


def run_strategic_mcts_smoke(config: StrategicMCTSSmokeConfig) -> dict[str, Any]:
    if config.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not config.simulations:
        raise ValueError("at least one simulation count is required")
    if config.backend not in {"python", "native"}:
        raise ValueError(f"unknown MCTS backend: {config.backend}")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    if config.backend == "native":
        from mini_tft.strategic.native import run_native_mcts_smoke

        native_result = run_native_mcts_smoke(
            episodes=config.episodes,
            seed=config.seed,
            simulations=config.simulations,
            max_depth=config.max_depth,
            rollout_steps=config.rollout_steps,
            prior_mode=config.prior_mode,
        )
        episode_rows = list(native_result["episode_rows"])
        decision_rows = list(native_result["decision_rows"])
    else:
        simulator_config = StrategicConfig()
        episode_rows = []
        decision_rows = []

        for name, policy in (("random", random_policy), ("heuristic", tft_heuristic_policy)):
            for episode in range(config.episodes):
                seed = config.seed + episode
                state, total_reward, steps = run_episode(policy, seed=seed, config=simulator_config)
                episode_rows.append(
                    _episode_row(
                        policy=name,
                        episode=episode,
                        seed=seed,
                        total_reward=total_reward,
                        steps=steps,
                        state=state,
                        config=simulator_config,
                        decisions=0,
                        elapsed_sec=0.0,
                        simulations=0,
                    )
                )

        for simulations in config.simulations:
            policy_name = f"mcts_{simulations}"
            for episode in range(config.episodes):
                seed = config.seed + episode
                row, decisions = _run_mcts_episode(
                    policy_name=policy_name,
                    episode=episode,
                    seed=seed,
                    simulations=simulations,
                    config=config,
                    simulator_config=simulator_config,
                )
                episode_rows.append(row)
                decision_rows.extend(decisions)

    elapsed_sec = perf_counter() - started

    policy_summaries = _summarize_policies(episode_rows, decision_rows)
    report = {
        "schema": "strategic-mcts-smoke/v1",
        "backend": config.backend,
        "status": "smoke_only" if _illegal_action_count(episode_rows) == 0 else "fail",
        "seed": config.seed,
        "episodes_per_policy": config.episodes,
        "simulations": list(config.simulations),
        "config": {
            "max_depth": config.max_depth,
            "rollout_steps": config.rollout_steps,
            "prior_mode": config.prior_mode,
        },
        "policy_summaries": policy_summaries,
        "comparison": _comparison(policy_summaries),
        "elapsed_sec": elapsed_sec,
        "artifacts": ["metrics.json", "decision.md", "paper_table.md", "episodes.jsonl"],
        "known_limits": [
            "simulator-backed MCTS uses the real strategic simulator, not learned dynamics",
            "placement_proxy is an elimination bucket, not real TFT placement",
            "this is a smoke run unless episode counts and seed ranges are expanded",
        ],
    }
    if decision_rows:
        report["artifacts"].append("decisions.jsonl")

    _write_json(config.out_dir / "metrics.json", report)
    _write_jsonl(config.out_dir / "episodes.jsonl", episode_rows)
    _write_jsonl(config.out_dir / "decisions.jsonl", decision_rows)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    (config.out_dir / "paper_table.md").write_text(_format_paper_table(report), encoding="utf-8")
    return report


def _run_mcts_episode(
    *,
    policy_name: str,
    episode: int,
    seed: int,
    simulations: int,
    config: StrategicMCTSSmokeConfig,
    simulator_config: StrategicConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = reset(seed=seed, config=simulator_config)
    planner = StrategicMCTSPlanner(
        config=StrategicMCTSConfig(
            simulations=simulations,
            max_depth=config.max_depth,
            rollout_steps=config.rollout_steps,
            prior_mode=config.prior_mode,
        ),
        simulator_config=simulator_config,
    )
    rows: list[dict[str, Any]] = []
    total_reward = 0.0
    steps = 0
    started = perf_counter()
    max_steps = simulator_config.max_round * (simulator_config.max_actions_per_round + 1)

    while not state.done and steps < max_steps:
        mask = legal_action_mask(state, simulator_config)
        decision = planner.plan(state)
        action = int(decision.selected_action)
        legal = 0 <= action < len(mask) and bool(mask[action])
        result = step(state, action, simulator_config)
        total_reward += result.reward
        rows.append(
            {
                "policy": policy_name,
                "episode": episode,
                "seed": seed,
                "step": steps,
                "round": _as_int(result.info["round"]),
                "action_id": action,
                "action": action_name(action),
                "legal": legal,
                "reward": float(result.reward),
                "ended_round": _as_bool(result.info["ended_round"]),
                "hp": _as_int(result.info["hp"]),
                "gold": _as_int(result.info["gold"]),
                "level": _as_int(result.info["level"]),
                "placement_proxy": _as_int(result.info["placement_proxy"]),
                "scenario_score": _as_float(result.info["scenario_score"]),
                "simulations": simulations,
                "mcts_elapsed_ms": decision.elapsed_ms,
                "mcts_max_depth": decision.max_depth,
                "visit_policy": decision.visit_policy.astype(float).tolist(),
                "action_visits": {
                    action_name(action_id): visits
                    for action_id, visits in decision.action_visits.items()
                },
                "action_values": {
                    action_name(action_id): value
                    for action_id, value in decision.action_values.items()
                },
            }
        )
        steps += 1

    elapsed_sec = perf_counter() - started
    return (
        _episode_row(
            policy=policy_name,
            episode=episode,
            seed=seed,
            total_reward=total_reward,
            steps=steps,
            state=state,
            config=simulator_config,
            decisions=len(rows),
            elapsed_sec=elapsed_sec,
            simulations=simulations,
        ),
        rows,
    )


def _episode_row(
    *,
    policy: str,
    episode: int,
    seed: int,
    total_reward: float,
    steps: int,
    state: Any,
    config: StrategicConfig,
    decisions: int,
    elapsed_sec: float,
    simulations: int,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "episode": episode,
        "seed": seed,
        "placement": placement_proxy(state, config),
        "final_round": int(state.round),
        "hp": int(state.hp),
        "final_reason": state.final_reason,
        "scenario_score": scenario_score(state, config),
        "illegal_actions": int(state.total_illegal_actions),
        "total_reward": float(total_reward),
        "steps": int(steps),
        "decisions": int(decisions),
        "simulations": int(simulations),
        "elapsed_sec": float(elapsed_sec),
    }


def _summarize_policies(
    episode_rows: Sequence[dict[str, Any]],
    decision_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in episode_rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)
    decisions_by_policy = Counter(str(row["policy"]) for row in decision_rows)
    return {
        policy: _summarize_rows(rows, int(decisions_by_policy[policy]))
        for policy, rows in sorted(by_policy.items())
    }


def _summarize_rows(rows: Sequence[dict[str, Any]], decision_count: int) -> dict[str, Any]:
    placements = [int(row["placement"]) for row in rows]
    final_rounds = [int(row["final_round"]) for row in rows]
    hps = [int(row["hp"]) for row in rows]
    scenario_scores = [float(row["scenario_score"]) for row in rows]
    rewards = [float(row["total_reward"]) for row in rows]
    elapsed_sec = sum(float(row["elapsed_sec"]) for row in rows)
    simulations = max(int(row["simulations"]) for row in rows)
    return {
        "episodes": len(rows),
        "simulations": simulations,
        "mean_placement": _mean(placements),
        "placement_counts": {str(key): int(value) for key, value in Counter(placements).items()},
        "mean_final_round": _mean(final_rounds),
        "mean_hp": _mean(hps),
        "death_rate": _rate(sum(str(row["final_reason"]) == "hp_zero" for row in rows), len(rows)),
        "survivor_rate": _rate(
            sum(str(row["final_reason"]) == "max_round" for row in rows),
            len(rows),
        ),
        "mean_scenario_score": _mean(scenario_scores),
        "mean_reward": _mean(rewards),
        "illegal_action_count": sum(int(row["illegal_actions"]) for row in rows),
        "total_decisions": decision_count,
        "elapsed_sec": elapsed_sec,
        "decisions_per_sec": decision_count / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "simulations_per_sec": (
            (decision_count * simulations) / elapsed_sec
            if elapsed_sec > 0.0 and simulations > 0
            else 0.0
        ),
    }


def _comparison(policy_summaries: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_by_mean_placement": min(
            policy_summaries,
            key=lambda name: float(policy_summaries[name]["mean_placement"]),
        ),
        "best_by_mean_scenario_score": max(
            policy_summaries,
            key=lambda name: float(policy_summaries[name]["mean_scenario_score"]),
        ),
        "best_by_mean_reward": max(
            policy_summaries,
            key=lambda name: float(policy_summaries[name]["mean_reward"]),
        ),
    }


def _illegal_action_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(int(row["illegal_actions"]) for row in rows)


def _format_decision(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Strategic MCTS Smoke",
            "",
            f"Status: `{report['status']}`",
            "",
            "## Evidence",
            "",
            f"- Seed: {report['seed']}",
            f"- Episodes per policy: {report['episodes_per_policy']}",
            f"- Simulation counts: {', '.join(str(value) for value in report['simulations'])}",
            f"- Elapsed sec: {report['elapsed_sec']:.3f}",
            "",
            _format_paper_table(report),
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )


def _format_paper_table(report: dict[str, Any]) -> str:
    lines = [
        (
            "| Policy | Sims | Mean placement | Mean final round | Death rate | "
            "Scenario score | Reward | Decisions/sec | Sims/sec |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, summary in report["policy_summaries"].items():
        lines.append(
            "| "
            f"{policy} | "
            f"{summary['simulations']} | "
            f"{summary['mean_placement']:.3f} | "
            f"{summary['mean_final_round']:.3f} | "
            f"{summary['death_rate']:.3f} | "
            f"{summary['mean_scenario_score']:.3f} | "
            f"{summary['mean_reward']:.3f} | "
            f"{summary['decisions_per_sec']:.2f} | "
            f"{summary['simulations_per_sec']:.2f} |"
        )
    return "\n".join(lines)


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return float(sum(values_list) / len(values_list)) if values_list else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _as_int(value: Any) -> int:
    return int(value) if value is not None else 0


def _as_float(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _as_bool(value: Any) -> bool:
    return bool(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/strategic_lane/mcts_smoke"))
    parser.add_argument("--backend", choices=["python", "native"], default="python")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulations", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--prior-mode", choices=["uniform", "heuristic"], default="uniform")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_strategic_mcts_smoke(
        StrategicMCTSSmokeConfig(
            out_dir=args.out_dir,
            backend=args.backend,
            episodes=args.episodes,
            seed=args.seed,
            simulations=tuple(args.simulations),
            max_depth=args.max_depth,
            rollout_steps=args.rollout_steps,
            prior_mode=args.prior_mode,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] != "smoke_only" else 0


if __name__ == "__main__":
    raise SystemExit(main())
