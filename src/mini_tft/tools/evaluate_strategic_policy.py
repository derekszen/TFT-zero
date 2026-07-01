"""Write deterministic strategic-lane policy evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_tft.strategic.adapters.analytics import summarize_episode_rows
from mini_tft.strategic.adapters.baselines import (
    StrategicPolicy,
    random_policy,
    tft_heuristic_policy,
    weakest_legal_policy,
    worst_first_policy,
)
from mini_tft.strategic.core import (
    NUM_ACTIONS,
    StrategicConfig,
    legal_action_mask,
    observe,
    placement_proxy,
    reset,
    scenario_score,
    state_signature,
    step,
)
from mini_tft.strategic.core.actions import action_name
from mini_tft.strategic.core.rules import run_episode

POLICY_BY_NAME: dict[str, StrategicPolicy] = {
    "heuristic": tft_heuristic_policy,
    "random": random_policy,
    "weakest_legal": weakest_legal_policy,
    "worst_first": worst_first_policy,
}
DEFAULT_POLICY_NAMES = ("heuristic", "random")


@dataclass(frozen=True)
class PolicySpec:
    name: str
    checkpoint: Path | None = None


@dataclass(frozen=True)
class StrategicPolicyEvaluationConfig:
    out_dir: Path
    policies: tuple[PolicySpec, ...] = tuple(PolicySpec(name) for name in DEFAULT_POLICY_NAMES)
    episodes: int = 32
    seed: int = 0
    verify_run_episode: bool = True


def run_strategic_policy_evaluation(
    config: StrategicPolicyEvaluationConfig,
) -> dict[str, Any]:
    """Evaluate named strategic policies and write auditable artifacts."""

    if config.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not config.policies:
        raise ValueError("at least one policy is required")

    _validate_unique_policies(config.policies)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    strategic_config = StrategicConfig()
    episode_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    policy_summaries: dict[str, Any] = {}

    for spec in config.policies:
        policy = _resolve_policy(spec)
        policy_episode_rows: list[dict[str, Any]] = []
        policy_action_rows: list[dict[str, Any]] = []
        for episode in range(config.episodes):
            episode_seed = config.seed + episode
            episode_row, rows = _run_traced_episode(
                policy,
                policy_name=spec.name,
                episode=episode,
                seed=episode_seed,
                config=strategic_config,
                verify_run_episode=config.verify_run_episode,
            )
            policy_episode_rows.append(episode_row)
            policy_action_rows.extend(rows)

        episode_rows.extend(policy_episode_rows)
        action_rows.extend(policy_action_rows)
        policy_summaries[spec.name] = _summarize_policy_rows(
            policy_episode_rows,
            policy_action_rows,
        )

    report = _build_report(config, episode_rows, action_rows, policy_summaries)
    _write_json(config.out_dir / "metrics.json", report)
    _write_jsonl(config.out_dir / "episodes.jsonl", episode_rows)
    _write_jsonl(config.out_dir / "actions.jsonl", action_rows)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    return report


def build_policy_specs(policy_names: Sequence[str] | None) -> tuple[PolicySpec, ...]:
    names = tuple(policy_names) if policy_names else DEFAULT_POLICY_NAMES
    return tuple(PolicySpec(name=name) for name in names)


def _run_traced_episode(
    policy: StrategicPolicy,
    *,
    policy_name: str,
    episode: int,
    seed: int,
    config: StrategicConfig,
    verify_run_episode: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = reset(seed=seed, config=config)
    total_reward = 0.0
    steps = 0
    rows: list[dict[str, Any]] = []
    max_steps = config.max_round * (config.max_actions_per_round + 1)

    while not state.done and steps < max_steps:
        observation = observe(state, config)
        mask = legal_action_mask(state, config)
        action = int(policy(state, mask, config))
        legal = 0 <= action < len(mask) and bool(mask[action])
        result = step(state, action, config)
        info: dict[str, Any] = dict(result.info)
        total_reward += result.reward
        rows.append(
            {
                "policy": policy_name,
                "episode": episode,
                "seed": seed,
                "step": steps,
                "round": int(info["round"]),
                "action_id": action,
                "action": action_name(action),
                "legal": legal,
                "legal_action_count": int(mask.sum()),
                "legal_actions": [
                    action_name(action_id) for action_id, is_legal in enumerate(mask) if is_legal
                ],
                "observation_dim": int(observation.shape[0]),
                "reward": float(result.reward),
                "ended_round": bool(info["ended_round"]),
                "hp": int(info["hp"]),
                "gold": int(info["gold"]),
                "level": int(info["level"]),
                "placement_proxy": int(info["placement_proxy"]),
                "scenario_score": float(info["scenario_score"]),
            }
        )
        steps += 1

    if verify_run_episode:
        reference_state, reference_reward, reference_steps = run_episode(
            policy,
            seed=seed,
            config=config,
        )
        if (
            state_signature(reference_state) != state_signature(state)
            or abs(reference_reward - total_reward) > 1e-9
            or reference_steps != steps
        ):
            raise RuntimeError(
                f"traced episode diverged from run_episode for {policy_name} seed {seed}"
            )

    episode_row = {
        "policy": policy_name,
        "episode": episode,
        "seed": seed,
        "placement": placement_proxy(state, config),
        "final_round": state.round,
        "hp": state.hp,
        "final_reason": state.final_reason,
        "scenario_score": scenario_score(state, config),
        "illegal_actions": state.total_illegal_actions,
        "total_reward": total_reward,
        "steps": steps,
    }
    return episode_row, rows


def _summarize_policy_rows(
    episode_rows: Sequence[dict[str, Any]],
    action_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    episode_summary = summarize_episode_rows(episode_rows)
    action_counts = _action_counts(action_rows)
    total_actions = sum(action_counts.values())
    scenario_scores = [float(row["scenario_score"]) for row in episode_rows]
    placements = [int(row["placement"]) for row in episode_rows]
    final_reasons = Counter(str(row["final_reason"]) for row in episode_rows)
    illegal_action_count = sum(int(row["illegal_actions"]) for row in episode_rows)

    return {
        **episode_summary,
        "mean_placement": _mean(placements),
        "mean_scenario_score": _mean(scenario_scores),
        "illegal_action_count": illegal_action_count,
        "total_actions": total_actions,
        "action_counts": action_counts,
        "action_rates": {
            action: _rate(count, total_actions) for action, count in action_counts.items()
        },
        "final_reason_counts": {
            reason: int(count) for reason, count in sorted(final_reasons.items())
        },
        "mean_reward": _mean(float(row["total_reward"]) for row in episode_rows),
        "mean_steps": _mean(int(row["steps"]) for row in episode_rows),
    }


def _build_report(
    config: StrategicPolicyEvaluationConfig,
    episode_rows: Sequence[dict[str, Any]],
    action_rows: Sequence[dict[str, Any]],
    policy_summaries: dict[str, Any],
) -> dict[str, Any]:
    illegal_action_count = sum(
        int(summary["illegal_action_count"]) for summary in policy_summaries.values()
    )
    status = "smoke_only" if illegal_action_count == 0 else "fail"
    return {
        "schema": "strategic-policy-eval/v1",
        "status": status,
        "seed": config.seed,
        "episodes_per_policy": config.episodes,
        "policies": [spec.name for spec in config.policies],
        "policy_summaries": policy_summaries,
        "overall": {
            **summarize_episode_rows(episode_rows),
            "illegal_action_count": illegal_action_count,
            "total_actions": len(action_rows),
        },
        "comparison": _comparison(policy_summaries),
        "artifacts": ["metrics.json", "decision.md", "episodes.jsonl", "actions.jsonl"],
        "future_extensions": [
            (
                "A strategic PPO checkpoint policy can be added by resolving a checkpoint "
                "into a StrategicPolicy that consumes observe(state) and legal_action_mask(state)."
            )
        ],
        "known_limits": [
            "placement_proxy is an elimination-timing bucket, not a real lobby placement",
            "scenario_score is a dense smoke metric, not a promoted quality claim",
            "PPO checkpoint loading is intentionally not coupled to trainer internals yet",
        ],
    }


def _comparison(policy_summaries: dict[str, Any]) -> dict[str, Any]:
    if not policy_summaries:
        return {}
    return {
        "best_by_mean_scenario_score": max(
            policy_summaries,
            key=lambda name: float(policy_summaries[name]["mean_scenario_score"]),
        ),
        "best_by_mean_placement": min(
            policy_summaries,
            key=lambda name: float(policy_summaries[name]["mean_placement"]),
        ),
    }


def _resolve_policy(spec: PolicySpec) -> StrategicPolicy:
    if spec.checkpoint is not None:
        raise NotImplementedError(
            "strategic PPO checkpoint policy loading is reserved for a future trainer adapter"
        )
    try:
        return POLICY_BY_NAME[spec.name]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICY_BY_NAME))
        message = f"unknown strategic policy {spec.name!r}; choose one of: {choices}"
        raise ValueError(message) from exc


def _validate_unique_policies(policies: Sequence[PolicySpec]) -> None:
    names = [spec.name for spec in policies]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate strategic policies are not supported: {', '.join(duplicates)}")


def _action_counts(action_rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(row["action"]) for row in action_rows)
    counts = {
        action_name(action_id): int(counter[action_name(action_id)])
        for action_id in range(NUM_ACTIONS)
    }
    for name, count in sorted(counter.items()):
        if name not in counts:
            counts[name] = int(count)
    return counts


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _format_decision(report: dict[str, Any]) -> str:
    lines = [
        "# Strategic Policy Evaluation",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Evidence",
        "",
        f"- Seed: {report['seed']}",
        f"- Episodes per policy: {report['episodes_per_policy']}",
        f"- Policies: {', '.join(report['policies'])}",
        f"- Artifacts: {', '.join(report['artifacts'])}",
        "",
        (
            "| Policy | Mean placement | Survivor rate | Mean final round | "
            "Mean scenario score | Illegal actions |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in report["policies"]:
        summary = report["policy_summaries"][policy]
        lines.append(
            "| "
            f"{policy} | "
            f"{summary['mean_placement']:.3f} | "
            f"{summary['survivor_rate']:.3f} | "
            f"{summary['mean_final_round']:.3f} | "
            f"{summary['mean_scenario_score']:.3f} | "
            f"{summary['illegal_action_count']} |"
        )

    lines.extend(
        [
            "",
            "## Action Rates",
            "",
        ]
    )
    for policy in report["policies"]:
        summary = report["policy_summaries"][policy]
        nonzero_rates = [
            f"{name}={rate:.3f}"
            for name, rate in summary["action_rates"].items()
            if rate > 0.0
        ]
        lines.append(f"- {policy}: {', '.join(nonzero_rates) if nonzero_rates else 'none'}")

    lines.extend(
        [
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
            "## Next",
            "",
            (
                "- Add a strategic PPO checkpoint loader when the trainer checkpoint "
                "contract is stable."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        action="append",
        choices=sorted(POLICY_BY_NAME),
        help="Policy to evaluate. May be repeated; defaults to heuristic and random.",
    )
    parser.add_argument("--no-run-episode-check", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_strategic_policy_evaluation(
        StrategicPolicyEvaluationConfig(
            out_dir=args.out_dir,
            policies=build_policy_specs(args.policy),
            episodes=args.episodes,
            seed=args.seed,
            verify_run_episode=not args.no_run_episode_check,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] != "smoke_only" else 0


if __name__ == "__main__":
    raise SystemExit(main())
