"""Repeated-seed policy-improvement gate for strategic MuZero Torch checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mini_tft.strategic.core.actions import NUM_ACTIONS, action_name
from mini_tft.tools.train_strategic_muzero_torch import (
    StrategicTorchPolicyEvalConfig,
    run_torch_checkpoint_policy_evaluation,
)

REQUIRED_POLICIES = ("random", "weakest_legal", "worst_first", "heuristic", "torch_muzero")
DEFAULT_SEEDS = (6100, 6200, 6300, 6400, 6500)


@dataclass(frozen=True)
class StrategicPolicyImprovementGateConfig:
    checkpoint_path: Path
    out_dir: Path
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    episodes_per_seed: int = 512
    device: str = "cpu"


def run_strategic_policy_improvement_gate(
    config: StrategicPolicyImprovementGateConfig,
) -> dict[str, Any]:
    """Run repeated policy evals and write an ACCEPT/REJECT promotion packet."""

    _validate_config(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    seed_reports: list[dict[str, Any]] = []
    loop_events: list[dict[str, Any]] = [
        {
            "timestamp": started_at,
            "attempt": 1,
            "stage": "start",
            "message": "policy improvement gate started",
            "validation": "checkpoint_exists",
            "next_action": "run repeated-seed policy evaluations",
        }
    ]

    seed_root = config.out_dir / "seed_evals"
    for seed in config.seeds:
        seed_out_dir = seed_root / f"seed_{seed}"
        report = run_torch_checkpoint_policy_evaluation(
            StrategicTorchPolicyEvalConfig(
                checkpoint_path=config.checkpoint_path,
                out_dir=seed_out_dir,
                episodes=config.episodes_per_seed,
                seed=seed,
                device=config.device,
            )
        )
        _validate_seed_report(report, seed)
        seed_reports.append(report)
        loop_events.append(
            {
                "timestamp": _utc_now(),
                "attempt": 1,
                "stage": "seed_eval",
                "seed": seed,
                "message": "completed seed evaluation",
                "status": report["status"],
                "out_dir": str(seed_out_dir),
                "validation": "seed metrics, actions, episodes, decision artifacts written",
                "next_action": "continue remaining seeds",
            }
        )

    aggregate = _aggregate_seed_reports(seed_reports)
    promotion = _promotion_decision(seed_reports, aggregate)
    finished_at = _utc_now()
    report = {
        "schema": "strategic-policy-improvement-gate/v1",
        "status": "pass" if promotion["verdict"] == "ACCEPT" else "reject",
        "verdict": promotion["verdict"],
        "started_at": started_at,
        "finished_at": finished_at,
        "checkpoint_path": str(config.checkpoint_path),
        "config": {
            "seeds": list(config.seeds),
            "episodes_per_seed": config.episodes_per_seed,
            "device": config.device,
        },
        "policies": list(REQUIRED_POLICIES),
        "seed_results": _seed_results(seed_reports),
        "aggregate_policy_summaries": aggregate,
        "promotion_decision": promotion,
        "artifacts": [
            "metrics.json",
            "decision.md",
            "final_report.md",
            "loop-state.json",
            "loop-run-log.md",
            "seed_evals/",
        ],
        "known_limits": [
            "placement_proxy is an elimination-timing bucket, not real TFT placement",
            "this evaluates greedy Torch policy-head actions, not model-backed search",
            "ACCEPT/REJECT is scoped to random and weakest_legal majority wins plus legality",
        ],
    }

    loop_events.append(
        {
            "timestamp": finished_at,
            "attempt": 1,
            "stage": "final_report",
            "message": "policy improvement gate finished",
            "verdict": promotion["verdict"],
            "validation": "metrics, decision, final_report, loop-state, loop-run-log written",
            "next_action": _suggested_loop_action(report),
        }
    )
    _write_json(config.out_dir / "metrics.json", report)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    (config.out_dir / "final_report.md").write_text(
        _format_final_report(report),
        encoding="utf-8",
    )
    _write_json(config.out_dir / "loop-state.json", _loop_state(report))
    (config.out_dir / "loop-run-log.md").write_text(
        _format_loop_log(loop_events),
        encoding="utf-8",
    )
    return report


def _validate_config(config: StrategicPolicyImprovementGateConfig) -> None:
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {config.checkpoint_path}")
    if config.episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be positive")
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be unique")


def _validate_seed_report(report: Mapping[str, Any], seed: int) -> None:
    policies = tuple(str(policy) for policy in report.get("policies", ()))
    missing = [policy for policy in REQUIRED_POLICIES if policy not in policies]
    if missing:
        raise RuntimeError(f"seed {seed} report is missing policies: {', '.join(missing)}")
    if int(report.get("seed", -1)) != int(seed):
        raise RuntimeError(f"seed report has wrong seed: expected {seed}, got {report.get('seed')}")


def _aggregate_seed_reports(seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for policy in REQUIRED_POLICIES:
        summaries = [
            dict(dict(report["policy_summaries"])[policy])
            for report in seed_reports
        ]
        aggregate[policy] = _aggregate_policy_summaries(summaries)
    return aggregate


def _aggregate_policy_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes = sum(int(summary["episodes"]) for summary in summaries)
    total_actions = sum(int(summary["total_actions"]) for summary in summaries)
    action_counts = Counter[str]()
    placement_counts = Counter[str]()
    final_reason_counts = Counter[str]()
    for summary in summaries:
        action_counts.update(
            {str(action): int(count) for action, count in dict(summary["action_counts"]).items()}
        )
        placement_counts.update(
            {
                str(placement): int(count)
                for placement, count in dict(summary["placement_counts"]).items()
            }
        )
        final_reason_counts.update(
            {
                str(reason): int(count)
                for reason, count in dict(summary["final_reason_counts"]).items()
            }
        )

    return {
        "seed_count": len(summaries),
        "episodes": episodes,
        "placement_counts": dict(sorted(placement_counts.items())),
        "survivor_rate": _weighted_mean(summaries, "survivor_rate"),
        "mean_final_round": _weighted_mean(summaries, "mean_final_round"),
        "median_final_round_mean": _weighted_mean(summaries, "median_final_round"),
        "min_final_round": min(int(summary["min_final_round"]) for summary in summaries),
        "max_final_round": max(int(summary["max_final_round"]) for summary in summaries),
        "mean_placement": _weighted_mean(summaries, "mean_placement"),
        "mean_scenario_score": _weighted_mean(summaries, "mean_scenario_score"),
        "mean_reward": _weighted_mean(summaries, "mean_reward"),
        "mean_steps": _weighted_mean(summaries, "mean_steps"),
        "illegal_action_count": sum(int(summary["illegal_action_count"]) for summary in summaries),
        "total_actions": total_actions,
        "action_counts": {
            action_name(action_id): int(action_counts[action_name(action_id)])
            for action_id in range(NUM_ACTIONS)
        },
        "action_rates": {
            action_name(action_id): _rate(int(action_counts[action_name(action_id)]), total_actions)
            for action_id in range(NUM_ACTIONS)
        },
        "final_reason_counts": dict(sorted(final_reason_counts.items())),
    }


def _promotion_decision(
    seed_reports: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    seed_count = len(seed_reports)
    required_seed_wins = seed_count // 2 + 1
    torch_summary = dict(aggregate["torch_muzero"])
    baseline_wins = {
        baseline: _count_seed_wins(seed_reports, baseline)
        for baseline in ("random", "weakest_legal", "worst_first", "heuristic")
    }
    criteria = {
        "required_policies_present": True,
        "torch_zero_illegal_actions": int(torch_summary["illegal_action_count"]) == 0,
        "beats_random_most_seeds": baseline_wins["random"] >= required_seed_wins,
        "beats_weakest_legal_most_seeds": baseline_wins["weakest_legal"] >= required_seed_wins,
        "beats_worst_first_most_seeds": baseline_wins["worst_first"] >= required_seed_wins,
        "aggregate_action_distribution_present": bool(torch_summary["action_counts"]),
    }
    failed_checks = [name for name, passed in criteria.items() if not passed]
    return {
        "verdict": "ACCEPT" if not failed_checks else "REJECT",
        "criteria": criteria,
        "failed_checks": failed_checks,
        "seed_count": seed_count,
        "required_seed_wins": required_seed_wins,
        "torch_seed_wins": baseline_wins,
        "comparison_order": [
            "lower mean_placement",
            "higher mean_scenario_score",
            "higher mean_reward",
            "higher mean_final_round",
            "higher survivor_rate",
        ],
    }


def _count_seed_wins(seed_reports: Sequence[Mapping[str, Any]], baseline: str) -> int:
    wins = 0
    for report in seed_reports:
        summaries = dict(report["policy_summaries"])
        if _policy_score(dict(summaries["torch_muzero"])) > _policy_score(
            dict(summaries[baseline])
        ):
            wins += 1
    return wins


def _policy_score(summary: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        -float(summary["mean_placement"]),
        float(summary["mean_scenario_score"]),
        float(summary["mean_reward"]),
        float(summary["mean_final_round"]),
        float(summary["survivor_rate"]),
    )


def _seed_results(seed_reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for report in seed_reports:
        summaries = dict(report["policy_summaries"])
        comparisons = {
            baseline: _policy_score(dict(summaries["torch_muzero"]))
            > _policy_score(dict(summaries[baseline]))
            for baseline in ("random", "weakest_legal", "worst_first", "heuristic")
        }
        results.append(
            {
                "seed": int(report["seed"]),
                "status": str(report["status"]),
                "episodes_per_policy": int(report["episodes_per_policy"]),
                "policies": list(report["policies"]),
                "torch_beats": comparisons,
                "policy_summaries": {
                    policy: summaries[policy] for policy in REQUIRED_POLICIES
                },
                "artifact_dir": str(Path("seed_evals") / f"seed_{report['seed']}"),
            }
        )
    return results


def _weighted_mean(summaries: Sequence[Mapping[str, Any]], key: str) -> float:
    denominator = sum(int(summary["episodes"]) for summary in summaries)
    if denominator <= 0:
        return 0.0
    numerator = sum(float(summary[key]) * int(summary["episodes"]) for summary in summaries)
    return float(numerator / denominator)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _loop_state(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "strategic-policy-improvement-loop-state/v1",
        "owner": "codex",
        "attempt": 1,
        "attempt_cap": 3,
        "current_stage": "final_report",
        "checkpoint_path": report["checkpoint_path"],
        "status": report["status"],
        "verdict": report["verdict"],
        "failed_checks": report["promotion_decision"]["failed_checks"],
        "next_action": _suggested_loop_action(report),
        "pause_kill_criteria": [
            "stop if checkpoint path is missing or unreadable",
            "stop if any seed eval emits illegal actions for torch_muzero",
            "stop if required artifact files are missing after the run",
            "stop after 3 failed implementation or validation attempts",
        ],
        "prune_rules": [
            "keep metrics.json, decision.md, final_report.md, loop-state.json, loop-run-log.md",
            "keep per-seed metrics, decisions, episodes, and action traces for audit",
            (
                "old policy_improvement_gate_* artifacts may be archived after a newer "
                "accepted artifact exists"
            ),
        ],
        "codex_allowance_check": {
            "source": "unknown",
            "checked_at": report["started_at"],
            "five_hour_window_remaining": "unknown",
            "weekly_usage": "unknown",
            "decision": "continue",
            "reason": "single bounded verification run with no GPU training",
        },
    }


def _format_decision(report: Mapping[str, Any]) -> str:
    promotion = dict(report["promotion_decision"])
    aggregate = dict(report["aggregate_policy_summaries"])
    lines = [
        "# Strategic Policy Improvement Gate",
        "",
        f"Verdict: `{report['verdict']}`",
        f"Status: `{report['status']}`",
        f"Suggested loop action: `{_suggested_loop_action(report)}`",
        "",
        "## Promotion Criteria",
        "",
    ]
    for name, passed in dict(promotion["criteria"]).items():
        lines.append(f"- {name}: `{bool(passed)}`")

    lines.extend(
        [
            "",
            "## Seed Wins",
            "",
            f"- Required wins: {promotion['required_seed_wins']} of {promotion['seed_count']}",
        ]
    )
    for baseline, wins in dict(promotion["torch_seed_wins"]).items():
        lines.append(f"- torch_muzero vs {baseline}: {wins}")

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            (
                "| Policy | Mean placement | Survivor rate | Mean final round | "
                "Mean scenario score | Mean reward | Illegal actions |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy in REQUIRED_POLICIES:
        summary = dict(aggregate[policy])
        lines.append(
            "| "
            f"{policy} | "
            f"{float(summary['mean_placement']):.3f} | "
            f"{float(summary['survivor_rate']):.3f} | "
            f"{float(summary['mean_final_round']):.3f} | "
            f"{float(summary['mean_scenario_score']):.3f} | "
            f"{float(summary['mean_reward']):.3f} | "
            f"{int(summary['illegal_action_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Failed Checks",
            "",
        ]
    )
    failed_checks = list(promotion["failed_checks"])
    lines.extend(f"- {check}" for check in failed_checks)
    if not failed_checks:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _format_final_report(report: Mapping[str, Any]) -> str:
    failed_checks = ", ".join(report["promotion_decision"]["failed_checks"]) or "none"
    return "\n".join(
        [
            "# Strategic Policy Improvement Final Report",
            "",
            f"Verdict: `{report['verdict']}`",
            "",
            "## Objective",
            "",
            (
                "Evaluate whether the trained Torch MuZero-style checkpoint beats "
                "random and weakest_legal baselines on most repeated seeds with zero "
                "illegal actions."
            ),
            "",
            "## Evidence",
            "",
            f"- Checkpoint: `{report['checkpoint_path']}`",
            f"- Seeds: {', '.join(str(seed) for seed in report['config']['seeds'])}",
            f"- Episodes per seed: {report['config']['episodes_per_seed']}",
            f"- Policies: {', '.join(report['policies'])}",
            f"- Failed checks: {failed_checks}",
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )


def _format_loop_log(events: Iterable[Mapping[str, Any]]) -> str:
    event_list = list(events)
    checked_at = event_list[0]["timestamp"] if event_list else _utc_now()
    lines = [
        "# Strategic Policy Improvement Loop Run Log",
        "",
        "## Codex Allowance Check",
        "Source: unknown",
        f"Checked at: {checked_at}",
        "5h window remaining: unknown",
        "Weekly usage: unknown",
        "Decision: continue",
        "",
        "## Runs",
        "",
    ]
    for event in event_list:
        detail = ", ".join(
            f"{key}={value}"
            for key, value in event.items()
            if key not in {"timestamp", "message"}
        )
        suffix = f" ({detail})" if detail else ""
        lines.append(f"- {event['timestamp']}: {event['message']}{suffix}")
    lines.append("")
    return "\n".join(lines)


def _suggested_loop_action(report: Mapping[str, Any]) -> str:
    if report["verdict"] == "ACCEPT":
        return "accept weak-baseline gate; do not claim heuristic superiority"
    return "retry after inspecting failed_checks and seed_results"


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Evaluation seed. May be repeated; defaults to the gate seed set.",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(args.seed) if args.seed is not None else DEFAULT_SEEDS
    report = run_strategic_policy_improvement_gate(
        StrategicPolicyImprovementGateConfig(
            checkpoint_path=args.checkpoint_path,
            out_dir=args.out_dir,
            seeds=seeds,
            episodes_per_seed=args.episodes_per_seed,
            device=args.device,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["verdict"] != "ACCEPT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
