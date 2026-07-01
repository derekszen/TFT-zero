"""Run checkpoint-guided strategic MCTS cache smoke and compare to heuristic priors."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mini_tft.tools.generate_strategic_muzero_cache import (
    StrategicMuZeroCacheRunConfig,
    run_strategic_muzero_cache_run,
)


@dataclass(frozen=True)
class CheckpointGuidedMCTSSmokeConfig:
    out_dir: Path
    checkpoint_path: Path
    episodes: int = 512
    max_rows: int = 8192
    seed: int = 0
    simulations: int = 64
    max_depth: int = 10
    rollout_steps: int = 6
    checkpoint_device: str = "cpu"
    skip_determinism_check: bool = False


def run_checkpoint_guided_mcts_smoke(
    config: CheckpointGuidedMCTSSmokeConfig,
) -> dict[str, Any]:
    if not config.checkpoint_path.exists():
        raise ValueError(f"checkpoint does not exist: {config.checkpoint_path}")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_prior_dir = config.out_dir / "checkpoint_prior_cache"
    checkpoint_full_dir = config.out_dir / "checkpoint_full_cache"
    heuristic_dir = config.out_dir / "heuristic_cache"
    checkpoint_prior_report = run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=checkpoint_prior_dir,
            episodes=config.episodes,
            max_rows=config.max_rows,
            seed=config.seed,
            backend="python",
            simulations=config.simulations,
            max_depth=config.max_depth,
            rollout_steps=config.rollout_steps,
            prior_mode="checkpoint",
            value_mode="heuristic",
            checkpoint_path=config.checkpoint_path,
            checkpoint_device=config.checkpoint_device,
            determinism_check=not config.skip_determinism_check,
        )
    )
    checkpoint_full_report = run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=checkpoint_full_dir,
            episodes=config.episodes,
            max_rows=config.max_rows,
            seed=config.seed,
            backend="python",
            simulations=config.simulations,
            max_depth=config.max_depth,
            rollout_steps=config.rollout_steps,
            prior_mode="checkpoint",
            value_mode="checkpoint",
            checkpoint_path=config.checkpoint_path,
            checkpoint_device=config.checkpoint_device,
            determinism_check=not config.skip_determinism_check,
        )
    )
    heuristic_report = run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=heuristic_dir,
            episodes=config.episodes,
            max_rows=config.max_rows,
            seed=config.seed,
            backend="python",
            simulations=config.simulations,
            max_depth=config.max_depth,
            rollout_steps=config.rollout_steps,
            prior_mode="heuristic",
            value_mode="heuristic",
            determinism_check=not config.skip_determinism_check,
        )
    )
    shutil.copyfile(checkpoint_prior_dir / "rows.jsonl", config.out_dir / "rows.jsonl")

    comparison = _comparison(
        checkpoint_prior_report,
        checkpoint_full_report,
        heuristic_report,
    )
    quality_checks = _quality_checks(
        checkpoint_prior_report,
        checkpoint_full_report,
        heuristic_report,
        comparison,
    )
    status = "pass" if all(quality_checks.values()) else "fail"
    report = {
        "schema": "checkpoint-guided-mcts-smoke/v1",
        "status": status,
        "claim_scope": "checkpoint_guided_strategic_mcts_v0",
        "not_claiming": [
            "full iterative MuZero self-play",
            "learned model transitions",
            "real TFT rank or patch quality",
        ],
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint_path": str(config.checkpoint_path),
        "config": {
            "episodes": config.episodes,
            "max_rows": config.max_rows,
            "seed": config.seed,
            "simulations": config.simulations,
            "max_depth": config.max_depth,
            "rollout_steps": config.rollout_steps,
            "checkpoint_device": config.checkpoint_device,
            "skip_determinism_check": config.skip_determinism_check,
        },
        "metrics": {
            "checkpoint_prior": checkpoint_prior_report["metrics"],
            "checkpoint_full": checkpoint_full_report["metrics"],
            "heuristic": heuristic_report["metrics"],
            "comparison": comparison,
            "quality_checks": quality_checks,
            "programmatic_checks_passed": sum(
                1 for passed in quality_checks.values() if passed
            ),
            "programmatic_checks_total": len(quality_checks),
        },
        "artifacts": [
            "metrics.json",
            "decision.md",
            "final_report.md",
            "loop-state.json",
            "loop-run-log.md",
            "rows.jsonl",
            "checkpoint_prior_cache/metrics.json",
            "checkpoint_prior_cache/rows.jsonl",
            "checkpoint_full_cache/metrics.json",
            "checkpoint_full_cache/rows.jsonl",
            "heuristic_cache/metrics.json",
            "heuristic_cache/rows.jsonl",
        ],
        "known_limits": [
            (
                "primary rows use checkpoint policy priors with heuristic leaf values; "
                "checkpoint leaf values are an ablation only"
            ),
            "simulator transitions remain canonical and not learned",
            "comparison is cache/search evidence, not full iterative promotion",
            "Python checkpoint callbacks are slower than the native C++ heuristic MCTS backend",
        ],
    }
    _write_json(config.out_dir / "metrics.json", report)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    (config.out_dir / "final_report.md").write_text(
        _format_final_report(report),
        encoding="utf-8",
    )
    _write_json(config.out_dir / "loop-state.json", _loop_state(config, report))
    (config.out_dir / "loop-run-log.md").write_text(
        _format_loop_log(report),
        encoding="utf-8",
    )
    return report


def _quality_checks(
    checkpoint_prior_report: Mapping[str, Any],
    checkpoint_full_report: Mapping[str, Any],
    heuristic_report: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, bool]:
    checkpoint_prior_metrics = _mapping(checkpoint_prior_report.get("metrics"))
    checkpoint_full_metrics = _mapping(checkpoint_full_report.get("metrics"))
    heuristic_metrics = _mapping(heuristic_report.get("metrics"))
    checkpoint_prior_cache = _mapping(checkpoint_prior_metrics.get("cache"))
    checkpoint_full_cache = _mapping(checkpoint_full_metrics.get("cache"))
    heuristic_cache = _mapping(heuristic_metrics.get("cache"))
    checkpoint_prior_search = _mapping(checkpoint_prior_metrics.get("search_smoke"))
    checkpoint_full_search = _mapping(checkpoint_full_metrics.get("search_smoke"))
    heuristic_search = _mapping(heuristic_metrics.get("search_smoke"))
    return {
        "checkpoint_prior_legal_action_rate_1": float(
            checkpoint_prior_cache.get("legal_action_rate", 0.0)
        )
        == 1.0,
        "checkpoint_prior_policy_target_valid_rate_1": float(
            checkpoint_prior_cache.get("policy_target_valid_rate", 0.0)
        )
        == 1.0,
        "checkpoint_prior_mcts_target_rate_1": float(
            checkpoint_prior_cache.get("mcts_target_rate", 0.0)
        )
        == 1.0,
        "checkpoint_prior_value_target_finite_rate_1": float(
            checkpoint_prior_cache.get("value_target_finite_rate", 0.0)
        )
        == 1.0,
        "checkpoint_full_legal_action_rate_1": float(
            checkpoint_full_cache.get("legal_action_rate", 0.0)
        )
        == 1.0,
        "checkpoint_full_policy_target_valid_rate_1": float(
            checkpoint_full_cache.get("policy_target_valid_rate", 0.0)
        )
        == 1.0,
        "checkpoint_full_mcts_target_rate_1": float(
            checkpoint_full_cache.get("mcts_target_rate", 0.0)
        )
        == 1.0,
        "checkpoint_full_value_target_finite_rate_1": float(
            checkpoint_full_cache.get("value_target_finite_rate", 0.0)
        )
        == 1.0,
        "heuristic_legal_action_rate_1": float(
            heuristic_cache.get("legal_action_rate", 0.0)
        )
        == 1.0,
        "heuristic_policy_target_valid_rate_1": float(
            heuristic_cache.get("policy_target_valid_rate", 0.0)
        )
        == 1.0,
        "heuristic_mcts_target_rate_1": float(heuristic_cache.get("mcts_target_rate", 0.0))
        == 1.0,
        "heuristic_value_target_finite_rate_1": float(
            heuristic_cache.get("value_target_finite_rate", 0.0)
        )
        == 1.0,
        "rows_match_across_modes": bool(comparison.get("rows_match_across_modes")),
        "checkpoint_prior_prior_mode_recorded": checkpoint_prior_search.get("prior_mode")
        == "checkpoint",
        "checkpoint_prior_value_mode_recorded": checkpoint_prior_search.get("value_mode")
        == "heuristic",
        "checkpoint_full_prior_mode_recorded": checkpoint_full_search.get("prior_mode")
        == "checkpoint",
        "checkpoint_full_value_mode_recorded": checkpoint_full_search.get("value_mode")
        == "checkpoint",
        "heuristic_prior_mode_recorded": heuristic_search.get("prior_mode") == "heuristic",
        "heuristic_value_mode_recorded": heuristic_search.get("value_mode") == "heuristic",
        "checkpoint_prior_decisions_per_sec_positive": float(
            comparison.get("checkpoint_prior_decisions_per_sec", 0.0)
        )
        > 0.0,
        "checkpoint_full_decisions_per_sec_positive": float(
            comparison.get("checkpoint_full_decisions_per_sec", 0.0)
        )
        > 0.0,
        "heuristic_decisions_per_sec_positive": float(
            comparison.get("heuristic_decisions_per_sec", 0.0)
        )
        > 0.0,
    }


def _comparison(
    checkpoint_prior_report: Mapping[str, Any],
    checkpoint_full_report: Mapping[str, Any],
    heuristic_report: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_prior_cache = _mapping(
        _mapping(checkpoint_prior_report.get("metrics")).get("cache")
    )
    checkpoint_full_cache = _mapping(
        _mapping(checkpoint_full_report.get("metrics")).get("cache")
    )
    heuristic_cache = _mapping(_mapping(heuristic_report.get("metrics")).get("cache"))
    checkpoint_prior_search = _mapping(
        _mapping(checkpoint_prior_report.get("metrics")).get("search_smoke")
    )
    checkpoint_full_search = _mapping(
        _mapping(checkpoint_full_report.get("metrics")).get("search_smoke")
    )
    heuristic_search = _mapping(_mapping(heuristic_report.get("metrics")).get("search_smoke"))
    checkpoint_prior_reward = float(checkpoint_prior_cache.get("mean_reward", 0.0))
    checkpoint_full_reward = float(checkpoint_full_cache.get("mean_reward", 0.0))
    heuristic_reward = float(heuristic_cache.get("mean_reward", 0.0))
    checkpoint_prior_value = float(checkpoint_prior_cache.get("mean_value_target", 0.0))
    checkpoint_full_value = float(checkpoint_full_cache.get("mean_value_target", 0.0))
    heuristic_value = float(heuristic_cache.get("mean_value_target", 0.0))
    checkpoint_prior_rows = int(checkpoint_prior_cache.get("rows", 0))
    checkpoint_full_rows = int(checkpoint_full_cache.get("rows", 0))
    heuristic_rows = int(heuristic_cache.get("rows", 0))
    return {
        "rows_match_across_modes": checkpoint_prior_rows
        == checkpoint_full_rows
        == heuristic_rows,
        "recommended_mode": "checkpoint_prior_heuristic_value",
        "checkpoint_prior_policy_target_source": "checkpoint_guided_mcts",
        "checkpoint_full_policy_target_source": "checkpoint_guided_mcts",
        "heuristic_policy_target_source": "mcts",
        "checkpoint_prior_mean_reward": checkpoint_prior_reward,
        "checkpoint_full_mean_reward": checkpoint_full_reward,
        "heuristic_mean_reward": heuristic_reward,
        "mean_reward_delta_checkpoint_prior_minus_heuristic": checkpoint_prior_reward
        - heuristic_reward,
        "mean_reward_delta_checkpoint_full_minus_heuristic": checkpoint_full_reward
        - heuristic_reward,
        "mean_reward_delta_checkpoint_full_minus_prior": checkpoint_full_reward
        - checkpoint_prior_reward,
        "checkpoint_prior_mean_value_target": checkpoint_prior_value,
        "checkpoint_full_mean_value_target": checkpoint_full_value,
        "heuristic_mean_value_target": heuristic_value,
        "mean_value_delta_checkpoint_prior_minus_heuristic": checkpoint_prior_value
        - heuristic_value,
        "mean_value_delta_checkpoint_full_minus_heuristic": checkpoint_full_value
        - heuristic_value,
        "mean_value_delta_checkpoint_full_minus_prior": checkpoint_full_value
        - checkpoint_prior_value,
        "checkpoint_prior_decisions_per_sec": float(
            checkpoint_prior_search.get("decisions_per_sec", 0.0)
        ),
        "checkpoint_full_decisions_per_sec": float(
            checkpoint_full_search.get("decisions_per_sec", 0.0)
        ),
        "heuristic_decisions_per_sec": float(heuristic_search.get("decisions_per_sec", 0.0)),
    }


def _format_decision(report: Mapping[str, Any]) -> str:
    comparison = _mapping(_mapping(report.get("metrics")).get("comparison"))
    metrics = _mapping(report.get("metrics"))
    prior_reward_delta = _float_metric(
        comparison,
        "mean_reward_delta_checkpoint_prior_minus_heuristic",
    )
    full_reward_delta = _float_metric(
        comparison,
        "mean_reward_delta_checkpoint_full_minus_heuristic",
    )
    return "\n".join(
        [
            "# Checkpoint-Guided Strategic MCTS",
            "",
            f"Status: `{report['status']}`",
            "",
            "Evidence:",
            f"- Checkpoint: `{report['checkpoint_path']}`",
            (
                "- Programmatic checks: "
                f"{metrics.get('programmatic_checks_passed')}/"
                f"{metrics.get('programmatic_checks_total')}"
            ),
            f"- Rows match across modes: {comparison.get('rows_match_across_modes')}",
            (
                "- Mean reward delta checkpoint-prior minus heuristic: "
                f"{prior_reward_delta:.6f}"
            ),
            (
                "- Mean reward delta checkpoint-full minus heuristic: "
                f"{full_reward_delta:.6f}"
            ),
            (
                "- Checkpoint-prior decisions/sec: "
                f"{float(comparison.get('checkpoint_prior_decisions_per_sec', 0.0)):.2f}"
            ),
            (
                "- Checkpoint-full decisions/sec: "
                f"{float(comparison.get('checkpoint_full_decisions_per_sec', 0.0)):.2f}"
            ),
            (
                "- Heuristic decisions/sec: "
                f"{float(comparison.get('heuristic_decisions_per_sec', 0.0)):.2f}"
            ),
            "",
            "Suggested action:",
            "- Use this as the Goal 2 gate before implementing the iterative V0 promotion loop.",
            "",
        ]
    )


def _format_final_report(report: Mapping[str, Any]) -> str:
    lines = [_format_decision(report), "Limits:"]
    lines.extend(f"- {limit}" for limit in report["known_limits"])
    lines.append("")
    return "\n".join(lines)


def _loop_state(
    config: CheckpointGuidedMCTSSmokeConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    status = "accepted" if report.get("status") == "pass" else "blocked"
    return {
        "schema": "loop-state/v1",
        "owner": "codex",
        "objective": "Verify checkpoint-guided strategic MCTS V0 cache generation.",
        "deliverable": "muzero_cache",
        "attempt_cap": 3,
        "attempt": 1,
        "status": status,
        "automation_level": "L1",
        "state_prune_rules": [
            "Keep the latest accepted or blocked state.",
            "Keep all failed quality-check summaries until a newer accepted run exists.",
            "Do not prune post-run judge verdict references.",
        ],
        "acceptance_criteria": [
            "checkpoint loads and supplies legal policy priors",
            (
                "primary checkpoint-prior plus heuristic-value MCTS writes cache rows "
                "with checkpoint metadata"
            ),
            "matched checkpoint-full and heuristic-prior ablation rows are generated",
            "zero illegal actions and finite targets in all caches",
            "programmatic quality checks all pass",
        ],
        "pause_criteria": [
            "post-run judge verdict is missing, malformed, or REJECT",
            "any programmatic quality check fails",
            "checkpoint-prior, checkpoint-full, or heuristic comparison artifact is missing",
        ],
        "kill_criteria": [
            "attempt cap exceeds 3",
            "the same quality-check failure repeats through the cap",
            "checkpoint loading or inference fails after checkpoint path is verified",
        ],
        "blocked_condition": [
            "missing checkpoint",
            "non-finite checkpoint policy/value output",
            "illegal actions or invalid policy/value targets",
            "missing matched heuristic comparison",
        ],
        "validation_commands": [
            (
                "python -m mini_tft.tools.checkpoint_guided_mcts_smoke "
                "--checkpoint <path> --strict"
            )
        ],
        "artifacts": report["artifacts"],
        "verifier": "pending_post_run_judge",
        "quality_checks": _mapping(_mapping(report.get("metrics")).get("quality_checks")),
        "config": {
            "checkpoint_path": str(config.checkpoint_path),
            "episodes": config.episodes,
            "max_rows": config.max_rows,
            "seed": config.seed,
            "simulations": config.simulations,
            "max_depth": config.max_depth,
            "rollout_steps": config.rollout_steps,
            "checkpoint_device": config.checkpoint_device,
            "skip_determinism_check": config.skip_determinism_check,
        },
        "current_next_action": (
            "run the read-only post-run judge"
            if status == "accepted"
            else "inspect failed quality checks before promotion"
        ),
    }


def _format_loop_log(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics"))
    return "\n".join(
        [
            f"## {report['created_at']} - checkpoint_guided_mcts_smoke - attempt 1/3",
            "",
            (
                "- Action: generated checkpoint-prior primary MCTS cache rows plus "
                "checkpoint-full and heuristic ablations."
            ),
            (
                "- Validation: programmatic checks "
                f"{metrics.get('programmatic_checks_passed')}/"
                f"{metrics.get('programmatic_checks_total')}; "
                f"status `{report['status']}`."
            ),
            "- Verifier verdict: pending post-run judge.",
            "- Next action: run the read-only post-run judge before promotion.",
            "",
        ]
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_metric(metrics: Mapping[str, Any], key: str) -> float:
    return float(metrics.get(key, 0.0))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/strategic_lane/checkpoint_guided_mcts_smoke"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--checkpoint-device", default="cpu")
    parser.add_argument("--skip-determinism-check", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_checkpoint_guided_mcts_smoke(
        CheckpointGuidedMCTSSmokeConfig(
            out_dir=args.out_dir,
            checkpoint_path=args.checkpoint,
            episodes=args.episodes,
            max_rows=args.max_rows,
            seed=args.seed,
            simulations=args.simulations,
            max_depth=args.max_depth,
            rollout_steps=args.rollout_steps,
            checkpoint_device=args.checkpoint_device,
            skip_determinism_check=args.skip_determinism_check,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
