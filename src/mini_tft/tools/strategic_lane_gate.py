"""Generate strategic-lane quality-gate artifacts."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mini_tft.strategic.adapters.baselines import random_policy, tft_heuristic_policy
from mini_tft.strategic.adapters.muzero_cache import CacheRow, cache_metrics, generate_cache
from mini_tft.strategic.adapters.puffer import run_benchmark
from mini_tft.strategic.adapters.web_demo import state_payload
from mini_tft.strategic.core import StrategicConfig, placement_proxy, reset, scenario_score
from mini_tft.strategic.core.rules import run_episode


@dataclass(frozen=True)
class StrategicLaneGateConfig:
    out_dir: Path = Path("artifacts/strategic_lane")
    seed: int = 0
    cache_episodes: int = 4
    cache_rows: int = 128
    benchmark_episodes: int = 16
    benchmark_batch_size: int = 256
    benchmark_repeats: int = 3
    calibration_seeds: int = 32


def run_gate(config: StrategicLaneGateConfig) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    puffer_dir = out_dir / "puffer_speed"
    cache_dir = out_dir / "muzero_cache"
    playable_dir = out_dir / "playable_demo"
    for path in (puffer_dir, cache_dir, playable_dir):
        path.mkdir(parents=True, exist_ok=True)

    puffer_result = run_benchmark(
        episodes=config.benchmark_episodes,
        batch_size=config.benchmark_batch_size,
        repeated_runs=config.benchmark_repeats,
        out_dir=puffer_dir,
    )

    cache_rows = generate_cache(
        episodes=config.cache_episodes,
        max_rows=config.cache_rows,
    )
    cache_report = _write_cache_artifacts(cache_rows, cache_dir, config)
    playable_report = _write_playable_artifacts(playable_dir, config)
    calibration = _calibrate(config)

    status = _overall_status(
        puffer_decision=puffer_result.decision,
        cache_report=cache_report,
        playable_report=playable_report,
        calibration=calibration,
    )
    report: dict[str, Any] = {
        "schema": "quality-gate/v1",
        "deliverable": "other",
        "status": status,
        "seed": config.seed,
        "commands": ["python -m mini_tft.tools.strategic_lane_gate"],
        "metrics": {
            "puffer_speed": puffer_result.to_dict(),
            "muzero_cache": cache_report["metrics"],
            "playable_demo": playable_report["metrics"],
            "calibration": calibration,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        },
        "artifacts": [
            "puffer_speed/metrics.json",
            "puffer_speed/decision.md",
            "muzero_cache/rows.jsonl",
            "muzero_cache/metrics.json",
            "muzero_cache/decision.md",
            "playable_demo/initial_payload.json",
            "playable_demo/metrics.json",
            "playable_demo/decision.md",
            "metrics.json",
            "decision.md",
            "final_report.md",
            "loop-state.json",
            "loop-run-log.md",
        ],
        "known_limits": [
            "strategic lane is a simplified TFT-shaped simulator, not a full TFT clone",
            "MuZero cache artifact is smoke data; no dynamics/policy model is promoted here",
            "playable demo artifact is a shared-rules payload, not a launched browser route",
            "puffer speed artifact is native vector throughput evidence, not PPO quality evidence",
            "placement_proxy is an elimination-timing bucket, not real lobby placement",
        ],
    }

    _write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    (out_dir / "final_report.md").write_text(_format_final_report(report), encoding="utf-8")
    _write_loop_state(out_dir, config, report)
    print(json.dumps(report, indent=2))
    return report


def _write_cache_artifacts(
    rows: list[CacheRow],
    out_dir: Path,
    config: StrategicLaneGateConfig,
) -> dict[str, Any]:
    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as file:
        for index, row in enumerate(rows):
            file.write(json.dumps(_cache_row_payload(index, row)) + "\n")

    metrics = cache_metrics(rows)
    status = "smoke_only" if metrics.get("legal_action_rate") == 1.0 else "fail"
    report: dict[str, Any] = {
        "schema": "quality-gate/v1",
        "deliverable": "muzero_cache",
        "status": status,
        "seed": config.seed,
        "commands": ["python -m mini_tft.tools.strategic_lane_gate"],
        "metrics": metrics,
        "artifacts": ["rows.jsonl", "metrics.json", "decision.md"],
        "known_limits": [
            "rows are policy/value cache smoke data",
            "this command does not train or promote a MuZero model",
        ],
    }
    _write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    return report


def _write_playable_artifacts(
    out_dir: Path,
    config: StrategicLaneGateConfig,
) -> dict[str, Any]:
    state = reset(seed=config.seed)
    payload = state_payload(state)
    _write_json(out_dir / "initial_payload.json", payload)
    metrics = {
        "round": payload["round"],
        "hp": payload["hp"],
        "gold": payload["gold"],
        "level": payload["level"],
        "legal_actions": len(payload["legal_actions"]),
        "has_enemy_strength": "enemy_strength" in payload,
        "has_placement_proxy": "placement_proxy" in payload,
        "has_scenario_score": "scenario_score" in payload,
        "has_role_item_slots": "role_item_slots" in payload,
        "stage_label": payload["stage_label"],
    }
    status = "smoke_only" if metrics["legal_actions"] > 0 else "fail"
    report: dict[str, Any] = {
        "schema": "quality-gate/v1",
        "deliverable": "playable_demo",
        "status": status,
        "seed": config.seed,
        "commands": ["python -m mini_tft.tools.strategic_lane_gate"],
        "metrics": metrics,
        "artifacts": ["initial_payload.json", "metrics.json", "decision.md"],
        "known_limits": ["browser route is not wired to this strategic payload yet"],
    }
    _write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    return report


def _calibrate(config: StrategicLaneGateConfig) -> dict[str, Any]:
    strategic_config = StrategicConfig()
    results: dict[str, Any] = {}
    for name, policy in (
        ("heuristic", tft_heuristic_policy),
        ("random", random_policy),
    ):
        placements: list[int] = []
        scenario_scores: list[float] = []
        rounds: list[int] = []
        hps: list[int] = []
        reasons: list[str | None] = []
        illegal_actions: list[int] = []
        last_episode_steps = 0
        for seed in range(config.calibration_seeds):
            state, _, last_episode_steps = run_episode(policy, seed=seed, config=strategic_config)
            placements.append(placement_proxy(state, strategic_config))
            scenario_scores.append(scenario_score(state, strategic_config))
            rounds.append(state.round)
            hps.append(state.hp)
            reasons.append(state.final_reason)
            illegal_actions.append(state.total_illegal_actions)
        reason_counts = Counter(reasons)
        death_count = reason_counts["hp_zero"]
        survivor_count = reason_counts["max_round"]
        results[name] = {
            "mean_placement": float(np.mean(placements)),
            "placement_counts": {
                str(key): int(value) for key, value in sorted(Counter(placements).items())
            },
            "mean_scenario_score": float(np.mean(scenario_scores)),
            "min_placement": int(np.min(placements)),
            "max_placement": int(np.max(placements)),
            "mean_final_round": float(np.mean(rounds)),
            "final_round_quantiles": {
                str(quantile): float(np.quantile(rounds, quantile))
                for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "mean_hp": float(np.mean(hps)),
            "min_hp": int(np.min(hps)),
            "max_hp": int(np.max(hps)),
            "final_reason_counts": {
                str(key): int(value) for key, value in sorted(reason_counts.items())
            },
            "death_rate": float(death_count / max(1, config.calibration_seeds)),
            "survivor_rate": float(survivor_count / max(1, config.calibration_seeds)),
            "illegal_actions": int(np.sum(illegal_actions)),
            "episodes": config.calibration_seeds,
            "last_episode_steps": last_episode_steps,
        }
    results["heuristic_death_rate_target"] = results["heuristic"]["death_rate"] >= 0.5
    results["enemy_model"] = "round_plus_rng_noise_pressure"
    return results


def _overall_status(
    *,
    puffer_decision: str,
    cache_report: dict[str, Any],
    playable_report: dict[str, Any],
    calibration: dict[str, Any],
) -> str:
    if cache_report["status"] == "fail" or playable_report["status"] == "fail":
        return "fail"
    if not calibration["heuristic_death_rate_target"]:
        return "fail"
    if puffer_decision == "fail":
        return "fail"
    return "smoke_only"


def _cache_row_payload(index: int, row: CacheRow) -> dict[str, Any]:
    return {
        "index": index,
        "observation": row.observation.astype(float).tolist(),
        "legal_mask": row.legal_mask.astype(bool).tolist(),
        "action": row.action,
        "reward": row.reward,
        "next_observation": row.next_observation.astype(float).tolist(),
        "done": row.done,
        "policy_target": row.policy_target.astype(float).tolist(),
        "value_target": row.value_target,
        "metadata": row.metadata,
    }


def _write_loop_state(
    out_dir: Path,
    config: StrategicLaneGateConfig,
    report: dict[str, Any],
) -> None:
    timestamp = datetime.now(UTC).isoformat()
    log_path = out_dir / "loop-run-log.md"
    attempt = _next_attempt_number(log_path)
    state = {
        "schema": "loop-state/v1",
        "owner": "mini_tft.tools.strategic_lane_gate",
        "objective": "strategic lane smoke gate",
        "deliverable": "other",
        "attempt_cap": 3,
        "attempt": attempt,
        "status": "running" if report["status"] != "fail" else "blocked",
        "current_next_action": "run read-only verifier before marking loop complete",
        "prune_rules": [
            "keep latest metrics, decisions, final report, and loop logs",
            "cache rows are reproducible from fixed seeds and may be regenerated",
            "do not prune verifier outputs until the next accepted strategic-lane gate",
        ],
        "codex_allowance_check": {
            "source": "unknown",
            "checked_at": timestamp,
            "5h_window_remaining": "unknown",
            "weekly_usage": "unknown",
            "decision": "continue",
        },
        "acceptance_criteria": [
            "fixed-seed Markov tests pass",
            "cache rows are exported",
            "puffer-speed artifact is written with native-vector parity evidence",
            "playable payload artifact is written",
            "heuristic death rate is at least 50 percent",
            "read-only verifier accepts the artifacts",
        ],
        "blocked_condition": [
            "illegal cache actions",
            "missing playable payload",
            "heuristic death rate below target",
            "puffer benchmark decision fail",
        ],
        "validation_commands": [
            "pytest -q tests/test_strategic_scaffold.py tests/test_strategic_core.py",
            (
                "ruff check src/mini_tft/strategic "
                "tests/test_strategic_scaffold.py tests/test_strategic_core.py"
            ),
            (
                "pyright src/mini_tft/strategic "
                "tests/test_strategic_scaffold.py tests/test_strategic_core.py"
            ),
        ],
        "artifacts": report["artifacts"],
        "verifier": "pending",
        "config": {
            "seed": config.seed,
            "cache_episodes": config.cache_episodes,
            "cache_rows": config.cache_rows,
            "benchmark_episodes": config.benchmark_episodes,
            "benchmark_batch_size": config.benchmark_batch_size,
            "benchmark_repeats": config.benchmark_repeats,
            "calibration_seeds": config.calibration_seeds,
        },
    }
    _write_json(out_dir / "loop-state.json", state)
    if not log_path.exists():
        log_path.write_text("# Strategic Lane Loop Run Log\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as file:
        if log_path.stat().st_size > 0:
            file.write("\n")
        file.write(
            "\n".join(
                [
                    f"## Attempt {attempt} - {timestamp}",
                    "",
                    "Codex Allowance Check:",
                    "- Source: unknown",
                    f"- Checked at: {timestamp}",
                    "- 5h window remaining: unknown",
                    "- Weekly usage: unknown",
                    "- Decision: continue",
                    "",
                    f"Action: generated strategic-lane smoke artifacts in `{out_dir}`.",
                    f"Validation: status `{report['status']}`.",
                    "Verifier: pending.",
                    "Next action: run read-only verifier before completion.",
                    "",
                ]
            )
        )


def _next_attempt_number(log_path: Path) -> int:
    if not log_path.exists():
        return 1
    text = log_path.read_text(encoding="utf-8")
    _, _, current_loop = text.rpartition("Verifier result: ACCEPT")
    return current_loop.count("## Attempt ") + 1


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_decision(report: dict[str, Any]) -> str:
    formula_lines = []
    if report["deliverable"] == "other":
        formula_lines = [
            "",
            "Placement proxy bucket:",
            "- Survive max_round -> 1",
            "- Die at round >= 36 -> 2",
            "- Die at round >= 32 -> 3",
            "- Die at round >= 29 -> 4",
            "- Die at round >= 25 -> 5",
            "- Die at round >= 18 -> 6",
            "- Die at round >= 11 -> 7",
            "- Die earlier -> 8",
            "",
            "Dense quality score:",
            "- scenario_score = 0.45 * round_frac + 0.25 * hp_frac + 0.30 * strength_ratio",
        ]
    return "\n".join(
        [
            "# Decision",
            "",
            f"Status: {report['status']}",
            "",
            "Evidence:",
            f"- Deliverable: {report['deliverable']}",
            f"- Seed: {report['seed']}",
            f"- Artifacts: {', '.join(report['artifacts'])}",
            *formula_lines,
            "",
            "Limits:",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
            "Next:",
            "- Use these smoke artifacts as the baseline for real MuZero trainer wiring.",
            "",
        ]
    )


def _format_final_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    calibration = metrics["calibration"]["heuristic"]
    puffer = metrics["puffer_speed"]
    cache = metrics["muzero_cache"]
    playable = metrics["playable_demo"]
    benchmark_label = "Native strategic Puffer vector"
    return "\n".join(
        [
            "# Strategic Lane Final Report",
            "",
            f"Status: `{report['status']}`",
            "",
            "## What Is Proven",
            "",
            "- The clean strategic rules can emit legal cache rows for MuZero-style data.",
            (
                "- The benchmark path writes native Puffer vector throughput metrics "
                "with scalar parity."
            ),
            "- The playable adapter can serialize a shared-rules state payload.",
            "- The heuristic baseline usually dies under the tuned enemy curve.",
            "",
            "## Key Metrics",
            "",
            f"- Cache rows: {cache['rows']}",
            f"- Cache legal action rate: {cache['legal_action_rate']:.3f}",
            f"- Puffer benchmark kind: {puffer['benchmark_kind']}",
            f"- {benchmark_label} speed ratio: {puffer['speedup']:.2f}x",
            f"- Puffer speed decision: {puffer['decision']}",
            f"- Semantic parity: {puffer['semantic_parity']}",
            f"- Puffer trainer: {puffer['puffer_trainer']}",
            f"- Playable legal actions: {playable['legal_actions']}",
            f"- Playable stage label: {playable['stage_label']}",
            f"- Heuristic mean placement: {calibration['mean_placement']:.3f}",
            f"- Heuristic placement counts: {calibration['placement_counts']}",
            f"- Heuristic mean scenario score: {calibration['mean_scenario_score']:.3f}",
            f"- Heuristic median final round: {calibration['final_round_quantiles']['0.5']:.1f}",
            f"- Heuristic death rate: {calibration['death_rate']:.3f}",
            f"- Heuristic survivor rate: {calibration['survivor_rate']:.3f}",
            f"- Heuristic mean HP: {calibration['mean_hp']:.3f}",
            "",
            "## Known Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
            "## Next",
            "",
            "- Add a browser route or mode that uses `state_payload` and strategic `step`.",
            "- Add a tiny policy/value/dynamics smoke trainer over `rows.jsonl`.",
            "- Use `--env-kind strategic --puffer-backend native` for PPO speed experiments.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/strategic_lane"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-episodes", type=int, default=4)
    parser.add_argument("--cache-rows", type=int, default=128)
    parser.add_argument("--benchmark-episodes", type=int, default=16)
    parser.add_argument("--benchmark-batch-size", type=int, default=256)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--calibration-seeds", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_gate(
        StrategicLaneGateConfig(
            out_dir=args.out_dir,
            seed=args.seed,
            cache_episodes=args.cache_episodes,
            cache_rows=args.cache_rows,
            benchmark_episodes=args.benchmark_episodes,
            benchmark_batch_size=args.benchmark_batch_size,
            benchmark_repeats=args.benchmark_repeats,
            calibration_seeds=args.calibration_seeds,
        )
    )


if __name__ == "__main__":
    main()
