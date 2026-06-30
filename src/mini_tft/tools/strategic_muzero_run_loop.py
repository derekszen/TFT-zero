"""Run the reusable strategic MuZero-style smoke loop end to end."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from mini_tft.tools.evaluate_strategic_policy import (
    StrategicPolicyEvaluationConfig,
    build_policy_specs,
    run_strategic_policy_evaluation,
)
from mini_tft.tools.generate_strategic_muzero_cache import (
    StrategicMuZeroCacheRunConfig,
    run_strategic_muzero_cache_run,
)
from mini_tft.tools.strategic_muzero_loop import (
    StrategicMuZeroLoopConfig,
    run_strategic_muzero_loop,
)
from mini_tft.tools.strategic_parity_matrix import (
    SCENARIO_BY_NAME,
    SCENARIOS,
    StrategicParityMatrixConfig,
    run_strategic_parity_matrix,
)
from mini_tft.tools.train_strategic_muzero_smoke import (
    StrategicMuZeroTrainSmokeConfig,
    run_strategic_muzero_train_smoke,
)

DEFAULT_OUT_DIR = Path("artifacts/strategic_lane/muzero_run_loop")


@dataclass(frozen=True)
class StrategicMuZeroRunLoopConfig:
    out_dir: Path = DEFAULT_OUT_DIR
    seed: int = 0
    attempt_cap: int = 3
    require_queue_ready: bool = True
    cache_episodes: int = 64
    cache_rows: int = 1024
    mcts_simulations: int = 16
    mcts_max_depth: int = 10
    mcts_rollout_steps: int = 6
    mcts_prior_mode: str = "heuristic"
    train_epochs: int = 24
    train_learning_rate: float = 0.03
    baseline_episodes: int = 32
    parity_seeds: tuple[int, ...] = (0, 1, 7, 19)
    parity_scenarios: tuple[str, ...] | None = None
    parity_fuzz_episodes: int = 0
    parity_fuzz_max_steps: int = 80
    cc: str = "cc"
    codex_allowance_source: str = "user"
    codex_five_hour_window_remaining: str = "ample"
    codex_weekly_usage: str = "ample"
    codex_allowance_decision: str = "continue"


def run_strategic_muzero_run_loop(
    config: StrategicMuZeroRunLoopConfig,
) -> dict[str, Any]:
    """Run parity, baselines, cache, train smoke, and the metric verifier."""

    if config.cache_rows <= 0:
        raise ValueError("cache_rows must be positive")
    if config.cache_episodes <= 0:
        raise ValueError("cache_episodes must be positive")
    if config.baseline_episodes <= 0:
        raise ValueError("baseline_episodes must be positive")

    started = perf_counter()
    timestamp = datetime.now(UTC).isoformat()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    parity_dir = config.out_dir / "parity_matrix"
    baseline_dir = config.out_dir / "policy_eval"
    cache_dir = config.out_dir / "cache"
    train_dir = config.out_dir / "train_smoke"
    gate_dir = config.out_dir / "gate"

    parity_report = run_strategic_parity_matrix(
        StrategicParityMatrixConfig(
            out_dir=parity_dir,
            seeds=config.parity_seeds,
            scenarios=_parity_scenarios(config.parity_scenarios),
            cc=config.cc,
            fuzz_episodes=config.parity_fuzz_episodes,
            fuzz_max_steps=config.parity_fuzz_max_steps,
        )
    )
    baseline_report = run_strategic_policy_evaluation(
        StrategicPolicyEvaluationConfig(
            out_dir=baseline_dir,
            policies=build_policy_specs(["heuristic", "random"]),
            episodes=config.baseline_episodes,
            seed=config.seed + 1_000,
        )
    )
    cache_report = run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=cache_dir,
            episodes=config.cache_episodes,
            max_rows=config.cache_rows,
            seed=config.seed,
            simulations=config.mcts_simulations,
            max_depth=config.mcts_max_depth,
            rollout_steps=config.mcts_rollout_steps,
            prior_mode=config.mcts_prior_mode,
        )
    )
    train_report = run_strategic_muzero_train_smoke(
        StrategicMuZeroTrainSmokeConfig(
            rows_path=cache_dir / "rows.jsonl",
            out_dir=train_dir,
            epochs=config.train_epochs,
            learning_rate=config.train_learning_rate,
            seed=config.seed + 2_000,
        )
    )
    gate_report = run_strategic_muzero_loop(
        StrategicMuZeroLoopConfig(
            out_dir=gate_dir,
            seed=config.seed,
            attempt_cap=config.attempt_cap,
            min_cache_rows=config.cache_rows,
            require_queue_ready=config.require_queue_ready,
            cache_metrics_path=cache_dir / "metrics.json",
            cache_rows_path=cache_dir / "rows.jsonl",
            train_metrics_path=train_dir / "metrics.json",
            baseline_metrics_path=baseline_dir / "metrics.json",
            parity_metrics_path=parity_dir / "metrics.json",
            codex_allowance_source=config.codex_allowance_source,
            codex_five_hour_window_remaining=config.codex_five_hour_window_remaining,
            codex_weekly_usage=config.codex_weekly_usage,
            codex_allowance_decision=config.codex_allowance_decision,
        )
    )

    verifier = _read_json(gate_dir / "verifier" / "metrics.json")
    report = _build_report(
        config=config,
        timestamp=timestamp,
        elapsed_sec=perf_counter() - started,
        parity_report=parity_report,
        baseline_report=baseline_report,
        cache_report=cache_report,
        train_report=train_report,
        gate_report=gate_report,
        verifier=verifier,
    )
    _write_json(config.out_dir / "metrics.json", report)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    (config.out_dir / "final_report.md").write_text(
        _format_final_report(report),
        encoding="utf-8",
    )
    verifier_dir = config.out_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    _write_json(verifier_dir / "metrics.json", verifier)
    (verifier_dir / "decision.md").write_text(
        _format_verifier_decision(verifier),
        encoding="utf-8",
    )
    _write_loop_state(config.out_dir, config, report, verifier, timestamp=timestamp)
    return report


def _parity_scenarios(names: tuple[str, ...] | None) -> tuple[Any, ...]:
    if names is None:
        return SCENARIOS
    return tuple(SCENARIO_BY_NAME[name] for name in names)


def _build_report(
    *,
    config: StrategicMuZeroRunLoopConfig,
    timestamp: str,
    elapsed_sec: float,
    parity_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    cache_report: Mapping[str, Any],
    train_report: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> dict[str, Any]:
    gate_metrics = _mapping(gate_report.get("metrics"))
    criteria = _criteria(verifier)
    status = "pass" if verifier.get("verdict") == "ACCEPT" else str(gate_report.get("status"))
    allowance = _allowance_check(config, timestamp)
    return {
        "schema": "strategic-muzero-run-loop/v1",
        "deliverable": "muzero_cache",
        "status": status,
        "seed": config.seed,
        "created_at": timestamp,
        "elapsed_sec": elapsed_sec,
        "commands": [_command(config)],
        "metrics": {
            "programmatic_criteria": criteria,
            "codex_allowance_check": allowance,
            "parity": {
                "status": parity_report.get("status"),
                "summary": parity_report.get("summary"),
                "fuzz": parity_report.get("fuzz"),
            },
            "baselines": gate_metrics.get("baselines", {}),
            "baseline_eval": {
                "status": baseline_report.get("status"),
                "episodes_per_policy": baseline_report.get("episodes_per_policy"),
                "policies": baseline_report.get("policies"),
                "comparison": baseline_report.get("comparison"),
            },
            "cache": gate_metrics.get("cache", {}),
            "search_smoke": gate_metrics.get("search_smoke", {}),
            "train_smoke": gate_metrics.get("train_smoke", {}),
            "determinism": gate_metrics.get("determinism", {}),
            "gate": {
                "status": gate_report.get("status"),
                "verifier": gate_report.get("verifier"),
            },
            "stage_statuses": {
                "parity_matrix": parity_report.get("status"),
                "policy_eval": baseline_report.get("status"),
                "cache": cache_report.get("status"),
                "train_smoke": train_report.get("status"),
                "gate": gate_report.get("status"),
            },
            "config": _config_metrics(config),
        },
        "artifacts": _artifacts(),
        "known_limits": [
            "the training stage is a tiny linear policy/value/dynamics smoke, not full MuZero",
            "the parity matrix is fixed-trace and optional fuzz evidence, not exhaustive proof",
            "the queue-ready gate proves artifact readiness, not policy quality",
        ],
    }


def _criteria(verifier: Mapping[str, Any]) -> dict[str, Any]:
    checks = [_criterion(check) for check in _as_list(verifier.get("checks"))]
    failed = [check for check in checks if not check["passed"]]
    return {
        "schema": "programmatic-criteria/v1",
        "source": "gate/verifier/metrics.json",
        "verdict": verifier.get("verdict"),
        "status": verifier.get("status"),
        "summary": verifier.get("summary"),
        "failed": len(failed),
        "checks": checks,
    }


def _criterion(check: Any) -> dict[str, Any]:
    row = _mapping(check)
    return {
        "name": str(row.get("name", "")),
        "passed": bool(row.get("passed")),
        "message": str(row.get("message", "")),
    }


def _config_metrics(config: StrategicMuZeroRunLoopConfig) -> dict[str, Any]:
    return {
        "attempt_cap": config.attempt_cap,
        "require_queue_ready": config.require_queue_ready,
        "cache_episodes": config.cache_episodes,
        "cache_rows": config.cache_rows,
        "mcts_simulations": config.mcts_simulations,
        "mcts_max_depth": config.mcts_max_depth,
        "mcts_rollout_steps": config.mcts_rollout_steps,
        "mcts_prior_mode": config.mcts_prior_mode,
        "train_epochs": config.train_epochs,
        "train_learning_rate": config.train_learning_rate,
        "baseline_episodes": config.baseline_episodes,
        "parity_seeds": list(config.parity_seeds),
        "parity_scenarios": list(config.parity_scenarios) if config.parity_scenarios else None,
        "parity_fuzz_episodes": config.parity_fuzz_episodes,
        "parity_fuzz_max_steps": config.parity_fuzz_max_steps,
        "cc": config.cc,
        "codex_allowance_source": config.codex_allowance_source,
        "codex_five_hour_window_remaining": config.codex_five_hour_window_remaining,
        "codex_weekly_usage": config.codex_weekly_usage,
        "codex_allowance_decision": config.codex_allowance_decision,
    }


def _artifacts() -> list[str]:
    return [
        "metrics.json",
        "decision.md",
        "final_report.md",
        "loop-state.json",
        "loop-run-log.md",
        "verifier/metrics.json",
        "verifier/decision.md",
        "parity_matrix/metrics.json",
        "parity_matrix/matrix.jsonl",
        "policy_eval/metrics.json",
        "policy_eval/episodes.jsonl",
        "policy_eval/actions.jsonl",
        "cache/metrics.json",
        "cache/rows.jsonl",
        "train_smoke/metrics.json",
        "train_smoke/train_smoke.npz",
        "gate/metrics.json",
        "gate/final_report.md",
        "gate/verifier/metrics.json",
    ]


def _write_loop_state(
    out_dir: Path,
    config: StrategicMuZeroRunLoopConfig,
    report: Mapping[str, Any],
    verifier: Mapping[str, Any],
    *,
    timestamp: str,
) -> None:
    log_path = out_dir / "loop-run-log.md"
    attempt = _next_attempt_number(log_path)
    allowance = _allowance_check(config, timestamp)
    next_action = _next_action(verifier, config.codex_allowance_decision)
    accepted = verifier.get("verdict") == "ACCEPT"
    state = {
        "schema": "loop-state/v1",
        "owner": "mini_tft.tools.strategic_muzero_run_loop",
        "objective": "run the reusable strategic MuZero-style artifact loop",
        "deliverable": "muzero_cache",
        "attempt_cap": config.attempt_cap,
        "attempt": attempt,
        "status": "accepted" if accepted else "blocked",
        "current_next_action": next_action,
        "prune_rules": [
            "keep top-level metrics, decisions, final reports, state, and run logs",
            "stage rows and checkpoints may be pruned only after an accepted replacement run",
        ],
        "codex_allowance_check": allowance,
        "acceptance_criteria": [],
        "programmatic_criteria_source": "gate/verifier/metrics.json",
        "blocked_condition": [
            "parity matrix fails",
            "baseline comparison is missing",
            "cache rows or MCTS targets fail verifier criteria",
            "train smoke losses are non-finite or checkpoint is missing",
            "gate verifier rejects after the attempt cap",
        ],
        "pause_criteria": [
            "Codex allowance status is unknown before starting a new longer loop",
            "weekly usage is at or above the soft pause threshold",
        ],
        "kill_criteria": [
            "same blocker persists through the attempt cap",
            "weekly usage is at or above the hard pause threshold",
            "verifier rejects after all concrete fixes have been attempted",
        ],
        "validation_commands": [
            "env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_muzero_run_loop --strict",
            "env -u UV_PYTHON uv run pytest -q tests/test_strategic_muzero_run_loop.py",
        ],
        "artifacts": report.get("artifacts", []),
        "verifier": "accept" if accepted else "reject",
        "config": _config_metrics(config),
    }
    criteria = _mapping(_mapping(report.get("metrics")).get("programmatic_criteria"))
    state["acceptance_criteria"] = [
        str(check.get("name"))
        for check in _as_list(criteria.get("checks"))
        if isinstance(check, Mapping)
    ]
    _write_json(out_dir / "loop-state.json", state)
    if not log_path.exists():
        log_path.write_text("# Strategic MuZero Run Loop Log\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as file:
        file.write(
            "\n".join(
                [
                    "",
                    f"## Attempt {attempt} - {timestamp}",
                    "",
                    "## Codex Allowance Check",
                    f"Source: {allowance['source']}",
                    f"Checked at: {timestamp}",
                    f"5h window remaining: {allowance['5h_window_remaining']}",
                    f"Weekly usage: {allowance['weekly_usage']}",
                    f"Decision: {allowance['decision']}",
                    "",
                    f"Action: ran reusable MuZero-style loop in `{out_dir}`.",
                    f"Validation: verifier `{verifier['verdict']}`; status `{report['status']}`.",
                    f"Suggested loop action: {next_action}",
                    "",
                ]
            )
        )


def _allowance_check(config: StrategicMuZeroRunLoopConfig, timestamp: str) -> dict[str, str]:
    return {
        "source": config.codex_allowance_source,
        "checked_at": timestamp,
        "5h_window_remaining": config.codex_five_hour_window_remaining,
        "weekly_usage": config.codex_weekly_usage,
        "decision": config.codex_allowance_decision,
    }


def _format_decision(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics"))
    criteria = _mapping(metrics.get("programmatic_criteria"))
    summary = _mapping(criteria.get("summary"))
    train = _mapping(metrics.get("train_smoke"))
    allowance = _mapping(metrics.get("codex_allowance_check"))
    return "\n".join(
        [
            "# Strategic MuZero Run Loop Decision",
            "",
            f"Status: `{report['status']}`",
            f"Verifier: `{criteria.get('verdict', 'not_run')}`",
            "",
            "Evidence:",
            f"- Programmatic checks passed: {summary.get('passed', 0)}/{summary.get('total', 0)}",
            f"- Failed checks: {summary.get('failed', 0)}",
            f"- Cache rows: {_mapping(metrics.get('cache')).get('rows', 0)}",
            f"- Train checkpoint: {train.get('checkpoint_path', '')}",
            f"- Parity status: {_mapping(metrics.get('parity')).get('status', 'not_run')}",
            f"- Codex allowance decision: {allowance.get('decision', 'unknown')}",
            "",
            "Suggested loop action:",
            f"- {_next_action(criteria, str(allowance.get('decision', 'continue')))}",
            "",
        ]
    )


def _format_final_report(report: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics"))
    criteria = _mapping(metrics.get("programmatic_criteria"))
    summary = _mapping(criteria.get("summary"))
    cache = _mapping(metrics.get("cache"))
    search = _mapping(metrics.get("search_smoke"))
    train = _mapping(metrics.get("train_smoke"))
    baselines = _mapping(metrics.get("baselines"))
    parity = _mapping(metrics.get("parity"))
    allowance = _mapping(metrics.get("codex_allowance_check"))
    baseline_policies = ", ".join(str(policy) for policy in _as_list(baselines.get("policies")))
    return "\n".join(
        [
            "# Strategic MuZero Run Loop Final Report",
            "",
            f"Status: `{report['status']}`",
            f"Verifier: `{criteria.get('verdict', 'not_run')}`",
            f"Elapsed sec: `{float(report.get('elapsed_sec', 0.0)):.3f}`",
            "",
            "## Key Metrics",
            "",
            f"- Programmatic checks: {summary.get('passed', 0)}/{summary.get('total', 0)} passed",
            f"- Cache rows: {cache.get('rows', 0)}",
            f"- MCTS target rate: {float(cache.get('mcts_target_rate', 0.0)):.6f}",
            f"- Legal action rate: {float(cache.get('legal_action_rate', 0.0)):.6f}",
            (
                "- Policy target valid rate: "
                f"{float(cache.get('policy_target_valid_rate', 0.0)):.6f}"
            ),
            (
                "- Illegal policy mass max: "
                f"{float(cache.get('illegal_policy_mass_max', 0.0)):.6g}"
            ),
            f"- Search decisions/sec: {float(search.get('decisions_per_sec', 0.0)):.2f}",
            f"- Train losses finite: {bool(train.get('losses_finite'))}",
            f"- Train checkpoint exists: {bool(train.get('checkpoint_exists'))}",
            f"- Baseline policies: {baseline_policies}",
            (
                "- Codex allowance: "
                f"{allowance.get('source', 'unknown')} / {allowance.get('decision', 'unknown')}"
            ),
            (
                "- Parity matrix: "
                f"{parity.get('status', 'not_run')} "
                f"({_mapping(parity.get('summary')).get('passed', 0)}/"
                f"{_mapping(parity.get('summary')).get('total_checks', 0)})"
            ),
            "",
            "## Failed Criteria",
            "",
            *_format_failed_criteria(criteria),
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in _as_list(report.get("known_limits"))],
            "",
        ]
    )


def _format_failed_criteria(criteria: Mapping[str, Any]) -> list[str]:
    failed = [
        _mapping(check)
        for check in _as_list(criteria.get("checks"))
        if isinstance(check, Mapping) and not bool(check.get("passed"))
    ]
    if not failed:
        return ["- None."]
    return [f"- {check.get('name')}: {check.get('message')}" for check in failed]


def _format_verifier_decision(verifier: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Strategic MuZero Run Loop Verifier",
            "",
            f"Verdict: `{verifier.get('verdict', 'not_run')}`",
            f"Status: `{verifier.get('status', 'not_run')}`",
            "",
            "Findings:",
            *_format_verifier_findings(verifier),
            "",
        ]
    )


def _format_verifier_findings(verifier: Mapping[str, Any]) -> list[str]:
    findings = _as_list(verifier.get("findings"))
    if not findings:
        return ["- None."]
    return [
        f"- {_mapping(finding).get('check')}: {_mapping(finding).get('message')}"
        for finding in findings
    ]


def _next_action(
    verifier_or_criteria: Mapping[str, Any],
    allowance_decision: str = "continue",
) -> str:
    if verifier_or_criteria.get("verdict") == "ACCEPT":
        if allowance_decision == "continue":
            return "queue the longer MuZero-style run with this artifact contract"
        return "provide Codex allowance status before starting the longer MuZero-style run"
    findings = _as_list(verifier_or_criteria.get("findings"))
    if findings:
        first = _mapping(findings[0])
        return f"fix `{first.get('check')}`: {first.get('message')}"
    checks = [
        _mapping(check)
        for check in _as_list(verifier_or_criteria.get("checks"))
        if isinstance(check, Mapping) and not bool(check.get("passed"))
    ]
    if checks:
        return f"fix `{checks[0].get('name')}`: {checks[0].get('message')}"
    return "inspect verifier output before queueing the longer MuZero-style run"


def _command(config: StrategicMuZeroRunLoopConfig) -> str:
    parts = [
        "env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_muzero_run_loop",
        f"--out-dir {config.out_dir}",
        f"--seed {config.seed}",
        f"--cache-episodes {config.cache_episodes}",
        f"--cache-rows {config.cache_rows}",
        f"--mcts-simulations {config.mcts_simulations}",
        f"--mcts-max-depth {config.mcts_max_depth}",
        f"--mcts-rollout-steps {config.mcts_rollout_steps}",
        f"--mcts-prior-mode {config.mcts_prior_mode}",
        f"--train-epochs {config.train_epochs}",
        f"--train-learning-rate {config.train_learning_rate}",
        f"--baseline-episodes {config.baseline_episodes}",
        f"--codex-allowance-source {config.codex_allowance_source}",
        f"--codex-5h-window-remaining {config.codex_five_hour_window_remaining}",
        f"--codex-weekly-usage {config.codex_weekly_usage}",
        f"--codex-allowance-decision {config.codex_allowance_decision}",
        *[f"--parity-seed {seed}" for seed in config.parity_seeds],
        "--strict",
    ]
    if config.parity_scenarios:
        parts.extend(f"--parity-scenario {name}" for name in config.parity_scenarios)
    if config.parity_fuzz_episodes:
        parts.append(f"--parity-fuzz-episodes {config.parity_fuzz_episodes}")
        parts.append(f"--parity-fuzz-max-steps {config.parity_fuzz_max_steps}")
    if not config.require_queue_ready:
        parts.append("--allow-smoke-only")
    return " ".join(parts)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _next_attempt_number(log_path: Path) -> int:
    if not log_path.exists():
        return 1
    return log_path.read_text(encoding="utf-8").count("## Attempt ") + 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempt-cap", type=int, default=3)
    parser.add_argument("--allow-smoke-only", action="store_true")
    parser.add_argument("--cache-episodes", type=int, default=64)
    parser.add_argument("--cache-rows", type=int, default=1024)
    parser.add_argument("--mcts-simulations", type=int, default=16)
    parser.add_argument("--mcts-max-depth", type=int, default=10)
    parser.add_argument("--mcts-rollout-steps", type=int, default=6)
    parser.add_argument("--mcts-prior-mode", choices=["uniform", "heuristic"], default="heuristic")
    parser.add_argument("--train-epochs", type=int, default=24)
    parser.add_argument("--train-learning-rate", type=float, default=0.03)
    parser.add_argument("--baseline-episodes", type=int, default=32)
    parser.add_argument("--parity-seed", type=int, action="append")
    parser.add_argument("--parity-scenario", choices=sorted(SCENARIO_BY_NAME), action="append")
    parser.add_argument("--parity-fuzz-episodes", type=int, default=0)
    parser.add_argument("--parity-fuzz-max-steps", type=int, default=80)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--codex-allowance-source", default="user")
    parser.add_argument("--codex-5h-window-remaining", default="ample")
    parser.add_argument("--codex-weekly-usage", default="ample")
    parser.add_argument(
        "--codex-allowance-decision",
        choices=["continue", "soft-pause", "hard-pause"],
        default="continue",
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_strategic_muzero_run_loop(
        StrategicMuZeroRunLoopConfig(
            out_dir=args.out_dir,
            seed=args.seed,
            attempt_cap=args.attempt_cap,
            require_queue_ready=not args.allow_smoke_only,
            cache_episodes=args.cache_episodes,
            cache_rows=args.cache_rows,
            mcts_simulations=args.mcts_simulations,
            mcts_max_depth=args.mcts_max_depth,
            mcts_rollout_steps=args.mcts_rollout_steps,
            mcts_prior_mode=args.mcts_prior_mode,
            train_epochs=args.train_epochs,
            train_learning_rate=args.train_learning_rate,
            baseline_episodes=args.baseline_episodes,
            parity_seeds=tuple(args.parity_seed) if args.parity_seed else (0, 1, 7, 19),
            parity_scenarios=tuple(args.parity_scenario) if args.parity_scenario else None,
            parity_fuzz_episodes=args.parity_fuzz_episodes,
            parity_fuzz_max_steps=args.parity_fuzz_max_steps,
            cc=args.cc,
            codex_allowance_source=args.codex_allowance_source,
            codex_five_hour_window_remaining=args.codex_5h_window_remaining,
            codex_weekly_usage=args.codex_weekly_usage,
            codex_allowance_decision=args.codex_allowance_decision,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
