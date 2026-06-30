"""Write and verify strategic MuZero-readiness loop artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from mini_tft.strategic.core import NUM_ACTIONS
from mini_tft.strategic.core.obs import observation_dim
from mini_tft.strategic.core.state import DEFAULT_STRATEGIC_CONFIG

EXPECTED_OBSERVATION_DIM = observation_dim(DEFAULT_STRATEGIC_CONFIG)
EXPECTED_ACTION_DIM = NUM_ACTIONS
DEFAULT_LOOP_OUT_DIR = Path("artifacts/strategic_lane/muzero_loop")


@dataclass(frozen=True)
class StrategicMuZeroLoopConfig:
    out_dir: Path = DEFAULT_LOOP_OUT_DIR
    seed: int = 0
    attempt_cap: int = 3
    min_cache_rows: int = 1024
    require_queue_ready: bool = True
    cache_metrics_path: Path | None = None
    cache_rows_path: Path | None = None
    mcts_metrics_path: Path | None = None
    train_metrics_path: Path | None = None
    baseline_metrics_path: Path | None = None
    parity_metrics_path: Path | None = None
    checkpoint_path: Path | None = None
    deterministic: bool | None = None
    mcts_target_rows: int | None = None
    mcts_target_rate: float | None = None
    policy_target_valid_rate: float | None = None
    policy_target_sum_error_max: float | None = None
    illegal_policy_mass_max: float | None = None
    value_target_finite_rate: float | None = None
    codex_allowance_source: str = "user"
    codex_five_hour_window_remaining: str = "ample"
    codex_weekly_usage: str = "ample"
    codex_allowance_decision: str = "continue"


def run_strategic_muzero_loop(config: StrategicMuZeroLoopConfig) -> dict[str, Any]:
    """Write the reusable MuZero loop scaffold and verifier output."""

    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "verifier").mkdir(parents=True, exist_ok=True)

    report = build_loop_report(config)
    verifier = verify_muzero_loop_metrics(
        report,
        require_queue_ready=config.require_queue_ready,
    )
    report["status"] = str(verifier["status"])
    report["verifier"] = _verifier_summary(verifier)

    _write_json(config.out_dir / "metrics.json", report)
    _write_json(config.out_dir / "verifier" / "metrics.json", verifier)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    (config.out_dir / "final_report.md").write_text(
        _format_final_report(report, verifier),
        encoding="utf-8",
    )
    (config.out_dir / "verifier" / "decision.md").write_text(
        _format_verifier_decision(verifier),
        encoding="utf-8",
    )
    _write_loop_state(config.out_dir, config, report, verifier)
    return report


def build_loop_report(config: StrategicMuZeroLoopConfig) -> dict[str, Any]:
    cache_source = _read_json_optional(config.cache_metrics_path)
    mcts_source = _read_json_optional(config.mcts_metrics_path)
    train_source = _read_json_optional(config.train_metrics_path)
    baseline_source = _read_json_optional(config.baseline_metrics_path)
    parity_source = _read_json_optional(config.parity_metrics_path)

    metrics = _empty_metrics(config)
    if cache_source is not None:
        _apply_cache_metrics(metrics, cache_source)
    if config.cache_rows_path is not None:
        _apply_cache_rows_jsonl(metrics, config.cache_rows_path)
    if mcts_source is not None:
        _apply_mcts_metrics(metrics, mcts_source)
    if train_source is not None:
        _apply_train_metrics(metrics, train_source)
    if baseline_source is not None:
        _apply_baseline_metrics(metrics, baseline_source)
    if parity_source is not None:
        _apply_parity_metrics(metrics, parity_source)
    _apply_overrides(metrics, config)

    source_artifacts = _source_artifacts(config)
    return {
        "schema": "quality-gate/v1",
        "deliverable": "muzero_cache",
        "status": "blocked",
        "seed": config.seed,
        "commands": [
            "env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_muzero_loop"
        ],
        "metrics": metrics,
        "artifacts": [
            "metrics.json",
            "decision.md",
            "final_report.md",
            "loop-state.json",
            "loop-run-log.md",
            "verifier/metrics.json",
            "verifier/decision.md",
            *source_artifacts,
        ],
        "known_limits": [
            "this loop verifies MuZero-readiness evidence; it does not train MuZero itself",
            "queue-ready pass requires MCTS policy targets in every cache row",
            "smoke artifacts are not promoted to policy quality without baseline comparison",
        ],
    }


def verify_muzero_loop_metrics(
    report: Mapping[str, Any],
    *,
    require_queue_ready: bool = True,
) -> dict[str, Any]:
    """Read-only metric verifier for the strategic MuZero loop."""

    checks: list[dict[str, Any]] = []
    metrics = _mapping(report.get("metrics"))
    acceptance = _mapping(metrics.get("acceptance"))
    min_cache_rows = _as_int(acceptance.get("min_cache_rows"), 1024)
    policy_tol = _as_float(acceptance.get("policy_sum_tolerance"), 1e-5)

    _check(
        checks,
        "schema",
        report.get("schema") == "quality-gate/v1",
        "metrics schema is quality-gate/v1",
        f"expected quality-gate/v1, got {report.get('schema')!r}",
    )
    _check(
        checks,
        "deliverable",
        report.get("deliverable") == "muzero_cache",
        "deliverable is muzero_cache",
        f"expected muzero_cache, got {report.get('deliverable')!r}",
    )

    action_surface = _mapping(metrics.get("action_surface"))
    _check_equal(
        checks,
        "observation_dim",
        _as_int(action_surface.get("observation_dim")),
        EXPECTED_OBSERVATION_DIM,
    )
    _check_equal(
        checks,
        "action_dim",
        _as_int(action_surface.get("action_dim")),
        EXPECTED_ACTION_DIM,
    )

    cache = _mapping(metrics.get("cache"))
    rows = _as_int(cache.get("rows"))
    mcts_target_rows = _as_int(cache.get("mcts_target_rows"))
    _check(
        checks,
        "cache_rows",
        rows >= min_cache_rows,
        f"cache rows {rows} meet minimum {min_cache_rows}",
        f"cache rows {rows} below minimum {min_cache_rows}",
    )
    _check(
        checks,
        "mcts_target_rows",
        rows > 0 and mcts_target_rows == rows,
        "every cache row has an MCTS visit-policy target",
        f"MCTS target rows {mcts_target_rows} do not match cache rows {rows}",
    )
    _check_rate(checks, "mcts_target_rate", cache.get("mcts_target_rate"), 1.0)
    _check_rate(checks, "legal_action_rate", cache.get("legal_action_rate"), 1.0)
    _check_rate(checks, "policy_target_valid_rate", cache.get("policy_target_valid_rate"), 1.0)
    _check_rate(checks, "value_target_finite_rate", cache.get("value_target_finite_rate"), 1.0)
    policy_error = _as_float(cache.get("policy_target_sum_error_max"), float("inf"))
    _check(
        checks,
        "policy_target_sum_error_max",
        policy_error <= policy_tol,
        f"policy target sum error {policy_error:.6g} within tolerance {policy_tol:.6g}",
        f"policy target sum error {policy_error:.6g} exceeds tolerance {policy_tol:.6g}",
    )
    illegal_mass = _as_float(cache.get("illegal_policy_mass_max"), float("inf"))
    _check(
        checks,
        "illegal_policy_mass_max",
        illegal_mass <= policy_tol,
        f"illegal policy mass {illegal_mass:.6g} within tolerance {policy_tol:.6g}",
        f"illegal policy mass {illegal_mass:.6g} exceeds tolerance {policy_tol:.6g}",
    )

    search = _mapping(metrics.get("search_smoke"))
    _check(
        checks,
        "mcts_search_smoke",
        bool(search.get("ran")) and _as_int(search.get("total_decisions")) > 0,
        "MCTS search smoke ran and produced decisions",
        "MCTS search smoke is missing or produced no decisions",
    )
    _check(
        checks,
        "mcts_search_illegal_actions",
        _as_int(search.get("illegal_action_count")) == 0,
        "MCTS search reported zero illegal actions",
        f"MCTS search illegal actions: {_as_int(search.get('illegal_action_count'))}",
    )

    train = _mapping(metrics.get("train_smoke"))
    _check(
        checks,
        "train_smoke_ran",
        bool(train.get("ran")),
        "tiny policy/value/dynamics train smoke ran",
        "tiny policy/value/dynamics train smoke did not run",
    )
    _check(
        checks,
        "train_losses_finite",
        bool(train.get("losses_finite")),
        "train smoke losses are finite",
        "train smoke losses are missing or non-finite",
    )

    deterministic = _mapping(metrics.get("determinism"))
    _check(
        checks,
        "determinism",
        bool(deterministic.get("fixed_seed_reproducible")),
        "fixed-seed trace/cache reproducibility is recorded",
        "fixed-seed trace/cache reproducibility is missing or false",
    )

    if require_queue_ready:
        baselines = _mapping(metrics.get("baselines"))
        _check(
            checks,
            "baseline_comparison",
            bool(baselines.get("present")) and bool(baselines.get("comparison_present")),
            "baseline comparison is present",
            "baseline comparison is missing",
        )
        policies = {str(policy) for policy in _as_list(baselines.get("policies"))}
        _check(
            checks,
            "baseline_policies",
            {"random", "heuristic"}.issubset(policies),
            "random and heuristic baselines are present",
            f"random/heuristic baselines missing from {sorted(policies)}",
        )
        _check(
            checks,
            "checkpoint",
            bool(train.get("checkpoint_exists")),
            "train smoke checkpoint exists",
            "train smoke checkpoint is missing",
        )
        parity = _mapping(metrics.get("parity"))
        _check(
            checks,
            "parity_matrix",
            bool(parity.get("present")) and parity.get("status") == "pass",
            "strategic Python/native/Ocean parity matrix passed",
            f"strategic parity matrix missing or not pass: {parity.get('status')!r}",
        )

    failed = [check for check in checks if not check["passed"]]
    status = _status_for_checks(failed, require_queue_ready=bool(require_queue_ready))
    return {
        "schema": "loop-verifier/v1",
        "verdict": "ACCEPT" if not failed else "REJECT",
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(),
        "require_queue_ready": require_queue_ready,
        "checks": checks,
        "findings": [
            {
                "level": "error",
                "check": str(check["name"]),
                "message": str(check["failure"]),
            }
            for check in failed
        ],
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "total": len(checks),
        },
    }


def _empty_metrics(config: StrategicMuZeroLoopConfig) -> dict[str, Any]:
    return {
        "acceptance": {
            "min_cache_rows": config.min_cache_rows,
            "policy_sum_tolerance": 1e-5,
            "require_queue_ready": config.require_queue_ready,
        },
        "action_surface": {
            "name": "strategic_11_macro",
            "observation_dim": EXPECTED_OBSERVATION_DIM,
            "action_dim": EXPECTED_ACTION_DIM,
        },
        "cache": {
            "rows": 0,
            "mcts_target_rows": 0,
            "mcts_target_rate": 0.0,
            "legal_action_rate": 0.0,
            "policy_target_valid_rate": 0.0,
            "policy_target_sum_error_max": None,
            "illegal_policy_mass_max": None,
            "value_target_finite_rate": 0.0,
        },
        "search_smoke": {
            "ran": False,
            "total_decisions": 0,
            "illegal_action_count": 0,
        },
        "train_smoke": {
            "ran": False,
            "losses_finite": False,
            "checkpoint_exists": False,
        },
        "baselines": {
            "present": False,
            "comparison_present": False,
            "policies": [],
        },
        "parity": {
            "present": False,
            "status": "not_run",
            "total_checks": 0,
            "failed": 0,
        },
        "determinism": {
            "fixed_seed_reproducible": False,
        },
    }


def _apply_cache_metrics(metrics: dict[str, Any], source: Mapping[str, Any]) -> None:
    source_metrics = _mapping(source.get("metrics", source))
    source_cache = _mapping(source_metrics.get("cache")) or source_metrics
    cache = _nested_metrics(metrics, "cache")
    action_surface = _nested_metrics(metrics, "action_surface")
    _copy_int(source_cache, cache, "rows")
    _copy_int(source_cache, action_surface, "observation_dim")
    _copy_int(source_cache, action_surface, "action_dim")
    for key in (
        "legal_action_rate",
        "mcts_target_rate",
        "policy_target_valid_rate",
        "policy_target_sum_error_max",
        "illegal_policy_mass_max",
        "value_target_finite_rate",
    ):
        _copy_float(source_cache, cache, key)
    _copy_int(source_cache, cache, "mcts_target_rows")
    for key in ("terminal_rows", "mean_reward", "mean_value_target"):
        if key in source_cache:
            cache[key] = source_cache[key]

    source_search = _mapping(source_metrics.get("search_smoke"))
    if source_search:
        metrics["search_smoke"] = source_search
    source_determinism = _mapping(source_metrics.get("determinism"))
    if source_determinism:
        metrics["determinism"] = source_determinism


def _apply_cache_rows_jsonl(metrics: dict[str, Any], path: Path) -> None:
    rows = _read_jsonl(path)
    cache = _nested_metrics(metrics, "cache")
    action_surface = _nested_metrics(metrics, "action_surface")
    policy_tol = _as_float(_mapping(metrics.get("acceptance")).get("policy_sum_tolerance"), 1e-5)
    legal_actions = 0
    valid_policy_targets = 0
    finite_value_targets = 0
    mcts_target_rows = 0
    policy_sum_errors: list[float] = []
    illegal_masses: list[float] = []
    observation_dims: set[int] = set()
    action_dims: set[int] = set()

    for row in rows:
        observation = _as_list(row.get("observation"))
        legal_mask = [bool(value) for value in _as_list(row.get("legal_mask"))]
        policy_target = [
            _as_float(value, float("nan"))
            for value in _as_list(row.get("policy_target"))
        ]
        action = _as_int(row.get("action"), -1)
        if observation:
            observation_dims.add(len(observation))
        if legal_mask:
            action_dims.add(len(legal_mask))
        if 0 <= action < len(legal_mask) and legal_mask[action]:
            legal_actions += 1

        target_sum = sum(policy_target)
        policy_sum_error = abs(target_sum - 1.0) if isfinite(target_sum) else float("inf")
        illegal_mass = sum(
            max(0.0, value)
            for value, legal in zip(policy_target, legal_mask, strict=False)
            if not legal and isfinite(value)
        )
        policy_sum_errors.append(policy_sum_error)
        illegal_masses.append(illegal_mass)
        if (
            len(policy_target) == len(legal_mask)
            and policy_target
            and all(isfinite(value) and value >= 0.0 for value in policy_target)
            and policy_sum_error <= policy_tol
            and illegal_mass <= policy_tol
        ):
            valid_policy_targets += 1
        if isfinite(_as_float(row.get("value_target"), float("nan"))):
            finite_value_targets += 1
        if _row_has_mcts_target(row):
            mcts_target_rows += 1

    row_count = len(rows)
    cache["rows"] = row_count
    cache["rows_source"] = str(path)
    cache["legal_action_rate"] = _rate(legal_actions, row_count)
    cache["policy_target_valid_rate"] = _rate(valid_policy_targets, row_count)
    cache["policy_target_sum_error_max"] = max(policy_sum_errors) if policy_sum_errors else None
    cache["illegal_policy_mass_max"] = max(illegal_masses) if illegal_masses else None
    cache["value_target_finite_rate"] = _rate(finite_value_targets, row_count)
    cache["mcts_target_rows"] = mcts_target_rows
    cache["mcts_target_rate"] = _rate(mcts_target_rows, row_count)
    cache["observation_dim_consistent"] = len(observation_dims) <= 1
    cache["action_dim_consistent"] = len(action_dims) <= 1
    if observation_dims:
        action_surface["observation_dim"] = (
            next(iter(observation_dims)) if len(observation_dims) == 1 else -1
        )
        cache["observation_dims"] = sorted(observation_dims)
    if action_dims:
        action_surface["action_dim"] = next(iter(action_dims)) if len(action_dims) == 1 else -1
        cache["action_dims"] = sorted(action_dims)


def _apply_mcts_metrics(metrics: dict[str, Any], source: Mapping[str, Any]) -> None:
    policy_summaries = _mapping(source.get("policy_summaries"))
    if not policy_summaries:
        policy_summaries = _mapping(_mapping(source.get("metrics")).get("policy_summaries"))
    total_decisions = 0
    illegal_actions = 0
    policies: list[str] = []
    for name, summary_value in policy_summaries.items():
        summary = _mapping(summary_value)
        if not str(name).startswith("mcts_"):
            continue
        policies.append(str(name))
        total_decisions += _as_int(summary.get("total_decisions"))
        illegal_actions += _as_int(summary.get("illegal_action_count"))
    search = _nested_metrics(metrics, "search_smoke")
    if total_decisions > 0:
        search["ran"] = True
        search["total_decisions"] = total_decisions
        search["illegal_action_count"] = illegal_actions
        search["policies"] = policies


def _apply_train_metrics(metrics: dict[str, Any], source: Mapping[str, Any]) -> None:
    source_metrics = _mapping(source.get("metrics", source))
    source_train = _mapping(source_metrics.get("train_smoke")) or source_metrics
    train = _nested_metrics(metrics, "train_smoke")
    train["ran"] = bool(source_train.get("ran", source.get("status") in {"pass", "smoke_only"}))
    train["losses_finite"] = bool(source_train.get("losses_finite", False))
    train["checkpoint_exists"] = bool(source_train.get("checkpoint_exists", False))
    for key in (
        "checkpoint_path",
        "policy_loss_initial",
        "policy_loss_final",
        "value_loss_initial",
        "value_loss_final",
        "dynamics_loss_initial",
        "dynamics_loss_final",
        "total_loss_initial",
        "total_loss_final",
        "epochs",
        "rows",
        "elapsed_sec",
    ):
        if key in source_train:
            train[key] = source_train[key]


def _apply_baseline_metrics(metrics: dict[str, Any], source: Mapping[str, Any]) -> None:
    baselines = _nested_metrics(metrics, "baselines")
    policies = _as_list(source.get("policies"))
    if not policies:
        policies = list(_mapping(source.get("policy_summaries")).keys())
    if not policies:
        policies = list(_mapping(_mapping(source.get("metrics")).get("policy_summaries")).keys())
    comparison = source.get("comparison")
    if comparison is None:
        comparison = _mapping(source.get("metrics")).get("comparison")
    baselines["present"] = bool(policies)
    baselines["comparison_present"] = bool(comparison)
    baselines["policies"] = [str(policy) for policy in policies]
    if comparison:
        baselines["comparison"] = comparison


def _apply_parity_metrics(metrics: dict[str, Any], source: Mapping[str, Any]) -> None:
    parity = _nested_metrics(metrics, "parity")
    summary = _mapping(source.get("summary"))
    parity["present"] = True
    parity["schema"] = source.get("schema")
    parity["status"] = str(source.get("status", "missing"))
    parity["oracle_backend"] = source.get("oracle_backend")
    parity["backends"] = _as_list(source.get("backends"))
    parity["total_checks"] = _as_int(summary.get("total_checks"))
    parity["passed"] = _as_int(summary.get("passed"))
    parity["failed"] = _as_int(summary.get("failed"))


def _apply_overrides(metrics: dict[str, Any], config: StrategicMuZeroLoopConfig) -> None:
    cache = _nested_metrics(metrics, "cache")
    for key, value in (
        ("mcts_target_rows", config.mcts_target_rows),
        ("mcts_target_rate", config.mcts_target_rate),
        ("policy_target_valid_rate", config.policy_target_valid_rate),
        ("policy_target_sum_error_max", config.policy_target_sum_error_max),
        ("illegal_policy_mass_max", config.illegal_policy_mass_max),
        ("value_target_finite_rate", config.value_target_finite_rate),
    ):
        if value is not None:
            cache[key] = value
    if config.checkpoint_path is not None:
        train = _nested_metrics(metrics, "train_smoke")
        train["checkpoint_path"] = str(config.checkpoint_path)
        train["checkpoint_exists"] = config.checkpoint_path.exists()
    if config.deterministic is not None:
        deterministic = _nested_metrics(metrics, "determinism")
        deterministic["fixed_seed_reproducible"] = config.deterministic


def _source_artifacts(config: StrategicMuZeroLoopConfig) -> list[str]:
    paths = [
        config.cache_metrics_path,
        config.cache_rows_path,
        config.mcts_metrics_path,
        config.train_metrics_path,
        config.baseline_metrics_path,
        config.parity_metrics_path,
        config.checkpoint_path,
    ]
    return [str(path) for path in paths if path is not None]


def _verifier_summary(verifier: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": verifier.get("schema"),
        "verdict": verifier.get("verdict"),
        "status": verifier.get("status"),
        "checked_at": verifier.get("checked_at"),
        "failed_checks": [
            str(finding["check"])
            for finding in _as_list(verifier.get("findings"))
            if isinstance(finding, dict) and "check" in finding
        ],
    }


def _write_loop_state(
    out_dir: Path,
    config: StrategicMuZeroLoopConfig,
    report: Mapping[str, Any],
    verifier: Mapping[str, Any],
) -> None:
    timestamp = datetime.now(UTC).isoformat()
    log_path = out_dir / "loop-run-log.md"
    attempt = _next_attempt_number(log_path)
    allowance = _allowance_check(config, timestamp)
    accepted = verifier.get("verdict") == "ACCEPT"
    state = {
        "schema": "loop-state/v1",
        "owner": "mini_tft.tools.strategic_muzero_loop",
        "objective": "verify strategic MuZero cache/search/train readiness",
        "deliverable": "muzero_cache",
        "attempt_cap": config.attempt_cap,
        "attempt": attempt,
        "status": "accepted" if accepted else "blocked",
        "current_next_action": _next_action(verifier, config.codex_allowance_decision),
        "prune_rules": [
            "keep metrics, decisions, final reports, loop logs, and verifier outputs",
            "cache rows and train checkpoints may be pruned only after accepted replacement",
        ],
        "codex_allowance_check": allowance,
        "acceptance_criteria": [
            f"cache rows >= {config.min_cache_rows}",
            "observation/action dimensions match the strategic 38x11 contract",
            "all cache actions are legal",
            "every cache row has an MCTS visit-policy target",
            "policy targets sum to one and place zero mass on illegal actions",
            "value targets are finite",
            "tiny policy/value/dynamics train smoke has finite losses",
            "fixed-seed trace/cache reproducibility is recorded",
            "strategic Python/native/Ocean parity matrix passes",
            "random and heuristic baselines are present before queue-ready pass",
            "read-only metric verifier accepts",
        ],
        "blocked_condition": [
            "missing MCTS-target cache rows",
            "illegal policy target mass",
            "non-finite value or train losses",
            "missing baseline comparison",
            "missing or failing strategic parity matrix",
            "missing train checkpoint for queue-ready pass",
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
            (
                "env -u UV_PYTHON uv run python -m "
                "mini_tft.tools.strategic_muzero_loop --strict"
            ),
            "env -u UV_PYTHON uv run pytest -q tests/test_strategic_muzero_loop.py",
        ],
        "artifacts": report.get("artifacts", []),
        "verifier": "accept" if accepted else "reject",
        "config": {
            "seed": config.seed,
            "min_cache_rows": config.min_cache_rows,
            "require_queue_ready": config.require_queue_ready,
            "cache_metrics_path": _path_or_none(config.cache_metrics_path),
            "cache_rows_path": _path_or_none(config.cache_rows_path),
            "mcts_metrics_path": _path_or_none(config.mcts_metrics_path),
            "train_metrics_path": _path_or_none(config.train_metrics_path),
            "baseline_metrics_path": _path_or_none(config.baseline_metrics_path),
            "parity_metrics_path": _path_or_none(config.parity_metrics_path),
        },
    }
    _write_json(out_dir / "loop-state.json", state)
    if not log_path.exists():
        log_path.write_text("# Strategic MuZero Loop Run Log\n", encoding="utf-8")
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
                    f"Action: generated MuZero-readiness loop artifacts in `{out_dir}`.",
                    f"Validation: verifier `{verifier['verdict']}`; status `{report['status']}`.",
                    f"Verifier: {verifier['verdict']}.",
                    f"Next action: {_next_action(verifier, config.codex_allowance_decision)}",
                    "",
                ]
            )
        )


def _allowance_check(config: StrategicMuZeroLoopConfig, timestamp: str) -> dict[str, str]:
    return {
        "source": config.codex_allowance_source,
        "checked_at": timestamp,
        "5h_window_remaining": config.codex_five_hour_window_remaining,
        "weekly_usage": config.codex_weekly_usage,
        "decision": config.codex_allowance_decision,
    }


def _format_decision(report: Mapping[str, Any]) -> str:
    verifier = _mapping(report.get("verifier"))
    return "\n".join(
        [
            "# Decision",
            "",
            f"Status: {report['status']}",
            "",
            "Evidence:",
            f"- Deliverable: {report['deliverable']}",
            f"- Seed: {report['seed']}",
            f"- Verifier verdict: {verifier.get('verdict', 'not_run')}",
            f"- Failed checks: {', '.join(_as_list(verifier.get('failed_checks'))) or 'none'}",
            "",
            "Limits:",
            *[f"- {limit}" for limit in _as_list(report.get("known_limits"))],
            "",
            "Next:",
            f"- {_next_action(verifier)}",
            "",
        ]
    )


def _format_final_report(report: Mapping[str, Any], verifier: Mapping[str, Any]) -> str:
    metrics = _mapping(report.get("metrics"))
    cache = _mapping(metrics.get("cache"))
    search = _mapping(metrics.get("search_smoke"))
    train = _mapping(metrics.get("train_smoke"))
    baselines = _mapping(metrics.get("baselines"))
    parity = _mapping(metrics.get("parity"))
    deterministic = _mapping(metrics.get("determinism"))
    return "\n".join(
        [
            "# Strategic MuZero Loop Final Report",
            "",
            f"Status: `{report['status']}`",
            f"Verifier: `{verifier['verdict']}`",
            "",
            "## Key Metrics",
            "",
            f"- Cache rows: {_as_int(cache.get('rows'))}",
            f"- MCTS target rows: {_as_int(cache.get('mcts_target_rows'))}",
            f"- Legal action rate: {_as_float(cache.get('legal_action_rate')):.6f}",
            (
                "- Policy target valid rate: "
                f"{_as_float(cache.get('policy_target_valid_rate')):.6f}"
            ),
            (
                "- Illegal policy mass max: "
                f"{_as_float(cache.get('illegal_policy_mass_max'), float('inf')):.6g}"
            ),
            f"- Value target finite rate: {_as_float(cache.get('value_target_finite_rate')):.6f}",
            f"- MCTS search decisions: {_as_int(search.get('total_decisions'))}",
            f"- Train smoke ran: {bool(train.get('ran'))}",
            f"- Train losses finite: {bool(train.get('losses_finite'))}",
            f"- Train checkpoint exists: {bool(train.get('checkpoint_exists'))}",
            f"- Baseline policies: {', '.join(_as_list(baselines.get('policies'))) or 'none'}",
            (
                "- Parity matrix: "
                f"{parity.get('status', 'not_run')} "
                f"({_as_int(parity.get('passed'))}/{_as_int(parity.get('total_checks'))})"
            ),
            f"- Deterministic: {bool(deterministic.get('fixed_seed_reproducible'))}",
            "",
            "## Verifier Findings",
            "",
            *_format_findings(verifier),
            "",
            "## What This Means",
            "",
            "- `pass` means the run has enough cache/search/train/baseline evidence to queue.",
            "- `smoke_only` or `blocked` must not be promoted to MuZero quality.",
            "",
        ]
    )


def _format_verifier_decision(verifier: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Verifier Decision",
            "",
            f"Verdict: {verifier['verdict']}",
            f"Status: {verifier['status']}",
            f"Require queue ready: {verifier['require_queue_ready']}",
            "",
            "Findings:",
            *_format_findings(verifier),
            "",
        ]
    )


def _format_findings(verifier: Mapping[str, Any]) -> list[str]:
    findings = _as_list(verifier.get("findings"))
    if not findings:
        return ["- None."]
    return [
        f"- {str(_mapping(finding).get('check'))}: {str(_mapping(finding).get('message'))}"
        for finding in findings
    ]


def _status_for_checks(failed: Sequence[Mapping[str, Any]], *, require_queue_ready: bool) -> str:
    if not failed:
        return "pass" if require_queue_ready else "smoke_only"
    missing_evidence = {
        "cache_rows",
        "mcts_target_rows",
        "mcts_search_smoke",
        "train_smoke_ran",
        "determinism",
        "baseline_comparison",
        "baseline_policies",
        "checkpoint",
        "parity_matrix",
    }
    hard_failures = {
        "schema",
        "deliverable",
        "observation_dim",
        "action_dim",
        "legal_action_rate",
        "policy_target_valid_rate",
        "value_target_finite_rate",
        "policy_target_sum_error_max",
        "illegal_policy_mass_max",
        "mcts_search_illegal_actions",
        "train_losses_finite",
    }
    failed_names = {str(check["name"]) for check in failed}
    if missing_evidence & failed_names:
        return "blocked"
    return "fail" if hard_failures & failed_names else "blocked"


def _next_action(verifier: Mapping[str, Any], allowance_decision: str = "continue") -> str:
    if verifier.get("verdict") == "ACCEPT":
        if allowance_decision == "continue":
            return "queue the overnight MuZero run using these accepted artifacts"
        return "provide Codex allowance status before queueing the overnight MuZero run"
    findings = _as_list(verifier.get("findings"))
    if not findings:
        return "inspect verifier output before queueing MuZero"
    first = _mapping(findings[0])
    return f"fix `{first.get('check')}`: {first.get('message')}"


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    success: str,
    failure: str,
) -> None:
    is_passed = bool(passed)
    checks.append(
        {
            "name": name,
            "passed": is_passed,
            "message": success if is_passed else failure,
            "success": success,
            "failure": failure,
        }
    )


def _check_equal(
    checks: list[dict[str, Any]],
    name: str,
    observed: int,
    expected: int,
) -> None:
    _check(
        checks,
        name,
        observed == expected,
        f"{name} is {expected}",
        f"expected {expected}, got {observed}",
    )


def _check_rate(
    checks: list[dict[str, Any]],
    name: str,
    value: Any,
    expected: float,
) -> None:
    observed = _as_float(value, float("nan"))
    _check(
        checks,
        name,
        observed == expected,
        f"{name} is {expected:.6f}",
        f"{name} expected {expected:.6f}, got {observed:.6f}",
    )


def _copy_int(source: Mapping[str, Any], target: dict[str, Any], key: str) -> None:
    if key in source:
        target[key] = _as_int(source[key])


def _copy_float(source: Mapping[str, Any], target: dict[str, Any], key: str) -> None:
    if key in source:
        target[key] = _as_float(source[key])


def _row_has_mcts_target(row: Mapping[str, Any]) -> bool:
    metadata = _mapping(row.get("metadata"))
    source = str(metadata.get("policy_target_source", metadata.get("target_source", ""))).lower()
    if source == "mcts" or "mcts" in source:
        return True
    return any(
        key in metadata
        for key in (
            "mcts_simulations",
            "mcts_visits",
            "mcts_visit_policy",
            "root_visits",
            "search_visits",
        )
    )


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested_metrics(metrics: dict[str, Any], key: str) -> dict[str, Any]:
    value = metrics.setdefault(key, {})
    if isinstance(value, dict):
        return value
    metrics[key] = {}
    return metrics[key]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _read_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _next_attempt_number(log_path: Path) -> int:
    if not log_path.exists():
        return 1
    return log_path.read_text(encoding="utf-8").count("## Attempt ") + 1


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_LOOP_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempt-cap", type=int, default=3)
    parser.add_argument("--min-cache-rows", type=int, default=1024)
    parser.add_argument("--allow-smoke-only", action="store_true")
    parser.add_argument("--cache-metrics", type=Path)
    parser.add_argument("--cache-rows-jsonl", type=Path)
    parser.add_argument("--mcts-metrics", type=Path)
    parser.add_argument("--train-metrics", type=Path)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--parity-metrics", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mcts-target-rows", type=int)
    parser.add_argument("--mcts-target-rate", type=float)
    parser.add_argument("--policy-target-valid-rate", type=float)
    parser.add_argument("--policy-target-sum-error-max", type=float)
    parser.add_argument("--illegal-policy-mass-max", type=float)
    parser.add_argument("--value-target-finite-rate", type=float)
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
    report = run_strategic_muzero_loop(
        StrategicMuZeroLoopConfig(
            out_dir=args.out_dir,
            seed=args.seed,
            attempt_cap=args.attempt_cap,
            min_cache_rows=args.min_cache_rows,
            require_queue_ready=not args.allow_smoke_only,
            cache_metrics_path=args.cache_metrics,
            cache_rows_path=args.cache_rows_jsonl,
            mcts_metrics_path=args.mcts_metrics,
            train_metrics_path=args.train_metrics,
            baseline_metrics_path=args.baseline_metrics,
            parity_metrics_path=args.parity_metrics,
            checkpoint_path=args.checkpoint,
            deterministic=True if args.deterministic else None,
            mcts_target_rows=args.mcts_target_rows,
            mcts_target_rate=args.mcts_target_rate,
            policy_target_valid_rate=args.policy_target_valid_rate,
            policy_target_sum_error_max=args.policy_target_sum_error_max,
            illegal_policy_mass_max=args.illegal_policy_mass_max,
            value_target_finite_rate=args.value_target_finite_rate,
            codex_allowance_source=args.codex_allowance_source,
            codex_five_hour_window_remaining=args.codex_5h_window_remaining,
            codex_weekly_usage=args.codex_weekly_usage,
            codex_allowance_decision=args.codex_allowance_decision,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and report["verifier"]["verdict"] != "ACCEPT":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
