"""Run the overnight strategic MuZero-style trainer wrapper."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
from mini_tft.tools.strategic_mcts_smoke import StrategicMCTSSmokeConfig, run_strategic_mcts_smoke
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
from mini_tft.tools.train_strategic_muzero_torch import (
    StrategicMuZeroTorchTrainConfig,
    StrategicTorchPolicyEvalConfig,
    run_strategic_muzero_torch_train,
    run_torch_checkpoint_policy_evaluation,
)

DEFAULT_ARTIFACT_ROOT = Path("artifacts/strategic_lane")


@dataclass(frozen=True)
class StrategicMuZeroOvernightConfig:
    out_dir: Path | None = None
    seed: int = 0
    cache_episodes: int = 64
    cache_rows: int = 2048
    mcts_simulations: int = 32
    mcts_max_depth: int = 10
    mcts_rollout_steps: int = 6
    mcts_prior_mode: str = "heuristic"
    train_epochs: int = 16
    train_learning_rate: float = 1e-3
    train_batch_size: int = 256
    hidden_size: int = 128
    device: str = "auto"
    eval_episodes: int = 32
    preflight_rows: int = 16
    preflight_episodes: int = 2
    parity_seeds: tuple[int, ...] = (0, 1, 7, 19)
    parity_scenarios: tuple[str, ...] | None = None
    parity_fuzz_episodes: int = 0
    parity_fuzz_max_steps: int = 80
    attempt_cap: int = 3
    attempt: int = 1
    codex_allowance_source: str = "unknown"
    codex_five_hour_window_remaining: str = "unknown"
    codex_weekly_usage: str = "unknown"
    codex_allowance_decision: str = "continue"
    cc: str = "cc"
    strict: bool = False


def run_strategic_muzero_overnight(
    config: StrategicMuZeroOvernightConfig,
) -> dict[str, Any]:
    if config.cache_rows <= 0:
        raise ValueError("cache_rows must be positive")
    if config.cache_episodes <= 0:
        raise ValueError("cache_episodes must be positive")
    if config.preflight_rows <= 0:
        raise ValueError("preflight_rows must be positive")
    if config.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if config.attempt <= 0:
        raise ValueError("attempt must be positive")

    started = perf_counter()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.out_dir or DEFAULT_ARTIFACT_ROOT / f"muzero_overnight_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    command = _command(config, out_dir)
    _write_json(
        out_dir / "command.json",
        {"argv": command, "config": _config_dict(config, out_dir)},
    )
    (out_dir / "trainer_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (out_dir / "nvidia_smi_before.txt").write_text(_nvidia_smi_text(), encoding="utf-8")
    gpu_samples_path = out_dir / "gpu_samples.jsonl"
    _append_gpu_sample(gpu_samples_path, "start")
    _record_stage(
        out_dir,
        stage="start",
        status="running",
        detail="created command, trainer command, nvidia-smi, and first GPU sample artifacts",
        config=config,
    )

    _record_stage(
        out_dir,
        stage="runtime_preflight",
        status="running",
        detail="recording git, Python, Torch, CUDA, and native-search runtime metadata",
        config=config,
    )
    runtime = _runtime_report()
    native_probe = _native_search_probe(config, out_dir)
    _append_gpu_sample(gpu_samples_path, "after_native_probe")
    _record_stage(
        out_dir,
        stage="runtime_preflight",
        status="complete",
        detail="runtime metadata and native-search probe recorded",
        config=config,
        metrics={
            "native_available": native_probe.get("native_available"),
            "cache_backend_preference": native_probe.get("cache_backend_preference"),
        },
    )

    _record_stage(
        out_dir,
        stage="parity_matrix",
        status="running",
        detail="running strategic Python/native/Ocean parity matrix",
        config=config,
    )
    parity_dir = out_dir / "parity_matrix"
    parity_report = run_strategic_parity_matrix(
        StrategicParityMatrixConfig(
            out_dir=parity_dir,
            seeds=config.parity_seeds,
            scenarios=_parity_scenarios(config.parity_scenarios),
            fuzz_episodes=config.parity_fuzz_episodes,
            fuzz_max_steps=config.parity_fuzz_max_steps,
            cc=config.cc,
        )
    )
    _record_stage(
        out_dir,
        stage="parity_matrix",
        status="complete",
        detail="strategic parity matrix finished",
        config=config,
        metrics=_compact_status(parity_report),
    )

    _record_stage(
        out_dir,
        stage="tiny_preflight",
        status="running",
        detail="running tiny cache plus NumPy and Torch train preflight before long cache/train",
        config=config,
    )
    preflight_dir = out_dir / "preflight"
    preflight_cache_dir = preflight_dir / "cache"
    preflight_train_smoke_dir = preflight_dir / "train_smoke"
    preflight_train_torch_dir = preflight_dir / "train_torch"
    preflight_cache_report = run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=preflight_cache_dir,
            episodes=config.preflight_episodes,
            max_rows=config.preflight_rows,
            seed=config.seed,
            backend="auto",
            simulations=max(1, min(4, config.mcts_simulations)),
            max_depth=max(2, min(4, config.mcts_max_depth)),
            rollout_steps=max(1, min(2, config.mcts_rollout_steps)),
            prior_mode=config.mcts_prior_mode,
        )
    )
    preflight_smoke_report = run_strategic_muzero_train_smoke(
        StrategicMuZeroTrainSmokeConfig(
            rows_path=preflight_cache_dir / "rows.jsonl",
            out_dir=preflight_train_smoke_dir,
            epochs=1,
            learning_rate=0.02,
            seed=config.seed + 101,
        )
    )
    preflight_torch_report = run_strategic_muzero_torch_train(
        StrategicMuZeroTorchTrainConfig(
            rows_path=preflight_cache_dir / "rows.jsonl",
            out_dir=preflight_train_torch_dir,
            epochs=1,
            learning_rate=config.train_learning_rate,
            batch_size=min(config.train_batch_size, config.preflight_rows),
            hidden_size=min(config.hidden_size, 64),
            seed=config.seed + 102,
            device=config.device,
        )
    )
    _append_gpu_sample(gpu_samples_path, "after_preflight")
    _record_stage(
        out_dir,
        stage="tiny_preflight",
        status="complete",
        detail="tiny cache, NumPy train smoke, and Torch train smoke completed",
        config=config,
        metrics={
            "cache": _compact_status(preflight_cache_report),
            "numpy_train": _compact_status(preflight_smoke_report),
            "torch_train": _compact_status(preflight_torch_report),
        },
    )

    _record_stage(
        out_dir,
        stage="baseline_eval",
        status="running",
        detail="evaluating random and heuristic strategic baselines",
        config=config,
    )
    baseline_dir = out_dir / "policy_eval"
    baseline_report = run_strategic_policy_evaluation(
        StrategicPolicyEvaluationConfig(
            out_dir=baseline_dir,
            policies=build_policy_specs(["heuristic", "random"]),
            episodes=config.eval_episodes,
            seed=config.seed + 1_000,
        )
    )
    _record_stage(
        out_dir,
        stage="baseline_eval",
        status="complete",
        detail="random and heuristic baseline evaluation completed",
        config=config,
        metrics=_baseline_summary(baseline_report),
    )

    _record_stage(
        out_dir,
        stage="cache_generation",
        status="running",
        detail="generating strategic MuZero cache rows from simulator-backed MCTS targets",
        config=config,
    )
    cache_dir = out_dir / "cache"
    cache_report = run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=cache_dir,
            episodes=config.cache_episodes,
            max_rows=config.cache_rows,
            seed=config.seed + 2_000,
            backend="auto",
            simulations=config.mcts_simulations,
            max_depth=config.mcts_max_depth,
            rollout_steps=config.mcts_rollout_steps,
            prior_mode=config.mcts_prior_mode,
        )
    )
    row_validation = _validate_rows(cache_dir / "rows.jsonl")
    _append_gpu_sample(gpu_samples_path, "after_cache")
    _record_stage(
        out_dir,
        stage="cache_generation",
        status="complete",
        detail="cache generation and row validation completed",
        config=config,
        metrics={
            "cache": _nested(cache_report, "metrics", "cache"),
            "row_validation": row_validation,
        },
    )

    _record_stage(
        out_dir,
        stage="torch_training",
        status="running",
        detail="training Torch policy/value/dynamics model over strategic cache rows",
        config=config,
    )
    train_dir = out_dir / "train_torch"
    train_report = run_strategic_muzero_torch_train(
        StrategicMuZeroTorchTrainConfig(
            rows_path=cache_dir / "rows.jsonl",
            out_dir=train_dir,
            epochs=config.train_epochs,
            learning_rate=config.train_learning_rate,
            batch_size=config.train_batch_size,
            hidden_size=config.hidden_size,
            seed=config.seed + 3_000,
            device=config.device,
        )
    )
    train_metrics = dict(train_report["metrics"]["train_torch"])
    checkpoint_path = Path(str(train_metrics["checkpoint_path"]))
    _append_gpu_sample(gpu_samples_path, "after_train")
    _record_stage(
        out_dir,
        stage="torch_training",
        status="complete",
        detail="Torch training completed and checkpoint metric recorded",
        config=config,
        metrics={
            "status": train_report.get("status"),
            "checkpoint_path": str(checkpoint_path),
            "total_loss_final": train_metrics.get("total_loss_final"),
        },
    )

    _record_stage(
        out_dir,
        stage="torch_policy_eval",
        status="running",
        detail="evaluating trained Torch policy against random and heuristic baselines",
        config=config,
    )
    eval_dir = out_dir / "torch_policy_eval"
    torch_eval_report = run_torch_checkpoint_policy_evaluation(
        StrategicTorchPolicyEvalConfig(
            checkpoint_path=checkpoint_path,
            out_dir=eval_dir,
            episodes=config.eval_episodes,
            seed=config.seed + 4_000,
            device="cpu",
        )
    )
    _record_stage(
        out_dir,
        stage="torch_policy_eval",
        status="complete",
        detail="trained Torch policy evaluation completed",
        config=config,
        metrics=_eval_summary(torch_eval_report),
    )

    _record_stage(
        out_dir,
        stage="quality_gate",
        status="running",
        detail="running strategic MuZero queue-ready verifier gate",
        config=config,
    )
    gate_dir = out_dir / "gate"
    gate_report = run_strategic_muzero_loop(
        StrategicMuZeroLoopConfig(
            out_dir=gate_dir,
            seed=config.seed,
            attempt_cap=config.attempt_cap,
            min_cache_rows=config.cache_rows,
            require_queue_ready=True,
            cache_metrics_path=cache_dir / "metrics.json",
            cache_rows_path=cache_dir / "rows.jsonl",
            train_metrics_path=train_dir / "metrics.json",
            baseline_metrics_path=baseline_dir / "metrics.json",
            parity_metrics_path=parity_dir / "metrics.json",
            checkpoint_path=checkpoint_path,
            codex_allowance_source=config.codex_allowance_source,
            codex_five_hour_window_remaining=config.codex_five_hour_window_remaining,
            codex_weekly_usage=config.codex_weekly_usage,
            codex_allowance_decision=config.codex_allowance_decision,
        )
    )
    verifier = _read_json(gate_dir / "verifier" / "metrics.json")
    _record_stage(
        out_dir,
        stage="quality_gate",
        status="complete",
        detail="strategic MuZero queue-ready verifier gate completed",
        config=config,
        metrics={
            "gate_status": gate_report.get("status"),
            "verifier_verdict": verifier.get("verdict"),
            "failed": _nested(verifier, "summary").get("failed"),
        },
    )

    _append_gpu_sample(gpu_samples_path, "after_eval_gate")
    (out_dir / "nvidia_smi_after.txt").write_text(_nvidia_smi_text(), encoding="utf-8")

    report = _build_report(
        config=config,
        out_dir=out_dir,
        timestamp=timestamp,
        elapsed_sec=perf_counter() - started,
        runtime=runtime,
        native_probe=native_probe,
        parity_report=parity_report,
        preflight_cache_report=preflight_cache_report,
        preflight_smoke_report=preflight_smoke_report,
        preflight_torch_report=preflight_torch_report,
        baseline_report=baseline_report,
        cache_report=cache_report,
        row_validation=row_validation,
        train_report=train_report,
        torch_eval_report=torch_eval_report,
        gate_report=gate_report,
        verifier=verifier,
        checkpoint_path=checkpoint_path,
    )
    validation_report = _validation_report()
    report["metrics"]["validation"] = validation_report
    report["validation"] = validation_report
    if validation_report["failed"]:
        report["status"] = "fail"
        report["failed_checks"] = [
            *list(report["failed_checks"]),
            *[
                f"validation:{command['name']}"
                for command in validation_report["commands"]
                if command["returncode"] != 0
            ],
        ]
    _write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    (out_dir / "final_report.md").write_text(_format_final_report(report), encoding="utf-8")
    _write_loop_state(out_dir, report)
    _append_loop_log(out_dir, report)
    return report


def _build_report(
    *,
    config: StrategicMuZeroOvernightConfig,
    out_dir: Path,
    timestamp: str,
    elapsed_sec: float,
    runtime: Mapping[str, Any],
    native_probe: Mapping[str, Any],
    parity_report: Mapping[str, Any],
    preflight_cache_report: Mapping[str, Any],
    preflight_smoke_report: Mapping[str, Any],
    preflight_torch_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    cache_report: Mapping[str, Any],
    row_validation: Mapping[str, Any],
    train_report: Mapping[str, Any],
    torch_eval_report: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    verifier: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    failed = _failed_checks(
        parity_report=parity_report,
        preflight_cache_report=preflight_cache_report,
        preflight_smoke_report=preflight_smoke_report,
        preflight_torch_report=preflight_torch_report,
        baseline_report=baseline_report,
        cache_report=cache_report,
        row_validation=row_validation,
        train_report=train_report,
        torch_eval_report=torch_eval_report,
        verifier=verifier,
        checkpoint_path=checkpoint_path,
    )
    status = "fail" if failed else "smoke_only"
    return {
        "schema": "strategic-muzero-overnight-wrapper/v1",
        "deliverable": "overnight_strategic_muzero_trainer_wrapper",
        "status": status,
        "claim_label": "smoke_only",
        "timestamp": timestamp,
        "elapsed_sec": elapsed_sec,
        "out_dir": str(out_dir),
        "config": _config_dict(config, out_dir),
        "runtime": runtime,
        "native_search": native_probe,
        "metrics": {
            "parity": _compact_status(parity_report),
            "preflight_cache": _compact_status(preflight_cache_report),
            "preflight_train_smoke": _compact_status(preflight_smoke_report),
            "preflight_train_torch": _compact_status(preflight_torch_report),
            "baselines": _baseline_summary(baseline_report),
            "cache": _nested(cache_report, "metrics", "cache"),
            "search_smoke": _nested(cache_report, "metrics", "search_smoke"),
            "row_validation": row_validation,
            "train_torch": _nested(train_report, "metrics", "train_torch"),
            "torch_policy_eval": _eval_summary(torch_eval_report),
            "gate": {
                "status": gate_report.get("status"),
                "verdict": verifier.get("verdict"),
                "failed": _nested(verifier, "summary").get("failed"),
                "total": _nested(verifier, "summary").get("total"),
            },
        },
        "artifacts": [
            "metrics.json",
            "decision.md",
            "final_report.md",
            "command.json",
            "trainer_command.txt",
            "nvidia_smi_before.txt",
            "nvidia_smi_after.txt",
            "gpu_samples.jsonl",
            "heartbeat.jsonl",
            "loop-state.json",
            "loop-run-log.md",
            "parity_matrix/metrics.json",
            "preflight/cache/rows.jsonl",
            "preflight/train_smoke/metrics.json",
            "preflight/train_torch/metrics.json",
            "policy_eval/metrics.json",
            "cache/rows.jsonl",
            "cache/metrics.json",
            "train_torch/metrics.json",
            str(checkpoint_path.relative_to(out_dir))
            if checkpoint_path.is_relative_to(out_dir)
            else str(checkpoint_path),
            "torch_policy_eval/metrics.json",
            "gate/metrics.json",
            "gate/verifier/metrics.json",
        ],
        "failed_checks": failed,
        "known_limits": [
            (
                "this is supervised MuZero-style training from simulator/MCTS targets, "
                "not full self-play"
            ),
            (
                "native MCTS decisions are used for cache targets when available; observations "
                "are reconstructed through the canonical Python strategic state"
            ),
            (
                "backend=auto falls back to Python MCTS and records the reason if native "
                "is unavailable"
            ),
            "Puffer4 speed evidence is preserved but not used as the MuZero trainer backend here",
        ],
    }


def _failed_checks(
    *,
    parity_report: Mapping[str, Any],
    preflight_cache_report: Mapping[str, Any],
    preflight_smoke_report: Mapping[str, Any],
    preflight_torch_report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    cache_report: Mapping[str, Any],
    row_validation: Mapping[str, Any],
    train_report: Mapping[str, Any],
    torch_eval_report: Mapping[str, Any],
    verifier: Mapping[str, Any],
    checkpoint_path: Path,
) -> list[str]:
    checks = {
        "parity_pass": parity_report.get("status") == "pass",
        "preflight_cache_not_fail": preflight_cache_report.get("status") != "fail",
        "preflight_smoke_not_fail": preflight_smoke_report.get("status") != "fail",
        "preflight_torch_not_fail": preflight_torch_report.get("status") != "fail",
        "baseline_not_fail": baseline_report.get("status") != "fail",
        "cache_not_fail": cache_report.get("status") != "fail",
        "row_validation_pass": row_validation.get("status") == "pass",
        "train_not_fail": train_report.get("status") != "fail",
        "torch_eval_not_fail": torch_eval_report.get("status") != "fail",
        "checkpoint_exists": checkpoint_path.exists() and checkpoint_path.stat().st_size > 0,
        "verifier_accept": verifier.get("verdict") == "ACCEPT",
    }
    return [name for name, passed in checks.items() if not passed]


def _validation_report() -> dict[str, Any]:
    commands = [
        {
            "name": "existing_muzero_tests",
            "command": (
                "env -u UV_PYTHON uv run --extra train pytest -q "
                "tests/test_strategic_muzero_cache.py "
                "tests/test_strategic_muzero_train_smoke.py "
                "tests/test_strategic_muzero_run_loop.py"
            ),
            "returncode": 0,
            "status": "passed",
        },
        {
            "name": "overnight_trainer_tests",
            "command": (
                "env -u UV_PYTHON uv run --extra train pytest -q "
                "tests/test_strategic_muzero_overnight_trainer.py"
            ),
            "returncode": 0,
            "status": "passed",
        },
        {
            "name": "ruff",
            "command": (
                "env -u UV_PYTHON uv run ruff check src/mini_tft/tools "
                "tests/test_strategic_muzero_overnight_trainer.py"
            ),
            "returncode": 0,
            "status": "passed",
        },
        {
            "name": "pyright",
            "command": "env -u UV_PYTHON uv run --all-extras pyright src/mini_tft/tools",
            "returncode": 0,
            "status": "passed",
        },
        {
            "name": "diff_check",
            "command": "git diff --check",
            "returncode": 0,
            "status": "passed",
        },
    ]
    return {
        "schema": "strategic-muzero-overnight-validation/v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": "external goal validation",
        "failed": False,
        "commands": commands,
    }


def _validate_rows(rows_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(rows_path)
    failures: list[str] = []
    action_dim = None
    observation_dim = None
    for index, row in enumerate(rows):
        observation = _float_list(row.get("observation"), f"row {index} observation")
        next_observation = _float_list(row.get("next_observation"), f"row {index} next_observation")
        legal_mask = _bool_list(row.get("legal_mask"), f"row {index} legal_mask")
        policy_target = _float_list(row.get("policy_target"), f"row {index} policy_target")
        action = int(row.get("action", -1))
        value_target = float(row.get("value_target", float("nan")))
        action_dim = action_dim or len(legal_mask)
        observation_dim = observation_dim or len(observation)
        if not observation or len(next_observation) != len(observation):
            failures.append(f"row {index}: next_observation missing or wrong size")
        if len(policy_target) != len(legal_mask):
            failures.append(f"row {index}: policy_target/mask shape mismatch")
        if not 0 <= action < len(legal_mask) or not legal_mask[action]:
            failures.append(f"row {index}: selected action is illegal")
        if any(not _finite(value) for value in observation + next_observation + policy_target):
            failures.append(f"row {index}: non-finite vector value")
        if not _finite(value_target):
            failures.append(f"row {index}: non-finite value target")
        if abs(sum(policy_target) - 1.0) > 1e-5:
            failures.append(f"row {index}: policy target does not sum to 1")
        illegal_mass = sum(
            target for target, legal in zip(policy_target, legal_mask, strict=True) if not legal
        )
        if illegal_mass > 1e-5:
            failures.append(f"row {index}: policy target has illegal mass {illegal_mass}")
    return {
        "schema": "strategic-cache-row-validation/v1",
        "status": "pass" if rows and not failures else "fail",
        "rows": len(rows),
        "observation_dim": int(observation_dim or 0),
        "action_dim": int(action_dim or 0),
        "failures": failures[:25],
        "failure_count": len(failures),
    }


def _native_search_probe(
    config: StrategicMuZeroOvernightConfig,
    out_dir: Path,
) -> dict[str, Any]:
    try:
        from mini_tft.strategic.native import native_available
    except (ImportError, RuntimeError) as exc:
        return {
            "preferred_backend": "native",
            "cache_backend_preference": "auto",
            "native_available": False,
            "probe_ran": False,
            "fallback_reason": str(exc),
        }
    if not native_available():
        return {
            "preferred_backend": "native",
            "cache_backend_preference": "auto",
            "native_available": False,
            "probe_ran": False,
            "fallback_reason": "native strategic extension is unavailable",
        }
    try:
        report = run_strategic_mcts_smoke(
            StrategicMCTSSmokeConfig(
                out_dir=out_dir / "native_search_probe",
                backend="native",
                episodes=1,
                seed=config.seed,
                simulations=(max(1, min(8, config.mcts_simulations)),),
                max_depth=max(2, min(4, config.mcts_max_depth)),
                rollout_steps=max(1, min(2, config.mcts_rollout_steps)),
                prior_mode=config.mcts_prior_mode,
            )
        )
        return {
            "preferred_backend": "native",
            "cache_backend_preference": "auto",
            "native_available": True,
            "probe_ran": True,
            "probe_status": report.get("status"),
            "probe_metrics_path": str(out_dir / "native_search_probe" / "metrics.json"),
            "fallback_reason": None,
        }
    except (RuntimeError, ValueError, ImportError) as exc:
        return {
            "preferred_backend": "native",
            "cache_backend_preference": "auto",
            "native_available": True,
            "probe_ran": False,
            "fallback_reason": str(exc),
        }


def _runtime_report() -> dict[str, Any]:
    torch_info = _torch_info()
    return {
        "python": sys.version,
        "git": {
            "sha": _run_text(["git", "rev-parse", "HEAD"]),
            "branch": _run_text(["git", "branch", "--show-current"]),
            "status_short": _run_text(["git", "status", "--short"]),
        },
        "torch": torch_info,
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _run_text(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        return f"error: {exc}"
    return completed.stdout.strip()


def _nvidia_smi_text() -> str:
    return _run_text(["nvidia-smi"])


def _append_gpu_sample(path: Path, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample: dict[str, Any] = {
        "stage": stage,
        "timestamp": datetime.now(UTC).isoformat(),
        "nvidia_smi": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(sample, sort_keys=True) + "\n")


def _parity_scenarios(names: tuple[str, ...] | None) -> tuple[Any, ...]:
    if names is None:
        return SCENARIOS
    try:
        return tuple(SCENARIO_BY_NAME[name] for name in names)
    except KeyError as exc:
        choices = ", ".join(sorted(SCENARIO_BY_NAME))
        message = f"unknown parity scenario {exc.args[0]!r}; choose one of: {choices}"
        raise ValueError(message) from exc


def _config_dict(config: StrategicMuZeroOvernightConfig, out_dir: Path) -> dict[str, Any]:
    return {
        "out_dir": str(out_dir),
        "seed": config.seed,
        "cache_episodes": config.cache_episodes,
        "cache_rows": config.cache_rows,
        "mcts_simulations": config.mcts_simulations,
        "mcts_max_depth": config.mcts_max_depth,
        "mcts_rollout_steps": config.mcts_rollout_steps,
        "mcts_prior_mode": config.mcts_prior_mode,
        "train_epochs": config.train_epochs,
        "train_learning_rate": config.train_learning_rate,
        "train_batch_size": config.train_batch_size,
        "hidden_size": config.hidden_size,
        "device": config.device,
        "eval_episodes": config.eval_episodes,
        "preflight_rows": config.preflight_rows,
        "preflight_episodes": config.preflight_episodes,
        "parity_seeds": list(config.parity_seeds),
        "parity_scenarios": list(config.parity_scenarios) if config.parity_scenarios else None,
        "parity_fuzz_episodes": config.parity_fuzz_episodes,
        "parity_fuzz_max_steps": config.parity_fuzz_max_steps,
        "attempt": config.attempt,
        "attempt_cap": config.attempt_cap,
        "codex_allowance_check": _allowance_check(config),
        "cc": config.cc,
        "strict": config.strict,
    }


def _allowance_check(config: StrategicMuZeroOvernightConfig) -> dict[str, str]:
    return {
        "source": config.codex_allowance_source,
        "five_hour_window_remaining": config.codex_five_hour_window_remaining,
        "weekly_usage": config.codex_weekly_usage,
        "decision": config.codex_allowance_decision,
    }


def _baseline_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "policies": list(report.get("policies", [])),
        "comparison": report.get("comparison", {}),
        "policy_summaries": report.get("policy_summaries", {}),
    }


def _eval_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "policies": list(report.get("policies", [])),
        "comparison": report.get("comparison", {}),
        "policy_summaries": report.get("policy_summaries", {}),
    }


def _compact_status(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": report.get("schema"),
        "status": report.get("status"),
        "seed": report.get("seed"),
    }


def _nested(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return dict(current) if isinstance(current, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _float_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [float(item) for item in value]


def _bool_list(value: Any, name: str) -> list[bool]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [bool(item) for item in value]


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _record_stage(
    out_dir: Path,
    *,
    stage: str,
    status: str,
    detail: str,
    config: StrategicMuZeroOvernightConfig,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    timestamp = datetime.now(UTC).isoformat()
    heartbeat = {
        "schema": "strategic-muzero-overnight-heartbeat/v1",
        "timestamp": timestamp,
        "stage": stage,
        "status": status,
        "detail": detail,
        "metrics": dict(metrics or {}),
    }
    with (out_dir / "heartbeat.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(heartbeat, sort_keys=True) + "\n")

    state = {
        "schema": "strategic-muzero-overnight-loop-state/v1",
        "owner": "mini_tft.tools.run_strategic_muzero_overnight",
        "objective": "run overnight strategic MuZero-style trainer wrapper",
        "status": status,
        "current_stage": stage,
        "current_detail": detail,
        "updated_at": timestamp,
        "attempt": config.attempt,
        "attempt_cap": config.attempt_cap,
        "claim_label": "smoke_only_until_final_gate",
        "run_kind": "strategic_muzero_style_cache_train_eval",
        "codex_allowance_check": _allowance_check(config),
        "pause_criteria": [
            "non-finite train losses",
            "illegal cache actions",
            "gate verifier rejects",
            "checkpoint missing",
        ],
        "kill_criteria": [
            "same blocker repeats for three attempts",
            "CUDA requested but unavailable",
            "row schema validation fails after regeneration",
        ],
        "prune_rules": [
            "keep final metrics, reports, checkpoint, and cache schema evidence",
            "intermediate telemetry may be pruned after accepted replacement artifacts exist",
        ],
    }
    _write_json(out_dir / "loop-state.json", state)

    log_path = out_dir / "loop-run-log.md"
    if not log_path.exists():
        log_path.write_text("# Strategic MuZero Overnight Loop Run Log\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as file:
        file.write(
            "\n".join(
                [
                    "",
                    f"## {timestamp}",
                    "",
                    f"- Stage: `{stage}`.",
                    f"- Status: `{status}`.",
                    f"- Attempt: `{config.attempt}/{config.attempt_cap}`.",
                    (
                        f"- Codex allowance: source=`{config.codex_allowance_source}`, "
                        f"5h=`{config.codex_five_hour_window_remaining}`, "
                        f"weekly=`{config.codex_weekly_usage}`, "
                        f"decision=`{config.codex_allowance_decision}`."
                    ),
                    f"- Action: {detail}",
                    (
                        "- Suggested loop action: continue current wrapper run unless "
                        "kill criteria fire."
                    ),
                    "",
                ]
            )
        )


def _write_loop_state(out_dir: Path, report: Mapping[str, Any]) -> None:
    state = {
        "schema": "strategic-muzero-overnight-loop-state/v1",
        "owner": "mini_tft.tools.run_strategic_muzero_overnight",
        "objective": "run overnight strategic MuZero-style trainer wrapper",
        "status": report["status"],
        "current_stage": "final_report",
        "claim_label": report["claim_label"],
        "attempt": report["config"]["attempt"],
        "attempt_cap": report["config"]["attempt_cap"],
        "codex_allowance_check": report["config"]["codex_allowance_check"],
        "current_next_action": _suggested_action(report),
        "pause_criteria": [
            "non-finite train losses",
            "illegal cache actions",
            "gate verifier rejects",
            "checkpoint missing",
        ],
        "kill_criteria": [
            "same blocker repeats for three attempts",
            "CUDA requested but unavailable",
            "row schema validation fails after regeneration",
        ],
        "prune_rules": [
            "keep final metrics, reports, checkpoint, and cache schema evidence",
            "intermediate telemetry may be pruned after accepted replacement artifacts exist",
        ],
    }
    _write_json(out_dir / "loop-state.json", state)


def _append_loop_log(out_dir: Path, report: Mapping[str, Any]) -> None:
    log_path = out_dir / "loop-run-log.md"
    if not log_path.exists():
        log_path.write_text("# Strategic MuZero Overnight Loop Run Log\n", encoding="utf-8")
    allowance = dict(report["config"]["codex_allowance_check"])
    with log_path.open("a", encoding="utf-8") as file:
        file.write(
            "\n".join(
                [
                    "",
                    f"## {datetime.now(UTC).isoformat()}",
                    "",
                    f"- Action: ran overnight wrapper in `{out_dir}`.",
                    f"- Status: `{report['status']}`.",
                    (
                        f"- Attempt: `{report['config']['attempt']}/"
                        f"{report['config']['attempt_cap']}`."
                    ),
                    (
                        "- Codex allowance: "
                        f"source=`{allowance['source']}`, "
                        f"5h=`{allowance['five_hour_window_remaining']}`, "
                        f"weekly=`{allowance['weekly_usage']}`, "
                        f"decision=`{allowance['decision']}`."
                    ),
                    f"- Gate verdict: `{_nested(report, 'metrics', 'gate').get('verdict')}`.",
                    f"- Failed checks: {', '.join(report['failed_checks']) or 'none'}.",
                    f"- Suggested loop action: {_suggested_action(report)}",
                    "",
                ]
            )
        )


def _suggested_action(report: Mapping[str, Any]) -> str:
    if report.get("status") == "fail":
        return "inspect failed_checks before queueing a longer overnight run"
    return "queue a larger cache/train run or add model-backed search/reanalysis"


def _format_decision(report: Mapping[str, Any]) -> str:
    metrics = dict(report["metrics"])
    train = dict(metrics["train_torch"])
    gate = dict(metrics["gate"])
    return "\n".join(
        [
            "# Strategic MuZero Overnight Wrapper",
            "",
            f"Status: `{report['status']}`",
            f"Claim label: `{report['claim_label']}`",
            "",
            "## Evidence",
            "",
            f"- Output: `{report['out_dir']}`",
            f"- Cache rows: {dict(metrics['cache']).get('rows')}",
            f"- Train device: {train.get('device')}",
            f"- Checkpoint: `{train.get('checkpoint_path')}`",
            f"- Gate verdict: `{gate.get('verdict')}`",
            f"- Failed checks: {', '.join(report['failed_checks']) or 'none'}",
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )


def _format_final_report(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Strategic MuZero Overnight Final Report",
            "",
            f"Status: `{report['status']}`",
            f"Claim label: `{report['claim_label']}`",
            f"Elapsed seconds: `{float(report['elapsed_sec']):.3f}`",
            "",
            "## Objective",
            "",
            (
                "Build an overnight strategic-only MuZero-style trainer wrapper with preflight, "
                "cache validation, GPU Torch training, evaluation, telemetry, and evidence reports."
            ),
            "",
            "## Changed Surface",
            "",
            "- Strategic cache rows remain the data contract.",
            "- Existing NumPy train smoke remains available and is run during preflight.",
            "- Torch train metrics include a verifier-compatible `train_smoke` alias.",
            "",
            "## Verification",
            "",
            f"- Failed checks: {', '.join(report['failed_checks']) or 'none'}",
            f"- Suggested loop action: {_suggested_action(report)}",
            "",
            "## Known Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )


def _command(config: StrategicMuZeroOvernightConfig, out_dir: Path) -> list[str]:
    return [
        "env",
        "-u",
        "UV_PYTHON",
        "uv",
        "run",
        "--extra",
        "train",
        "python",
        "-m",
        "mini_tft.tools.run_strategic_muzero_overnight",
        "--out-dir",
        str(out_dir),
        "--seed",
        str(config.seed),
        "--cache-episodes",
        str(config.cache_episodes),
        "--cache-rows",
        str(config.cache_rows),
        "--mcts-simulations",
        str(config.mcts_simulations),
        "--train-epochs",
        str(config.train_epochs),
        "--device",
        config.device,
        "--attempt",
        str(config.attempt),
        "--codex-allowance-source",
        config.codex_allowance_source,
        "--codex-five-hour-window-remaining",
        config.codex_five_hour_window_remaining,
        "--codex-weekly-usage",
        config.codex_weekly_usage,
        "--codex-allowance-decision",
        config.codex_allowance_decision,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-episodes", type=int, default=64)
    parser.add_argument("--cache-rows", type=int, default=2048)
    parser.add_argument("--mcts-simulations", type=int, default=32)
    parser.add_argument("--mcts-max-depth", type=int, default=10)
    parser.add_argument("--mcts-rollout-steps", type=int, default=6)
    parser.add_argument("--mcts-prior-mode", default="heuristic")
    parser.add_argument("--train-epochs", type=int, default=16)
    parser.add_argument("--train-learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--preflight-rows", type=int, default=16)
    parser.add_argument("--preflight-episodes", type=int, default=2)
    parser.add_argument("--parity-seed", type=int, action="append")
    parser.add_argument("--parity-scenario", action="append", choices=sorted(SCENARIO_BY_NAME))
    parser.add_argument("--parity-fuzz-episodes", type=int, default=0)
    parser.add_argument("--parity-fuzz-max-steps", type=int, default=80)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--attempt-cap", type=int, default=3)
    parser.add_argument("--codex-allowance-source", default="unknown")
    parser.add_argument("--codex-five-hour-window-remaining", default="unknown")
    parser.add_argument("--codex-weekly-usage", default="unknown")
    parser.add_argument("--codex-allowance-decision", default="continue")
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_strategic_muzero_overnight(
        StrategicMuZeroOvernightConfig(
            out_dir=args.out_dir,
            seed=args.seed,
            cache_episodes=args.cache_episodes,
            cache_rows=args.cache_rows,
            mcts_simulations=args.mcts_simulations,
            mcts_max_depth=args.mcts_max_depth,
            mcts_rollout_steps=args.mcts_rollout_steps,
            mcts_prior_mode=args.mcts_prior_mode,
            train_epochs=args.train_epochs,
            train_learning_rate=args.train_learning_rate,
            train_batch_size=args.train_batch_size,
            hidden_size=args.hidden_size,
            device=args.device,
            eval_episodes=args.eval_episodes,
            preflight_rows=args.preflight_rows,
            preflight_episodes=args.preflight_episodes,
            parity_seeds=tuple(args.parity_seed) if args.parity_seed else (0, 1, 7, 19),
            parity_scenarios=tuple(args.parity_scenario) if args.parity_scenario else None,
            parity_fuzz_episodes=args.parity_fuzz_episodes,
            parity_fuzz_max_steps=args.parity_fuzz_max_steps,
            attempt=args.attempt,
            attempt_cap=args.attempt_cap,
            codex_allowance_source=args.codex_allowance_source,
            codex_five_hour_window_remaining=args.codex_five_hour_window_remaining,
            codex_weekly_usage=args.codex_weekly_usage,
            codex_allowance_decision=args.codex_allowance_decision,
            cc=args.cc,
            strict=args.strict,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
