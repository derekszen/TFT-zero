"""Stage 4 MuZero-lite dynamics/search bootstrap runner."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import LobbyPolicy, PlayerOrderMode
from mini_tft.search.distill import load_student_policy, write_json
from mini_tft.search.dynamics import (
    ModelBackedRootPlanner,
    ModelSearchConfig,
    generate_transition_dataset,
    load_transition_dataset,
    parse_stage4_hidden_sizes,
    train_dynamics_model,
    validate_transition_dataset_schema,
)
from mini_tft.search.mcts import MCTSConfig
from mini_tft.tools.lobby_diagnostics import (
    _named_policy,
    _opponent_policy,
    compare_trace_files,
    run_lobby_trace_export,
)
from mini_tft.tools.stage2_mcts import DEFAULT_FROZEN_POOL, DEFAULT_PROMOTED_CHECKPOINT

DEFAULT_STAGE2_METRICS = Path("artifacts/muzero_stage/stage2/metrics.json")
DEFAULT_STAGE3_DECISION = Path("artifacts/muzero_stage/stage3/decision.md")
DEFAULT_STAGE3_METRICS = Path("artifacts/muzero_stage/stage3/metrics.json")
DEFAULT_STAGE3_PROMOTED = Path("artifacts/muzero_stage/stage3/promoted/student_stage3.pt")


def run_stage4_loop(
    *,
    out_dir: Path = Path("artifacts/muzero_stage/stage4"),
    stage2_metrics: Path = DEFAULT_STAGE2_METRICS,
    stage3_decision: Path = DEFAULT_STAGE3_DECISION,
    stage3_metrics: Path = DEFAULT_STAGE3_METRICS,
    stage3_checkpoint: Path = DEFAULT_STAGE3_PROMOTED,
    ppo_checkpoint: Path = DEFAULT_PROMOTED_CHECKPOINT,
    frozen_pool: Sequence[Path] = DEFAULT_FROZEN_POOL,
    attempts: int = 12,
    max_wall_clock_sec: float = 8.0 * 60.0 * 60.0,
    teacher_episodes: int = 1,
    eval_episodes: int = 3,
    seed: int = 51000,
    eval_seed: int = 41000,
    player_count: int = 8,
    max_actions_per_player: int | None = 8,
    teacher_simulations: int = 4,
    teacher_rollout_rounds: int = 1,
    teacher_max_root_actions: int = 4,
    teacher_prior_improvement_margin: float = 0.25,
    model_simulations: int = 8,
    model_max_root_actions: int = 8,
    model_prior_improvement_margin: float = 1.0,
    simulator_value_guard: bool = False,
    simulator_value_threshold: float = 0.05,
    simulator_value_max_actions: int = 12,
    train_epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    hidden_sizes: Sequence[int] = (128, 128),
    latent_dim: int = 64,
    dynamics_coef: float = 1.0,
    reward_coef: float = 1.0,
    value_coef: float = 0.5,
    policy_coef: float = 1.0,
    device: str = "cpu",
    player_order_mode: PlayerOrderMode = "fixed",
) -> dict[str, Any]:
    """Run the capped Stage 4 data/train/eval loop."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if max_wall_clock_sec <= 0.0:
        raise ValueError("max_wall_clock_sec must be positive")
    stage3 = _require_stage3_pass(stage3_decision, stage3_metrics, stage3_checkpoint)
    stage2 = _load_stage2_metrics(stage2_metrics)
    dirs = _stage4_dirs(out_dir)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    resource_check = _resource_check(out_dir, device=device)
    if resource_check["decision"] == "hard-pause":
        raise RuntimeError(f"unsafe resources for Stage 4 run: {resource_check}")

    loop_log = out_dir / "loop-run-log.md"
    loop_state_path = out_dir / "loop-state.json"
    config = EnvConfig(seed=seed)
    teacher_config = MCTSConfig(
        simulations=teacher_simulations,
        rollout_rounds=teacher_rollout_rounds,
        max_root_actions=teacher_max_root_actions,
        max_actions_per_player=max_actions_per_player,
        prior_improvement_margin=teacher_prior_improvement_margin,
    )
    model_search_config = ModelSearchConfig(
        simulations=model_simulations,
        max_root_actions=model_max_root_actions,
        prior_improvement_margin=model_prior_improvement_margin,
        simulator_value_guard=simulator_value_guard,
        simulator_value_threshold=simulator_value_threshold,
        simulator_value_max_actions=simulator_value_max_actions,
    )
    mixed_opponent = _named_policy("mixed")
    stage3_policy = load_student_policy(stage3_checkpoint, device=device)
    frozen_opponent = _opponent_policy(
        opponent_policy_name="mixed",
        opponent_checkpoints=frozen_pool,
        device=device,
    )

    dataset_paths: list[Path] = []
    attempts_report: list[dict[str, Any]] = []
    best_checkpoint: Path | None = None
    best_score = float("inf")
    best_prediction_loss = float("inf")
    best_pass_report: dict[str, Any] | None = None
    final_classification = "BLOCKED_DYNAMICS"
    started = perf_counter()
    _append_loop_log(
        loop_log,
        "# Stage 4 Dynamics-Search Bootstrap Loop\n\n"
        "## Codex Allowance Check\n"
        "Source: unknown\n"
        f"Checked at: {_now_iso()}\n"
        "5h window remaining: unknown\n"
        "Weekly usage: unknown\n"
        "Decision: continue\n\n"
        f"Resource check: `{resource_check['decision']}`\n",
    )

    for attempt in range(1, attempts + 1):
        if perf_counter() - started > max_wall_clock_sec:
            break

        attempt_seed = seed + attempt * 1000
        mixed_dataset = dirs["datasets"] / f"iter_{attempt:02d}_mixed.npz"
        frozen_dataset = dirs["datasets"] / f"iter_{attempt:02d}_frozen.npz"
        mixed_dataset_metrics = generate_transition_dataset(
            out_path=mixed_dataset,
            suite_name="mixed",
            episodes=teacher_episodes,
            seed=attempt_seed,
            player_count=player_count,
            max_actions_per_player=max_actions_per_player,
            env_config=config,
            mcts_config=teacher_config,
            opponent_policy=mixed_opponent,
            prior_policy=stage3_policy,
            rollout_policy=stage3_policy,
            player_order_mode=player_order_mode,
        )
        frozen_dataset_metrics = generate_transition_dataset(
            out_path=frozen_dataset,
            suite_name="frozen",
            episodes=teacher_episodes,
            seed=attempt_seed + 500,
            player_count=player_count,
            max_actions_per_player=max_actions_per_player,
            env_config=config,
            mcts_config=teacher_config,
            opponent_policy=frozen_opponent,
            prior_policy=stage3_policy,
            rollout_policy=stage3_policy,
            player_order_mode=player_order_mode,
        )
        for dataset_path in (mixed_dataset, frozen_dataset):
            validate_transition_dataset_schema(
                load_transition_dataset(dataset_path),
                env_config=config,
            )
            dataset_paths.append(dataset_path)

        checkpoint = dirs["checkpoints"] / f"dynamics_iter_{attempt:02d}.pt"
        train_report = train_dynamics_model(
            dataset_paths=dataset_paths,
            out_path=checkpoint,
            epochs=train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_sizes=hidden_sizes,
            latent_dim=latent_dim,
            dynamics_coef=dynamics_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            policy_coef=policy_coef,
            seed=seed + attempt,
            device=device,
            resume=best_checkpoint,
        )
        eval_report = run_model_search_eval(
            checkpoint=checkpoint,
            out_dir=dirs["evals"] / f"iter_{attempt:02d}",
            stage3_checkpoint=stage3_checkpoint,
            ppo_checkpoint=ppo_checkpoint,
            frozen_pool=frozen_pool,
            stage2_metrics=stage2,
            episodes=eval_episodes,
            seed=eval_seed,
            player_count=player_count,
            max_actions_per_player=max_actions_per_player,
            config=EnvConfig(seed=eval_seed),
            search_config=model_search_config,
            device=device,
            player_order_mode=player_order_mode,
        )
        classification = classify_stage4(train_report=train_report, eval_report=eval_report)
        score = _model_score(eval_report)
        prediction_loss = float(train_report.get("final_next_obs_loss") or float("inf"))
        if score < best_score or (
            score == best_score and prediction_loss < best_prediction_loss
        ):
            best_score = score
            best_prediction_loss = prediction_loss
            best_checkpoint = checkpoint
        if classification in {"PASS_STAGE_4", "PASS_STAGE_4_STRONG"}:
            best_pass_report = {
                "classification": classification,
                "checkpoint": str(checkpoint),
                "train": train_report,
                "eval": eval_report,
                "score": score,
            }
            promoted_path = dirs["promoted"] / "dynamics_stage4.pt"
            shutil.copy2(checkpoint, promoted_path)
            shutil.copy2(
                checkpoint.with_suffix(".metrics.json"),
                promoted_path.with_suffix(".metrics.json"),
            )
            final_classification = classification

        next_action = _next_action(classification, attempt=attempt, attempts=attempts)
        attempt_report = {
            "attempt": attempt,
            "classification": classification,
            "checkpoint": str(checkpoint),
            "mixed_dataset": mixed_dataset_metrics,
            "frozen_dataset": frozen_dataset_metrics,
            "train": train_report,
            "eval": eval_report,
            "score": score,
            "next_action": next_action,
        }
        attempts_report.append(attempt_report)
        _append_loop_log(loop_log, format_loop_attempt(attempt_report))
        _write_loop_state(
            loop_state_path,
            {
                "attempt_cap": attempts,
                "attempt": attempt,
                "classification": classification,
                "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
                "best_score": best_score,
                "best_prediction_loss": best_prediction_loss,
                "current_checkpoint": str(checkpoint),
                "next_action": next_action,
            },
        )
        if classification == "PASS_STAGE_4_STRONG":
            break

    final_attempt = attempts_report[-1] if attempts_report else None
    if best_pass_report is None and final_attempt is not None:
        final_classification = str(final_attempt["classification"])
        final_eval = final_attempt["eval"]
        final_train = final_attempt["train"]
    elif best_pass_report is not None:
        final_eval = best_pass_report["eval"]
        final_train = best_pass_report["train"]
    else:
        final_eval = None
        final_train = None

    report: dict[str, Any] = {
        "status": (
            "pass"
            if final_classification in {"PASS_STAGE_4", "PASS_STAGE_4_STRONG"}
            else "fail"
        ),
        "kind": "stage4_dynamics_search_loop",
        "classification": final_classification,
        "stage2_metrics": str(stage2_metrics),
        "stage3_decision": str(stage3_decision),
        "stage3_metrics": str(stage3_metrics),
        "stage3_checkpoint": str(stage3_checkpoint),
        "ppo_checkpoint": str(ppo_checkpoint),
        "frozen_pool": [str(path) for path in frozen_pool],
        "resource_check": resource_check,
        "config": {
            "attempt_cap": attempts,
            "max_wall_clock_sec": max_wall_clock_sec,
            "teacher_episodes": teacher_episodes,
            "eval_episodes": eval_episodes,
            "seed": seed,
            "eval_seed": eval_seed,
            "player_count": player_count,
            "max_actions_per_player": max_actions_per_player,
            "teacher_simulations": teacher_simulations,
            "teacher_rollout_rounds": teacher_rollout_rounds,
            "teacher_max_root_actions": teacher_max_root_actions,
            "teacher_prior_improvement_margin": teacher_prior_improvement_margin,
            "model_simulations": model_simulations,
            "model_max_root_actions": model_max_root_actions,
            "model_prior_improvement_margin": model_prior_improvement_margin,
            "simulator_value_guard": simulator_value_guard,
            "simulator_value_threshold": simulator_value_threshold,
            "simulator_value_max_actions": simulator_value_max_actions,
            "train_epochs": train_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_sizes": [int(size) for size in hidden_sizes],
            "latent_dim": latent_dim,
            "dynamics_coef": dynamics_coef,
            "reward_coef": reward_coef,
            "value_coef": value_coef,
            "policy_coef": policy_coef,
            "device": device,
            "player_order_mode": player_order_mode,
        },
        "stage3_entry": stage3,
        "artifacts": {
            "datasets": str(dirs["datasets"]),
            "checkpoints": str(dirs["checkpoints"]),
            "evals": str(dirs["evals"]),
            "loops": str(dirs["loops"]),
            "promoted": str(dirs["promoted"]),
            "decision": str(out_dir / "decision.md"),
            "metrics": str(out_dir / "metrics.json"),
            "loop_run_log": str(loop_log),
            "loop_state": str(loop_state_path),
        },
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "best_pass": best_pass_report,
        "attempts": attempts_report,
        "final_eval": final_eval,
        "final_train": final_train,
        "elapsed_sec": perf_counter() - started,
    }
    verifier = verify_stage4_report(report)
    report["verifier"] = verifier
    if report["status"] == "pass" and verifier["status"] != "pass":
        report["status"] = "fail"
        report["classification"] = verifier["classification"]
    write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(format_stage4_decision(report), encoding="utf-8")
    return report


def run_model_search_eval(
    *,
    checkpoint: Path,
    out_dir: Path,
    stage3_checkpoint: Path,
    ppo_checkpoint: Path,
    frozen_pool: Sequence[Path],
    stage2_metrics: dict[str, Any],
    episodes: int,
    seed: int,
    player_count: int,
    max_actions_per_player: int | None,
    config: EnvConfig,
    search_config: ModelSearchConfig,
    device: str,
    player_order_mode: PlayerOrderMode,
) -> dict[str, Any]:
    """Evaluate model-backed search against Stage 3, PPO, and frozen pools."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stage3_policy = load_student_policy(stage3_checkpoint, device=device)
    mixed_opponent = _named_policy("mixed")
    frozen_opponent = _opponent_policy(
        opponent_policy_name="mixed",
        opponent_checkpoints=frozen_pool,
        device=device,
    )
    model_mixed = _timed_model_trace_export(
        checkpoint=checkpoint,
        out_dir=out_dir / "model_mixed",
        trace_name="model_mixed",
        episodes=episodes,
        seed=seed,
        opponent_policy=mixed_opponent,
        opponent_policy_name="mixed",
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        search_config=search_config,
        stage3_policy=stage3_policy,
        device=device,
        player_order_mode=player_order_mode,
    )
    model_frozen = _timed_model_trace_export(
        checkpoint=checkpoint,
        out_dir=out_dir / "model_frozen",
        trace_name="model_frozen",
        episodes=episodes,
        seed=seed,
        opponent_policy=frozen_opponent,
        opponent_policy_name="frozen_checkpoint_pool",
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        search_config=search_config,
        stage3_policy=stage3_policy,
        device=device,
        player_order_mode=player_order_mode,
    )
    stage3_mixed = _timed_policy_trace_export(
        out_dir=out_dir / "stage3_mixed",
        trace_name="stage3_mixed",
        episodes=episodes,
        seed=seed,
        hero_policy=stage3_policy,
        hero_policy_name="stage3_student",
        opponent_policy=mixed_opponent,
        opponent_policy_name="mixed",
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        device=device,
        player_order_mode=player_order_mode,
    )
    stage3_frozen = _timed_policy_trace_export(
        out_dir=out_dir / "stage3_frozen",
        trace_name="stage3_frozen",
        episodes=episodes,
        seed=seed,
        hero_policy=stage3_policy,
        hero_policy_name="stage3_student",
        opponent_policy=frozen_opponent,
        opponent_policy_name="frozen_checkpoint_pool",
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        device=device,
        player_order_mode=player_order_mode,
    )
    ppo_mixed = run_lobby_trace_export(
        episodes=episodes,
        seed=seed,
        checkpoint=ppo_checkpoint,
        opponent_policy_name="mixed",
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        device=device,
        config=config,
        out_dir=out_dir / "ppo_mixed",
        trace_name="ppo_mixed",
        player_order_mode=player_order_mode,
    )
    ppo_frozen = run_lobby_trace_export(
        episodes=episodes,
        seed=seed,
        checkpoint=ppo_checkpoint,
        opponent_checkpoints=frozen_pool,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        device=device,
        config=config,
        out_dir=out_dir / "ppo_frozen",
        trace_name="ppo_frozen",
        player_order_mode=player_order_mode,
    )
    stage3_mixed_compare = compare_trace_files(
        baseline_trace=Path(stage3_mixed["artifacts"]["trace"]),
        candidate_trace=Path(model_mixed["artifacts"]["trace"]),
        out_dir=out_dir / "compare_model_vs_stage3_mixed",
    )
    stage3_frozen_compare = compare_trace_files(
        baseline_trace=Path(stage3_frozen["artifacts"]["trace"]),
        candidate_trace=Path(model_frozen["artifacts"]["trace"]),
        out_dir=out_dir / "compare_model_vs_stage3_frozen",
    )
    ppo_mixed_compare = compare_trace_files(
        baseline_trace=Path(ppo_mixed["artifacts"]["trace"]),
        candidate_trace=Path(model_mixed["artifacts"]["trace"]),
        out_dir=out_dir / "compare_model_vs_ppo_mixed",
    )
    ppo_frozen_compare = compare_trace_files(
        baseline_trace=Path(ppo_frozen["artifacts"]["trace"]),
        candidate_trace=Path(model_frozen["artifacts"]["trace"]),
        out_dir=out_dir / "compare_model_vs_ppo_frozen",
    )
    report: dict[str, Any] = {
        "status": "pass",
        "kind": "stage4_model_search_eval",
        "checkpoint": str(checkpoint),
        "model_mixed": _summary(model_mixed),
        "model_frozen": _summary(model_frozen),
        "stage3_mixed": _summary(stage3_mixed),
        "stage3_frozen": _summary(stage3_frozen),
        "ppo_mixed": _summary(ppo_mixed),
        "ppo_frozen": _summary(ppo_frozen),
        "model_vs_stage3_mixed": _compare_summary(stage3_mixed_compare),
        "model_vs_stage3_frozen": _compare_summary(stage3_frozen_compare),
        "model_vs_ppo_mixed": _compare_summary(ppo_mixed_compare),
        "model_vs_ppo_frozen": _compare_summary(ppo_frozen_compare),
        "simulator_mcts_reference": {
            "mixed": stage2_metrics["mcts_mixed_eval"],
            "frozen": stage2_metrics["mcts_frozen_eval"],
        },
        "artifacts": {
            "metrics": str(out_dir / "metrics.json"),
            "model_mixed": str(out_dir / "model_mixed"),
            "model_frozen": str(out_dir / "model_frozen"),
            "stage3_mixed": str(out_dir / "stage3_mixed"),
            "stage3_frozen": str(out_dir / "stage3_frozen"),
            "ppo_mixed": str(out_dir / "ppo_mixed"),
            "ppo_frozen": str(out_dir / "ppo_frozen"),
            "compare_model_vs_stage3_mixed": str(out_dir / "compare_model_vs_stage3_mixed"),
            "compare_model_vs_stage3_frozen": str(out_dir / "compare_model_vs_stage3_frozen"),
            "compare_model_vs_ppo_mixed": str(out_dir / "compare_model_vs_ppo_mixed"),
            "compare_model_vs_ppo_frozen": str(out_dir / "compare_model_vs_ppo_frozen"),
        },
    }
    write_json(out_dir / "metrics.json", report)
    return report


def classify_stage4(*, train_report: dict[str, Any], eval_report: dict[str, Any]) -> str:
    """Classify one Stage 4 attempt under the objective gates."""

    if not train_report.get("beats_persistence_next_obs", False):
        return "BLOCKED_DYNAMICS"
    if not _loss_improved(train_report, "reward") or not _loss_improved(train_report, "value"):
        return "BLOCKED_DYNAMICS"
    if (
        eval_report["model_mixed"]["illegal_actions"] != 0
        or eval_report["model_frozen"]["illegal_actions"] != 0
    ):
        return "BLOCKED_SEARCH"
    mixed_delta = eval_report["model_vs_stage3_mixed"]["scalar"]
    frozen_delta = eval_report["model_vs_stage3_frozen"]["scalar"]
    if (
        mixed_delta["mean_placement"] is None
        or frozen_delta["mean_placement"] is None
        or mixed_delta["mean_final_hp"] is None
        or frozen_delta["mean_final_hp"] is None
    ):
        return "BLOCKED_POLICY"
    if (
        mixed_delta["mean_placement"] > 0.10
        or frozen_delta["mean_placement"] > 0.25
        or mixed_delta["mean_final_hp"] < -5.0
        or frozen_delta["mean_final_hp"] < -5.0
    ):
        return "BLOCKED_POLICY"
    if not _speed_gate(eval_report):
        return "BLOCKED_SPEED"
    frozen_uplift = frozen_delta["mean_placement"] <= -0.10
    hp_uplift = (
        mixed_delta["mean_placement"] <= 0.0
        and frozen_delta["mean_placement"] <= 0.0
        and (mixed_delta["mean_final_hp"] >= 5.0 or frozen_delta["mean_final_hp"] >= 5.0)
    )
    if frozen_uplift or hp_uplift:
        return "PASS_STAGE_4_STRONG"
    return "PASS_STAGE_4"


def verify_stage4_report(report: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed verifier for final Stage 4 artifacts and gates."""

    errors: list[str] = []
    artifacts = report.get("artifacts", {})
    for key in ("datasets", "checkpoints", "evals", "loops", "promoted"):
        if not Path(str(artifacts.get(key, ""))).is_dir():
            errors.append(f"missing artifact directory: {key}")
    for key in ("decision", "metrics", "loop_run_log", "loop_state"):
        path = Path(str(artifacts.get(key, "")))
        if key in {"decision", "metrics"}:
            continue
        if not path.is_file():
            errors.append(f"missing artifact file: {key}")
    dataset_paths = sorted(Path(str(artifacts.get("datasets", ""))).glob("*.npz"))
    if not dataset_paths:
        errors.append("no transition datasets")
    else:
        for dataset_path in dataset_paths:
            try:
                validate_transition_dataset_schema(load_transition_dataset(dataset_path))
            except Exception as exc:  # pragma: no cover - verifier should preserve reason
                errors.append(f"{dataset_path}: {exc}")
    if report.get("classification") in {"PASS_STAGE_4", "PASS_STAGE_4_STRONG"}:
        final_train = report.get("final_train") or {}
        final_eval = report.get("final_eval") or {}
        if not final_train.get("beats_persistence_next_obs", False):
            errors.append("final model does not beat persistence next-state baseline")
        if classify_stage4(train_report=final_train, eval_report=final_eval) not in {
            "PASS_STAGE_4",
            "PASS_STAGE_4_STRONG",
        }:
            errors.append("final eval no longer satisfies pass gates")
        promoted = Path(str(artifacts.get("promoted", ""))) / "dynamics_stage4.pt"
        if not promoted.is_file():
            errors.append(f"missing promoted checkpoint: {promoted}")
    return {
        "status": "pass" if not errors else "reject",
        "classification": "BLOCKED_DYNAMICS" if errors else report.get("classification"),
        "errors": errors,
    }


def format_loop_attempt(report: dict[str, Any]) -> str:
    eval_report = report["eval"]
    train_report = report["train"]
    mixed_delta = eval_report["model_vs_stage3_mixed"]["scalar"]
    frozen_delta = eval_report["model_vs_stage3_frozen"]["scalar"]
    return "\n".join(
        [
            f"## Attempt {report['attempt']}",
            "",
            f"- timestamp: `{_now_iso()}`",
            "- action: `generate_transition_data -> train_dynamics -> evaluate_model_search`",
            f"- classification: `{report['classification']}`",
            f"- checkpoint: `{report['checkpoint']}`",
            f"- rows mixed/frozen: {report['mixed_dataset']['rows']} / "
            f"{report['frozen_dataset']['rows']}",
            f"- final loss: {train_report['final_loss']}",
            f"- next-obs loss vs persistence: {train_report['final_next_obs_loss']} / "
            f"{train_report['persistence_next_obs_mse']}",
            f"- model vs Stage 3 mixed placement delta: "
            f"{_fmt(mixed_delta.get('mean_placement'))}",
            f"- model vs Stage 3 frozen placement delta: "
            f"{_fmt(frozen_delta.get('mean_placement'))}",
            f"- model vs Stage 3 mixed HP delta: {_fmt(mixed_delta.get('mean_final_hp'))}",
            f"- model vs Stage 3 frozen HP delta: {_fmt(frozen_delta.get('mean_final_hp'))}",
            f"- model min decisions/sec: {_model_min_dps(eval_report):.3f}",
            f"- simulator MCTS max decisions/sec: {_teacher_max_dps(eval_report):.3f}",
            f"- validation: `{report['classification']}`",
            "- verifier verdict: `pending_final_verifier`",
            f"- next action: `{report['next_action']}`",
            "",
        ]
    )


def format_stage4_decision(report: dict[str, Any]) -> str:
    final_eval = report.get("final_eval") or {}
    final_train = report.get("final_train") or {}
    mixed_delta = final_eval.get("model_vs_stage3_mixed", {}).get("scalar", {})
    frozen_delta = final_eval.get("model_vs_stage3_frozen", {}).get("scalar", {})
    verifier = report.get("verifier", {})
    return "\n".join(
        [
            "# Stage 4 Dynamics-Search Gate Decision",
            "",
            f"**Decision:** `{report['classification']}`",
            "",
            "This stage trains a MuZero-lite representation/dynamics/prediction model "
            "from simulator transitions and evaluates model-backed root search. It is "
            "not yet full MuZero self-play.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Attempts run | {len(report['attempts'])} |",
            f"| Best checkpoint | `{report['best_checkpoint']}` |",
            f"| Next-obs loss | {_fmt(final_train.get('final_next_obs_loss'))} |",
            f"| Persistence next-obs loss | {_fmt(final_train.get('persistence_next_obs_mse'))} |",
            f"| Reward loss improved | {_loss_improved(final_train, 'reward')} |",
            f"| Value loss improved | {_loss_improved(final_train, 'value')} |",
            f"| Mixed placement delta vs Stage 3 | "
            f"{_fmt(mixed_delta.get('mean_placement'))} |",
            f"| Frozen placement delta vs Stage 3 | "
            f"{_fmt(frozen_delta.get('mean_placement'))} |",
            f"| Mixed HP delta vs Stage 3 | {_fmt(mixed_delta.get('mean_final_hp'))} |",
            f"| Frozen HP delta vs Stage 3 | {_fmt(frozen_delta.get('mean_final_hp'))} |",
            f"| Model min decisions/sec | {_fmt(_model_min_dps(final_eval))} |",
            f"| Simulator MCTS max decisions/sec | {_fmt(_teacher_max_dps(final_eval))} |",
            f"| Verifier | `{verifier.get('status', 'missing')}` |",
            "",
            "## Required Artifacts",
            "",
            f"- `{report['artifacts']['datasets']}`",
            f"- `{report['artifacts']['checkpoints']}`",
            f"- `{report['artifacts']['evals']}`",
            f"- `{report['artifacts']['loops']}`",
            f"- `{report['artifacts']['promoted']}`",
            f"- `{report['artifacts']['loop_run_log']}`",
            f"- `{report['artifacts']['loop_state']}`",
            "",
        ]
    )


def _timed_model_trace_export(
    *,
    checkpoint: Path,
    out_dir: Path,
    trace_name: str,
    episodes: int,
    seed: int,
    opponent_policy: LobbyPolicy,
    opponent_policy_name: str,
    player_count: int,
    max_actions_per_player: int | None,
    config: EnvConfig,
    search_config: ModelSearchConfig,
    stage3_policy: LobbyPolicy,
    device: str,
    player_order_mode: PlayerOrderMode,
) -> dict[str, Any]:
    planner = ModelBackedRootPlanner(
        checkpoint_path=checkpoint,
        config=search_config,
        device=device,
        prior_policy=stage3_policy,
    )
    hero_policy = _planner_policy_with_episode_context(planner, seed)
    started = perf_counter()
    report = run_lobby_trace_export(
        episodes=episodes,
        seed=seed,
        hero_policy_name="stage4_model_search",
        opponent_policy_name=opponent_policy_name,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        out_dir=out_dir,
        trace_name=trace_name,
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
        player_order_mode=player_order_mode,
    )
    elapsed_sec = perf_counter() - started
    report["model_search"] = {
        "simulations": search_config.simulations,
        "max_root_actions": search_config.max_root_actions,
        "search_decisions": len(planner.traces),
        "selected_from_prior_rate": _rate(
            sum(1 for row in planner.traces if bool(row.get("selected_from_prior"))),
            len(planner.traces),
        ),
    }
    report["speed"] = {
        "elapsed_sec": elapsed_sec,
        "decisions_per_sec": float(report["decisions"]) / elapsed_sec if elapsed_sec > 0 else 0.0,
        "model_search_decisions_per_sec": (
            len(planner.traces) / elapsed_sec if elapsed_sec > 0 else 0.0
        ),
    }
    write_json(out_dir / "metrics.json", report)
    return report


def _timed_policy_trace_export(
    *,
    out_dir: Path,
    trace_name: str,
    episodes: int,
    seed: int,
    hero_policy: LobbyPolicy,
    hero_policy_name: str,
    opponent_policy: LobbyPolicy,
    opponent_policy_name: str,
    player_count: int,
    max_actions_per_player: int | None,
    config: EnvConfig,
    device: str,
    player_order_mode: PlayerOrderMode,
) -> dict[str, Any]:
    started = perf_counter()
    report = run_lobby_trace_export(
        episodes=episodes,
        seed=seed,
        hero_policy_name=hero_policy_name,
        opponent_policy_name=opponent_policy_name,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        out_dir=out_dir,
        trace_name=trace_name,
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
        device=device,
        player_order_mode=player_order_mode,
    )
    elapsed_sec = perf_counter() - started
    report["speed"] = {
        "elapsed_sec": elapsed_sec,
        "decisions_per_sec": float(report["decisions"]) / elapsed_sec if elapsed_sec > 0 else 0.0,
    }
    write_json(out_dir / "metrics.json", report)
    return report


def _planner_policy_with_episode_context(
    planner: ModelBackedRootPlanner,
    base_seed: int,
) -> LobbyPolicy:
    current_seed: int | None = None

    def policy(
        player_id: int,
        state: Any,
        mask: Any,
        data: Any,
        env_config: Any,
        rng: Any,
    ) -> int:
        nonlocal current_seed
        if current_seed != state.seed:
            current_seed = int(state.seed)
            planner.set_episode_context(
                episode=max(0, current_seed - base_seed),
                seed=current_seed,
            )
        if player_id == 0:
            return int(planner.policy(player_id, state, mask, data, env_config, rng))
        raise ValueError("model search planner received a non-hero player")

    return policy


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    speed = report.get("speed", {})
    model_search = report.get("model_search", {})
    return {
        "status": report["status"],
        "episodes": report["episodes"],
        "decisions": report["decisions"],
        "illegal_actions": report["total_illegal_actions"],
        "mean_placement": summary["mean_placement"],
        "mean_final_hp": summary["mean_final_hp"],
        "mean_final_board_strength": summary["mean_final_board_strength"],
        "roll_rate": summary["chosen_roll_rate"],
        "buy_xp_rate": summary["chosen_buy_xp_rate"],
        "buy_shop_rate": summary["chosen_buy_shop_rate"],
        "end_turn_rate": summary["chosen_end_turn_rate"],
        "decisions_per_sec": speed.get("decisions_per_sec"),
        "model_search_decisions_per_sec": speed.get("model_search_decisions_per_sec"),
        "elapsed_sec": speed.get("elapsed_sec"),
        "search_decisions": model_search.get("search_decisions"),
        "selected_from_prior_rate": model_search.get("selected_from_prior_rate"),
    }


def _compare_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "scalar": report["deltas"]["scalar"],
        "category_rates": report["deltas"]["category_rates"],
        "bucket_rates": report["deltas"]["bucket_rates"],
    }


def _loss_improved(report: dict[str, Any], name: str) -> bool:
    initial = report.get(f"initial_{name}_loss")
    final = report.get(f"final_{name}_loss")
    return initial is not None and final is not None and float(final) < float(initial)


def _speed_gate(eval_report: dict[str, Any]) -> bool:
    return _model_min_dps(eval_report) >= 2.0 * _teacher_max_dps(eval_report)


def _model_min_dps(eval_report: dict[str, Any]) -> float:
    values = [
        eval_report.get("model_mixed", {}).get("model_search_decisions_per_sec"),
        eval_report.get("model_frozen", {}).get("model_search_decisions_per_sec"),
    ]
    numeric = [float(value) for value in values if value is not None]
    return min(numeric) if numeric else 0.0


def _teacher_max_dps(eval_report: dict[str, Any]) -> float:
    reference = eval_report.get("simulator_mcts_reference", {})
    values = [
        reference.get("mixed", {}).get("mcts_decisions_per_sec"),
        reference.get("frozen", {}).get("mcts_decisions_per_sec"),
    ]
    numeric = [float(value) for value in values if value is not None]
    return max(numeric) if numeric else float("inf")


def _model_score(eval_report: dict[str, Any]) -> float:
    mixed = eval_report["model_vs_stage3_mixed"]["scalar"]
    frozen = eval_report["model_vs_stage3_frozen"]["scalar"]
    return float(
        mixed["mean_placement"]
        + frozen["mean_placement"]
        - 0.02 * mixed["mean_final_hp"]
        - 0.02 * frozen["mean_final_hp"]
    )


def _stage4_dirs(out_dir: Path) -> dict[str, Path]:
    return {
        "datasets": out_dir / "datasets",
        "checkpoints": out_dir / "checkpoints",
        "evals": out_dir / "evals",
        "loops": out_dir / "loops",
        "promoted": out_dir / "promoted",
    }


def _resource_check(out_dir: Path, *, device: str) -> dict[str, Any]:
    usage = shutil.disk_usage(out_dir.parent if out_dir.parent.exists() else Path("."))
    free_gb = usage.free / (1024**3)
    gpu = _gpu_resource_check()
    decision = "continue" if free_gb >= 5.0 else "soft-pause"
    if device.startswith("cuda") and gpu.get("status") == "pass":
        free_mb = float(gpu.get("free_mb", 0.0))
        if free_mb < 2048.0:
            decision = "hard-pause"
    return {
        "disk_free_gb": free_gb,
        "gpu": gpu,
        "decision": decision,
    }


def _gpu_resource_check() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"status": "unknown"}
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return {"status": "unknown"}
    first = lines[0].split(",")
    if len(first) < 2:
        return {"status": "unknown", "raw": output}
    return {
        "status": "pass",
        "free_mb": float(first[0].strip()),
        "total_mb": float(first[1].strip()),
    }


def _require_stage3_pass(
    stage3_decision: Path,
    stage3_metrics: Path,
    stage3_checkpoint: Path,
) -> dict[str, Any]:
    if not stage3_decision.is_file():
        raise FileNotFoundError(f"missing Stage 3 decision: {stage3_decision}")
    if not stage3_metrics.is_file():
        raise FileNotFoundError(f"missing Stage 3 metrics: {stage3_metrics}")
    if not stage3_checkpoint.is_file():
        raise FileNotFoundError(f"missing Stage 3 promoted checkpoint: {stage3_checkpoint}")
    metrics = json.loads(stage3_metrics.read_text(encoding="utf-8"))
    if metrics.get("classification") != "PASS_STAGE_3":
        raise RuntimeError(f"Stage 3 did not pass: {metrics.get('classification')}")
    return {
        "classification": metrics.get("classification"),
        "checkpoint": str(stage3_checkpoint),
        "best_checkpoint": metrics.get("best_checkpoint"),
    }


def _load_stage2_metrics(stage2_metrics: Path) -> dict[str, Any]:
    if not stage2_metrics.is_file():
        raise FileNotFoundError(f"missing Stage 2 metrics: {stage2_metrics}")
    metrics = json.loads(stage2_metrics.read_text(encoding="utf-8"))
    if metrics.get("classification") != "PASS_TO_STAGE_3":
        raise RuntimeError(f"Stage 2 did not pass: {metrics.get('classification')}")
    return metrics


def _append_loop_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _write_loop_state(path: Path, payload: dict[str, Any]) -> None:
    payload = {
        "owner": "stage4_dynamics_search_loop",
        "schema": "stage4_loop_state/v1",
        "prune_rule": (
            "keep all promoted checkpoints and per-attempt state; prune smoke dirs manually"
        ),
        "pause_criteria": [
            "resource_check decision is hard-pause",
            "weekly Codex usage reaches hard pause threshold",
            "three consecutive verifier rejects on the same blocker",
        ],
        "kill_criteria": [
            "illegal actions become nonzero after a code change",
            "dataset schema validation fails",
            "CUDA or disk resource check reports unsafe state",
        ],
        **payload,
    }
    write_json(path, payload)
    loops_dir = path.parent / "loops"
    loops_dir.mkdir(parents=True, exist_ok=True)
    write_json(loops_dir / f"attempt_{payload['attempt']:02d}_state.json", payload)


def _next_action(classification: str, *, attempt: int, attempts: int) -> str:
    if classification == "PASS_STAGE_4_STRONG":
        return "ready_for_stage5_self_play"
    if classification == "PASS_STAGE_4":
        return "continue_for_strong_pass" if attempt < attempts else "ready_for_stage5_self_play"
    if attempt >= attempts:
        return "inspect_stage4_blocker"
    return "continue_loop"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/muzero_stage/stage4"))
    parser.add_argument("--stage2-metrics", type=Path, default=DEFAULT_STAGE2_METRICS)
    parser.add_argument("--stage3-decision", type=Path, default=DEFAULT_STAGE3_DECISION)
    parser.add_argument("--stage3-metrics", type=Path, default=DEFAULT_STAGE3_METRICS)
    parser.add_argument("--stage3-checkpoint", type=Path, default=DEFAULT_STAGE3_PROMOTED)
    parser.add_argument("--ppo-checkpoint", type=Path, default=DEFAULT_PROMOTED_CHECKPOINT)
    parser.add_argument("--frozen-pool", type=Path, action="append", default=None)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--max-wall-clock-sec", type=float, default=8.0 * 60.0 * 60.0)
    parser.add_argument("--teacher-episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=51000)
    parser.add_argument("--eval-seed", type=int, default=41000)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=8)
    parser.add_argument("--teacher-simulations", type=int, default=4)
    parser.add_argument("--teacher-rollout-rounds", type=int, default=1)
    parser.add_argument("--teacher-max-root-actions", type=int, default=4)
    parser.add_argument("--teacher-prior-improvement-margin", type=float, default=0.25)
    parser.add_argument("--model-simulations", type=int, default=8)
    parser.add_argument("--model-max-root-actions", type=int, default=8)
    parser.add_argument("--model-prior-improvement-margin", type=float, default=1.0)
    parser.add_argument("--simulator-value-guard", action="store_true")
    parser.add_argument("--simulator-value-threshold", type=float, default=0.05)
    parser.add_argument("--simulator-value-max-actions", type=int, default=12)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dynamics-coef", type=float, default=1.0)
    parser.add_argument("--reward-coef", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--policy-coef", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--player-order-mode",
        choices=["fixed", "random", "rotating"],
        default="fixed",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_stage4_loop(
        out_dir=args.out_dir,
        stage2_metrics=args.stage2_metrics,
        stage3_decision=args.stage3_decision,
        stage3_metrics=args.stage3_metrics,
        stage3_checkpoint=args.stage3_checkpoint,
        ppo_checkpoint=args.ppo_checkpoint,
        frozen_pool=tuple(args.frozen_pool or DEFAULT_FROZEN_POOL),
        attempts=args.attempts,
        max_wall_clock_sec=args.max_wall_clock_sec,
        teacher_episodes=args.teacher_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        eval_seed=args.eval_seed,
        player_count=args.players,
        max_actions_per_player=args.max_actions_per_player,
        teacher_simulations=args.teacher_simulations,
        teacher_rollout_rounds=args.teacher_rollout_rounds,
        teacher_max_root_actions=args.teacher_max_root_actions,
        teacher_prior_improvement_margin=args.teacher_prior_improvement_margin,
        model_simulations=args.model_simulations,
        model_max_root_actions=args.model_max_root_actions,
        model_prior_improvement_margin=args.model_prior_improvement_margin,
        simulator_value_guard=args.simulator_value_guard,
        simulator_value_threshold=args.simulator_value_threshold,
        simulator_value_max_actions=args.simulator_value_max_actions,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_sizes=parse_stage4_hidden_sizes(args.hidden_sizes),
        latent_dim=args.latent_dim,
        dynamics_coef=args.dynamics_coef,
        reward_coef=args.reward_coef,
        value_coef=args.value_coef,
        policy_coef=args.policy_coef,
        device=args.device,
        player_order_mode=args.player_order_mode,
    )
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "decision": report["artifacts"]["decision"],
                "metrics": report["artifacts"]["metrics"],
                "attempts": len(report["attempts"]),
                "best_checkpoint": report["best_checkpoint"],
                "promoted": str(Path(report["artifacts"]["promoted"]) / "dynamics_stage4.pt"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
