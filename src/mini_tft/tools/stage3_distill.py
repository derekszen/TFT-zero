"""Stage 3 search-distillation bootstrap runner."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import LobbyPolicy, PlayerOrderMode
from mini_tft.search.distill import (
    generate_teacher_dataset,
    load_search_dataset,
    load_student_policy,
    parse_hidden_sizes,
    train_student,
    validate_dataset_schema,
    write_json,
)
from mini_tft.search.mcts import MCTSConfig
from mini_tft.tools.lobby_diagnostics import (
    _checkpoint_policy,
    _named_policy,
    _opponent_policy,
    compare_trace_files,
    run_lobby_trace_export,
)
from mini_tft.tools.stage2_mcts import (
    DEFAULT_FROZEN_POOL,
    DEFAULT_PROMOTED_CHECKPOINT,
)

DEFAULT_STAGE2_DECISION = Path("artifacts/muzero_stage/stage2/decision.md")
DEFAULT_STAGE2_METRICS = Path("artifacts/muzero_stage/stage2/metrics.json")


def run_stage3_loop(
    *,
    out_dir: Path = Path("artifacts/muzero_stage/stage3"),
    stage2_decision: Path = DEFAULT_STAGE2_DECISION,
    stage2_metrics: Path = DEFAULT_STAGE2_METRICS,
    promoted_checkpoint: Path = DEFAULT_PROMOTED_CHECKPOINT,
    frozen_pool: Sequence[Path] = DEFAULT_FROZEN_POOL,
    attempts: int = 8,
    teacher_episodes: int = 1,
    eval_episodes: int = 3,
    seed: int = 41000,
    eval_seed: int = 31000,
    player_count: int = 8,
    max_actions_per_player: int | None = 8,
    simulations: int = 4,
    rollout_rounds: int = 1,
    max_root_actions: int = 4,
    prior_improvement_margin: float = 0.25,
    train_epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    value_coef: float = 0.5,
    selected_action_coef: float = 1.0,
    hidden_sizes: Sequence[int] = (128, 128),
    device: str = "cpu",
    player_order_mode: PlayerOrderMode = "fixed",
) -> dict[str, Any]:
    """Run the capped Stage 3 data/train/eval loop."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    _require_stage2_pass(stage2_decision, stage2_metrics)
    dirs = _stage3_dirs(out_dir)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    loop_log = out_dir / "loop-run-log.md"
    loop_state_path = out_dir / "loop-state.json"
    storage = _resource_check(out_dir)
    stage2 = json.loads(stage2_metrics.read_text(encoding="utf-8"))
    config = EnvConfig(seed=seed)
    mcts_config = MCTSConfig(
        simulations=simulations,
        rollout_rounds=rollout_rounds,
        max_root_actions=max_root_actions,
        max_actions_per_player=max_actions_per_player,
        prior_improvement_margin=prior_improvement_margin,
    )
    mixed_opponent = _named_policy("mixed")
    promoted_policy = _checkpoint_policy(promoted_checkpoint, device=device)
    frozen_opponent = _opponent_policy(
        opponent_policy_name="mixed",
        opponent_checkpoints=frozen_pool,
        device=device,
    )

    dataset_paths: list[Path] = []
    best_checkpoint: Path | None = None
    best_score = float("inf")
    attempts_report: list[dict[str, Any]] = []
    final_classification = "BLOCKED_POLICY"
    final_eval: dict[str, Any] | None = None
    started = perf_counter()
    _append_loop_log(
        loop_log,
        f"# Stage 3 Search-Distillation Loop\n\nResource check: `{storage['decision']}`\n",
    )

    for attempt in range(1, attempts + 1):
        attempt_seed = seed + attempt * 1000
        mixed_dataset = dirs["datasets"] / f"iter_{attempt:02d}_mixed.npz"
        frozen_dataset = dirs["datasets"] / f"iter_{attempt:02d}_frozen.npz"
        mixed_dataset_metrics = generate_teacher_dataset(
            out_path=mixed_dataset,
            suite_name="mixed",
            episodes=teacher_episodes,
            seed=attempt_seed,
            player_count=player_count,
            max_actions_per_player=max_actions_per_player,
            env_config=config,
            mcts_config=mcts_config,
            opponent_policy=mixed_opponent,
            prior_policy=promoted_policy,
            rollout_policy=promoted_policy,
            player_order_mode=player_order_mode,
        )
        frozen_dataset_metrics = generate_teacher_dataset(
            out_path=frozen_dataset,
            suite_name="frozen",
            episodes=teacher_episodes,
            seed=attempt_seed + 500,
            player_count=player_count,
            max_actions_per_player=max_actions_per_player,
            env_config=config,
            mcts_config=mcts_config,
            opponent_policy=frozen_opponent,
            prior_policy=promoted_policy,
            rollout_policy=promoted_policy,
            player_order_mode=player_order_mode,
        )
        for dataset_path in (mixed_dataset, frozen_dataset):
            validate_dataset_schema(load_search_dataset(dataset_path), env_config=config)
            dataset_paths.append(dataset_path)

        checkpoint = dirs["checkpoints"] / f"student_iter_{attempt:02d}.pt"
        train_report = train_student(
            dataset_paths=dataset_paths,
            out_path=checkpoint,
            epochs=train_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            value_coef=value_coef,
            selected_action_coef=selected_action_coef,
            hidden_sizes=hidden_sizes,
            seed=seed + attempt,
            device=device,
            resume=best_checkpoint,
        )
        eval_report = run_student_eval(
            checkpoint=checkpoint,
            out_dir=dirs["evals"] / f"iter_{attempt:02d}",
            promoted_checkpoint=promoted_checkpoint,
            frozen_pool=frozen_pool,
            stage2_metrics=stage2,
            episodes=eval_episodes,
            seed=eval_seed,
            player_count=player_count,
            max_actions_per_player=max_actions_per_player,
            config=EnvConfig(seed=eval_seed),
            device=device,
            player_order_mode=player_order_mode,
        )
        classification = classify_stage3(eval_report)
        score = _student_score(eval_report)
        if score < best_score:
            best_score = score
            best_checkpoint = checkpoint
        if classification == "PASS_STAGE_3":
            promoted_path = dirs["promoted"] / "student_stage3.pt"
            shutil.copy2(checkpoint, promoted_path)
            shutil.copy2(
                checkpoint.with_suffix(".metrics.json"),
                promoted_path.with_suffix(".metrics.json"),
            )
            final_classification = classification
            final_eval = eval_report
            next_action = "ready_for_stage4_dynamics"
        else:
            next_action = "continue_loop" if attempt < attempts else "inspect_policy_gap"

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
                "current_checkpoint": str(checkpoint),
                "next_action": next_action,
            },
        )
        if classification == "PASS_STAGE_3":
            break

    if final_eval is None and attempts_report:
        final_eval = attempts_report[-1]["eval"]
        final_classification = attempts_report[-1]["classification"]

    report: dict[str, Any] = {
        "status": "pass" if final_classification == "PASS_STAGE_3" else "fail",
        "kind": "stage3_search_distillation_loop",
        "classification": final_classification,
        "stage2_decision": str(stage2_decision),
        "stage2_metrics": str(stage2_metrics),
        "promoted_checkpoint": str(promoted_checkpoint),
        "frozen_pool": [str(path) for path in frozen_pool],
        "resource_check": storage,
        "config": {
            "attempt_cap": attempts,
            "teacher_episodes": teacher_episodes,
            "eval_episodes": eval_episodes,
            "seed": seed,
            "eval_seed": eval_seed,
            "player_count": player_count,
            "max_actions_per_player": max_actions_per_player,
            "simulations": simulations,
            "rollout_rounds": rollout_rounds,
            "max_root_actions": max_root_actions,
            "prior_improvement_margin": prior_improvement_margin,
            "train_epochs": train_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "value_coef": value_coef,
            "selected_action_coef": selected_action_coef,
            "hidden_sizes": [int(size) for size in hidden_sizes],
            "device": device,
            "player_order_mode": player_order_mode,
        },
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
        "attempts": attempts_report,
        "final_eval": final_eval,
        "elapsed_sec": perf_counter() - started,
    }
    write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(format_stage3_decision(report), encoding="utf-8")
    return report


def run_student_eval(
    *,
    checkpoint: Path,
    out_dir: Path,
    promoted_checkpoint: Path,
    frozen_pool: Sequence[Path],
    stage2_metrics: dict[str, Any],
    episodes: int,
    seed: int,
    player_count: int,
    max_actions_per_player: int | None,
    config: EnvConfig,
    device: str,
    player_order_mode: PlayerOrderMode,
) -> dict[str, Any]:
    """Evaluate a student checkpoint against mixed/frozen and PPO baselines."""

    out_dir.mkdir(parents=True, exist_ok=True)
    student_policy = load_student_policy(checkpoint, device=device)
    mixed_opponent = _named_policy("mixed")
    frozen_opponent = _opponent_policy(
        opponent_policy_name="mixed",
        opponent_checkpoints=frozen_pool,
        device=device,
    )
    mixed = _timed_trace_export(
        out_dir=out_dir / "student_mixed",
        trace_name="student_mixed",
        episodes=episodes,
        seed=seed,
        hero_policy=student_policy,
        opponent_policy=mixed_opponent,
        opponent_policy_name="mixed",
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        player_order_mode=player_order_mode,
    )
    frozen = _timed_trace_export(
        out_dir=out_dir / "student_frozen",
        trace_name="student_frozen",
        episodes=episodes,
        seed=seed,
        hero_policy=student_policy,
        opponent_policy=frozen_opponent,
        opponent_policy_name="frozen_checkpoint_pool",
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        player_order_mode=player_order_mode,
    )
    ppo_mixed = run_lobby_trace_export(
        episodes=episodes,
        seed=seed,
        checkpoint=promoted_checkpoint,
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
        checkpoint=promoted_checkpoint,
        opponent_checkpoints=frozen_pool,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        device=device,
        config=config,
        out_dir=out_dir / "ppo_frozen",
        trace_name="ppo_frozen",
        player_order_mode=player_order_mode,
    )
    mixed_compare = compare_trace_files(
        baseline_trace=Path(ppo_mixed["artifacts"]["trace"]),
        candidate_trace=Path(mixed["artifacts"]["trace"]),
        out_dir=out_dir / "compare_mixed_vs_ppo",
    )
    frozen_compare = compare_trace_files(
        baseline_trace=Path(ppo_frozen["artifacts"]["trace"]),
        candidate_trace=Path(frozen["artifacts"]["trace"]),
        out_dir=out_dir / "compare_frozen_vs_ppo",
    )
    report: dict[str, Any] = {
        "status": "pass",
        "kind": "stage3_student_eval",
        "checkpoint": str(checkpoint),
        "student_mixed": _summary(mixed),
        "student_frozen": _summary(frozen),
        "ppo_mixed": _summary(ppo_mixed),
        "ppo_frozen": _summary(ppo_frozen),
        "mixed_vs_ppo": _compare_summary(mixed_compare),
        "frozen_vs_ppo": _compare_summary(frozen_compare),
        "stage2_teacher": {
            "mixed": stage2_metrics["mcts_mixed_eval"],
            "frozen": stage2_metrics["mcts_frozen_eval"],
        },
        "artifacts": {
            "metrics": str(out_dir / "metrics.json"),
            "student_mixed": str(out_dir / "student_mixed"),
            "student_frozen": str(out_dir / "student_frozen"),
            "ppo_mixed": str(out_dir / "ppo_mixed"),
            "ppo_frozen": str(out_dir / "ppo_frozen"),
            "compare_mixed_vs_ppo": str(out_dir / "compare_mixed_vs_ppo"),
            "compare_frozen_vs_ppo": str(out_dir / "compare_frozen_vs_ppo"),
        },
    }
    write_json(out_dir / "metrics.json", report)
    return report


def classify_stage3(eval_report: dict[str, Any]) -> str:
    """Classify one Stage 3 student eval under the objective gates."""

    if (
        eval_report["student_mixed"]["illegal_actions"] != 0
        or eval_report["student_frozen"]["illegal_actions"] != 0
    ):
        return "BLOCKED_POLICY"
    mixed_delta = eval_report["mixed_vs_ppo"]["scalar"]
    frozen_delta = eval_report["frozen_vs_ppo"]["scalar"]
    if (
        mixed_delta["mean_placement"] > 0.25
        or frozen_delta["mean_placement"] > 0.50
        or mixed_delta["mean_final_hp"] < -5.0
        or frozen_delta["mean_final_hp"] < -5.0
    ):
        return "BLOCKED_POLICY"
    mixed_student = eval_report["student_mixed"]
    frozen_student = eval_report["student_frozen"]
    ppo_mixed = eval_report["ppo_mixed"]
    ppo_frozen = eval_report["ppo_frozen"]
    teacher_mixed = eval_report["stage2_teacher"]["mixed"]
    teacher_frozen = eval_report["stage2_teacher"]["frozen"]
    mixed_strong = mixed_student["mean_placement"] <= ppo_mixed["mean_placement"] + 0.10
    frozen_strong = (
        frozen_student["mean_placement"] <= ppo_frozen["mean_placement"] + 0.25
        and frozen_student["mean_final_hp"] >= ppo_frozen["mean_final_hp"]
    )
    speed_strong = (
        mixed_student["decisions_per_sec"] >= 2.0 * teacher_mixed["mcts_decisions_per_sec"]
        and frozen_student["decisions_per_sec"] >= 2.0 * teacher_frozen["mcts_decisions_per_sec"]
    )
    if mixed_strong and frozen_strong and speed_strong:
        return "PASS_STAGE_3"
    if mixed_strong and frozen_strong and not speed_strong:
        return "BLOCKED_SPEED"
    return "BLOCKED_POLICY"


def format_loop_attempt(report: dict[str, Any]) -> str:
    eval_report = report["eval"]
    return "\n".join(
        [
            f"## Attempt {report['attempt']}",
            "",
            f"- classification: `{report['classification']}`",
            f"- checkpoint: `{report['checkpoint']}`",
            f"- rows mixed/frozen: {report['mixed_dataset']['rows']} / "
            f"{report['frozen_dataset']['rows']}",
            f"- train final loss: {report['train']['final_loss']}",
            f"- train selected-action acc: {report['train']['final_selected_action_acc']}",
            f"- mixed placement delta vs PPO: "
            f"{eval_report['mixed_vs_ppo']['scalar']['mean_placement']:.3f}",
            f"- frozen placement delta vs PPO: "
            f"{eval_report['frozen_vs_ppo']['scalar']['mean_placement']:.3f}",
            f"- mixed HP delta vs PPO: "
            f"{eval_report['mixed_vs_ppo']['scalar']['mean_final_hp']:.3f}",
            f"- frozen HP delta vs PPO: "
            f"{eval_report['frozen_vs_ppo']['scalar']['mean_final_hp']:.3f}",
            f"- next action: `{report['next_action']}`",
            "",
        ]
    )


def format_stage3_decision(report: dict[str, Any]) -> str:
    final_eval = report["final_eval"] or {}
    mixed_delta = final_eval.get("mixed_vs_ppo", {}).get("scalar", {})
    frozen_delta = final_eval.get("frozen_vs_ppo", {}).get("scalar", {})
    return "\n".join(
        [
            "# Stage 3 Search-Distillation Gate Decision",
            "",
            f"**Decision:** `{report['classification']}`",
            "",
            "This stage distills PPO-guided MCTS traces into a standalone policy/value "
            "student. It is still search-distillation bootstrap work, not a learned "
            "dynamics model and not full MuZero.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Attempts run | {len(report['attempts'])} |",
            f"| Best checkpoint | `{report['best_checkpoint']}` |",
            f"| Mixed placement delta vs PPO | "
            f"{_fmt(mixed_delta.get('mean_placement'))} |",
            f"| Frozen placement delta vs PPO | "
            f"{_fmt(frozen_delta.get('mean_placement'))} |",
            f"| Mixed HP delta vs PPO | {_fmt(mixed_delta.get('mean_final_hp'))} |",
            f"| Frozen HP delta vs PPO | {_fmt(frozen_delta.get('mean_final_hp'))} |",
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


def _timed_trace_export(
    *,
    out_dir: Path,
    trace_name: str,
    episodes: int,
    seed: int,
    hero_policy: LobbyPolicy,
    opponent_policy: LobbyPolicy,
    opponent_policy_name: str,
    player_count: int,
    max_actions_per_player: int | None,
    config: EnvConfig,
    player_order_mode: PlayerOrderMode,
) -> dict[str, Any]:
    started = perf_counter()
    report = run_lobby_trace_export(
        episodes=episodes,
        seed=seed,
        hero_policy_name="stage3_student",
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
    report["speed"] = {
        "elapsed_sec": elapsed_sec,
        "decisions_per_sec": float(report["decisions"]) / elapsed_sec if elapsed_sec > 0 else 0.0,
        "episodes_per_sec": episodes / elapsed_sec if elapsed_sec > 0 else 0.0,
    }
    write_json(out_dir / "metrics.json", report)
    return report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    speed = report.get("speed", {})
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
        "elapsed_sec": speed.get("elapsed_sec"),
    }


def _compare_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "scalar": report["deltas"]["scalar"],
        "category_rates": report["deltas"]["category_rates"],
        "bucket_rates": report["deltas"]["bucket_rates"],
    }


def _student_score(eval_report: dict[str, Any]) -> float:
    mixed = eval_report["mixed_vs_ppo"]["scalar"]
    frozen = eval_report["frozen_vs_ppo"]["scalar"]
    return float(
        mixed["mean_placement"]
        + frozen["mean_placement"]
        - 0.02 * mixed["mean_final_hp"]
        - 0.02 * frozen["mean_final_hp"]
    )


def _stage3_dirs(out_dir: Path) -> dict[str, Path]:
    return {
        "datasets": out_dir / "datasets",
        "checkpoints": out_dir / "checkpoints",
        "evals": out_dir / "evals",
        "loops": out_dir / "loops",
        "promoted": out_dir / "promoted",
    }


def _resource_check(out_dir: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(out_dir.parent if out_dir.parent.exists() else Path("."))
    free_gb = usage.free / (1024**3)
    return {
        "disk_free_gb": free_gb,
        "decision": "continue" if free_gb >= 5.0 else "soft-pause",
    }


def _require_stage2_pass(stage2_decision: Path, stage2_metrics: Path) -> None:
    if not stage2_decision.is_file():
        raise FileNotFoundError(f"missing Stage 2 decision: {stage2_decision}")
    if not stage2_metrics.is_file():
        raise FileNotFoundError(f"missing Stage 2 metrics: {stage2_metrics}")
    metrics = json.loads(stage2_metrics.read_text(encoding="utf-8"))
    if metrics.get("classification") != "PASS_TO_STAGE_3":
        raise RuntimeError(f"Stage 2 did not pass: {metrics.get('classification')}")


def _append_loop_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _write_loop_state(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
    loops_dir = path.parent / "loops"
    loops_dir.mkdir(parents=True, exist_ok=True)
    write_json(loops_dir / f"attempt_{payload['attempt']:02d}_state.json", payload)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/muzero_stage/stage3"))
    parser.add_argument("--stage2-decision", type=Path, default=DEFAULT_STAGE2_DECISION)
    parser.add_argument("--stage2-metrics", type=Path, default=DEFAULT_STAGE2_METRICS)
    parser.add_argument("--promoted-checkpoint", type=Path, default=DEFAULT_PROMOTED_CHECKPOINT)
    parser.add_argument("--frozen-pool", type=Path, action="append", default=None)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--teacher-episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--eval-seed", type=int, default=31000)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--rollout-rounds", type=int, default=1)
    parser.add_argument("--max-root-actions", type=int, default=4)
    parser.add_argument("--prior-improvement-margin", type=float, default=0.25)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--selected-action-coef", type=float, default=1.0)
    parser.add_argument("--hidden-sizes", default="128,128")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--player-order-mode",
        choices=["fixed", "random", "rotating"],
        default="fixed",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_stage3_loop(
        out_dir=args.out_dir,
        stage2_decision=args.stage2_decision,
        stage2_metrics=args.stage2_metrics,
        promoted_checkpoint=args.promoted_checkpoint,
        frozen_pool=tuple(args.frozen_pool or DEFAULT_FROZEN_POOL),
        attempts=args.attempts,
        teacher_episodes=args.teacher_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        eval_seed=args.eval_seed,
        player_count=args.players,
        max_actions_per_player=args.max_actions_per_player,
        simulations=args.simulations,
        rollout_rounds=args.rollout_rounds,
        max_root_actions=args.max_root_actions,
        prior_improvement_margin=args.prior_improvement_margin,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        value_coef=args.value_coef,
        selected_action_coef=args.selected_action_coef,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
