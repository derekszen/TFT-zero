"""Stage 2 simulator-backed MCTS scaffold runner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import LobbyPolicy, PlayerOrderMode, fast_level_lobby_policy
from mini_tft.search.mcts import MCTSConfig, RootMCTSPlanner
from mini_tft.tools.lobby_diagnostics import (
    _checkpoint_policy,
    _named_policy,
    _opponent_policy,
    compare_trace_files,
    run_lobby_trace_export,
)

DEFAULT_PROMOTED_CHECKPOINT = Path(
    "checkpoints/ppo_lobby_frozenpool_probe10k2_lr1e5_20260625T1052_h256.zip"
)
DEFAULT_FROZEN_POOL = (
    Path("checkpoints/ppo_lobby_500k_from_1m_h256.zip"),
    Path("checkpoints/tmp_ppo_baseline_50k_from_500k_h256.zip"),
    Path("checkpoints/ppo_lobby_merged_50k_from_500k_seed3300_h256.zip"),
    Path("checkpoints/ppo_lobby_merged_50k_low_lr_from_seed3300_h256.zip"),
)
DEFAULT_STAGE1_DECISION = Path("artifacts/muzero_stage/stage1/decision.md")


def run_stage2_suite(
    *,
    out_dir: Path = Path("artifacts/muzero_stage/stage2"),
    stage1_decision: Path = DEFAULT_STAGE1_DECISION,
    promoted_checkpoint: Path = DEFAULT_PROMOTED_CHECKPOINT,
    frozen_pool: Sequence[Path] = DEFAULT_FROZEN_POOL,
    smoke_episodes: int = 1,
    eval_episodes: int = 3,
    seed: int = 31000,
    player_count: int = 8,
    max_actions_per_player: int | None = 8,
    simulations: int = 4,
    rollout_rounds: int = 1,
    max_root_actions: int = 4,
    prior_improvement_margin: float = 0.25,
    device: str = "cpu",
    player_order_mode: PlayerOrderMode = "fixed",
) -> dict[str, Any]:
    """Run the bounded Stage 2 scaffold gate and write required artifacts."""

    _require_stage1_pass(stage1_decision)
    out_dir.mkdir(parents=True, exist_ok=True)
    search_trace_dir = out_dir / "search_traces"
    search_trace_dir.mkdir(parents=True, exist_ok=True)
    config = EnvConfig(seed=seed)

    mixed_opponent = _named_policy("mixed")
    promoted_policy = _checkpoint_policy(promoted_checkpoint, device=device)
    frozen_opponent = _opponent_policy(
        opponent_policy_name="mixed",
        opponent_checkpoints=frozen_pool,
        device=device,
    )
    mcts_config = MCTSConfig(
        simulations=simulations,
        rollout_rounds=rollout_rounds,
        max_root_actions=max_root_actions,
        max_actions_per_player=max_actions_per_player,
        prior_improvement_margin=prior_improvement_margin,
    )

    smoke = run_mcts_trace_export(
        name="mcts_smoke",
        out_dir=out_dir / "mcts_smoke",
        search_trace_path=search_trace_dir / "mcts_smoke.jsonl",
        episodes=smoke_episodes,
        seed=seed,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        env_config=config,
        mcts_config=mcts_config,
        opponent_policy=mixed_opponent,
        prior_policy=promoted_policy,
        rollout_policy=promoted_policy,
        opponent_policy_name="mixed",
        player_order_mode=player_order_mode,
    )
    mixed = run_mcts_trace_export(
        name="mcts_mixed_eval",
        out_dir=out_dir / "mcts_mixed_eval",
        search_trace_path=search_trace_dir / "mcts_mixed_eval.jsonl",
        episodes=eval_episodes,
        seed=seed,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        env_config=config,
        mcts_config=mcts_config,
        opponent_policy=mixed_opponent,
        prior_policy=promoted_policy,
        rollout_policy=promoted_policy,
        opponent_policy_name="mixed",
        player_order_mode=player_order_mode,
    )
    frozen = run_mcts_trace_export(
        name="mcts_frozen_eval",
        out_dir=out_dir / "mcts_frozen_eval",
        search_trace_path=search_trace_dir / "mcts_frozen_eval.jsonl",
        episodes=eval_episodes,
        seed=seed,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        env_config=config,
        mcts_config=mcts_config,
        opponent_policy=frozen_opponent,
        prior_policy=promoted_policy,
        rollout_policy=promoted_policy,
        opponent_policy_name="frozen_checkpoint_pool",
        player_order_mode=player_order_mode,
    )

    compare = run_mcts_vs_ppo_compare(
        out_dir=out_dir / "mcts_vs_ppo_compare",
        promoted_checkpoint=promoted_checkpoint,
        frozen_pool=frozen_pool,
        mcts_mixed_trace=Path(mixed["artifacts"]["trace"]),
        mcts_frozen_trace=Path(frozen["artifacts"]["trace"]),
        episodes=eval_episodes,
        seed=seed,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        env_config=config,
        device=device,
        player_order_mode=player_order_mode,
    )

    classification = classify_stage2(
        smoke=smoke,
        mixed=mixed,
        frozen=frozen,
        compare=compare,
    )
    report: dict[str, Any] = {
        "status": "pass" if classification == "PASS_TO_STAGE_3" else "fail",
        "kind": "stage2_mcts_gate",
        "classification": classification,
        "stage1_decision": str(stage1_decision),
        "promoted_checkpoint": str(promoted_checkpoint),
        "frozen_pool": [str(path) for path in frozen_pool],
        "config": {
            "smoke_episodes": smoke_episodes,
            "eval_episodes": eval_episodes,
            "seed": seed,
            "player_count": player_count,
            "max_actions_per_player": max_actions_per_player,
            "simulations": simulations,
            "rollout_rounds": rollout_rounds,
            "max_root_actions": max_root_actions,
            "prior_improvement_margin": prior_improvement_margin,
            "player_order_mode": player_order_mode,
        },
        "artifacts": {
            "mcts_smoke": str(out_dir / "mcts_smoke"),
            "mcts_mixed_eval": str(out_dir / "mcts_mixed_eval"),
            "mcts_frozen_eval": str(out_dir / "mcts_frozen_eval"),
            "mcts_vs_ppo_compare": str(out_dir / "mcts_vs_ppo_compare"),
            "search_traces": str(search_trace_dir),
            "metrics": str(out_dir / "metrics.json"),
            "decision": str(out_dir / "decision.md"),
        },
        "mcts_smoke": _gate_summary(smoke),
        "mcts_mixed_eval": _gate_summary(mixed),
        "mcts_frozen_eval": _gate_summary(frozen),
        "mcts_vs_ppo_compare": compare,
    }
    _write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(
        format_stage2_decision(report),
        encoding="utf-8",
    )
    return report


def run_mcts_trace_export(
    *,
    name: str,
    out_dir: Path,
    search_trace_path: Path,
    episodes: int,
    seed: int,
    player_count: int,
    max_actions_per_player: int | None,
    env_config: EnvConfig,
    mcts_config: MCTSConfig,
    opponent_policy: LobbyPolicy,
    opponent_policy_name: str,
    player_order_mode: PlayerOrderMode,
    prior_policy: LobbyPolicy | None = None,
    rollout_policy: LobbyPolicy = fast_level_lobby_policy,
) -> dict[str, Any]:
    """Evaluate MCTS as player 0 and export lobby plus search traces."""

    planner = RootMCTSPlanner(
        hero_player_id=0,
        config=mcts_config,
        opponent_policy=opponent_policy,
        rollout_policy=rollout_policy,
        prior_policy=prior_policy,
    )
    hero_policy = _planner_policy_with_episode_context(planner, seed)
    start = perf_counter()
    report = run_lobby_trace_export(
        episodes=episodes,
        seed=seed,
        hero_policy_name=name,
        opponent_policy_name=opponent_policy_name,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=env_config,
        out_dir=out_dir,
        trace_name=name,
        hero_policy=hero_policy,
        opponent_policy=opponent_policy,
        player_order_mode=player_order_mode,
    )
    elapsed_sec = perf_counter() - start
    _write_jsonl(search_trace_path, planner.traces)
    report["kind"] = "stage2_mcts_eval"
    report["name"] = name
    report["player_order_mode"] = player_order_mode
    report["mcts"] = {
        "simulations": mcts_config.simulations,
        "rollout_rounds": mcts_config.rollout_rounds,
        "max_root_actions": mcts_config.max_root_actions,
        "prior_improvement_margin": mcts_config.prior_improvement_margin,
        "search_decisions": len(planner.traces),
        "search_trace": str(search_trace_path),
        "search_illegal_actions": sum(int(row["illegal_actions"]) for row in planner.traces),
        "selected_from_prior_rate": _rate(
            sum(1 for row in planner.traces if bool(row.get("selected_from_prior"))),
            len(planner.traces),
        ),
    }
    report["speed"] = {
        "elapsed_sec": elapsed_sec,
        "episodes_per_sec": episodes / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        "decisions_per_sec": (
            float(report["decisions"]) / elapsed_sec if elapsed_sec > 0.0 else 0.0
        ),
        "mcts_decisions_per_sec": (
            len(planner.traces) / elapsed_sec if elapsed_sec > 0.0 else 0.0
        ),
        "simulations_per_sec": (
            (len(planner.traces) * mcts_config.simulations) / elapsed_sec
            if elapsed_sec > 0.0
            else 0.0
        ),
    }
    _write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(format_mcts_eval_markdown(report), encoding="utf-8")
    return report


def run_mcts_vs_ppo_compare(
    *,
    out_dir: Path,
    promoted_checkpoint: Path,
    frozen_pool: Sequence[Path],
    mcts_mixed_trace: Path,
    mcts_frozen_trace: Path,
    episodes: int,
    seed: int,
    player_count: int,
    max_actions_per_player: int | None,
    env_config: EnvConfig,
    device: str,
    player_order_mode: PlayerOrderMode,
) -> dict[str, Any]:
    """Run PPO on matching suites and compare trace distributions to MCTS."""

    out_dir.mkdir(parents=True, exist_ok=True)
    ppo_mixed = run_lobby_trace_export(
        episodes=episodes,
        seed=seed,
        checkpoint=promoted_checkpoint,
        opponent_policy_name="mixed",
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        device=device,
        config=env_config,
        out_dir=out_dir / "ppo_mixed_eval",
        trace_name="ppo_mixed_eval",
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
        config=env_config,
        out_dir=out_dir / "ppo_frozen_eval",
        trace_name="ppo_frozen_eval",
        player_order_mode=player_order_mode,
    )
    mixed_compare = compare_trace_files(
        baseline_trace=Path(ppo_mixed["artifacts"]["trace"]),
        candidate_trace=mcts_mixed_trace,
        out_dir=out_dir / "mixed",
    )
    frozen_compare = compare_trace_files(
        baseline_trace=Path(ppo_frozen["artifacts"]["trace"]),
        candidate_trace=mcts_frozen_trace,
        out_dir=out_dir / "frozen",
    )
    report: dict[str, Any] = {
        "status": "pass",
        "kind": "stage2_mcts_vs_ppo_compare",
        "ppo_mixed": _gate_summary(ppo_mixed),
        "ppo_frozen": _gate_summary(ppo_frozen),
        "mixed_compare": _compare_summary(mixed_compare),
        "frozen_compare": _compare_summary(frozen_compare),
        "artifacts": {
            "metrics": str(out_dir / "metrics.json"),
            "decision": str(out_dir / "decision.md"),
            "ppo_mixed_eval": str(out_dir / "ppo_mixed_eval"),
            "ppo_frozen_eval": str(out_dir / "ppo_frozen_eval"),
            "mixed_compare": str(out_dir / "mixed"),
            "frozen_compare": str(out_dir / "frozen"),
        },
    }
    _write_json(out_dir / "metrics.json", report)
    (out_dir / "decision.md").write_text(format_compare_markdown(report), encoding="utf-8")
    return report


def classify_stage2(
    *,
    smoke: dict[str, Any],
    mixed: dict[str, Any],
    frozen: dict[str, Any],
    compare: dict[str, Any],
) -> str:
    """Map Stage 2 gate metrics into the required decision labels."""

    reports = (smoke, mixed, frozen)
    if any(report["status"] != "pass" for report in reports):
        return "BLOCKED_SIMULATOR"
    if any(int(report["total_illegal_actions"]) != 0 for report in reports):
        return "BLOCKED_SIMULATOR"
    if any(int(report["mcts"]["search_illegal_actions"]) != 0 for report in reports):
        return "BLOCKED_SIMULATOR"
    if any(float(report["speed"]["mcts_decisions_per_sec"]) <= 0.05 for report in reports):
        return "BLOCKED_SPEED"
    if mixed["summary"]["chosen_end_turn_rate"] > 0.95 and frozen["summary"][
        "chosen_end_turn_rate"
    ] > 0.95:
        return "BLOCKED_POLICY"
    mixed_delta = compare["mixed_compare"]["scalar"]
    frozen_delta = compare["frozen_compare"]["scalar"]
    if _is_policy_regression(mixed_delta) or _is_policy_regression(frozen_delta):
        return "BLOCKED_POLICY"
    return "PASS_TO_STAGE_3"


def format_mcts_eval_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    speed = report["speed"]
    mcts = report["mcts"]
    return "\n".join(
        [
            f"# Stage 2 MCTS Evaluation: {report['name']}",
            "",
            f"**Status:** `{report['status']}`",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Episodes | {report['episodes']} |",
            f"| Decisions | {report['decisions']} |",
            f"| Search decisions | {mcts['search_decisions']} |",
            f"| Simulations per decision | {mcts['simulations']} |",
            f"| Rollout rounds | {mcts['rollout_rounds']} |",
            f"| Prior improvement margin | {mcts['prior_improvement_margin']:.3f} |",
            f"| Selected from prior rate | {mcts['selected_from_prior_rate']:.3f} |",
            f"| Illegal actions | {report['total_illegal_actions']} |",
            f"| Search illegal actions | {mcts['search_illegal_actions']} |",
            f"| Mean placement | {summary['mean_placement']:.3f} |",
            f"| Mean final HP | {summary['mean_final_hp']:.3f} |",
            f"| Mean final board strength | {summary['mean_final_board_strength']:.3f} |",
            f"| Roll rate | {summary['chosen_roll_rate']:.3f} |",
            f"| Buy-XP rate | {summary['chosen_buy_xp_rate']:.3f} |",
            f"| Buy/shop rate | {summary['chosen_buy_shop_rate']:.3f} |",
            f"| End-turn rate | {summary['chosen_end_turn_rate']:.3f} |",
            f"| Wall-clock seconds | {speed['elapsed_sec']:.3f} |",
            f"| MCTS decisions/sec | {speed['mcts_decisions_per_sec']:.3f} |",
            f"| Simulations/sec | {speed['simulations_per_sec']:.3f} |",
            "",
            f"Search trace: `{mcts['search_trace']}`",
            "",
        ]
    )


def format_compare_markdown(report: dict[str, Any]) -> str:
    mixed = report["mixed_compare"]["scalar"]
    frozen = report["frozen_compare"]["scalar"]
    return "\n".join(
        [
            "# Stage 2 MCTS vs PPO Comparison",
            "",
            f"**Status:** `{report['status']}`",
            "",
            "| Suite | Placement delta | HP delta | Roll delta | Buy-XP delta | End-turn delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            _compare_row("Mixed", mixed),
            _compare_row("Frozen", frozen),
            "",
        ]
    )


def format_stage2_decision(report: dict[str, Any]) -> str:
    mixed = report["mcts_mixed_eval"]
    frozen = report["mcts_frozen_eval"]
    smoke = report["mcts_smoke"]
    mixed_delta = report["mcts_vs_ppo_compare"]["mixed_compare"]["scalar"]
    frozen_delta = report["mcts_vs_ppo_compare"]["frozen_compare"]["scalar"]
    interpretation = (
        "The simulator-backed scaffold is mechanically working, but the current "
        "search policy regresses hard versus PPO. Fix the MCTS policy/value setup "
        "before Stage 3."
        if report["classification"] == "BLOCKED_POLICY"
        else "The scaffold is mechanically working and did not hit a gate blocker."
    )
    return "\n".join(
        [
            "# Stage 2 Simulator-Backed MCTS Gate Decision",
            "",
            f"**Decision:** `{report['classification']}`",
            "",
            "The Stage 2 scaffold can clone lobby states, run legal-action simulator "
            "branches, export per-decision search traces, and compare MCTS against "
            "the promoted PPO checkpoint on matching mixed/frozen seeds.",
            "",
            interpretation,
            "",
            "| Check | Result |",
            "| --- | ---: |",
            f"| Smoke illegal actions | {smoke['illegal_actions']} |",
            f"| Mixed illegal actions | {mixed['illegal_actions']} |",
            f"| Frozen illegal actions | {frozen['illegal_actions']} |",
            f"| Smoke search illegal actions | {smoke['search_illegal_actions']} |",
            f"| Mixed search illegal actions | {mixed['search_illegal_actions']} |",
            f"| Frozen search illegal actions | {frozen['search_illegal_actions']} |",
            f"| Mixed mean placement | {mixed['mean_placement']:.3f} |",
            f"| Frozen mean placement | {frozen['mean_placement']:.3f} |",
            "| Mixed placement delta vs PPO | "
            f"{_format_optional(mixed_delta['mean_placement'])} |",
            f"| Mixed HP delta vs PPO | {_format_optional(mixed_delta['mean_final_hp'])} |",
            "| Frozen placement delta vs PPO | "
            f"{_format_optional(frozen_delta['mean_placement'])} |",
            f"| Frozen HP delta vs PPO | {_format_optional(frozen_delta['mean_final_hp'])} |",
            f"| Mixed MCTS decisions/sec | {mixed['mcts_decisions_per_sec']:.3f} |",
            f"| Frozen MCTS decisions/sec | {frozen['mcts_decisions_per_sec']:.3f} |",
            "",
            "## Required Artifacts",
            "",
            f"- `{report['artifacts']['mcts_smoke']}`",
            f"- `{report['artifacts']['mcts_mixed_eval']}`",
            f"- `{report['artifacts']['mcts_frozen_eval']}`",
            f"- `{report['artifacts']['mcts_vs_ppo_compare']}`",
            f"- `{report['artifacts']['search_traces']}`",
            "",
        ]
    )


def _planner_policy_with_episode_context(
    planner: RootMCTSPlanner,
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
        return int(planner.policy(player_id, state, mask, data, env_config, rng))

    return policy


def _gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    speed = report.get("speed", {})
    mcts = report.get("mcts", {})
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
        "elapsed_sec": speed.get("elapsed_sec"),
        "mcts_decisions_per_sec": speed.get("mcts_decisions_per_sec"),
        "simulations_per_sec": speed.get("simulations_per_sec"),
        "search_decisions": mcts.get("search_decisions"),
        "search_illegal_actions": mcts.get("search_illegal_actions", 0),
        "selected_from_prior_rate": mcts.get("selected_from_prior_rate"),
    }


def _compare_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "scalar": report["deltas"]["scalar"],
        "category_rates": report["deltas"]["category_rates"],
        "bucket_rates": report["deltas"]["bucket_rates"],
    }


def _compare_row(label: str, scalar: dict[str, float | None]) -> str:
    return (
        f"| {label} | {_format_optional(scalar['mean_placement'])} | "
        f"{_format_optional(scalar['mean_final_hp'])} | "
        f"{_format_optional(scalar['chosen_roll_rate'])} | "
        f"{_format_optional(scalar['chosen_buy_xp_rate'])} | "
        f"{_format_optional(scalar['chosen_end_turn_rate'])} |"
    )


def _is_policy_regression(scalar: dict[str, float | None]) -> bool:
    placement_delta = scalar.get("mean_placement")
    hp_delta = scalar.get("mean_final_hp")
    board_delta = scalar.get("mean_final_board_strength")
    return (
        (placement_delta is not None and placement_delta >= 1.0)
        or (hp_delta is not None and hp_delta <= -20.0)
        or (board_delta is not None and board_delta <= -50.0)
    )


def _require_stage1_pass(stage1_decision: Path) -> None:
    if not stage1_decision.is_file():
        raise FileNotFoundError(f"missing Stage 1 decision artifact: {stage1_decision}")
    text = stage1_decision.read_text(encoding="utf-8")
    if "`PASS_TO_STAGE_2`" not in text and "PASS_TO_STAGE_2" not in text:
        raise RuntimeError(f"Stage 1 did not pass to Stage 2: {stage1_decision}")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/muzero_stage/stage2"))
    parser.add_argument("--stage1-decision", type=Path, default=DEFAULT_STAGE1_DECISION)
    parser.add_argument("--promoted-checkpoint", type=Path, default=DEFAULT_PROMOTED_CHECKPOINT)
    parser.add_argument("--frozen-pool", type=Path, action="append", default=None)
    parser.add_argument("--smoke-episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--rollout-rounds", type=int, default=1)
    parser.add_argument("--max-root-actions", type=int, default=4)
    parser.add_argument("--prior-improvement-margin", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--player-order-mode",
        choices=["fixed", "random", "rotating"],
        default="fixed",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_stage2_suite(
        out_dir=args.out_dir,
        stage1_decision=args.stage1_decision,
        promoted_checkpoint=args.promoted_checkpoint,
        frozen_pool=tuple(args.frozen_pool or DEFAULT_FROZEN_POOL),
        smoke_episodes=args.smoke_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        player_count=args.players,
        max_actions_per_player=args.max_actions_per_player,
        simulations=args.simulations,
        rollout_rounds=args.rollout_rounds,
        max_root_actions=args.max_root_actions,
        prior_improvement_margin=args.prior_improvement_margin,
        device=args.device,
        player_order_mode=args.player_order_mode,
    )
    print(json.dumps(_gate_summary_for_stdout(report), indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def _gate_summary_for_stdout(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": report["classification"],
        "decision": report["artifacts"]["decision"],
        "metrics": report["artifacts"]["metrics"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
