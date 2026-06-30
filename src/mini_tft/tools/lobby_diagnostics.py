"""Stage-1 lobby diagnostics for PPO plateau investigation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import (
    Action,
    action_name,
    is_buy_shop_action,
    is_move_bench_to_board_action,
    is_move_board_to_bench_action,
    is_sell_bench_action,
)
from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import Set1LobbyState
from mini_tft.core.lobby_step import LobbyPolicy, PlayerOrderMode
from mini_tft.core.set_data import GameData
from mini_tft.rl.checkpoint_policy import CheckpointFormat, load_lobby_checkpoint_policy
from mini_tft.rl.lobby_env import LOBBY_POLICY_BY_NAME, MiniTFTLobbyEnv, MiniTFTLobbySnapshot


@dataclass(frozen=True)
class RollProbeCase:
    episode: int
    seed: int
    decision_index: int
    context: dict[str, Any]
    snapshot: MiniTFTLobbySnapshot


def run_lobby_trace_export(
    *,
    episodes: int = 10,
    seed: int = 1000,
    hero_policy_name: str = "fast_level",
    opponent_policy_name: str = "mixed",
    checkpoint: Path | None = None,
    opponent_checkpoints: Sequence[Path] = (),
    player_count: int = 8,
    max_actions_per_player: int | None = None,
    device: str = "cpu",
    checkpoint_format: CheckpointFormat = "auto",
    config: EnvConfig | None = None,
    out_dir: Path | None = None,
    trace_name: str = "trace",
    hero_policy: LobbyPolicy | None = None,
    opponent_policy: LobbyPolicy | None = None,
    player_order_mode: PlayerOrderMode = "random",
) -> dict[str, Any]:
    """Export per-hero-decision lobby traces and return a summary report."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    base_config = config or EnvConfig(seed=seed)
    resolved_hero = hero_policy or _hero_policy(
        hero_policy_name=hero_policy_name,
        checkpoint=checkpoint,
        device=device,
        checkpoint_format=checkpoint_format,
    )
    resolved_opponent = opponent_policy or _opponent_policy(
        opponent_policy_name=opponent_policy_name,
        opponent_checkpoints=opponent_checkpoints,
        device=device,
        checkpoint_format=checkpoint_format,
    )

    rows: list[dict[str, Any]] = []
    total_illegal_actions = 0
    total_fights = 0

    for episode in range(episodes):
        episode_seed = seed + episode
        episode_rows: list[dict[str, Any]] = []
        policy = _make_trace_policy(
            episode=episode,
            episode_seed=episode_seed,
            episode_rows=episode_rows,
            hero_policy=resolved_hero,
            opponent_policy=resolved_opponent,
        )

        env = MiniTFTLobbyEnv(
            seed=episode_seed,
            config=base_config,
            player_count=player_count,
            policy=policy,
            max_actions_per_player=max_actions_per_player,
            player_order_mode=player_order_mode,
        )
        _obs, _info = env.reset(seed=episode_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _obs, _reward, terminated, truncated, info = env.step(None)
            action_step = info.get("action_step_result")
            if action_step is not None:
                total_illegal_actions += int(action_step.illegal_actions)
                total_fights += int(action_step.resolved_fights)

        summary = env.episode_summary(0)
        for row in episode_rows:
            row["outcome"] = {
                "placement": _summary_int(summary, "placement"),
                "final_hp": _summary_int(summary, "final_hp"),
                "survived_round": _summary_int(summary, "survived_round"),
                "final_board_strength": _summary_float(summary, "final_board_strength"),
                "final_reason": summary["final_reason"],
            }
        rows.extend(episode_rows)
        env.close()

    summary = summarize_trace_rows(rows)
    report: dict[str, Any] = {
        "status": "pass" if total_illegal_actions == 0 else "fail",
        "kind": "lobby_trace_export",
        "episodes": episodes,
        "seed": seed,
        "player_count": player_count,
        "player_order_mode": player_order_mode,
        "hero_policy": str(checkpoint) if checkpoint is not None else hero_policy_name,
        "opponent_policy": (
            "frozen_checkpoint_pool" if opponent_checkpoints else opponent_policy_name
        ),
        "opponent_checkpoints": [str(path) for path in opponent_checkpoints],
        "decisions": len(rows),
        "total_illegal_actions": total_illegal_actions,
        "total_fights": total_fights,
        "summary": summary,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = out_dir / f"{trace_name}.jsonl"
        _write_jsonl(trace_path, rows)
        report["artifacts"] = {
            "trace": str(trace_path),
            "metrics": str(out_dir / "metrics.json"),
            "decision": str(out_dir / "decision.md"),
        }
        _write_json(out_dir / "metrics.json", report)
        (out_dir / "decision.md").write_text(format_trace_export_markdown(report), encoding="utf-8")
    return report


def summarize_trace_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exported trace rows into stable diagnostic metrics."""

    action_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    round_bucket_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    gold_bucket_counts: Counter[str] = Counter()
    hp_bucket_counts: Counter[str] = Counter()
    episode_outcomes: dict[int, dict[str, Any]] = {}

    for row in rows:
        action_counts[str(row["action_name"])] += 1
        category_counts[str(row["action_category"])] += 1
        round_bucket_counts[str(row["round_bucket"])] += 1
        level_counts[str(row["level"])] += 1
        gold_bucket_counts[str(row["gold_bucket"])] += 1
        hp_bucket_counts[str(row["hp_bucket"])] += 1
        outcome = row.get("outcome", {})
        if outcome:
            episode = int(row["episode"])
            previous = episode_outcomes.get(episode)
            if previous is not None and previous != outcome:
                raise ValueError(f"conflicting outcomes for episode {episode}")
            episode_outcomes[episode] = dict(outcome)

    decisions = len(rows)
    placement_values = [
        float(outcome["placement"])
        for outcome in episode_outcomes.values()
        if "placement" in outcome
    ]
    final_hp_values = [
        float(outcome["final_hp"])
        for outcome in episode_outcomes.values()
        if "final_hp" in outcome
    ]
    board_strength_values = [
        float(outcome["final_board_strength"])
        for outcome in episode_outcomes.values()
        if "final_board_strength" in outcome
    ]
    return {
        "decisions": decisions,
        "episodes": len({int(row["episode"]) for row in rows}) if rows else 0,
        "outcome_episodes": len(episode_outcomes),
        "outcome_aggregation": "episode",
        "action_counts": dict(sorted(action_counts.items())),
        "action_rates": _rates(action_counts, decisions),
        "category_counts": dict(sorted(category_counts.items())),
        "category_rates": _rates(category_counts, decisions),
        "round_bucket_counts": dict(sorted(round_bucket_counts.items())),
        "round_bucket_rates": _rates(round_bucket_counts, decisions),
        "level_counts": dict(sorted(level_counts.items())),
        "level_rates": _rates(level_counts, decisions),
        "gold_bucket_counts": dict(sorted(gold_bucket_counts.items())),
        "gold_bucket_rates": _rates(gold_bucket_counts, decisions),
        "hp_bucket_counts": dict(sorted(hp_bucket_counts.items())),
        "hp_bucket_rates": _rates(hp_bucket_counts, decisions),
        "mean_placement": _mean_or_none(placement_values),
        "mean_final_hp": _mean_or_none(final_hp_values),
        "mean_final_board_strength": _mean_or_none(board_strength_values),
        "legal_roll_rate": _mean_bool(rows, "legal_roll"),
        "legal_buy_xp_rate": _mean_bool(rows, "legal_buy_xp"),
        "chosen_roll_rate": _action_rate(rows, int(Action.ROLL)),
        "chosen_buy_xp_rate": _action_rate(rows, int(Action.BUY_XP)),
        "chosen_end_turn_rate": _action_rate(rows, int(Action.END_TURN)),
        "chosen_buy_shop_rate": _category_rate(rows, "buy_shop"),
    }


def compare_trace_files(
    *,
    baseline_trace: Path,
    candidate_trace: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare two exported trace JSONL files."""

    baseline_rows = _read_jsonl(baseline_trace)
    candidate_rows = _read_jsonl(candidate_trace)
    baseline = summarize_trace_rows(baseline_rows)
    candidate = summarize_trace_rows(candidate_rows)
    action_rate_deltas = _metric_deltas(
        baseline.get("action_rates", {}),
        candidate.get("action_rates", {}),
    )
    category_rate_deltas = _metric_deltas(
        baseline.get("category_rates", {}),
        candidate.get("category_rates", {}),
    )
    bucket_rate_deltas = {
        "round": _metric_deltas(
            baseline.get("round_bucket_rates", {}),
            candidate.get("round_bucket_rates", {}),
        ),
        "level": _metric_deltas(
            baseline.get("level_rates", {}),
            candidate.get("level_rates", {}),
        ),
        "gold": _metric_deltas(
            baseline.get("gold_bucket_rates", {}),
            candidate.get("gold_bucket_rates", {}),
        ),
        "hp": _metric_deltas(
            baseline.get("hp_bucket_rates", {}),
            candidate.get("hp_bucket_rates", {}),
        ),
    }
    scalar_deltas = {
        key: _nullable_delta(baseline.get(key), candidate.get(key))
        for key in (
            "mean_placement",
            "mean_final_hp",
            "mean_final_board_strength",
            "chosen_roll_rate",
            "chosen_buy_xp_rate",
            "chosen_end_turn_rate",
            "chosen_buy_shop_rate",
            "legal_roll_rate",
            "legal_buy_xp_rate",
        )
    }
    report: dict[str, Any] = {
        "status": "pass",
        "kind": "lobby_trace_compare",
        "baseline_trace": str(baseline_trace),
        "candidate_trace": str(candidate_trace),
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {
            "scalar": scalar_deltas,
            "action_rates": action_rate_deltas,
            "category_rates": category_rate_deltas,
            "bucket_rates": bucket_rate_deltas,
        },
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report["artifacts"] = {
            "metrics": str(out_dir / "metrics.json"),
            "decision": str(out_dir / "decision.md"),
        }
        _write_json(out_dir / "metrics.json", report)
        (out_dir / "decision.md").write_text(
            format_trace_compare_markdown(report),
            encoding="utf-8",
        )
    return report


def run_roll_value_probe(
    *,
    episodes: int = 10,
    seed: int = 1000,
    max_states: int = 100,
    min_states: int = 100,
    hero_policy_name: str = "fast_level",
    opponent_policy_name: str = "mixed",
    checkpoint: Path | None = None,
    opponent_checkpoints: Sequence[Path] = (),
    player_count: int = 8,
    max_actions_per_player: int | None = None,
    device: str = "cpu",
    checkpoint_format: CheckpointFormat = "auto",
    config: EnvConfig | None = None,
    out_dir: Path | None = None,
    hero_policy: LobbyPolicy | None = None,
    opponent_policy: LobbyPolicy | None = None,
) -> dict[str, Any]:
    """Probe roll-vs-no-roll value from cloned legal-roll lobby states."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    if min_states <= 0:
        raise ValueError("min_states must be positive")

    base_config = config or EnvConfig(seed=seed)
    resolved_hero = hero_policy or _hero_policy(
        hero_policy_name=hero_policy_name,
        checkpoint=checkpoint,
        device=device,
        checkpoint_format=checkpoint_format,
    )
    resolved_opponent = opponent_policy or _opponent_policy(
        opponent_policy_name=opponent_policy_name,
        opponent_checkpoints=opponent_checkpoints,
        device=device,
        checkpoint_format=checkpoint_format,
    )
    cases = _collect_roll_probe_cases(
        episodes=episodes,
        seed=seed,
        max_states=max_states,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=base_config,
        hero_policy=resolved_hero,
        opponent_policy=resolved_opponent,
    )
    results = [
        _evaluate_roll_probe_case(
            case,
            player_count=player_count,
            max_actions_per_player=max_actions_per_player,
            config=base_config,
            hero_policy=resolved_hero,
            opponent_policy=resolved_opponent,
        )
        for case in cases
    ]
    roll_advantages = [
        float(result["roll_advantage_vs_best_no_roll"])
        for result in results
        if result["roll_advantage_vs_best_no_roll"] is not None
    ]
    branch_values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for branch, payload in result["branches"].items():
            branch_values[branch].append(float(payload["value"]))

    report: dict[str, Any] = {
        "status": "pass" if len(results) >= min_states else "fail",
        "kind": "roll_value_probe",
        "episodes": episodes,
        "seed": seed,
        "player_count": player_count,
        "requested_states": max_states,
        "min_states": min_states,
        "sampled_states": len(results),
        "mean_roll_advantage_vs_best_no_roll": _mean_or_none(roll_advantages),
        "roll_better_than_best_no_roll_rate": _positive_rate(roll_advantages),
        "branch_mean_values": {
            branch: float(np.mean(values)) for branch, values in sorted(branch_values.items())
        },
        "results": results,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report["artifacts"] = {
            "metrics": str(out_dir / "metrics.json"),
            "decision": str(out_dir / "decision.md"),
        }
        _write_json(out_dir / "metrics.json", report)
        (out_dir / "decision.md").write_text(format_roll_probe_markdown(report), encoding="utf-8")
    return report


def format_trace_export_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# Stage 1 Lobby Trace Export",
            "",
            f"**Status:** `{report['status']}`",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Episodes | {report['episodes']} |",
            f"| Decisions | {report['decisions']} |",
            f"| Illegal actions | {report['total_illegal_actions']} |",
            f"| Mean placement | {_format_optional(summary['mean_placement'])} |",
            f"| Mean final HP | {_format_optional(summary['mean_final_hp'])} |",
            f"| Chosen roll rate | {summary['chosen_roll_rate']:.3f} |",
            f"| Chosen buy-shop rate | {summary['chosen_buy_shop_rate']:.3f} |",
            f"| Chosen buy-XP rate | {summary['chosen_buy_xp_rate']:.3f} |",
            f"| Chosen end-turn rate | {summary['chosen_end_turn_rate']:.3f} |",
            "",
        ]
    )


def format_trace_compare_markdown(report: dict[str, Any]) -> str:
    scalar = report["deltas"]["scalar"]
    lines = [
        "# Stage 1 Lobby Trace Comparison",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
        f"| Mean placement | {_format_optional(scalar['mean_placement'])} |",
        f"| Mean final HP | {_format_optional(scalar['mean_final_hp'])} |",
        f"| Buy-shop rate | {_format_optional(scalar['chosen_buy_shop_rate'])} |",
        f"| Roll rate | {_format_optional(scalar['chosen_roll_rate'])} |",
        f"| Buy-XP rate | {_format_optional(scalar['chosen_buy_xp_rate'])} |",
        f"| End-turn rate | {_format_optional(scalar['chosen_end_turn_rate'])} |",
        "",
        "## Action Category Drift",
        "",
        "| Category | Delta |",
        "| --- | ---: |",
    ]
    lines.extend(_format_delta_rows(report["deltas"]["category_rates"]))
    lines.extend(["", "## Bucket Drift", ""])
    bucket_rates = report["deltas"]["bucket_rates"]
    for label in ("round", "level", "gold", "hp"):
        lines.extend(
            [
                f"### {label.title()}",
                "",
                "| Bucket | Delta |",
                "| --- | ---: |",
                *_format_delta_rows(bucket_rates[label]),
                "",
            ]
        )
    return "\n".join(lines)


def format_roll_probe_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage 1 Roll-Value Probe",
            "",
            f"**Status:** `{report['status']}`",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Sampled states | {report['sampled_states']} |",
            f"| Min states | {report['min_states']} |",
            "| Mean roll advantage vs best no-roll | "
            f"{_format_optional(report['mean_roll_advantage_vs_best_no_roll'])} |",
            "| Roll better than best no-roll rate | "
            f"{_format_optional(report['roll_better_than_best_no_roll_rate'])} |",
            "",
        ]
    )


def _collect_roll_probe_cases(
    *,
    episodes: int,
    seed: int,
    max_states: int,
    player_count: int,
    max_actions_per_player: int | None,
    config: EnvConfig,
    hero_policy: LobbyPolicy,
    opponent_policy: LobbyPolicy,
) -> list[RollProbeCase]:
    cases: list[RollProbeCase] = []
    for episode in range(episodes):
        if len(cases) >= max_states:
            break
        episode_seed = seed + episode
        env_holder: list[MiniTFTLobbyEnv] = []
        policy = _make_roll_probe_policy(
            cases=cases,
            max_states=max_states,
            episode=episode,
            episode_seed=episode_seed,
            env_holder=env_holder,
            hero_policy=hero_policy,
            opponent_policy=opponent_policy,
        )

        env = MiniTFTLobbyEnv(
            seed=episode_seed,
            config=config,
            player_count=player_count,
            policy=policy,
            max_actions_per_player=max_actions_per_player,
            player_order_mode="random",
        )
        env_holder.append(env)
        _obs, _info = env.reset(seed=episode_seed)
        terminated = truncated = False
        while not (terminated or truncated) and len(cases) < max_states:
            _obs, _reward, terminated, truncated, _info = env.step(None)
        env.close()
    return cases


def _make_trace_policy(
    *,
    episode: int,
    episode_seed: int,
    episode_rows: list[dict[str, Any]],
    hero_policy: LobbyPolicy,
    opponent_policy: LobbyPolicy,
) -> LobbyPolicy:
    decision_index = 0

    def policy(
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        cfg: EnvConfig,
        rng: np.random.Generator,
    ) -> int:
        nonlocal decision_index
        if player_id == 0:
            chosen = int(hero_policy(player_id, state, mask, data, cfg, rng))
            episode_rows.append(
                _decision_context(
                    episode=episode,
                    episode_seed=episode_seed,
                    decision_index=decision_index,
                    state=state,
                    mask=mask,
                    data=data,
                    action=chosen,
                )
            )
            decision_index += 1
            return chosen
        return int(opponent_policy(player_id, state, mask, data, cfg, rng))

    return policy


def _make_roll_probe_policy(
    *,
    cases: list[RollProbeCase],
    max_states: int,
    episode: int,
    episode_seed: int,
    env_holder: list[MiniTFTLobbyEnv],
    hero_policy: LobbyPolicy,
    opponent_policy: LobbyPolicy,
) -> LobbyPolicy:
    decision_index = 0

    def policy(
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        cfg: EnvConfig,
        rng: np.random.Generator,
    ) -> int:
        nonlocal decision_index
        if player_id == 0:
            chosen = int(hero_policy(player_id, state, mask, data, cfg, rng))
            if mask[int(Action.ROLL)] and len(cases) < max_states:
                cases.append(
                    RollProbeCase(
                        episode=episode,
                        seed=episode_seed,
                        decision_index=decision_index,
                        context=_decision_context(
                            episode=episode,
                            episode_seed=episode_seed,
                            decision_index=decision_index,
                            state=state,
                            mask=mask,
                            data=data,
                            action=chosen,
                        ),
                        snapshot=env_holder[0].clone_state(),
                    )
                )
            decision_index += 1
            return chosen
        return int(opponent_policy(player_id, state, mask, data, cfg, rng))

    return policy


def _evaluate_roll_probe_case(
    case: RollProbeCase,
    *,
    player_count: int,
    max_actions_per_player: int | None,
    config: EnvConfig,
    hero_policy: LobbyPolicy,
    opponent_policy: LobbyPolicy,
) -> dict[str, Any]:
    legal_actions = set(int(action) for action in case.context["legal_actions"])
    branch_actions = {
        "roll": int(Action.ROLL),
        "end_turn": int(Action.END_TURN),
    }
    if int(Action.BUY_XP) in legal_actions:
        branch_actions["buy_xp"] = int(Action.BUY_XP)

    branches = {
        branch: _evaluate_forced_branch(
            case.snapshot,
            forced_action=action,
            seed=case.seed,
            player_count=player_count,
            max_actions_per_player=max_actions_per_player,
            config=config,
            hero_policy=hero_policy,
            opponent_policy=opponent_policy,
        )
        for branch, action in branch_actions.items()
        if action in legal_actions
    }
    no_roll_values = [
        float(payload["value"])
        for branch, payload in branches.items()
        if branch != "roll"
    ]
    roll_value = branches.get("roll", {}).get("value")
    roll_advantage = (
        float(roll_value) - max(no_roll_values)
        if roll_value is not None and no_roll_values
        else None
    )
    return {
        "episode": case.episode,
        "seed": case.seed,
        "decision_index": case.decision_index,
        "context": case.context,
        "branches": branches,
        "roll_advantage_vs_best_no_roll": roll_advantage,
    }


def _evaluate_forced_branch(
    snapshot: MiniTFTLobbySnapshot,
    *,
    forced_action: int,
    seed: int,
    player_count: int,
    max_actions_per_player: int | None,
    config: EnvConfig,
    hero_policy: LobbyPolicy,
    opponent_policy: LobbyPolicy,
) -> dict[str, Any]:
    forced_used = False

    def policy(
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        cfg: EnvConfig,
        rng: np.random.Generator,
    ) -> int:
        nonlocal forced_used
        if player_id == 0:
            if not forced_used:
                forced_used = True
                return forced_action
            return int(hero_policy(player_id, state, mask, data, cfg, rng))
        return int(opponent_policy(player_id, state, mask, data, cfg, rng))

    env = MiniTFTLobbyEnv(
        seed=seed,
        config=config,
        player_count=player_count,
        policy=policy,
        max_actions_per_player=max_actions_per_player,
        player_order_mode="random",
    )
    env.reset(seed=seed)
    env.restore_state(snapshot)
    terminated = truncated = False
    illegal_actions = 0
    while not (terminated or truncated):
        _obs, _reward, terminated, truncated, info = env.step(None)
        action_step = info.get("action_step_result")
        if action_step is not None:
            illegal_actions += int(action_step.illegal_actions)
    summary = env.episode_summary(0)
    value = _branch_value(summary, player_count=player_count, config=config)
    env.close()
    return {
        "action": forced_action,
        "action_name": action_name(forced_action),
        "value": value,
        "placement": _summary_int(summary, "placement"),
        "final_hp": _summary_int(summary, "final_hp"),
        "survived_round": _summary_int(summary, "survived_round"),
        "final_board_strength": _summary_float(summary, "final_board_strength"),
        "illegal_actions": illegal_actions,
    }


def _decision_context(
    *,
    episode: int,
    episode_seed: int,
    decision_index: int,
    state: Set1LobbyState,
    mask: NDArray[np.bool_],
    data: GameData,
    action: int,
) -> dict[str, Any]:
    player = state.players[0]
    legal_actions = [int(index) for index in np.flatnonzero(mask)]
    shop_costs = [
        int(data.units[unit_id].cost) if unit_id in data.units else 0 for unit_id in player.shop
    ]
    return {
        "episode": episode,
        "seed": episode_seed,
        "decision_index": decision_index,
        "round": state.round,
        "round_bucket": _round_bucket(state.round),
        "level": player.level,
        "xp": player.xp,
        "gold": player.gold,
        "gold_bucket": _gold_bucket(player.gold),
        "hp": player.hp,
        "hp_bucket": _hp_bucket(player.hp),
        "bench_size": sum(unit is not None for unit in player.bench),
        "board_size": sum(unit is not None for unit in player.board),
        "shop_unit_ids": [int(unit_id) for unit_id in player.shop],
        "shop_unit_names": [
            data.units[unit_id].name if unit_id in data.units else None for unit_id in player.shop
        ],
        "shop_costs": shop_costs,
        "shop_total_cost": int(sum(shop_costs)),
        "shop_max_cost": int(max(shop_costs) if shop_costs else 0),
        "legal_actions": legal_actions,
        "legal_action_count": len(legal_actions),
        "legal_roll": bool(mask[int(Action.ROLL)]),
        "legal_buy_xp": bool(mask[int(Action.BUY_XP)]),
        "legal_end_turn": bool(mask[int(Action.END_TURN)]),
        "legal_buy_count": sum(
            1 for index in legal_actions if is_buy_shop_action(index)
        ),
        "legal_sell_count": sum(
            1 for index in legal_actions if is_sell_bench_action(index)
        ),
        "action": int(action),
        "action_name": action_name(action),
        "action_category": _action_category(action),
        "total_rolls": player.total_rolls,
        "total_xp_buys": player.total_xp_buys,
        "total_units_bought": player.total_units_bought,
        "total_units_sold": player.total_units_sold,
    }


def _hero_policy(
    *,
    hero_policy_name: str,
    checkpoint: Path | None,
    device: str,
    checkpoint_format: CheckpointFormat = "auto",
) -> LobbyPolicy:
    if checkpoint is not None:
        return _checkpoint_policy(
            checkpoint,
            checkpoint_format=checkpoint_format,
            device=device,
        )
    return _named_policy(hero_policy_name)


def _opponent_policy(
    *,
    opponent_policy_name: str,
    opponent_checkpoints: Sequence[Path],
    device: str,
    checkpoint_format: CheckpointFormat = "auto",
) -> LobbyPolicy:
    if opponent_checkpoints:
        policies = tuple(
            _checkpoint_policy(
                checkpoint,
                checkpoint_format=checkpoint_format,
                device=device,
            )
            for checkpoint in opponent_checkpoints
        )

        def policy(
            player_id: int,
            state: Set1LobbyState,
            mask: NDArray[np.bool_],
            data: GameData,
            config: EnvConfig,
            rng: np.random.Generator,
        ) -> int:
            index = (player_id - 1) % len(policies)
            return int(policies[index](player_id, state, mask, data, config, rng))

        return policy
    return _named_policy(opponent_policy_name)


def _checkpoint_policy(
    checkpoint: Path,
    *,
    checkpoint_format: CheckpointFormat = "auto",
    device: str,
) -> LobbyPolicy:
    return load_lobby_checkpoint_policy(
        checkpoint,
        checkpoint_format=checkpoint_format,
        device=device,
    )


def _named_policy(name: str) -> LobbyPolicy:
    try:
        return LOBBY_POLICY_BY_NAME[name]
    except KeyError as exc:
        choices = ", ".join(sorted(LOBBY_POLICY_BY_NAME))
        raise ValueError(f"unknown lobby policy {name!r}; choose one of: {choices}") from exc


def _branch_value(
    summary: dict[str, int | float | str | None],
    *,
    player_count: int,
    config: EnvConfig,
) -> float:
    placement = _summary_float(summary, "placement")
    final_hp = max(0.0, _summary_float(summary, "final_hp"))
    board_strength = max(0.0, _summary_float(summary, "final_board_strength"))
    placement_score = (player_count + 1.0 - placement) / player_count
    hp_score = final_hp / max(1.0, float(config.starting_hp))
    return float(placement_score + 0.25 * hp_score + 0.001 * board_strength)


def _action_category(action: int) -> str:
    if action == int(Action.END_TURN):
        return "end_turn"
    if action == int(Action.ROLL):
        return "roll"
    if action == int(Action.BUY_XP):
        return "buy_xp"
    if is_buy_shop_action(action):
        return "buy_shop"
    if is_sell_bench_action(action):
        return "sell_bench"
    if action == int(Action.FIELD_BEST_BOARD):
        return "macro_field_best_board"
    if action == int(Action.SLAM_BEST_ITEM):
        return "macro_slam_best_item"
    if is_move_bench_to_board_action(action):
        return "move_bench_to_board"
    if is_move_board_to_bench_action(action):
        return "move_board_to_bench"
    return "unknown"


def _round_bucket(round_number: int) -> str:
    if round_number <= 10:
        return "01-10"
    if round_number <= 20:
        return "11-20"
    if round_number <= 30:
        return "21-30"
    return "31+"


def _gold_bucket(gold: int) -> str:
    if gold < 10:
        return "00-09"
    if gold < 20:
        return "10-19"
    if gold < 30:
        return "20-29"
    if gold < 50:
        return "30-49"
    return "50+"


def _hp_bucket(hp: int) -> str:
    if hp <= 0:
        return "000"
    if hp < 25:
        return "001-024"
    if hp < 50:
        return "025-049"
    if hp < 75:
        return "050-074"
    return "075+"


def _rates(counter: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {key: float(value / total) for key, value in sorted(counter.items())}


def _metric_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    keys = set(baseline) | set(candidate)
    return {
        key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in sorted(keys)
    }


def _format_delta_rows(deltas: dict[str, float]) -> list[str]:
    return [f"| {key} | {_format_optional(value)} |" for key, value in sorted(deltas.items())]


def _nullable_delta(baseline: Any, candidate: Any) -> float | None:
    if baseline is None or candidate is None:
        return None
    return float(candidate) - float(baseline)


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _positive_rate(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values) > 0.0))


def _mean_bool(rows: Sequence[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


def _action_rate(rows: Sequence[dict[str, Any]], action: int) -> float:
    if not rows:
        return 0.0
    return float(np.mean([int(row["action"]) == action for row in rows]))


def _category_rate(rows: Sequence[dict[str, Any]], category: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([str(row["action_category"]) == category for row in rows]))


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _summary_int(summary: dict[str, int | float | str | None], key: str) -> int:
    value = summary[key]
    if value is None or isinstance(value, str):
        raise RuntimeError(f"summary field {key!r} is not numeric: {value!r}")
    return int(value)


def _summary_float(summary: dict[str, int | float | str | None], key: str) -> float:
    value = summary[key]
    if value is None or isinstance(value, str):
        raise RuntimeError(f"summary field {key!r} is not numeric: {value!r}")
    return float(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-traces")
    _add_policy_args(export_parser)
    export_parser.add_argument("--out-dir", type=Path, required=True)
    export_parser.add_argument("--trace-name", default="trace")

    compare_parser = subparsers.add_parser("compare-traces")
    compare_parser.add_argument("--baseline-trace", type=Path, required=True)
    compare_parser.add_argument("--candidate-trace", type=Path, required=True)
    compare_parser.add_argument("--out-dir", type=Path, required=True)

    probe_parser = subparsers.add_parser("roll-probe")
    _add_policy_args(probe_parser)
    probe_parser.add_argument("--max-states", type=int, default=100)
    probe_parser.add_argument("--min-states", type=int, default=100)
    probe_parser.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "export-traces":
        report = run_lobby_trace_export(
            episodes=args.episodes,
            seed=args.seed,
            hero_policy_name=args.hero_policy,
            opponent_policy_name=args.opponent_policy,
            checkpoint=args.checkpoint,
            opponent_checkpoints=tuple(args.opponent_checkpoint or ()),
            player_count=args.players,
            max_actions_per_player=args.max_actions_per_player,
            device=args.device,
            checkpoint_format=args.checkpoint_format,
            out_dir=args.out_dir,
            trace_name=args.trace_name,
        )
    elif args.command == "compare-traces":
        report = compare_trace_files(
            baseline_trace=args.baseline_trace,
            candidate_trace=args.candidate_trace,
            out_dir=args.out_dir,
        )
    elif args.command == "roll-probe":
        report = run_roll_value_probe(
            episodes=args.episodes,
            seed=args.seed,
            max_states=args.max_states,
            min_states=args.min_states,
            hero_policy_name=args.hero_policy,
            opponent_policy_name=args.opponent_policy,
            checkpoint=args.checkpoint,
            opponent_checkpoints=tuple(args.opponent_checkpoint or ()),
            player_count=args.players,
            max_actions_per_player=args.max_actions_per_player,
            device=args.device,
            checkpoint_format=args.checkpoint_format,
            out_dir=args.out_dir,
        )
    else:
        raise AssertionError(f"unhandled command: {args.command}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


def _add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--hero-policy", choices=sorted(LOBBY_POLICY_BY_NAME), default="fast_level")
    parser.add_argument("--opponent-policy", choices=sorted(LOBBY_POLICY_BY_NAME), default="mixed")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "sb3", "puffer"],
        default="auto",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Frozen opponent checkpoint. Repeat to build a cycling opponent pool.",
    )
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=None)
    parser.add_argument("--device", default="cpu")


if __name__ == "__main__":
    raise SystemExit(main())
