"""Evaluate one hero policy in a simplified 8-player MiniTFT lobby."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import Action
from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import Set1LobbyState
from mini_tft.core.lobby_step import (
    LobbyActionRecord,
    LobbyPolicy,
    fast_level_lobby_policy,
    mixed_lobby_policy,
    random_lobby_policy,
    tempo_lobby_policy,
)
from mini_tft.core.masks import mask_without_oracle_macro_actions
from mini_tft.core.set_data import GameData
from mini_tft.rl.checkpoint_policy import (
    CheckpointFormat,
    load_lobby_checkpoint_policy,
    resolve_checkpoint_format,
)
from mini_tft.rl.lobby_env import MiniTFTLobbyEnv

POLICY_BY_NAME: dict[str, LobbyPolicy] = {
    "fast_level": fast_level_lobby_policy,
    "mixed": mixed_lobby_policy,
    "tempo": tempo_lobby_policy,
    "random": random_lobby_policy,
}


def run_lobby_evaluation(
    *,
    episodes: int = 100,
    seed: int = 1000,
    hero_policy_name: str = "fast_level",
    opponent_policy_name: str = "fast_level",
    checkpoint: Path | None = None,
    checkpoint_format: CheckpointFormat = "auto",
    player_count: int = 8,
    max_actions_per_player: int | None = None,
    device: str = "cpu",
    config: EnvConfig | None = None,
    allow_hero_macro_actions: bool = True,
    max_hero_macro_action_rate: float | None = None,
) -> dict[str, Any]:
    """Run seeded lobbies and summarize player-0 placement and HP metrics."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if player_count < 2:
        raise ValueError("player_count must be at least 2")
    if (
        max_hero_macro_action_rate is not None
        and not 0.0 <= max_hero_macro_action_rate <= 1.0
    ):
        raise ValueError("max_hero_macro_action_rate must be between 0.0 and 1.0")

    base_config = config or EnvConfig(seed=seed)
    resolved_checkpoint_format = (
        resolve_checkpoint_format(checkpoint, checkpoint_format)
        if checkpoint is not None
        else None
    )
    hero_policy = (
        load_lobby_checkpoint_policy(
            checkpoint,
            checkpoint_format=checkpoint_format,
            device=device,
        )
        if checkpoint is not None
        else _named_policy(hero_policy_name)
    )
    if not allow_hero_macro_actions:
        hero_policy = _macro_filtered_policy(hero_policy)
    opponent_policy = _named_policy(opponent_policy_name)
    mixed_policy = _mixed_policy(hero_policy, opponent_policy)

    hero_summaries: list[dict[str, Any]] = []
    lobby_rounds: list[int] = []
    lobby_final_hp: list[float] = []
    lobby_final_placements: list[int] = []
    total_actions = 0
    total_fights = 0
    total_illegal_actions = 0
    hero_actions = 0
    total_field_best_board_actions = 0
    total_slam_best_item_actions = 0
    hero_field_best_board_actions = 0
    hero_slam_best_item_actions = 0

    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        env = MiniTFTLobbyEnv(
            seed=episode_seed,
            config=base_config,
            player_count=player_count,
            policy=mixed_policy,
            max_actions_per_player=max_actions_per_player,
            player_order_mode="random",
        )
        _obs, _info = env.reset(seed=episode_seed)
        terminated = truncated = False
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            _obs, _reward, terminated, truncated, last_info = env.step(None)
            action_step = last_info.get("action_step_result")
            if action_step is not None:
                total_actions += int(action_step.action_count)
                total_fights += int(action_step.resolved_fights)
                total_illegal_actions += int(action_step.illegal_actions)
                hero_actions += _count_actions(action_step.actions, player_id=0)
                total_field_best_board_actions += _count_actions(
                    action_step.actions,
                    action=int(Action.FIELD_BEST_BOARD),
                )
                total_slam_best_item_actions += _count_actions(
                    action_step.actions,
                    action=int(Action.SLAM_BEST_ITEM),
                )
                hero_field_best_board_actions += _count_actions(
                    action_step.actions,
                    action=int(Action.FIELD_BEST_BOARD),
                    player_id=0,
                )
                hero_slam_best_item_actions += _count_actions(
                    action_step.actions,
                    action=int(Action.SLAM_BEST_ITEM),
                    player_id=0,
                )

        state = env.state
        if state is None:
            raise RuntimeError("lobby evaluation finished without state")
        if 0 not in state.placements:
            raise RuntimeError("lobby evaluation finished without hero placement")

        hero_summaries.append(env.episode_summary(0))
        lobby_rounds.append(state.round)
        all_summaries: list[dict[str, Any]] = env.lobby_summaries()
        lobby_final_hp.extend(float(summary["final_hp"]) for summary in all_summaries)
        lobby_final_placements.extend(int(summary["placement"]) for summary in all_summaries)
        env.close()

    placements = np.asarray([summary["placement"] for summary in hero_summaries], dtype=float)
    final_hp = np.asarray([summary["final_hp"] for summary in hero_summaries], dtype=float)
    survived_rounds = np.asarray(
        [summary["survived_round"] for summary in hero_summaries],
        dtype=float,
    )
    board_strengths = np.asarray(
        [summary["final_board_strength"] for summary in hero_summaries],
        dtype=float,
    )
    placement_histogram = {
        str(placement): int(np.sum(placements == placement))
        for placement in range(1, player_count + 1)
    }
    total_macro_actions = total_field_best_board_actions + total_slam_best_item_actions
    hero_macro_actions = hero_field_best_board_actions + hero_slam_best_item_actions
    hero_macro_action_rate = _rate(hero_macro_actions, hero_actions)
    status = (
        "fail"
        if (
            max_hero_macro_action_rate is not None
            and hero_macro_action_rate > max_hero_macro_action_rate
        )
        else "pass"
    )

    return {
        "status": status,
        "episodes": episodes,
        "seed": seed,
        "player_count": player_count,
        "hero_policy": str(checkpoint) if checkpoint is not None else hero_policy_name,
        "checkpoint_format": resolved_checkpoint_format,
        "device": device,
        "opponent_policy": opponent_policy_name,
        "allow_hero_macro_actions": allow_hero_macro_actions,
        "max_hero_macro_action_rate": max_hero_macro_action_rate,
        "mean_placement": float(np.mean(placements)),
        "median_placement": float(np.median(placements)),
        "top1_rate": float(np.mean(placements == 1)),
        "top4_rate": float(np.mean(placements <= min(4, player_count))),
        "mean_final_hp": float(np.mean(final_hp)),
        "mean_survived_round": float(np.mean(survived_rounds)),
        "mean_final_board_strength": float(np.mean(board_strengths)),
        "mean_lobby_rounds": float(np.mean(lobby_rounds)),
        "mean_lobby_final_hp": float(np.mean(lobby_final_hp)),
        "mean_lobby_placement": float(np.mean(lobby_final_placements)),
        "total_actions": total_actions,
        "total_fights": total_fights,
        "total_illegal_actions": total_illegal_actions,
        "hero_actions": hero_actions,
        "total_macro_actions": total_macro_actions,
        "total_field_best_board_actions": total_field_best_board_actions,
        "total_slam_best_item_actions": total_slam_best_item_actions,
        "hero_macro_actions": hero_macro_actions,
        "hero_field_best_board_actions": hero_field_best_board_actions,
        "hero_slam_best_item_actions": hero_slam_best_item_actions,
        "hero_macro_action_rate": hero_macro_action_rate,
        "placement_histogram": placement_histogram,
    }


def format_markdown(report: dict[str, Any]) -> str:
    histogram = report["placement_histogram"]
    rows = [
        "# MiniTFT Lobby Policy Evaluation",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Episodes | {report['episodes']} |",
        f"| Players | {report['player_count']} |",
        f"| Hero policy | `{report['hero_policy']}` |",
        f"| Checkpoint format | `{report['checkpoint_format']}` |",
        f"| Opponent policy | `{report['opponent_policy']}` |",
        f"| Mean placement | {report['mean_placement']:.3f} |",
        f"| Median placement | {report['median_placement']:.3f} |",
        f"| Top-1 rate | {report['top1_rate']:.3f} |",
        f"| Top-4 rate | {report['top4_rate']:.3f} |",
        f"| Mean final HP | {report['mean_final_hp']:.3f} |",
        f"| Mean survived round | {report['mean_survived_round']:.3f} |",
        f"| Mean final board strength | {report['mean_final_board_strength']:.3f} |",
        f"| Mean lobby rounds | {report['mean_lobby_rounds']:.3f} |",
        f"| Total actions | {report['total_actions']} |",
        f"| Hero actions | {report['hero_actions']} |",
        f"| Hero macro actions allowed | {report['allow_hero_macro_actions']} |",
        f"| Total macro actions | {report['total_macro_actions']} |",
        f"| Total field_best_board actions | {report['total_field_best_board_actions']} |",
        f"| Total slam_best_item actions | {report['total_slam_best_item_actions']} |",
        f"| Hero macro actions | {report['hero_macro_actions']} |",
        f"| Hero field_best_board actions | {report['hero_field_best_board_actions']} |",
        f"| Hero slam_best_item actions | {report['hero_slam_best_item_actions']} |",
        f"| Hero macro action rate | {report['hero_macro_action_rate']:.3f} |",
        f"| Total fights | {report['total_fights']} |",
        f"| Illegal actions | {report['total_illegal_actions']} |",
        "",
        "## Placement Histogram",
        "",
        "| Placement | Count |",
        "| ---: | ---: |",
    ]
    rows.extend(f"| {placement} | {count} |" for placement, count in histogram.items())
    rows.append("")
    return "\n".join(rows)


def _named_policy(name: str) -> LobbyPolicy:
    try:
        return POLICY_BY_NAME[name]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICY_BY_NAME))
        raise ValueError(f"unknown lobby policy {name!r}; choose one of: {choices}") from exc


def _mixed_policy(hero_policy: LobbyPolicy, opponent_policy: LobbyPolicy) -> LobbyPolicy:
    def policy(
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        config: EnvConfig,
        rng: np.random.Generator,
    ) -> int:
        if player_id == 0:
            return hero_policy(player_id, state, mask, data, config, rng)
        return opponent_policy(player_id, state, mask, data, config, rng)

    return policy


def _macro_filtered_policy(policy: LobbyPolicy) -> LobbyPolicy:
    def filtered_policy(
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        config: EnvConfig,
        rng: np.random.Generator,
    ) -> int:
        return policy(
            player_id,
            state,
            mask_without_oracle_macro_actions(mask),
            data,
            config,
            rng,
        )

    return filtered_policy


def _count_actions(
    actions: Iterable[LobbyActionRecord],
    *,
    action: int | None = None,
    player_id: int | None = None,
) -> int:
    return sum(
        1
        for record in actions
        if (action is None or record.action == action)
        and (player_id is None or record.player_id == player_id)
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--hero-policy", choices=sorted(POLICY_BY_NAME), default="fast_level")
    parser.add_argument("--opponent-policy", choices=sorted(POLICY_BY_NAME), default="fast_level")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "sb3", "puffer"],
        default="auto",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=None)
    parser.add_argument(
        "--disallow-hero-macro-actions",
        action="store_true",
        help="Remove field_best_board and slam_best_item from the evaluated hero policy mask.",
    )
    parser.add_argument(
        "--max-hero-macro-action-rate",
        type=float,
        default=None,
        help="Fail the report when player-0 selected macro action rate exceeds this value.",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_lobby_evaluation(
        episodes=args.episodes,
        seed=args.seed,
        hero_policy_name=args.hero_policy,
        opponent_policy_name=args.opponent_policy,
        checkpoint=args.checkpoint,
        checkpoint_format=args.checkpoint_format,
        player_count=args.players,
        max_actions_per_player=args.max_actions_per_player,
        device=args.device,
        allow_hero_macro_actions=not args.disallow_hero_macro_actions,
        max_hero_macro_action_rate=args.max_hero_macro_action_rate,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_markdown(report), end="")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
