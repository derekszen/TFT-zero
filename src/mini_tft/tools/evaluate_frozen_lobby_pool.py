"""Evaluate a PPO hero against a frozen checkpoint opponent pool."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import Action
from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import Set1LobbyState
from mini_tft.core.lobby_step import LobbyActionRecord, LobbyPolicy
from mini_tft.core.set_data import GameData
from mini_tft.rl.checkpoint_policy import CheckpointFormat, load_lobby_checkpoint_policy
from mini_tft.rl.lobby_env import MiniTFTLobbyEnv


def run_frozen_lobby_pool_evaluation(
    *,
    hero_checkpoint: Path,
    opponent_checkpoints: Sequence[Path],
    episodes: int = 100,
    seed: int = 1000,
    player_count: int = 8,
    max_actions_per_player: int | None = None,
    device: str = "cpu",
    hero_checkpoint_format: CheckpointFormat = "auto",
    opponent_checkpoint_format: CheckpointFormat = "auto",
    name: str = "frozen_lobby_pool",
    config: EnvConfig | None = None,
) -> dict[str, Any]:
    """Run seeded lobbies with player 0 against cycling frozen PPO checkpoints."""

    hero_policy = _checkpoint_policy(
        hero_checkpoint,
        checkpoint_format=hero_checkpoint_format,
        device=device,
    )
    opponent_policies = tuple(
        _checkpoint_policy(
            checkpoint,
            checkpoint_format=opponent_checkpoint_format,
            device=device,
        )
        for checkpoint in opponent_checkpoints
    )
    return run_lobby_policy_pool_evaluation(
        name=name,
        hero_policy=hero_policy,
        hero_policy_name=str(hero_checkpoint),
        opponent_policies=opponent_policies,
        opponent_policy_names=[str(checkpoint) for checkpoint in opponent_checkpoints],
        episodes=episodes,
        seed=seed,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=config,
        device=device,
    )


def run_lobby_policy_pool_evaluation(
    *,
    name: str,
    hero_policy: LobbyPolicy,
    hero_policy_name: str,
    opponent_policies: Sequence[LobbyPolicy],
    opponent_policy_names: Sequence[str],
    episodes: int = 100,
    seed: int = 1000,
    player_count: int = 8,
    max_actions_per_player: int | None = None,
    config: EnvConfig | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run seeded lobbies with an explicit player-0 policy and opponent pool."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if player_count < 2:
        raise ValueError("player_count must be at least 2")
    if not opponent_policies:
        raise ValueError("opponent_policies must not be empty")
    if len(opponent_policies) != len(opponent_policy_names):
        raise ValueError("opponent_policies and opponent_policy_names must match")

    base_config = config or EnvConfig(seed=seed)
    pool_policy = _pooled_policy(hero_policy, tuple(opponent_policies))

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
            policy=pool_policy,
            max_actions_per_player=max_actions_per_player,
            player_order_mode="random",
        )
        _obs, _info = env.reset(seed=episode_seed)
        terminated = truncated = False
        last_info: dict[str, Any] = {}
        while not (terminated or truncated):
            _obs, _reward, terminated, truncated, last_info = env.step(None)
            action_step = last_info.get("action_step_result")
            if action_step is None:
                continue
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
            raise RuntimeError("lobby pool evaluation finished without state")
        if 0 not in state.placements:
            raise RuntimeError("lobby pool evaluation finished without hero placement")

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
    total_macro_actions = total_field_best_board_actions + total_slam_best_item_actions
    hero_macro_actions = hero_field_best_board_actions + hero_slam_best_item_actions

    return {
        "status": "pass",
        "name": name,
        "episodes": episodes,
        "seed": seed,
        "player_count": player_count,
        "device": device,
        "hero_policy": hero_policy_name,
        "opponent_pool": list(opponent_policy_names),
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
        "hero_macro_action_rate": _rate(hero_macro_actions, hero_actions),
        "placement_histogram": {
            str(placement): int(np.sum(placements == placement))
            for placement in range(1, player_count + 1)
        },
    }


def format_markdown(report: dict[str, Any]) -> str:
    histogram = report["placement_histogram"]
    rows = [
        "# MiniTFT Frozen Lobby Pool Evaluation",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Name | `{report['name']}` |",
        f"| Episodes | {report['episodes']} |",
        f"| Players | {report['player_count']} |",
        f"| Device | `{report['device']}` |",
        f"| Hero policy | `{report['hero_policy']}` |",
        f"| Opponent pool size | {len(report['opponent_pool'])} |",
        f"| Mean placement | {report['mean_placement']:.3f} |",
        f"| Median placement | {report['median_placement']:.3f} |",
        f"| Top-1 rate | {report['top1_rate']:.3f} |",
        f"| Top-4 rate | {report['top4_rate']:.3f} |",
        f"| Mean final HP | {report['mean_final_hp']:.3f} |",
        f"| Mean survived round | {report['mean_survived_round']:.3f} |",
        f"| Mean final board strength | {report['mean_final_board_strength']:.3f} |",
        f"| Mean lobby rounds | {report['mean_lobby_rounds']:.3f} |",
        f"| Hero actions | {report['hero_actions']} |",
        f"| Hero macro actions | {report['hero_macro_actions']} |",
        f"| Hero macro action rate | {report['hero_macro_action_rate']:.3f} |",
        f"| Total fights | {report['total_fights']} |",
        f"| Illegal actions | {report['total_illegal_actions']} |",
        "",
        "## Opponent Pool",
        "",
    ]
    rows.extend(f"- `{checkpoint}`" for checkpoint in report["opponent_pool"])
    rows.extend(
        [
            "",
            "## Placement Histogram",
            "",
            "| Placement | Count |",
            "| ---: | ---: |",
        ]
    )
    rows.extend(f"| {placement} | {count} |" for placement, count in histogram.items())
    rows.append("")
    return "\n".join(rows)


def _pooled_policy(
    hero_policy: LobbyPolicy,
    opponent_policies: tuple[LobbyPolicy, ...],
) -> LobbyPolicy:
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
        opponent_policy = opponent_policies[(player_id - 1) % len(opponent_policies)]
        return opponent_policy(player_id, state, mask, data, config, rng)

    return policy


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
    parser.add_argument("--hero-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Frozen opponent checkpoint. Repeat to build a cycling opponent pool.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--hero-checkpoint-format",
        choices=["auto", "sb3", "puffer"],
        default="auto",
    )
    parser.add_argument(
        "--opponent-checkpoint-format",
        choices=["auto", "sb3", "puffer"],
        default="auto",
    )
    parser.add_argument("--name", default="frozen_lobby_pool")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_frozen_lobby_pool_evaluation(
        name=args.name,
        hero_checkpoint=args.hero_checkpoint,
        opponent_checkpoints=args.opponent_checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        player_count=args.players,
        max_actions_per_player=args.max_actions_per_player,
        device=args.device,
        hero_checkpoint_format=args.hero_checkpoint_format,
        opponent_checkpoint_format=args.opponent_checkpoint_format,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
