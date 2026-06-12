"""Smoke benchmark for all-player Set-1 lobby stepping."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import lobby_player_summary, new_lobby_state
from mini_tft.core.lobby_step import step_lobby_round_with_policies, tempo_lobby_policy
from mini_tft.core.set_data import load_set


def run_lobby_step_smoke(
    *,
    lobbies: int = 100,
    max_rounds: int = 8,
    max_actions_per_player: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    config = EnvConfig(
        seed=seed,
        starting_gold=10,
        max_actions_per_round=max_actions_per_player,
    )
    data = load_set(config.dataset)
    started = time.perf_counter()
    resolved_rounds = 0
    resolved_fights = 0
    actions = 0
    illegal_actions = 0
    forced_players = 0
    final_active: list[int] = []
    final_hp: list[int] = []
    final_board_strength: list[float] = []

    for lobby_index in range(lobbies):
        lobby_seed = seed + lobby_index * 10_000
        rng = np.random.default_rng(lobby_seed)
        lobby = new_lobby_state(config, data, seed=lobby_seed)

        for _ in range(max_rounds):
            if lobby.done:
                break
            result = step_lobby_round_with_policies(
                lobby,
                data,
                config,
                rng,
                tempo_lobby_policy,
                max_actions_per_player=max_actions_per_player,
                player_order_mode="random",
            )
            resolved_rounds += 1
            resolved_fights += result.resolved_fights
            actions += result.action_count
            illegal_actions += result.illegal_actions
            forced_players += len(result.forced_players)

        final_active.append(len(lobby.active_player_ids()))
        for player_id in range(lobby.player_count):
            summary = lobby_player_summary(lobby, player_id, data)
            final_hp.append(summary.final_hp)
            final_board_strength.append(summary.final_board_strength)

    elapsed = time.perf_counter() - started
    status = "pass"
    if resolved_rounds == 0 or resolved_fights == 0 or actions == 0 or illegal_actions:
        status = "fail"

    return {
        "status": status,
        "lobbies": lobbies,
        "max_rounds": max_rounds,
        "max_actions_per_player": max_actions_per_player,
        "resolved_rounds": resolved_rounds,
        "resolved_fights": resolved_fights,
        "actions": actions,
        "illegal_actions": illegal_actions,
        "forced_players": forced_players,
        "elapsed_sec": elapsed,
        "rounds_per_sec": resolved_rounds / elapsed if elapsed else 0.0,
        "fights_per_sec": resolved_fights / elapsed if elapsed else 0.0,
        "actions_per_sec": actions / elapsed if elapsed else 0.0,
        "mean_actions_per_player_round": (
            actions / (resolved_rounds * 8) if resolved_rounds else 0.0
        ),
        "mean_final_active_players": float(np.mean(final_active)) if final_active else 0.0,
        "mean_final_hp": float(np.mean(final_hp)) if final_hp else 0.0,
        "mean_final_board_strength": (
            float(np.mean(final_board_strength)) if final_board_strength else 0.0
        ),
    }


def format_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Set1 Lobby 8-Player Step Smoke Report",
            "",
            f"**Status:** `{report['status']}`",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Lobbies | {report['lobbies']} |",
            f"| Max rounds per lobby | {report['max_rounds']} |",
            f"| Max actions per player | {report['max_actions_per_player']} |",
            f"| Resolved lobby rounds | {report['resolved_rounds']} |",
            f"| Resolved pair fights | {report['resolved_fights']} |",
            f"| Player actions | {report['actions']} |",
            f"| Illegal actions | {report['illegal_actions']} |",
            f"| Forced player ends | {report['forced_players']} |",
            f"| Seconds | {report['elapsed_sec']:.6f} |",
            f"| Rounds/sec | {report['rounds_per_sec']:.1f} |",
            f"| Fights/sec | {report['fights_per_sec']:.1f} |",
            f"| Actions/sec | {report['actions_per_sec']:.1f} |",
            f"| Mean actions/player-round | {report['mean_actions_per_player_round']:.2f} |",
            f"| Mean final active players | {report['mean_final_active_players']:.2f} |",
            f"| Mean final HP | {report['mean_final_hp']:.2f} |",
            f"| Mean final board strength | {report['mean_final_board_strength']:.2f} |",
            "",
        ]
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lobbies", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_lobby_step_smoke(
        lobbies=args.lobbies,
        max_rounds=args.max_rounds,
        max_actions_per_player=args.max_actions_per_player,
        seed=args.seed,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_markdown(report), end="")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
