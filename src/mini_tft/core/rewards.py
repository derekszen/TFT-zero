"""Reward shaping for V0."""

from __future__ import annotations


def action_reward(is_end_turn: bool) -> float:
    return 0.0 if is_end_turn else -0.001


def illegal_action_reward() -> float:
    return -1.0


def end_turn_reward(
    won: bool,
    damage: int,
    board_strength_delta: float,
    terminated: bool,
    survived_max_round: bool,
    hp: int,
) -> float:
    reward = 0.5 if won else -(damage * 0.1)
    reward += board_strength_delta * 0.005
    if terminated and hp <= 0:
        reward -= 10.0
    if survived_max_round:
        reward += 20.0 + (hp * 0.2)
    return reward
