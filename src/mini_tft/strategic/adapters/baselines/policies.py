"""Baseline policies for the strategic lane."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mini_tft.strategic.core.actions import StrategicAction
from mini_tft.strategic.core.rules import board_strength, enemy_strength_for_round
from mini_tft.strategic.core.state import StrategicConfig, StrategicState

StrategicPolicy = Callable[[StrategicState, NDArray[np.bool_], StrategicConfig], int]


def first_legal_policy(
    _state: StrategicState,
    mask: NDArray[np.bool_],
    _config: StrategicConfig,
) -> int:
    legal = np.flatnonzero(mask)
    return int(legal[0]) if len(legal) else int(StrategicAction.HOLD)


def random_policy(
    state: StrategicState,
    mask: NDArray[np.bool_],
    _config: StrategicConfig,
) -> int:
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return int(StrategicAction.HOLD)
    index = (state.rng_key + state.round + state.action_count) % len(legal)
    return int(legal[int(index)])


def tft_heuristic_policy(
    state: StrategicState,
    mask: NDArray[np.bool_],
    config: StrategicConfig,
) -> int:
    pressure = enemy_strength_for_round(state.round, config)
    strength = board_strength(state, config)

    if mask[StrategicAction.FIELD_STRONGEST] and state.action_count >= 1:
        return int(StrategicAction.FIELD_STRONGEST)

    for action in (
        StrategicAction.SLAM_CARRY_ITEM,
        StrategicAction.SLAM_TANK_ITEM,
        StrategicAction.SLAM_SUPPORT_ITEM,
    ):
        if mask[action] and strength < pressure * 0.95:
            return int(action)

    if mask[StrategicAction.BUY_BEST_UPGRADE]:
        return int(StrategicAction.BUY_BEST_UPGRADE)
    if mask[StrategicAction.BUY_HIGHEST_COST] and (state.gold < 20 or strength < pressure * 1.05):
        return int(StrategicAction.BUY_HIGHEST_COST)
    if mask[StrategicAction.BUY_BEST_SYNERGY]:
        return int(StrategicAction.BUY_BEST_SYNERGY)
    if mask[StrategicAction.LEVEL] and _should_level(state):
        return int(StrategicAction.LEVEL)
    if mask[StrategicAction.ROLL] and state.gold >= 16 and strength < pressure * 0.85:
        return int(StrategicAction.ROLL)
    if mask[StrategicAction.FIELD_STRONGEST]:
        return int(StrategicAction.FIELD_STRONGEST)
    if (
        mask[StrategicAction.GREED_ECON]
        and state.gold >= 20
        and state.hp >= 60
        and strength >= pressure
    ):
        return int(StrategicAction.GREED_ECON)
    return int(StrategicAction.HOLD)


def _should_level(state: StrategicState) -> bool:
    if state.level < 4 and state.round >= 3:
        return True
    if state.level < 6 and state.round >= 9 and state.gold >= 8:
        return True
    if state.level < 8 and state.round >= 18 and state.gold >= 20:
        return True
    return False
