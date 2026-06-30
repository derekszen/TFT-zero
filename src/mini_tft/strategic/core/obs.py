"""Observation encoding for the strategic lane."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.strategic.core.state import (
    DEFAULT_STRATEGIC_CONFIG,
    ROLE_COUNT,
    UNIT_BY_ID,
    UNIT_COUNT,
    StrategicConfig,
    StrategicState,
)


def observation_dim(config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG) -> int:
    return (
        8
        + config.shop_size
        + (UNIT_COUNT + 1)
        + config.max_level
        + ROLE_COUNT
        + ROLE_COUNT
    )


def observe(
    state: StrategicState,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> NDArray[np.float32]:
    values: list[float] = [
        state.round / config.max_round,
        state.hp / config.starting_hp,
        state.gold / 100.0,
        state.level / config.max_level,
        state.xp / max(1.0, float(config.max_level * 2)),
        state.action_count / max(1, config.max_actions_per_round),
        state.last_board_strength / 200.0,
        state.last_enemy_strength / 200.0,
    ]
    values.extend((state.shop / max(UNIT_BY_ID)).astype(np.float32).tolist())
    values.extend((state.owned / 9.0).astype(np.float32).tolist())
    values.extend((state.fielded / max(UNIT_BY_ID)).astype(np.float32).tolist())
    values.extend((state.role_items / 5.0).astype(np.float32).tolist())
    values.extend((state.role_item_slots / config.max_role_item_slots).astype(np.float32).tolist())
    return np.asarray(values, dtype=np.float32)
