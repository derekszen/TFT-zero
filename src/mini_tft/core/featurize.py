"""Flat observation encoding."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData
from mini_tft.core.state import GameState, UnitInstance
from mini_tft.core.traits import trait_count_vector

OBS_CLIP_LOW = -10.0
OBS_CLIP_HIGH = 10.0


def observation_dim(data: GameData, config: EnvConfig) -> int:
    return (
        5
        + config.shop_size
        + config.bench_size
        + config.bench_size
        + config.max_level
        + config.max_level
        + config.item_bench_size
        + len(data.trait_ids)
    )


def featurize_state(
    state: GameState,
    data: GameData,
    config: EnvConfig,
) -> NDArray[np.float32]:
    values: list[float] = [
        state.round / config.max_round,
        state.hp / config.starting_hp,
        state.gold / 100.0,
        state.level / config.max_level,
        state.xp / 100.0,
    ]
    values.extend(_unit_ids(state.shop, data, config.shop_size))
    bench_ids = [unit.unit_id if unit else 0 for unit in state.bench]
    board_ids = [unit.unit_id if unit else 0 for unit in state.board]
    values.extend(_unit_ids(bench_ids, data, config.bench_size))
    values.extend(_stars(state.bench, config.bench_size))
    values.extend(_unit_ids(board_ids, data, config.max_level))
    values.extend(_stars(state.board, config.max_level))
    values.extend(_item_ids(state.item_bench, data, config.item_bench_size))
    values.extend(trait_count_vector(state.board, data))
    return np.asarray(values, dtype=np.float32)


def _unit_ids(unit_ids: list[int], data: GameData, size: int) -> list[float]:
    padded = [*unit_ids[:size], *([0] * max(0, size - len(unit_ids)))]
    return [unit_id / data.max_unit_id for unit_id in padded]


def _stars(units: list[UnitInstance | None], size: int) -> list[float]:
    padded = [*units[:size], *([None] * max(0, size - len(units)))]
    return [0.0 if unit is None else unit.stars / 3.0 for unit in padded]


def _item_ids(item_ids: list[int], data: GameData, size: int) -> list[float]:
    padded = [*item_ids[:size], *([0] * max(0, size - len(item_ids)))]
    return [item_id / data.max_item_id for item_id in padded]
