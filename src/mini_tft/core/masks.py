"""Legal action masking."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import (
    BUY_SHOP_OFFSET,
    NUM_ACTIONS,
    SELL_BENCH_OFFSET,
    Action,
    move_bench_to_board_action,
    move_board_to_bench_action,
)
from mini_tft.core.board import would_change_best_board
from mini_tft.core.config import EnvConfig
from mini_tft.core.ids import EMPTY
from mini_tft.core.items import first_combinable_recipe, is_completed_item
from mini_tft.core.set_data import GameData
from mini_tft.core.state import GameState


def legal_action_mask(
    state: GameState,
    data: GameData,
    config: EnvConfig,
) -> NDArray[np.bool_]:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    if state.done:
        return mask

    mask[Action.END_TURN] = True
    board_count = sum(unit is not None for unit in state.board)
    has_owned_unit = board_count > 0 or any(unit is not None for unit in state.bench)
    mask[Action.ROLL] = state.gold >= config.roll_cost
    mask[Action.BUY_XP] = (
        state.gold >= config.xp_buy_cost
        and state.level < config.max_level
        and has_owned_unit
    )

    has_bench_space = any(slot is None for slot in state.bench)
    for shop_index, unit_id in enumerate(state.shop):
        action_index = BUY_SHOP_OFFSET + shop_index
        if unit_id != EMPTY and has_bench_space and state.gold >= data.units[unit_id].cost:
            mask[action_index] = True

    for bench_index, unit in enumerate(state.bench):
        action_index = SELL_BENCH_OFFSET + bench_index
        if unit is not None:
            mask[action_index] = True

    has_units = any(unit is not None for unit in [*state.bench, *state.board])
    mask[Action.FIELD_BEST_BOARD] = has_units and would_change_best_board(state, data, config)

    has_completed_item = any(is_completed_item(item_id, data) for item_id in state.item_bench)
    has_combinable_components = first_combinable_recipe(state.item_bench, data) is not None
    has_target = any(
        unit is not None and len(unit.items) < config.max_items_on_unit
        for unit in state.board
    )
    mask[Action.SLAM_BEST_ITEM] = (has_completed_item and has_target) or has_combinable_components

    has_board_room = board_count < state.level
    for bench_index, unit in enumerate(state.bench):
        if unit is None:
            continue
        for board_index, board_unit in enumerate(state.board):
            if board_unit is not None or has_board_room:
                mask[move_bench_to_board_action(bench_index, board_index)] = True

    for board_index, unit in enumerate(state.board):
        if unit is None:
            continue
        for bench_index in range(len(state.bench)):
            mask[move_board_to_bench_action(board_index, bench_index)] = True
    return mask
