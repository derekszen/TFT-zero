"""Greedy board-strength heuristic bot."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.bots.base import bot_context, end_turn, is_legal, shop_buy_action
from mini_tft.core.actions import Action
from mini_tft.core.env import MiniTFTEnv


class GreedyBoardBot:
    name = "GreedyBoardBot"

    def act(
        self,
        env: MiniTFTEnv,
        obs: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> int:
        del obs, rng
        context = bot_context(env)
        buy = _best_affordable_shop_action(env, context.mask)
        if buy is not None:
            return buy
        if is_legal(context.mask, Action.SLAM_BEST_ITEM):
            return int(Action.SLAM_BEST_ITEM)
        if is_legal(context.mask, Action.FIELD_BEST_BOARD):
            return int(Action.FIELD_BEST_BOARD)
        return end_turn()


def _best_affordable_shop_action(
    env: MiniTFTEnv,
    mask: NDArray[np.bool_],
) -> int | None:
    assert env.state is not None
    candidates: list[tuple[float, int]] = []
    for index, unit_id in enumerate(env.state.shop):
        action = shop_buy_action(index)
        if is_legal(mask, action):
            candidates.append((env.data.units[unit_id].base_power, action))
    if not candidates:
        return None
    return max(candidates)[1]
