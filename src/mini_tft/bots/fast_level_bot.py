"""Fast-level heuristic bot."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.bots.base import bot_context, end_turn, is_legal, shop_buy_action
from mini_tft.core.actions import Action
from mini_tft.core.env import MiniTFTEnv


class FastLevelBot:
    name = "FastLevelBot"

    def act(
        self,
        env: MiniTFTEnv,
        obs: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> int:
        del obs, rng
        context = bot_context(env)
        if is_legal(context.mask, Action.SLAM_BEST_ITEM):
            return int(Action.SLAM_BEST_ITEM)
        if is_legal(context.mask, Action.FIELD_BEST_BOARD):
            return int(Action.FIELD_BEST_BOARD)
        if (
            is_legal(context.mask, Action.BUY_XP)
            and (context.state.level < 6 or context.state.gold >= 50)
        ):
            return int(Action.BUY_XP)
        buy = _highest_cost_buy(env, context.mask)
        if buy is not None:
            return buy
        return end_turn()


def _highest_cost_buy(env: MiniTFTEnv, mask: NDArray[np.bool_]) -> int | None:
    assert env.state is not None
    candidates: list[tuple[float, int]] = []
    for index, unit_id in enumerate(env.state.shop):
        action = shop_buy_action(index)
        if is_legal(mask, action):
            unit = env.data.units[unit_id]
            candidates.append((unit.cost * 100.0 + unit.base_power, action))
    if not candidates:
        return None
    return max(candidates)[1]
