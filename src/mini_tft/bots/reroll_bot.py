"""Low-cost reroll heuristic bot."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.bots.base import bot_context, end_turn, is_legal, owned_unit_ids, shop_buy_action
from mini_tft.core.actions import Action
from mini_tft.core.env import MiniTFTEnv


class RerollBot:
    name = "RerollBot"

    def act(
        self,
        env: MiniTFTEnv,
        obs: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> int:
        del obs, rng
        context = bot_context(env)
        buy = _low_cost_copy_buy(env, context.mask)
        if buy is not None:
            return buy
        if is_legal(context.mask, Action.FIELD_BEST_BOARD):
            return int(Action.FIELD_BEST_BOARD)
        if (
            context.state.level <= 5
            and context.state.gold > 50
            and is_legal(context.mask, Action.ROLL)
        ):
            return int(Action.ROLL)
        return end_turn()


def _low_cost_copy_buy(env: MiniTFTEnv, mask: NDArray[np.bool_]) -> int | None:
    assert env.state is not None
    owned = set(owned_unit_ids(env.state))
    candidates: list[tuple[float, int]] = []
    for index, unit_id in enumerate(env.state.shop):
        action = shop_buy_action(index)
        if not is_legal(mask, action):
            continue
        unit = env.data.units[unit_id]
        if unit.cost <= 2 or unit_id in owned:
            candidates.append((10.0 if unit_id in owned else 0.0, action))
    if not candidates:
        return None
    return max(candidates)[1]
