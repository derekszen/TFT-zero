"""Economy-first heuristic bot."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.bots.base import bot_context, end_turn, is_legal, owned_unit_ids, shop_buy_action
from mini_tft.core.actions import Action
from mini_tft.core.env import MiniTFTEnv


class EconBot:
    name = "EconBot"

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
        buy = _pair_or_synergy_buy(env, context.mask)
        if buy is not None:
            return buy
        if context.state.gold >= 50 and is_legal(context.mask, Action.BUY_XP):
            return int(Action.BUY_XP)
        return end_turn()


def _pair_or_synergy_buy(env: MiniTFTEnv, mask: NDArray[np.bool_]) -> int | None:
    assert env.state is not None
    owned = owned_unit_ids(env.state)
    owned_traits = {
        trait
        for unit_id in owned
        for trait in env.data.units[unit_id].traits
    }
    candidates: list[tuple[float, int]] = []
    for index, unit_id in enumerate(env.state.shop):
        action = shop_buy_action(index)
        if not is_legal(mask, action):
            continue
        unit = env.data.units[unit_id]
        pair_bonus = 10.0 if unit_id in owned else 0.0
        trait_bonus = sum(1.0 for trait in unit.traits if trait in owned_traits)
        if pair_bonus or trait_bonus:
            candidates.append((pair_bonus + trait_bonus + unit.base_power / 10.0, action))
    if not candidates:
        return None
    return max(candidates)[1]
