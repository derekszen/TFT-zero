"""Trait-commit heuristic bot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mini_tft.bots.base import bot_context, end_turn, is_legal, shop_buy_action
from mini_tft.core.actions import Action
from mini_tft.core.env import MiniTFTEnv


@dataclass
class TraitCommitBot:
    target_trait: str = "ranger"
    name: str = "TraitCommitBot"

    def act(
        self,
        env: MiniTFTEnv,
        obs: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> int:
        del obs, rng
        context = bot_context(env)
        buy = self._target_trait_buy(env, context.mask)
        if buy is not None:
            return buy
        if is_legal(context.mask, Action.SLAM_BEST_ITEM):
            return int(Action.SLAM_BEST_ITEM)
        if is_legal(context.mask, Action.FIELD_BEST_BOARD):
            return int(Action.FIELD_BEST_BOARD)
        if context.state.gold > 20 and is_legal(context.mask, Action.ROLL):
            return int(Action.ROLL)
        return end_turn()

    def _target_trait_buy(self, env: MiniTFTEnv, mask: NDArray[np.bool_]) -> int | None:
        assert env.state is not None
        candidates: list[tuple[float, int]] = []
        for index, unit_id in enumerate(env.state.shop):
            action = shop_buy_action(index)
            if not is_legal(mask, action):
                continue
            unit = env.data.units[unit_id]
            if self.target_trait in unit.traits:
                candidates.append((unit.cost * 100.0 + unit.base_power, action))
        if not candidates:
            return None
        return max(candidates)[1]
