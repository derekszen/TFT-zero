"""Shared bot helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import BUY_SHOP_OFFSET, NUM_ACTIONS, Action
from mini_tft.core.env import MiniTFTEnv
from mini_tft.core.state import GameState


class BaseBot(Protocol):
    name: str

    def act(
        self,
        env: MiniTFTEnv,
        obs: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> int:
        """Choose one legal action."""


@dataclass
class BotContext:
    env: MiniTFTEnv
    state: GameState
    mask: NDArray[np.bool_]


def bot_context(env: MiniTFTEnv) -> BotContext:
    if env.state is None:
        raise RuntimeError("Bot requires a reset environment.")
    return BotContext(env=env, state=env.state, mask=env.action_masks())


def is_legal(mask: NDArray[np.bool_], action: int) -> bool:
    return 0 <= action < NUM_ACTIONS and bool(mask[action])


def shop_buy_action(index: int) -> int:
    return BUY_SHOP_OFFSET + index


def owned_unit_ids(state: GameState) -> list[int]:
    return [
        unit.unit_id
        for unit in [*state.board, *state.bench]
        if unit is not None
    ]


def end_turn() -> int:
    return int(Action.END_TURN)
