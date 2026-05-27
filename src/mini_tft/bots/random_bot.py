"""Random legal policy with bounded board-move noise."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.bots.base import bot_context
from mini_tft.core.actions import Action
from mini_tft.core.env import MiniTFTEnv


class RandomBot:
    name = "RandomBot"

    def act(
        self,
        env: MiniTFTEnv,
        obs: NDArray[np.float32],
        rng: np.random.Generator,
    ) -> int:
        del obs
        context = bot_context(env)
        legal = np.flatnonzero(context.mask)
        if len(legal) == 0:
            raise RuntimeError("No legal actions available.")

        core_legal = legal[legal <= Action.SLAM_BEST_ITEM]
        placement_legal = legal[legal > Action.SLAM_BEST_ITEM]
        if len(core_legal) == 0:
            return int(rng.choice(legal))
        if len(placement_legal) > 0 and rng.random() < 0.2:
            return int(rng.choice(placement_legal))
        return int(rng.choice(core_legal))
