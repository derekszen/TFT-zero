"""Uniform random legal policy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mini_tft.bots.base import bot_context
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
        return int(rng.choice(legal))
