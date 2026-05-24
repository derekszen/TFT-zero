"""Gymnasium wrapper surface for training code."""

from __future__ import annotations

from mini_tft.core.config import EnvConfig
from mini_tft.core.env import MiniTFTEnv


class MiniTFTGymEnv(MiniTFTEnv):
    """Alias subclass kept for RL scripts and registration."""

    def __init__(self, seed: int | None = None, config: EnvConfig | None = None) -> None:
        super().__init__(config or EnvConfig(seed=seed))
