"""Optional PufferLib wrappers for MiniTFT Gymnasium environments."""

from __future__ import annotations

import importlib
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from mini_tft.core.actions import NUM_ACTIONS
from mini_tft.core.config import EnvConfig
from mini_tft.rl.gym_env import MiniTFTGymEnv
from mini_tft.rl.lobby_env import MiniTFTLobbyEnv

PUFFER_INSTALL_MESSAGE = (
    "PufferLib is optional. Install `pufferlib` in the active environment before "
    "using MiniTFT Puffer training or vectorization."
)


class MiniTFTMaskInObsEnv(gym.Env[NDArray[np.float32], int]):
    """Gymnasium env that appends the legal-action mask to every observation."""

    metadata = MiniTFTGymEnv.metadata

    def __init__(self, seed: int | None = None, config: EnvConfig | None = None) -> None:
        self.env = MiniTFTGymEnv(seed=seed, config=config)
        base_space = cast(spaces.Box, self.env.observation_space)
        if base_space.shape is None:
            raise ValueError("MiniTFT observation space must have a fixed shape")
        self.base_observation_dim = int(base_space.shape[0])
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    np.full(
                        self.base_observation_dim,
                        float(base_space.low.min()),
                        dtype=np.float32,
                    ),
                    np.zeros(NUM_ACTIONS, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.full(
                        self.base_observation_dim,
                        float(base_space.high.max()),
                        dtype=np.float32,
                    ),
                    np.ones(NUM_ACTIONS, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

    @property
    def state(self):
        return self.env.state

    @property
    def data(self):
        return self.env.data

    @property
    def config(self):
        return self.env.config

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._with_mask(obs), info

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self._with_mask(obs), reward, terminated, truncated, info

    def action_masks(self) -> NDArray[np.bool_]:
        return self.env.action_masks()

    def render(self) -> str:
        return self.env.render()

    def episode_summary(self) -> dict[str, int | float | bool | str | None]:
        return self.env.episode_summary()

    def _with_mask(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        mask = self.action_masks().astype(np.float32)
        return np.concatenate([obs.astype(np.float32), mask])


def require_pufferlib() -> Any:
    try:
        pufferlib = importlib.import_module("pufferlib")
        importlib.import_module("pufferlib.emulation")
    except ImportError as exc:  # pragma: no cover - exercised when optional dep is absent
        raise RuntimeError(PUFFER_INSTALL_MESSAGE) from exc
    return pufferlib


def make_masked_gym_env(
    seed: int | None = None,
    config: EnvConfig | None = None,
) -> MiniTFTMaskInObsEnv:
    return MiniTFTMaskInObsEnv(seed=seed, config=config)


def make_puffer_env(seed: int | None = None, config: EnvConfig | None = None) -> Any:
    """Return a PufferLib-emulated env for single-player MiniTFT."""

    pufferlib = require_pufferlib()
    return pufferlib.emulation.GymnasiumPufferEnv(make_masked_gym_env(seed=seed, config=config))


def make_lobby_puffer_env(
    *,
    seed: int | None = None,
    config: EnvConfig | None = None,
    player_count: int = 8,
) -> Any:
    """Return a PufferLib-emulated env for the Python lobby reference."""

    pufferlib = require_pufferlib()
    return pufferlib.emulation.GymnasiumPufferEnv(
        MiniTFTLobbyEnv(seed=seed, config=config, player_count=player_count)
    )
