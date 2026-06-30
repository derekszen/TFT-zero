"""Optional PufferLib wrappers for MiniTFT Gymnasium environments."""

from __future__ import annotations

import importlib
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from mini_tft.core.actions import NUM_ACTIONS as CORE_NUM_ACTIONS
from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import LobbyPolicy, PlayerOrderMode
from mini_tft.rl.gym_env import MiniTFTGymEnv
from mini_tft.rl.lobby_env import (
    LobbyObservation,
    LobbyStepMode,
    MiniTFTLobbyEnv,
    MiniTFTLobbyHeroEnv,
)
from mini_tft.strategic.adapters.puffer.vector_env import StrategicBatchEnv
from mini_tft.strategic.core import (
    NUM_ACTIONS as STRATEGIC_NUM_ACTIONS,
)
from mini_tft.strategic.core import (
    StrategicConfig,
)
from mini_tft.strategic.core import (
    legal_action_mask as strategic_legal_action_mask,
)
from mini_tft.strategic.core import (
    observation_dim as strategic_observation_dim,
)
from mini_tft.strategic.core import (
    observe as strategic_observe,
)
from mini_tft.strategic.core import (
    reset as strategic_reset,
)
from mini_tft.strategic.core import (
    step as strategic_step,
)

PUFFER_INSTALL_MESSAGE = (
    "PufferLib legacy wrappers are optional. Install a PufferLib version that exposes "
    "`pufferlib.emulation`/`pufferlib.PufferEnv`, or use the PufferLib 4 Ocean C env."
)

_NativePufferEnvBase: Any
try:
    _NativePufferEnvBase = cast(Any, importlib.import_module("pufferlib").PufferEnv)
except (AttributeError, ImportError):  # pragma: no cover - optional/legacy dep split
    _NativePufferEnvBase = object


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
                    np.zeros(CORE_NUM_ACTIONS, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.full(
                        self.base_observation_dim,
                        float(base_space.high.max()),
                        dtype=np.float32,
                    ),
                    np.ones(CORE_NUM_ACTIONS, dtype=np.float32),
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


class MiniTFTLobbyHeroMaskInObsEnv(gym.Env[NDArray[np.float32], int]):
    """Single-agent lobby env that appends the legal-action mask."""

    metadata = MiniTFTLobbyHeroEnv.metadata

    def __init__(
        self,
        *,
        seed: int | None = None,
        config: EnvConfig | None = None,
        player_count: int = 8,
        opponent_policy: LobbyPolicy,
        max_actions_per_player: int | None = None,
        allow_oracle_macro_actions: bool = True,
    ) -> None:
        self.env = MiniTFTLobbyHeroEnv(
            seed=seed,
            config=config,
            player_count=player_count,
            opponent_policy=opponent_policy,
            max_actions_per_player=max_actions_per_player,
            allow_oracle_macro_actions=allow_oracle_macro_actions,
        )
        base_space = cast(spaces.Box, self.env.observation_space)
        if base_space.shape is None:
            raise ValueError("MiniTFT lobby hero observation space must have a fixed shape")
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
                    np.zeros(CORE_NUM_ACTIONS, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.full(
                        self.base_observation_dim,
                        float(base_space.high.max()),
                        dtype=np.float32,
                    ),
                    np.ones(CORE_NUM_ACTIONS, dtype=np.float32),
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

    def episode_summary(self) -> dict[str, int | float | str | None]:
        return self.env.episode_summary()

    def _with_mask(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        mask = self.action_masks().astype(np.float32)
        return np.concatenate([obs.astype(np.float32), mask])


class StrategicMaskInObsEnv(gym.Env[NDArray[np.float32], int]):
    """Strategic-lane env that appends the legal-action mask to observations."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int | None = None,
        config: StrategicConfig | None = None,
    ) -> None:
        self.config = config or StrategicConfig()
        self._seed = 0 if seed is None else int(seed)
        self.state = strategic_reset(seed=self._seed, config=self.config)
        self.base_observation_dim = strategic_observation_dim(self.config)
        self.action_space = spaces.Discrete(STRATEGIC_NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    np.full(self.base_observation_dim, -np.inf, dtype=np.float32),
                    np.zeros(STRATEGIC_NUM_ACTIONS, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.full(self.base_observation_dim, np.inf, dtype=np.float32),
                    np.ones(STRATEGIC_NUM_ACTIONS, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        del options
        if seed is not None:
            self._seed = int(seed)
        self.state = strategic_reset(seed=self._seed, config=self.config)
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        result = strategic_step(self.state, int(action), self.config)
        info = self._info()
        info.update(result.info)
        return self._observation(), result.reward, result.terminated, result.truncated, info

    def action_masks(self) -> NDArray[np.bool_]:
        return strategic_legal_action_mask(self.state, self.config)

    def _observation(self) -> NDArray[np.float32]:
        return np.concatenate(
            [
                strategic_observe(self.state, self.config),
                self.action_masks().astype(np.float32),
            ]
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.action_masks(),
            "round": self.state.round,
            "hp": self.state.hp,
            "gold": self.state.gold,
            "level": self.state.level,
            "final_reason": self.state.final_reason,
        }


class MiniTFTLobbyHeroNativePufferEnv(_NativePufferEnvBase):
    """Native Puffer vector env for independent player-0 lobby hero episodes."""

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int | None = None,
        config: EnvConfig | None = None,
        player_count: int = 8,
        opponent_policy: LobbyPolicy,
        max_actions_per_player: int | None = None,
        allow_oracle_macro_actions: bool = True,
        buf: Any = None,
    ) -> None:
        if _NativePufferEnvBase is object:
            raise RuntimeError(PUFFER_INSTALL_MESSAGE)
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")

        self.num_agents = int(num_envs)
        self.envs = [
            MiniTFTLobbyHeroMaskInObsEnv(
                seed=None if seed is None else seed + env_index,
                config=config,
                player_count=player_count,
                opponent_policy=opponent_policy,
                max_actions_per_player=max_actions_per_player,
                allow_oracle_macro_actions=allow_oracle_macro_actions,
            )
            for env_index in range(self.num_agents)
        ]
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.base_observation_dim = self.envs[0].base_observation_dim
        self._next_reset_seeds = np.arange(
            (seed if seed is not None else 0) + 10_000,
            (seed if seed is not None else 0) + 10_000 + self.num_agents,
            dtype=np.int64,
        )
        cast(Any, super()).__init__(buf)

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
        infos: list[dict[str, Any]] = []
        for env_index, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + env_index
            obs, info = env.reset(seed=env_seed)
            self.observations[env_index] = np.asarray(obs, dtype=np.float32).reshape(-1)
            infos.append(info)
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True
        return self.observations, infos

    def step(
        self,
        actions: NDArray[np.int64] | list[int] | tuple[int, ...],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        action_array = np.asarray(actions, dtype=np.int64).reshape(self.num_agents)
        infos: list[dict[str, Any]] = []
        for env_index, action in enumerate(action_array):
            obs, reward, terminated, truncated, info = self.envs[env_index].step(int(action))
            self.rewards[env_index] = float(reward)
            self.terminals[env_index] = bool(terminated)
            self.truncations[env_index] = bool(truncated)
            if bool(terminated or truncated):
                info = dict(info)
                info["terminal_observation"] = np.asarray(obs, dtype=np.float32).copy()
                obs, reset_info = self.envs[env_index].reset(
                    seed=int(self._next_reset_seeds[env_index])
                )
                self._next_reset_seeds[env_index] += self.num_agents
                info["reset_info"] = reset_info
            self.observations[env_index] = np.asarray(obs, dtype=np.float32).reshape(-1)
            infos.append(info)
        self.masks[:] = True
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def action_masks(self) -> NDArray[np.bool_]:
        return np.stack([env.action_masks() for env in self.envs]).astype(np.bool_)

    def close(self) -> None:
        for env in self.envs:
            env.close()


class StrategicNativePufferEnv(_NativePufferEnvBase):
    """Native Puffer vector env for the clean strategic lane."""

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int | None = None,
        config: StrategicConfig | None = None,
        buf: Any = None,
    ) -> None:
        if _NativePufferEnvBase is object:
            raise RuntimeError(PUFFER_INSTALL_MESSAGE)
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")

        self.num_agents = int(num_envs)
        self.config = config or StrategicConfig()
        self.batch = StrategicBatchEnv(
            num_envs=self.num_agents,
            seed=seed if seed is not None else 0,
            config=self.config,
        )
        self.base_observation_dim = self.batch.base_observation_dim
        self.single_observation_space = spaces.Box(
            low=np.concatenate(
                [
                    np.full(self.base_observation_dim, -np.inf, dtype=np.float32),
                    np.zeros(STRATEGIC_NUM_ACTIONS, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    np.full(self.base_observation_dim, np.inf, dtype=np.float32),
                    np.ones(STRATEGIC_NUM_ACTIONS, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )
        self.single_action_space = spaces.Discrete(STRATEGIC_NUM_ACTIONS)
        self._next_reset_seeds = np.arange(
            (seed if seed is not None else 0) + 10_000,
            (seed if seed is not None else 0) + 10_000 + self.num_agents,
            dtype=np.int64,
        )
        cast(Any, super()).__init__(buf)

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
        observations = self.batch.reset(seed=seed if seed is not None else 0)
        self.observations[:] = observations
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        self.masks[:] = True
        return self.observations, self._infos(legal=True, ended_round=False)

    def step(
        self,
        actions: NDArray[np.int64] | list[int] | tuple[int, ...],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        result = self.batch.step(
            np.asarray(actions, dtype=np.int64).reshape(self.num_agents),
            include_infos=False,
        )
        infos = result.infos
        observations = result.observations.copy()
        done = np.logical_or(result.terminals, result.truncations)
        if bool(done.any()):
            done_indices = np.flatnonzero(done).astype(np.int64)
            for env_index in done_indices:
                infos[int(env_index)] = dict(infos[int(env_index)])
                infos[int(env_index)]["terminal_observation"] = observations[int(env_index)].copy()
            reset_seeds = self._next_reset_seeds[done_indices].copy()
            reset_observations = self.batch.reset(seed=reset_seeds, indices=done_indices)
            self._next_reset_seeds[done_indices] += self.num_agents
            for env_index in done_indices:
                infos[int(env_index)]["reset_info"] = self._info(int(env_index))
                observations[int(env_index)] = reset_observations[int(env_index)]

        self.observations[:] = observations
        self.rewards[:] = result.rewards
        self.terminals[:] = result.terminals
        self.truncations[:] = result.truncations
        self.masks[:] = True
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def action_masks(self) -> NDArray[np.bool_]:
        return self.batch.cached_action_mask

    def close(self) -> None:
        return None

    def _infos(self, *, legal: bool, ended_round: bool) -> list[dict[str, Any]]:
        return [
            {
                **self._info(env_index),
                "legal_action": legal,
                "ended_round": ended_round,
            }
            for env_index in range(self.num_agents)
        ]

    def _info(self, env_index: int) -> dict[str, Any]:
        return {
            "action_mask": self.action_masks()[env_index],
            "round": int(self.batch.round[env_index]),
            "hp": int(self.batch.hp[env_index]),
            "gold": int(self.batch.gold[env_index]),
            "level": int(self.batch.level[env_index]),
            "final_reason": self.batch.final_reason(env_index),
        }


def flatten_lobby_observation(obs: LobbyObservation) -> NDArray[np.float32]:
    """Flatten a lobby observation into trainable float features."""

    return np.concatenate(
        [
            obs["players"].astype(np.float32).reshape(-1),
            obs["action_masks"].astype(np.float32).reshape(-1),
            obs["active"].astype(np.float32).reshape(-1),
            obs["round"].astype(np.float32).reshape(-1),
            obs["hp"].astype(np.float32).reshape(-1),
            obs["placements"].astype(np.float32).reshape(-1),
        ]
    )


class MiniTFTLobbyFlatEnv(gym.Env[NDArray[np.float32], Any]):
    """Lobby wrapper with flat float observations and flat action plans."""

    metadata = MiniTFTLobbyEnv.metadata

    def __init__(
        self,
        *,
        seed: int | None = None,
        config: EnvConfig | None = None,
        player_count: int = 8,
        step_mode: LobbyStepMode = "policy",
        policy: LobbyPolicy | None = None,
        max_actions_per_player: int | None = None,
        player_order_mode: PlayerOrderMode = "random",
    ) -> None:
        if step_mode != "policy":
            raise ValueError("MiniTFTLobbyFlatEnv only supports policy step mode")
        self.env = MiniTFTLobbyEnv(
            seed=seed,
            config=config,
            player_count=player_count,
            step_mode=step_mode,
            policy=policy,
            max_actions_per_player=max_actions_per_player,
            player_order_mode=player_order_mode,
        )
        self.player_count = self.env.player_count
        self.max_actions_per_player = self.env.max_actions_per_player
        self.action_space = spaces.MultiDiscrete(
            np.full(
                self.player_count * self.max_actions_per_player,
                CORE_NUM_ACTIONS,
                dtype=np.int64,
            )
        )
        self.observation_space = _flat_lobby_observation_space(
            cast(spaces.Dict, self.env.observation_space)
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

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return flatten_lobby_observation(obs), info

    def step(
        self,
        action: NDArray[np.int64] | list[int] | tuple[int, ...],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action_plan = np.asarray(action, dtype=np.int64).reshape(
            self.player_count,
            self.max_actions_per_player,
        )
        obs, reward, terminated, truncated, info = self.env.step(action_plan)
        return flatten_lobby_observation(obs), reward, terminated, truncated, info

    def observe(self, agent: str | int) -> NDArray[np.float32]:
        return self.env.observe(agent)

    def action_mask(self, agent: str | int) -> NDArray[np.bool_]:
        return self.env.action_mask(agent)

    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, snapshot):
        obs, info = self.env.restore_state(snapshot)
        return flatten_lobby_observation(obs), info

    def state_signature(self) -> tuple[Any, ...]:
        return self.env.state_signature()

    def episode_summary(self, player_id: int = 0) -> dict[str, int | float | str | None]:
        return self.env.episode_summary(player_id)

    def lobby_summaries(self) -> list[dict[str, int | float | str | None]]:
        return self.env.lobby_summaries()


def _flat_lobby_observation_space(space: spaces.Dict) -> spaces.Box:
    players = cast(spaces.Box, space["players"])
    action_masks = cast(spaces.MultiBinary, space["action_masks"])
    active = cast(spaces.MultiBinary, space["active"])
    round_space = cast(spaces.Box, space["round"])
    hp = cast(spaces.Box, space["hp"])
    placements = cast(spaces.Box, space["placements"])

    low = np.concatenate(
        [
            np.asarray(players.low, dtype=np.float32).reshape(-1),
            np.zeros(int(np.prod(action_masks.shape)), dtype=np.float32),
            np.zeros(int(np.prod(active.shape)), dtype=np.float32),
            np.asarray(round_space.low, dtype=np.float32).reshape(-1),
            np.asarray(hp.low, dtype=np.float32).reshape(-1),
            np.asarray(placements.low, dtype=np.float32).reshape(-1),
        ]
    )
    high = np.concatenate(
        [
            np.asarray(players.high, dtype=np.float32).reshape(-1),
            np.ones(int(np.prod(action_masks.shape)), dtype=np.float32),
            np.ones(int(np.prod(active.shape)), dtype=np.float32),
            np.asarray(round_space.high, dtype=np.float32).reshape(-1),
            np.asarray(hp.high, dtype=np.float32).reshape(-1),
            np.asarray(placements.high, dtype=np.float32).reshape(-1),
        ]
    )
    return spaces.Box(low=low, high=high, dtype=np.float32)


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


def make_strategic_puffer_env(
    seed: int | None = None,
    config: StrategicConfig | None = None,
) -> Any:
    """Return a PufferLib-emulated env for strategic-lane PPO."""

    pufferlib = require_pufferlib()
    return pufferlib.emulation.GymnasiumPufferEnv(
        StrategicMaskInObsEnv(seed=seed, config=config)
    )


def make_lobby_puffer_env(
    *,
    seed: int | None = None,
    config: EnvConfig | None = None,
    player_count: int = 8,
) -> Any:
    """Return a PufferLib-emulated env for the Python lobby reference."""

    pufferlib = require_pufferlib()
    return pufferlib.emulation.GymnasiumPufferEnv(
        MiniTFTLobbyFlatEnv(seed=seed, config=config, player_count=player_count)
    )


def make_lobby_hero_puffer_env(
    *,
    seed: int | None = None,
    config: EnvConfig | None = None,
    player_count: int = 8,
    opponent_policy: LobbyPolicy,
    max_actions_per_player: int | None = None,
    allow_oracle_macro_actions: bool = True,
) -> Any:
    """Return a PufferLib-emulated env for player-0 lobby PPO."""

    pufferlib = require_pufferlib()
    return pufferlib.emulation.GymnasiumPufferEnv(
        MiniTFTLobbyHeroMaskInObsEnv(
            seed=seed,
            config=config,
            player_count=player_count,
            opponent_policy=opponent_policy,
            max_actions_per_player=max_actions_per_player,
            allow_oracle_macro_actions=allow_oracle_macro_actions,
        )
    )


def make_lobby_hero_native_puffer_env(
    *,
    num_envs: int,
    seed: int | None = None,
    config: EnvConfig | None = None,
    player_count: int = 8,
    opponent_policy: LobbyPolicy,
    max_actions_per_player: int | None = None,
    allow_oracle_macro_actions: bool = True,
) -> MiniTFTLobbyHeroNativePufferEnv:
    """Return a native PufferLib env for batched player-0 lobby PPO."""

    require_pufferlib()
    return MiniTFTLobbyHeroNativePufferEnv(
        num_envs=num_envs,
        seed=seed,
        config=config,
        player_count=player_count,
        opponent_policy=opponent_policy,
        max_actions_per_player=max_actions_per_player,
        allow_oracle_macro_actions=allow_oracle_macro_actions,
    )


def make_strategic_native_puffer_env(
    *,
    num_envs: int,
    seed: int | None = None,
    config: StrategicConfig | None = None,
) -> StrategicNativePufferEnv:
    """Return a native PufferLib env for batched strategic-lane PPO."""

    require_pufferlib()
    return StrategicNativePufferEnv(num_envs=num_envs, seed=seed, config=config)
