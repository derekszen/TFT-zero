"""Gymnasium wrapper for the Set-1-like MiniTFT lobby."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from mini_tft.core.actions import NUM_ACTIONS, Action
from mini_tft.core.config import EnvConfig
from mini_tft.core.featurize import OBS_CLIP_HIGH, OBS_CLIP_LOW, featurize_state, observation_dim
from mini_tft.core.lobby import (
    DEFAULT_LOBBY_PLAYERS,
    LobbyCombatResult,
    Set1LobbyState,
    clone_lobby_state,
    lobby_action_mask,
    lobby_player_summary,
    lobby_signature,
    new_lobby_state,
    resolve_lobby_round,
)
from mini_tft.core.lobby_step import (
    LobbyPolicy,
    LobbyRoundStepResult,
    PlayerOrderMode,
    apply_lobby_action,
    fast_level_lobby_policy,
    mixed_lobby_policy,
    random_lobby_policy,
    step_lobby_round_with_policies,
    tempo_lobby_policy,
)
from mini_tft.core.set_data import GameData, load_set

type AgentId = str
type LobbyObservation = dict[str, NDArray[np.float32] | NDArray[np.bool_]]
type LobbyStepMode = Literal["combat", "policy"]


@dataclass(frozen=True)
class MiniTFTLobbySnapshot:
    """Clone/restore payload for deterministic lobby branch search."""

    state: Set1LobbyState
    rng_state: dict[str, Any]
    rotation_offset: int = 0


class MiniTFTLobbyEnv(gym.Env[LobbyObservation, Any]):
    """Gym/PettingZoo-like lobby wrapper with optional all-player policy stepping."""

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        *,
        seed: int | None = None,
        config: EnvConfig | None = None,
        player_count: int = DEFAULT_LOBBY_PLAYERS,
        step_mode: LobbyStepMode = "policy",
        policy: LobbyPolicy | None = None,
        max_actions_per_player: int | None = None,
        player_order_mode: PlayerOrderMode = "random",
    ) -> None:
        self.config = config or EnvConfig(seed=seed)
        self.data: GameData = load_set(self.config.dataset)
        self.player_count = player_count
        self.step_mode: LobbyStepMode = step_mode
        self.policy = policy
        self.max_actions_per_player = max_actions_per_player or self.config.max_actions_per_round
        self.player_order_mode: PlayerOrderMode = player_order_mode
        self._rotation_offset = 0
        self.possible_agents = [f"player_{index}" for index in range(player_count)]
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.state: Set1LobbyState | None = None

        obs_dim = observation_dim(self.data, self.config)
        if self.step_mode == "combat":
            self.action_space = spaces.MultiDiscrete([player_count + 1] * player_count)
        elif self.step_mode == "policy":
            action_shape = np.full(
                (player_count, max(1, self.max_actions_per_player)),
                NUM_ACTIONS,
            )
            self.action_space = spaces.MultiDiscrete(action_shape)
        else:
            raise ValueError(f"unknown lobby step mode: {self.step_mode}")
        self.observation_space = spaces.Dict(
            {
                "players": spaces.Box(
                    low=OBS_CLIP_LOW,
                    high=OBS_CLIP_HIGH,
                    shape=(player_count, obs_dim),
                    dtype=np.float32,
                ),
                "action_masks": spaces.MultiBinary((player_count, NUM_ACTIONS)),
                "active": spaces.MultiBinary(player_count),
                "round": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "hp": spaces.Box(
                    low=0.0,
                    high=float(self.config.starting_hp),
                    shape=(player_count,),
                    dtype=np.float32,
                ),
                "placements": spaces.Box(
                    low=0.0,
                    high=float(player_count),
                    shape=(player_count,),
                    dtype=np.float32,
                ),
            }
        )

    @property
    def agents(self) -> list[AgentId]:
        state = self._require_state()
        return [self.possible_agents[player_id] for player_id in state.active_player_ids()]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[LobbyObservation, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        actual_seed = self.config.seed if seed is None else seed
        if actual_seed is None:
            actual_seed = int(np.random.default_rng().integers(0, np.iinfo(np.int64).max))
        self.rng = np.random.default_rng(actual_seed)
        self._rotation_offset = 0
        self.state = new_lobby_state(
            self.config,
            self.data,
            seed=actual_seed,
            player_count=self.player_count,
        )
        return self._observe_all(), self._info()

    def step(
        self,
        action: Any,
    ) -> tuple[LobbyObservation, float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        if state.done:
            raise RuntimeError("Lobby is done. Call reset() before step().")

        if self.step_mode == "policy":
            return self._step_policy(action)

        matchups = self._preferred_matchups(action)
        results = resolve_lobby_round(
            state,
            self.data,
            self.config,
            self.rng,
            matchups=matchups,
        )
        reward = self._mean_hp_delta_reward(results)
        terminated = state.done and state.final_reason == "one_player_remaining"
        truncated = state.done and state.final_reason == "max_round"
        return self._observe_all(), reward, terminated, truncated, self._info(results)

    def observe(self, agent: AgentId | int) -> NDArray[np.float32]:
        state = self._require_state()
        player_id = self._agent_index(agent)
        return featurize_state(state.players[player_id], self.data, self.config)

    def action_mask(self, agent: AgentId | int) -> NDArray[np.bool_]:
        state = self._require_state()
        return lobby_action_mask(state, self._agent_index(agent), self.data, self.config)

    def clone_state(self) -> MiniTFTLobbySnapshot:
        state = self._require_state()
        return MiniTFTLobbySnapshot(
            state=clone_lobby_state(state),
            rng_state=deepcopy(dict(self.rng.bit_generator.state)),
            rotation_offset=self._rotation_offset,
        )

    def restore_state(
        self,
        snapshot: MiniTFTLobbySnapshot,
    ) -> tuple[LobbyObservation, dict[str, Any]]:
        self.state = clone_lobby_state(snapshot.state)
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = deepcopy(snapshot.rng_state)
        self._rotation_offset = snapshot.rotation_offset
        return self._observe_all(), self._info()

    def state_signature(self) -> tuple[Any, ...]:
        return lobby_signature(self._require_state())

    def episode_summary(self, player_id: int = 0) -> dict[str, int | float | str | None]:
        summary = lobby_player_summary(self._require_state(), player_id, self.data)
        return cast(dict[str, int | float | str | None], asdict(summary))

    def lobby_summaries(self) -> list[dict[str, int | float | str | None]]:
        return [self.episode_summary(player_id) for player_id in range(self.player_count)]

    def _observe_all(self) -> LobbyObservation:
        state = self._require_state()
        player_obs = np.stack(
            [featurize_state(player, self.data, self.config) for player in state.players]
        ).astype(np.float32)
        masks = np.stack(
            [
                lobby_action_mask(state, player_id, self.data, self.config)
                for player_id in range(self.player_count)
            ]
        )
        active = np.asarray(
            [not player.done for player in state.players],
            dtype=np.bool_,
        )
        hp = np.asarray([player.hp for player in state.players], dtype=np.float32)
        placements = np.asarray(
            [state.placements.get(player_id, 0) for player_id in range(self.player_count)],
            dtype=np.float32,
        )
        return {
            "players": player_obs,
            "action_masks": masks.astype(np.bool_),
            "active": active,
            "round": np.asarray([state.round / self.config.max_round], dtype=np.float32),
            "hp": hp,
            "placements": placements,
        }

    def _info(
        self,
        results: list[LobbyCombatResult] | None = None,
        action_step: LobbyRoundStepResult | None = None,
    ) -> dict[str, Any]:
        state = self._require_state()
        return {
            "agents": self.agents,
            "possible_agents": list(self.possible_agents),
            "round": state.round,
            "placements": dict(state.placements),
            "last_matchups": list(state.last_matchups),
            "combat_results": list(results or []),
            "action_step_result": action_step,
            "player_order_mode": self.player_order_mode,
            "rotation_offset": self._rotation_offset,
            "final_reason": state.final_reason,
            "player_summaries": self.lobby_summaries() if state.done else None,
            "signature": self.state_signature(),
        }

    def _step_policy(
        self,
        action: Any,
    ) -> tuple[LobbyObservation, float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        policy, matchups, player_order = self._policy_step_inputs(action)
        action_step = step_lobby_round_with_policies(
            state,
            self.data,
            self.config,
            self.rng,
            policy,
            max_actions_per_player=self.max_actions_per_player,
            player_order=player_order,
            player_order_mode=self.player_order_mode,
            rotation_offset=self._rotation_offset,
            matchups=matchups,
        )
        if self.player_order_mode == "rotating" and action_step.player_order:
            self._rotation_offset += 1

        results = list(action_step.combat_results)
        reward = self._mean_hp_delta_reward(results)
        terminated = state.done and state.final_reason == "one_player_remaining"
        truncated = state.done and state.final_reason == "max_round"
        return (
            self._observe_all(),
            reward,
            terminated,
            truncated,
            self._info(results, action_step),
        )

    def _policy_step_inputs(
        self,
        action: Any,
    ) -> tuple[LobbyPolicy, list[tuple[int, int]] | None, tuple[int, ...] | None]:
        if isinstance(action, dict):
            raw_actions = action.get("actions")
            raw_policy = action.get("policy", self.policy)
            raw_matchups = action.get("matchups")
            raw_player_order = action.get("player_order")
            matchups = self._coerce_matchups(raw_matchups)
            player_order = (
                tuple(int(player_id) for player_id in raw_player_order)
                if raw_player_order is not None
                else None
            )
            if raw_policy is not None:
                return cast(LobbyPolicy, raw_policy), matchups, player_order
            return self._action_plan_policy(raw_actions), matchups, player_order

        policy = self.policy if self.policy is not None else self._action_plan_policy(action)
        return policy, None, None

    def _action_plan_policy(self, action_plan: Any) -> LobbyPolicy:
        if action_plan is None:
            plan = np.full(
                (self.player_count, self.max_actions_per_player),
                int(Action.END_TURN),
                dtype=np.int64,
            )
        else:
            plan = np.asarray(action_plan, dtype=np.int64)
            if plan.ndim == 1:
                plan = plan.reshape(self.player_count, 1)
            if plan.ndim != 2 or plan.shape[0] != self.player_count:
                raise ValueError("policy step action plan must have one row per lobby player")

        cursors = np.zeros(self.player_count, dtype=np.int64)

        def policy(
            player_id: int,
            _state: Set1LobbyState,
            _mask: NDArray[np.bool_],
            _data: GameData,
            _config: EnvConfig,
            _rng: np.random.Generator,
        ) -> int:
            cursor = int(cursors[player_id])
            cursors[player_id] += 1
            if cursor >= plan.shape[1]:
                return int(Action.END_TURN)
            return int(plan[player_id, cursor])

        return policy

    def _preferred_matchups(self, action: Any) -> list[tuple[int, int]] | None:
        if action is None:
            return None

        preferences = np.asarray(action, dtype=np.int64).reshape(-1)
        if preferences.size != self.player_count or np.all(preferences == 0):
            return None

        state = self._require_state()
        active = set(state.active_player_ids())
        used: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for player_id in sorted(active):
            if player_id in used:
                continue
            desired = int(preferences[player_id]) - 1
            if desired in active and desired != player_id and desired not in used:
                pairs.append((player_id, desired))
                used.update({player_id, desired})

        if len(used) != len(active):
            return None
        return pairs

    def _coerce_matchups(self, action: Any) -> list[tuple[int, int]] | None:
        if action is None:
            return None
        pairs = np.asarray(action, dtype=np.int64)
        if pairs.ndim == 2 and pairs.shape[1] == 2:
            return [(int(left), int(right)) for left, right in pairs]
        return self._preferred_matchups(action)

    def _mean_hp_delta_reward(self, results: list[LobbyCombatResult]) -> float:
        if not results:
            return 0.0
        return -float(np.mean([result.damage for result in results])) * 0.01

    def _agent_index(self, agent: AgentId | int) -> int:
        if isinstance(agent, int):
            player_id = agent
        else:
            if not agent.startswith("player_"):
                raise ValueError(f"Unknown agent id: {agent}")
            player_id = int(agent.removeprefix("player_"))
        if not 0 <= player_id < self.player_count:
            raise ValueError(f"agent index out of range: {agent}")
        return player_id

    def _require_state(self) -> Set1LobbyState:
        if self.state is None:
            raise RuntimeError("Call reset() before using the lobby environment.")
        return self.state


LOBBY_POLICY_BY_NAME: dict[str, LobbyPolicy] = {
    "fast_level": fast_level_lobby_policy,
    "mixed": mixed_lobby_policy,
    "random": random_lobby_policy,
    "tempo": tempo_lobby_policy,
}


class MiniTFTLobbyHeroEnv(gym.Env[NDArray[np.float32], int]):
    """Single-agent training wrapper for player 0 in a scripted lobby."""

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        *,
        seed: int | None = None,
        config: EnvConfig | None = None,
        player_count: int = DEFAULT_LOBBY_PLAYERS,
        opponent_policy: LobbyPolicy = tempo_lobby_policy,
        max_actions_per_player: int | None = None,
    ) -> None:
        self.config = config or EnvConfig(seed=seed)
        self.data: GameData = load_set(self.config.dataset)
        self.player_count = player_count
        self.opponent_policy = opponent_policy
        self.max_actions_per_player = max_actions_per_player or self.config.max_actions_per_round
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.state: Set1LobbyState | None = None
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=OBS_CLIP_LOW,
            high=OBS_CLIP_HIGH,
            shape=(observation_dim(self.data, self.config),),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        actual_seed = self.config.seed if seed is None else seed
        if actual_seed is None:
            actual_seed = int(np.random.default_rng().integers(0, np.iinfo(np.int64).max))
        self.rng = np.random.default_rng(actual_seed)
        self.state = new_lobby_state(
            self.config,
            self.data,
            seed=actual_seed,
            player_count=self.player_count,
        )
        return self._observe(), self._info()

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        if state.done or state.players[0].done:
            raise RuntimeError("Lobby hero episode is done. Call reset() before step().")

        action = int(action)
        mask = self.action_masks()
        legal = 0 <= action < len(mask) and bool(mask[action])
        reward = 0.0
        end_round = False
        action_step: LobbyRoundStepResult | None = None

        if not legal:
            state.players[0].total_illegal_actions += 1
            reward -= 1.0
            end_round = True
        elif action == Action.END_TURN:
            reward -= 0.01
            end_round = True
        else:
            applied = apply_lobby_action(state, 0, action, self.data, self.config, self.rng)
            if applied:
                state.players[0].round_action_count += 1
                reward += self._action_reward(action)
            else:
                state.players[0].total_illegal_actions += 1
                reward -= 0.5
                end_round = True
            if state.players[0].round_action_count >= self.max_actions_per_player:
                reward -= 0.05
                end_round = True

        if end_round and not state.done:
            round_reward, action_step = self._finish_round()
            reward += round_reward

        hero_done = state.done or state.players[0].done
        terminated = hero_done and state.final_reason != "max_round"
        truncated = hero_done and state.final_reason == "max_round"
        return self._observe(), float(reward), terminated, truncated, self._info(action_step)

    def action_masks(self) -> NDArray[np.bool_]:
        state = self._require_state()
        return lobby_action_mask(state, 0, self.data, self.config)

    def episode_summary(self) -> dict[str, int | float | str | None]:
        summary = lobby_player_summary(self._require_state(), 0, self.data)
        return cast(dict[str, int | float | str | None], asdict(summary))

    def _finish_round(self) -> tuple[float, LobbyRoundStepResult]:
        state = self._require_state()
        hero = state.players[0]
        before_hp = hero.hp
        before_strength = hero.last_board_strength
        opponents = tuple(player_id for player_id in state.active_player_ids() if player_id != 0)
        action_step = step_lobby_round_with_policies(
            state,
            self.data,
            self.config,
            self.rng,
            self.opponent_policy,
            max_actions_per_player=self.max_actions_per_player,
            player_order=opponents,
            player_order_mode="fixed",
        )

        reward = (state.players[0].hp - before_hp) * 0.08
        reward += (state.players[0].last_board_strength - before_strength) * 0.003
        for result in action_step.combat_results:
            if result.player_a == 0 or result.player_b == 0:
                reward += 0.5 if result.winner == 0 else -(result.damage * 0.1)
                break
        if state.players[0].done:
            reward += self._placement_reward(int(state.placements.get(0, self.player_count)))
        return reward, action_step

    def _action_reward(self, action: int) -> float:
        if Action.BUY_SHOP_0 <= action <= Action.BUY_SHOP_4:
            return 0.04
        if action == Action.FIELD_BEST_BOARD:
            return 0.08
        if action == Action.SLAM_BEST_ITEM:
            return 0.03
        if action == Action.BUY_XP:
            return 0.02
        if action == Action.ROLL:
            return -0.01
        return 0.0

    def _placement_reward(self, placement: int) -> float:
        score = float(self.player_count + 1 - placement)
        if placement == 1:
            score += 6.0
        elif placement <= min(4, self.player_count):
            score += 3.0
        return score

    def _observe(self) -> NDArray[np.float32]:
        return featurize_state(self._require_state().players[0], self.data, self.config)

    def _info(self, action_step: LobbyRoundStepResult | None = None) -> dict[str, Any]:
        state = self._require_state()
        return {
            "action_mask": self.action_masks(),
            "round": state.round,
            "hp": state.players[0].hp,
            "placement": state.placements.get(0),
            "final_reason": state.players[0].final_reason or state.final_reason,
            "action_step_result": action_step,
        }

    def _require_state(self) -> Set1LobbyState:
        if self.state is None:
            raise RuntimeError("Call reset() before using the lobby hero environment.")
        return self.state


Set1LobbySnapshot = MiniTFTLobbySnapshot
Set1LobbyGymEnv = MiniTFTLobbyEnv
