"""Simulator-backed root MCTS scaffold for MiniTFT lobby decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import (
    BUY_SHOP_OFFSET,
    NUM_ACTIONS,
    Action,
    action_name,
    is_buy_shop_action,
    is_move_bench_to_board_action,
    is_move_board_to_bench_action,
    is_sell_bench_action,
)
from mini_tft.core.combat import board_strength
from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import Set1LobbyState, clone_lobby_state, lobby_player_summary
from mini_tft.core.lobby_step import (
    LobbyPolicy,
    fast_level_lobby_policy,
    step_lobby_round_with_policies,
)
from mini_tft.core.set_data import GameData

RAW_PRIMITIVE_LEGACY_ACTION_SURFACE = "raw_primitive_legacy"
CANDIDATE_CHOICE_ACTION_SURFACE = "candidate_choice"


@dataclass(frozen=True)
class MCTSConfig:
    """Small root-search configuration for the Stage 2 scaffold."""

    action_surface: str = RAW_PRIMITIVE_LEGACY_ACTION_SURFACE
    simulations: int = 8
    rollout_rounds: int = 1
    exploration: float = 1.4
    max_root_actions: int = 8
    max_actions_per_player: int | None = None
    prior_improvement_margin: float = 0.0


@dataclass
class ActionStats:
    """Visit/value accumulator for one root action."""

    action: int
    visits: int = 0
    value_sum: float = 0.0
    illegal_actions: int = 0
    rollout_depth_sum: int = 0
    max_rollout_depth: int = 0

    @property
    def mean_value(self) -> float | None:
        if self.visits <= 0:
            return None
        return self.value_sum / self.visits


@dataclass(frozen=True)
class SimulationResult:
    """Outcome from one forced root-action branch."""

    value: float
    illegal_actions: int
    rollout_depth: int


@dataclass(frozen=True)
class MCTSDecision:
    """Planner decision plus diagnostics for trace export."""

    selected_action: int
    legal_actions: tuple[int, ...]
    considered_actions: tuple[int, ...]
    action_stats: tuple[ActionStats, ...]
    simulations_run: int
    elapsed_ms: float
    illegal_actions: int
    max_rollout_depth: int
    trace: dict[str, Any]


class RootMCTSPlanner:
    """Root-action MCTS/bandit planner backed by cloned lobby simulations."""

    def __init__(
        self,
        *,
        hero_player_id: int = 0,
        config: MCTSConfig | None = None,
        opponent_policy: LobbyPolicy = fast_level_lobby_policy,
        rollout_policy: LobbyPolicy = fast_level_lobby_policy,
        prior_policy: LobbyPolicy | None = None,
    ) -> None:
        self.hero_player_id = hero_player_id
        self.config = config or MCTSConfig()
        if self.config.action_surface != RAW_PRIMITIVE_LEGACY_ACTION_SURFACE:
            raise ValueError(
                "RootMCTSPlanner currently supports only raw primitive legacy actions; "
                "candidate-choice search requires the candidate-board wrapper surface "
                "to be wired into the search stack."
            )
        if self.config.simulations <= 0:
            raise ValueError("MCTS simulations must be positive")
        if self.config.rollout_rounds <= 0:
            raise ValueError("MCTS rollout_rounds must be positive")
        if self.config.max_root_actions <= 0:
            raise ValueError("MCTS max_root_actions must be positive")
        self.opponent_policy = opponent_policy
        self.rollout_policy = rollout_policy
        self.prior_policy = prior_policy
        self.traces: list[dict[str, Any]] = []
        self._episode: int | None = None
        self._episode_seed: int | None = None
        self._decision_index = 0

    def set_episode_context(self, *, episode: int, seed: int) -> None:
        self._episode = episode
        self._episode_seed = seed
        self._decision_index = 0

    def policy(
        self,
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> int:
        """LobbyPolicy-compatible entrypoint for player-0 planning."""

        if player_id != self.hero_player_id:
            return int(self.opponent_policy(player_id, state, mask, data, env_config, rng))
        decision = self.plan(
            state=state,
            mask=mask,
            data=data,
            env_config=env_config,
            rng=rng,
        )
        self.traces.append(decision.trace)
        self._decision_index += 1
        return decision.selected_action

    def plan(
        self,
        *,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> MCTSDecision:
        """Search legal root actions from the current lobby decision point."""

        start = perf_counter()
        legal_actions = tuple(int(action) for action in np.flatnonzero(mask))
        if not legal_actions:
            return self._empty_decision(start)

        prior_action = self._prior_action(state, mask, data, env_config, rng)
        candidates = self._candidate_actions(state, mask, data, prior_action=prior_action)
        stats = {action: ActionStats(action=action) for action in candidates}
        local_rng = _clone_rng(rng)
        simulations_run = 0

        for simulation_index in range(self.config.simulations):
            action = self._select_action(stats, simulations_run)
            branch_rng = np.random.default_rng(
                int(local_rng.integers(0, np.iinfo(np.int64).max))
            )
            result = self._simulate_root_action(
                root_state=state,
                action=action,
                data=data,
                env_config=env_config,
                rng=branch_rng,
            )
            action_stats = stats[action]
            action_stats.visits += 1
            action_stats.value_sum += result.value
            action_stats.illegal_actions += result.illegal_actions
            action_stats.rollout_depth_sum += result.rollout_depth
            action_stats.max_rollout_depth = max(
                action_stats.max_rollout_depth,
                result.rollout_depth,
            )
            simulations_run = simulation_index + 1

        selected_action = self._select_final_action(stats, prior_action)
        elapsed_ms = (perf_counter() - start) * 1000.0
        action_stats_tuple = tuple(stats[action] for action in candidates)
        illegal_actions = sum(item.illegal_actions for item in action_stats_tuple)
        max_rollout_depth = max(
            (item.max_rollout_depth for item in action_stats_tuple),
            default=0,
        )
        trace = self._trace_row(
            state=state,
            legal_actions=legal_actions,
            considered_actions=candidates,
            selected_action=selected_action,
            action_stats=action_stats_tuple,
            prior_action=prior_action,
            simulations_run=simulations_run,
            elapsed_ms=elapsed_ms,
            illegal_actions=illegal_actions,
            max_rollout_depth=max_rollout_depth,
        )
        return MCTSDecision(
            selected_action=selected_action,
            legal_actions=legal_actions,
            considered_actions=candidates,
            action_stats=action_stats_tuple,
            simulations_run=simulations_run,
            elapsed_ms=elapsed_ms,
            illegal_actions=illegal_actions,
            max_rollout_depth=max_rollout_depth,
            trace=trace,
        )

    def _empty_decision(self, start: float) -> MCTSDecision:
        elapsed_ms = (perf_counter() - start) * 1000.0
        trace = {
            "kind": "mcts_decision",
            "action_surface": self.config.action_surface,
            "episode": self._episode,
            "seed": self._episode_seed,
            "decision_index": self._decision_index,
            "selected_action": int(Action.END_TURN),
            "selected_action_name": action_name(int(Action.END_TURN)),
            "legal_actions": [],
            "considered_actions": [],
            "action_stats": [],
            "simulations_run": 0,
            "elapsed_ms": elapsed_ms,
            "illegal_actions": 0,
            "max_rollout_depth": 0,
        }
        return MCTSDecision(
            selected_action=int(Action.END_TURN),
            legal_actions=(),
            considered_actions=(),
            action_stats=(),
            simulations_run=0,
            elapsed_ms=elapsed_ms,
            illegal_actions=0,
            max_rollout_depth=0,
            trace=trace,
        )

    def _candidate_actions(
        self,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        *,
        prior_action: int | None,
    ) -> tuple[int, ...]:
        legal_actions = [int(action) for action in np.flatnonzero(mask)]
        ranked = sorted(
            [action for action in legal_actions if action != prior_action],
            key=lambda action: _root_action_prior(state, self.hero_player_id, action, data),
            reverse=True,
        )
        if prior_action is not None:
            return (prior_action, *ranked[: max(0, self.config.max_root_actions - 1)])
        return tuple(ranked[: min(self.config.max_root_actions, len(ranked))])

    def _prior_action(
        self,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> int | None:
        if self.prior_policy is None:
            return None
        action = int(
            self.prior_policy(
                self.hero_player_id,
                state,
                mask,
                data,
                env_config,
                _clone_rng(rng),
            )
        )
        if not _is_legal(mask, action):
            return None
        return action

    def _select_action(
        self,
        stats: dict[int, ActionStats],
        total_visits: int,
    ) -> int:
        for action, item in stats.items():
            if item.visits == 0:
                return action
        log_total = np.log(max(2, total_visits + 1))
        return max(
            stats.values(),
            key=lambda item: (
                (item.mean_value or 0.0)
                + self.config.exploration * np.sqrt(log_total / item.visits),
                -item.action,
            ),
        ).action

    def _select_final_action(
        self,
        stats: dict[int, ActionStats],
        prior_action: int | None,
    ) -> int:
        best = max(
            stats.values(),
            key=lambda item: (
                item.mean_value if item.mean_value is not None else float("-inf"),
                item.visits,
                -item.action,
            ),
        )
        if prior_action is None or prior_action not in stats:
            return best.action
        prior = stats[prior_action]
        if prior.mean_value is None or best.mean_value is None:
            return prior_action
        if best.action == prior_action:
            return prior_action
        if best.mean_value < prior.mean_value + self.config.prior_improvement_margin:
            return prior_action
        return best.action

    def _simulate_root_action(
        self,
        *,
        root_state: Set1LobbyState,
        action: int,
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> SimulationResult:
        branch_state = clone_lobby_state(root_state)
        illegal_actions = 0
        rollout_depth = 0
        forced_used = False

        def branch_policy(
            player_id: int,
            state: Set1LobbyState,
            mask: NDArray[np.bool_],
            game_data: GameData,
            config: EnvConfig,
            policy_rng: np.random.Generator,
        ) -> int:
            nonlocal forced_used
            if player_id == self.hero_player_id:
                if not forced_used:
                    forced_used = True
                    return action if _is_legal(mask, action) else int(Action.END_TURN)
                return int(
                    self.rollout_policy(player_id, state, mask, game_data, config, policy_rng)
                )
            return int(
                self.opponent_policy(player_id, state, mask, game_data, config, policy_rng)
            )

        while rollout_depth < self.config.rollout_rounds and not branch_state.done:
            result = step_lobby_round_with_policies(
                branch_state,
                data,
                env_config,
                rng,
                branch_policy if rollout_depth == 0 else self._full_rollout_policy,
                max_actions_per_player=self.config.max_actions_per_player,
                player_order=tuple(branch_state.active_player_ids()),
                player_order_mode="fixed",
            )
            illegal_actions += int(result.illegal_actions)
            rollout_depth += 1

        value = _state_value(
            branch_state,
            self.hero_player_id,
            data,
            env_config,
        )
        value -= float(illegal_actions)
        return SimulationResult(
            value=value,
            illegal_actions=illegal_actions,
            rollout_depth=rollout_depth,
        )

    def _full_rollout_policy(
        self,
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        config: EnvConfig,
        rng: np.random.Generator,
    ) -> int:
        if player_id == self.hero_player_id:
            return int(self.rollout_policy(player_id, state, mask, data, config, rng))
        return int(self.opponent_policy(player_id, state, mask, data, config, rng))

    def _trace_row(
        self,
        *,
        state: Set1LobbyState,
        legal_actions: tuple[int, ...],
        considered_actions: tuple[int, ...],
        selected_action: int,
        action_stats: tuple[ActionStats, ...],
        prior_action: int | None,
        simulations_run: int,
        elapsed_ms: float,
        illegal_actions: int,
        max_rollout_depth: int,
    ) -> dict[str, Any]:
        player = state.players[self.hero_player_id]
        return {
            "kind": "mcts_decision",
            "action_surface": self.config.action_surface,
            "episode": self._episode,
            "seed": self._episode_seed,
            "decision_index": self._decision_index,
            "round": state.round,
            "level": player.level,
            "gold": player.gold,
            "hp": player.hp,
            "bench_size": sum(unit is not None for unit in player.bench),
            "board_size": sum(unit is not None for unit in player.board),
            "selected_action": int(selected_action),
            "selected_action_name": action_name(selected_action),
            "prior_action": prior_action,
            "prior_action_name": action_name(prior_action) if prior_action is not None else None,
            "selected_from_prior": selected_action == prior_action,
            "prior_improvement_margin": self.config.prior_improvement_margin,
            "legal_actions": list(legal_actions),
            "considered_actions": list(considered_actions),
            "action_stats": [
                {
                    "action": item.action,
                    "action_name": action_name(item.action),
                    "visits": item.visits,
                    "mean_value": item.mean_value,
                    "value_sum": item.value_sum,
                    "illegal_actions": item.illegal_actions,
                    "mean_rollout_depth": (
                        item.rollout_depth_sum / item.visits if item.visits else None
                    ),
                    "max_rollout_depth": item.max_rollout_depth,
                }
                for item in action_stats
            ],
            "simulations_requested": self.config.simulations,
            "simulations_run": simulations_run,
            "elapsed_ms": elapsed_ms,
            "illegal_actions": illegal_actions,
            "max_rollout_depth": max_rollout_depth,
        }


def _is_legal(mask: NDArray[np.bool_], action: int) -> bool:
    return 0 <= int(action) < NUM_ACTIONS and bool(mask[int(action)])


def _root_action_prior(
    state: Set1LobbyState,
    player_id: int,
    action: int,
    data: GameData,
) -> float:
    player = state.players[player_id]
    if action == int(Action.FIELD_BEST_BOARD):
        return 1000.0
    if action == int(Action.SLAM_BEST_ITEM):
        return 950.0
    if is_buy_shop_action(action):
        shop_index = action - BUY_SHOP_OFFSET
        unit_id = player.shop[shop_index]
        unit = data.units.get(unit_id)
        return 800.0 + (unit.cost * 20.0 + unit.base_power if unit is not None else 0.0)
    if action == int(Action.BUY_XP):
        return 700.0 + max(0.0, 8.0 - float(player.level))
    if action == int(Action.ROLL):
        return 650.0 + min(float(player.gold), 50.0) * 0.1
    if action == int(Action.END_TURN):
        return 600.0
    if is_move_bench_to_board_action(action):
        return 500.0
    if is_move_board_to_bench_action(action):
        return 450.0
    if is_sell_bench_action(action):
        return 300.0
    return 0.0


def _state_value(
    state: Set1LobbyState,
    player_id: int,
    data: GameData,
    config: EnvConfig,
) -> float:
    player = state.players[player_id]
    summary = lobby_player_summary(state, player_id, data)
    placement = float(summary.placement)
    if not state.done and not player.done:
        placement = _live_pseudo_placement(state, player_id, data)
    placement_score = (state.player_count + 1.0 - placement) / state.player_count
    hp_score = max(0.0, float(player.hp)) / max(1.0, float(config.starting_hp))
    strength = board_strength(player.board, data).strength
    strength_score = min(1.0, max(0.0, strength) / 400.0)
    alive_bonus = 0.15 if not player.done else 0.0
    return placement_score + 0.30 * hp_score + 0.20 * strength_score + alive_bonus


def _live_pseudo_placement(
    state: Set1LobbyState,
    player_id: int,
    data: GameData,
) -> float:
    standings = sorted(
        state.active_player_ids(),
        key=lambda index: (
            state.players[index].hp,
            board_strength(state.players[index].board, data).strength,
            -index,
        ),
        reverse=True,
    )
    try:
        return float(standings.index(player_id) + 1)
    except ValueError:
        return float(state.placements.get(player_id, state.player_count))


def _clone_rng(rng: np.random.Generator) -> np.random.Generator:
    cloned = np.random.default_rng()
    cloned.bit_generator.state = deepcopy(rng.bit_generator.state)
    return cloned


def forced_action_rollout_value(
    *,
    state: Set1LobbyState,
    action: int,
    data: GameData,
    env_config: EnvConfig,
    rng: np.random.Generator,
    opponent_policy: LobbyPolicy = fast_level_lobby_policy,
    rollout_policy: LobbyPolicy = fast_level_lobby_policy,
    player_id: int = 0,
    rollout_rounds: int = 1,
    max_actions_per_player: int | None = None,
) -> SimulationResult:
    """Test helper exposing the same forced-action branch evaluator."""

    planner = RootMCTSPlanner(
        hero_player_id=player_id,
        config=MCTSConfig(
            simulations=1,
            rollout_rounds=rollout_rounds,
            max_actions_per_player=max_actions_per_player,
        ),
        opponent_policy=opponent_policy,
        rollout_policy=rollout_policy,
    )
    return planner._simulate_root_action(
        root_state=state,
        action=action,
        data=data,
        env_config=env_config,
        rng=rng,
    )
