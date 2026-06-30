"""Simulator-backed MCTS for the clean strategic lane."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.strategic.adapters.baselines import StrategicPolicy, tft_heuristic_policy
from mini_tft.strategic.core.actions import NUM_ACTIONS, StrategicAction, action_name
from mini_tft.strategic.core.rules import (
    legal_action_mask,
    placement_proxy,
    scenario_score,
    step,
)
from mini_tft.strategic.core.state import (
    DEFAULT_STRATEGIC_CONFIG,
    StrategicConfig,
    StrategicState,
)


@dataclass(frozen=True)
class StrategicMCTSConfig:
    simulations: int = 16
    max_depth: int = 12
    rollout_steps: int = 8
    exploration: float = 1.4
    gamma: float = 0.97
    prior_mode: str = "uniform"


@dataclass
class SearchNode:
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, SearchNode] = field(default_factory=dict)
    terminal: bool = False

    @property
    def mean_value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.value_sum / self.visits


@dataclass(frozen=True)
class StrategicMCTSDecision:
    selected_action: int
    visit_policy: NDArray[np.float32]
    action_values: dict[int, float]
    action_visits: dict[int, int]
    simulations_run: int
    elapsed_ms: float
    max_depth: int
    trace: dict[str, Any]


class StrategicMCTSPlanner:
    """Small full-tree MCTS backed by cloned strategic simulator states."""

    def __init__(
        self,
        *,
        config: StrategicMCTSConfig | None = None,
        simulator_config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
        rollout_policy: StrategicPolicy = tft_heuristic_policy,
    ) -> None:
        self.config = config or StrategicMCTSConfig()
        if self.config.simulations <= 0:
            raise ValueError("simulations must be positive")
        if self.config.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.config.rollout_steps < 0:
            raise ValueError("rollout_steps must be non-negative")
        if self.config.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        self.simulator_config = simulator_config
        self.rollout_policy = rollout_policy
        self.max_depth_seen = 0

    def policy(
        self,
        state: StrategicState,
        mask: NDArray[np.bool_],
        config: StrategicConfig,
    ) -> int:
        del mask, config
        return self.plan(state).selected_action

    def plan(self, state: StrategicState) -> StrategicMCTSDecision:
        started = perf_counter()
        root = SearchNode()
        self.max_depth_seen = 0
        self._expand(root, state)

        simulations_run = 0
        for _ in range(self.config.simulations):
            sim_state = state.clone()
            self._simulate(root, sim_state, depth=0)
            simulations_run += 1

        selected_action = self._select_final_action(root, state)
        elapsed_ms = (perf_counter() - started) * 1000.0
        visit_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action_values: dict[int, float] = {}
        action_visits: dict[int, int] = {}
        for action, child in sorted(root.children.items()):
            visit_policy[action] = child.visits
            action_values[action] = child.mean_value
            action_visits[action] = child.visits
        if float(visit_policy.sum()) > 0.0:
            visit_policy /= float(visit_policy.sum())
        trace = {
            "kind": "strategic_mcts_decision",
            "simulations": self.config.simulations,
            "selected_action": selected_action,
            "selected_action_name": action_name(selected_action),
            "legal_actions": [
                int(action)
                for action in np.flatnonzero(legal_action_mask(state, self.simulator_config))
            ],
            "action_visits": {
                action_name(action): visits for action, visits in action_visits.items()
            },
            "action_values": {
                action_name(action): value for action, value in action_values.items()
            },
            "elapsed_ms": elapsed_ms,
            "max_depth": self.max_depth_seen,
        }
        return StrategicMCTSDecision(
            selected_action=selected_action,
            visit_policy=visit_policy,
            action_values=action_values,
            action_visits=action_visits,
            simulations_run=simulations_run,
            elapsed_ms=elapsed_ms,
            max_depth=self.max_depth_seen,
            trace=trace,
        )

    def _simulate(self, node: SearchNode, state: StrategicState, *, depth: int) -> float:
        self.max_depth_seen = max(self.max_depth_seen, depth)
        if state.done:
            value = self._value(state)
            node.visits += 1
            node.value_sum += value
            node.terminal = True
            return value
        if depth >= self.config.max_depth:
            value = self._rollout_value(state)
            node.visits += 1
            node.value_sum += value
            return value
        if not node.children:
            self._expand(node, state)
            value = self._rollout_value(state)
            node.visits += 1
            node.value_sum += value
            return value

        action = self._select_child(node)
        child = node.children[action]
        result = step(state, action, self.simulator_config)
        value = float(result.reward)
        if not (result.terminated or result.truncated):
            value += self.config.gamma * self._simulate(child, state, depth=depth + 1)
        else:
            value += self.config.gamma * self._value(state)
            child.visits += 1
            child.value_sum += value
            child.terminal = True
        node.visits += 1
        node.value_sum += value
        return value

    def _expand(self, node: SearchNode, state: StrategicState) -> None:
        mask = legal_action_mask(state, self.simulator_config)
        legal_actions = [int(action) for action in np.flatnonzero(mask)]
        priors = self._priors(state, legal_actions)
        node.children = {
            action: SearchNode(prior=float(priors[index]))
            for index, action in enumerate(legal_actions)
        }

    def _select_child(self, node: SearchNode) -> int:
        parent_visits = max(1, node.visits)
        log_parent = log(parent_visits + 1)
        best_score = -float("inf")
        best_action = int(StrategicAction.HOLD)
        for action, child in node.children.items():
            exploit = child.mean_value
            explore = self.config.exploration * child.prior * sqrt(log_parent / (1 + child.visits))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _select_final_action(self, root: SearchNode, state: StrategicState) -> int:
        if not root.children:
            mask = legal_action_mask(state, self.simulator_config)
            legal = np.flatnonzero(mask)
            return int(legal[0]) if len(legal) else int(StrategicAction.HOLD)
        return max(
            root.children,
            key=lambda action: (root.children[action].visits, root.children[action].mean_value),
        )

    def _rollout_value(self, state: StrategicState) -> float:
        rollout_state = state.clone()
        value = 0.0
        discount = 1.0
        for _ in range(self.config.rollout_steps):
            if rollout_state.done:
                break
            mask = legal_action_mask(rollout_state, self.simulator_config)
            action = int(self.rollout_policy(rollout_state, mask, self.simulator_config))
            result = step(rollout_state, action, self.simulator_config)
            value += discount * float(result.reward)
            discount *= self.config.gamma
            if result.terminated or result.truncated:
                break
        return value + discount * self._value(rollout_state)

    def _value(self, state: StrategicState) -> float:
        score = scenario_score(state, self.simulator_config)
        placement = placement_proxy(state, self.simulator_config)
        placement_score = (8.0 - float(placement)) / 7.0
        return float(0.65 * score + 0.35 * placement_score)

    def _priors(self, state: StrategicState, legal_actions: list[int]) -> list[float]:
        if not legal_actions:
            return []
        if self.config.prior_mode == "heuristic":
            mask = legal_action_mask(state, self.simulator_config)
            heuristic_action = int(tft_heuristic_policy(state, mask, self.simulator_config))
            priors = np.full(len(legal_actions), 0.15 / len(legal_actions), dtype=np.float32)
            if heuristic_action in legal_actions:
                priors[legal_actions.index(heuristic_action)] += 0.85
            priors /= float(priors.sum())
            return [float(value) for value in priors.tolist()]
        return [1.0 / len(legal_actions)] * len(legal_actions)
