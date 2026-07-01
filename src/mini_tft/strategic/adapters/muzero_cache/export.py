"""MuZero-style cache export for the strategic lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.strategic.adapters.baselines import StrategicPolicy, tft_heuristic_policy
from mini_tft.strategic.adapters.mcts import StrategicMCTSConfig, StrategicMCTSPlanner
from mini_tft.strategic.core.obs import observe
from mini_tft.strategic.core.rules import legal_action_mask, reset, scenario_score, step
from mini_tft.strategic.core.state import DEFAULT_STRATEGIC_CONFIG, StrategicConfig


@dataclass(frozen=True)
class CacheRow:
    observation: NDArray[np.float32]
    legal_mask: NDArray[np.bool_]
    action: int
    reward: float
    next_observation: NDArray[np.float32]
    done: bool
    policy_target: NDArray[np.float32]
    value_target: float
    metadata: dict[str, int | float | bool | str | None]


def generate_cache(
    *,
    episodes: int = 4,
    max_rows: int = 256,
    policy: StrategicPolicy = tft_heuristic_policy,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> list[CacheRow]:
    rows: list[CacheRow] = []
    for episode in range(episodes):
        state = reset(seed=episode, config=config)
        episode_rows: list[CacheRow] = []
        while not state.done and len(rows) + len(episode_rows) < max_rows:
            obs = observe(state, config)
            mask = legal_action_mask(state, config)
            action = int(policy(state, mask, config))
            result = step(state, action, config)
            next_obs = observe(state, config)
            policy_target = np.zeros(mask.shape, dtype=np.float32)
            if 0 <= action < len(policy_target):
                policy_target[action] = 1.0
            episode_rows.append(
                CacheRow(
                    observation=obs,
                    legal_mask=mask,
                    action=action,
                    reward=result.reward,
                    next_observation=next_obs,
                    done=result.terminated or result.truncated,
                    policy_target=policy_target,
                    value_target=0.0,
                    metadata={
                        "episode": episode,
                        "round": result.info["round"],
                        "hp": result.info["hp"],
                        "placement_proxy": result.info["placement_proxy"],
                        "scenario_score": scenario_score(state, config),
                        "legal_action": result.info["legal_action"],
                    },
                )
            )
        _assign_returns(episode_rows)
        rows.extend(episode_rows)
        if len(rows) >= max_rows:
            break
    return rows


def generate_mcts_cache(
    *,
    episodes: int = 4,
    max_rows: int = 256,
    seed: int = 0,
    mcts_config: StrategicMCTSConfig | None = None,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> list[CacheRow]:
    rows: list[CacheRow] = []
    planner = StrategicMCTSPlanner(
        config=mcts_config or StrategicMCTSConfig(),
        simulator_config=config,
    )
    planner_config = planner.config
    for episode in range(episodes):
        state = reset(seed=seed + episode, config=config)
        episode_rows: list[CacheRow] = []
        while not state.done and len(rows) + len(episode_rows) < max_rows:
            obs = observe(state, config)
            mask = legal_action_mask(state, config)
            decision = planner.plan(state)
            action = int(decision.selected_action)
            result = step(state, action, config)
            next_obs = observe(state, config)
            policy_target_source = (
                "checkpoint_guided_mcts"
                if planner_config.prior_mode == "checkpoint"
                or planner_config.value_mode == "checkpoint"
                else "mcts"
            )
            episode_rows.append(
                CacheRow(
                    observation=obs,
                    legal_mask=mask,
                    action=action,
                    reward=result.reward,
                    next_observation=next_obs,
                    done=result.terminated or result.truncated,
                    policy_target=decision.visit_policy.astype(np.float32, copy=True),
                    value_target=0.0,
                    metadata={
                        "episode": episode,
                        "seed": seed + episode,
                        "round": result.info["round"],
                        "hp": result.info["hp"],
                        "placement_proxy": result.info["placement_proxy"],
                        "scenario_score": scenario_score(state, config),
                        "legal_action": result.info["legal_action"],
                        "policy_target_source": policy_target_source,
                        "mcts_simulations": decision.simulations_run,
                        "mcts_max_depth": decision.max_depth,
                        "mcts_prior_mode": planner_config.prior_mode,
                        "mcts_value_mode": planner_config.value_mode,
                        "checkpoint_path": planner_config.checkpoint_path,
                        "mcts_elapsed_ms": decision.elapsed_ms,
                        "mcts_root_visits": int(sum(decision.action_visits.values())),
                    },
                )
            )
        _assign_returns(episode_rows)
        rows.extend(episode_rows)
        if len(rows) >= max_rows:
            break
    return rows


def cache_metrics(rows: list[CacheRow]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "legal_action_rate": 0.0, "mean_value_target": 0.0}
    legal = [
        bool(row.legal_mask[row.action])
        for row in rows
        if 0 <= row.action < len(row.legal_mask)
    ]
    policy_sums = np.asarray([float(row.policy_target.sum()) for row in rows], dtype=np.float64)
    illegal_policy_masses = np.asarray(
        [
            float(np.clip(row.policy_target[~row.legal_mask], 0.0, None).sum())
            if row.policy_target.shape == row.legal_mask.shape
            else float("inf")
            for row in rows
        ],
        dtype=np.float64,
    )
    target_valid = [
        row.policy_target.shape == row.legal_mask.shape
        and bool(np.all(np.isfinite(row.policy_target)))
        and bool(np.all(row.policy_target >= 0.0))
        and abs(float(row.policy_target.sum()) - 1.0) <= 1e-5
        and (
            float(np.clip(row.policy_target[~row.legal_mask], 0.0, None).sum()) <= 1e-5
            if row.policy_target.shape == row.legal_mask.shape
            else False
        )
        for row in rows
    ]
    values = np.asarray([float(row.value_target) for row in rows], dtype=np.float64)
    mcts_targets = [_is_mcts_target(row) for row in rows]
    return {
        "rows": len(rows),
        "observation_dim": int(rows[0].observation.shape[0]),
        "action_dim": int(rows[0].legal_mask.shape[0]),
        "legal_action_rate": float(np.mean(legal)) if legal else 0.0,
        "policy_target_valid_rate": float(np.mean(target_valid)),
        "policy_target_sum_error_max": float(np.max(np.abs(policy_sums - 1.0))),
        "illegal_policy_mass_max": float(np.max(illegal_policy_masses)),
        "value_target_finite_rate": float(np.mean(np.isfinite(values))),
        "mcts_target_rows": int(sum(mcts_targets)),
        "mcts_target_rate": float(np.mean(mcts_targets)),
        "mean_reward": float(np.mean([row.reward for row in rows])),
        "mean_value_target": float(np.mean([row.value_target for row in rows])),
        "terminal_rows": int(sum(row.done for row in rows)),
    }


def _assign_returns(rows: list[CacheRow], gamma: float = 0.97) -> None:
    value = 0.0
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        value = row.reward + gamma * value
        rows[index] = CacheRow(
            observation=row.observation,
            legal_mask=row.legal_mask,
            action=row.action,
            reward=row.reward,
            next_observation=row.next_observation,
            done=row.done,
            policy_target=row.policy_target,
            value_target=float(value),
            metadata=row.metadata,
        )


def cache_row_payload(index: int, row: CacheRow) -> dict[str, Any]:
    return {
        "index": index,
        "observation": row.observation.astype(float).tolist(),
        "legal_mask": row.legal_mask.astype(bool).tolist(),
        "action": row.action,
        "reward": row.reward,
        "next_observation": row.next_observation.astype(float).tolist(),
        "done": row.done,
        "policy_target": row.policy_target.astype(float).tolist(),
        "value_target": row.value_target,
        "metadata": row.metadata,
    }


def _is_mcts_target(row: CacheRow) -> bool:
    source = str(row.metadata.get("policy_target_source", "")).lower()
    if "mcts" in source:
        return True
    return any(str(key).startswith("mcts_") for key in row.metadata)
