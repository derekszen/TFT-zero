"""MuZero-style cache export for the strategic lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.strategic.adapters.baselines import StrategicPolicy, tft_heuristic_policy
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


def cache_metrics(rows: list[CacheRow]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0, "legal_action_rate": 0.0, "mean_value_target": 0.0}
    legal = [
        bool(row.legal_mask[row.action])
        for row in rows
        if 0 <= row.action < len(row.legal_mask)
    ]
    return {
        "rows": len(rows),
        "observation_dim": int(rows[0].observation.shape[0]),
        "action_dim": int(rows[0].legal_mask.shape[0]),
        "legal_action_rate": float(np.mean(legal)) if legal else 0.0,
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
