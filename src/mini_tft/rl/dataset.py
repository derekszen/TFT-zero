"""Dataset loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RolloutDataset:
    obs: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float32]
    next_obs: NDArray[np.float32]
    dones: NDArray[np.bool_]
    masks: NDArray[np.bool_]
    episode_ids: NDArray[np.int64]
    step_idxs: NDArray[np.int64]
    returns: NDArray[np.float32]


def load_dataset(path: str | Path) -> RolloutDataset:
    data = np.load(path)
    dataset = RolloutDataset(
        obs=data["obs"].astype(np.float32),
        actions=data["actions"].astype(np.int64),
        rewards=data["rewards"].astype(np.float32),
        next_obs=data["next_obs"].astype(np.float32),
        dones=data["dones"].astype(np.bool_),
        masks=data["masks"].astype(np.bool_),
        episode_ids=data["episode_ids"].astype(np.int64),
        step_idxs=data["step_idxs"].astype(np.int64),
        returns=data["returns"].astype(np.float32),
    )
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: RolloutDataset) -> None:
    n = len(dataset.actions)
    arrays = [
        dataset.obs,
        dataset.rewards,
        dataset.next_obs,
        dataset.dones,
        dataset.masks,
        dataset.episode_ids,
        dataset.step_idxs,
        dataset.returns,
    ]
    if any(len(array) != n for array in arrays):
        raise ValueError("all dataset arrays must have the same leading length")
    if dataset.obs.shape != dataset.next_obs.shape:
        raise ValueError("obs and next_obs must have matching shapes")
    if dataset.masks.ndim != 2:
        raise ValueError("masks must be a 2D array")
