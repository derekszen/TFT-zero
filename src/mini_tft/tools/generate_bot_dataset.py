"""Generate compressed rollout data from heuristic bots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mini_tft.bots import (
    EconBot,
    FastLevelBot,
    GreedyBoardBot,
    RandomBot,
    RerollBot,
    TraitCommitBot,
)
from mini_tft.bots.base import BaseBot
from mini_tft.core.config import EnvConfig
from mini_tft.rl.gym_env import MiniTFTGymEnv


def default_bots() -> list[BaseBot]:
    return [
        RandomBot(),
        GreedyBoardBot(),
        EconBot(),
        RerollBot(),
        FastLevelBot(),
        TraitCommitBot("ranger", "TraitCommitBot[ranger]"),
        TraitCommitBot("noble", "TraitCommitBot[noble]"),
        TraitCommitBot("glacial", "TraitCommitBot[glacial]"),
    ]


def generate_dataset(
    episodes: int,
    output: Path,
    seed: int = 0,
    bots: list[BaseBot] | None = None,
) -> None:
    bots = bots or default_bots()
    rng = np.random.default_rng(seed)
    obs_rows = []
    action_rows = []
    reward_rows = []
    next_obs_rows = []
    done_rows = []
    mask_rows = []
    episode_id_rows = []
    step_idx_rows = []
    return_rows = []

    for episode_id in range(episodes):
        bot = bots[episode_id % len(bots)]
        episode_seed = seed + episode_id
        env = MiniTFTGymEnv(config=EnvConfig(seed=episode_seed))
        bot_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        obs, info = env.reset(seed=episode_seed)
        episode_indices: list[int] = []
        episode_rewards: list[float] = []
        terminated = truncated = False
        step_idx = 0

        while not (terminated or truncated):
            mask = info["action_mask"]
            action = bot.act(env, obs, bot_rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs_rows.append(obs)
            action_rows.append(action)
            reward_rows.append(reward)
            next_obs_rows.append(next_obs)
            done_rows.append(terminated or truncated)
            mask_rows.append(mask)
            episode_id_rows.append(episode_id)
            step_idx_rows.append(step_idx)
            episode_indices.append(len(reward_rows) - 1)
            episode_rewards.append(reward)
            obs = next_obs
            step_idx += 1

        returns = _returns_to_go(episode_rewards)
        for index, value in zip(episode_indices, returns, strict=True):
            while len(return_rows) <= index:
                return_rows.append(0.0)
            return_rows[index] = value

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        obs=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.int64),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        next_obs=np.asarray(next_obs_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.bool_),
        masks=np.asarray(mask_rows, dtype=np.bool_),
        episode_ids=np.asarray(episode_id_rows, dtype=np.int64),
        step_idxs=np.asarray(step_idx_rows, dtype=np.int64),
        returns=np.asarray(return_rows, dtype=np.float32),
    )


def _returns_to_go(rewards: list[float]) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for index in reversed(range(len(rewards))):
        running += rewards[index]
        returns[index] = running
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("rollouts/bot_dataset_v0.npz"))
    args = parser.parse_args()
    generate_dataset(args.episodes, args.out, args.seed)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
