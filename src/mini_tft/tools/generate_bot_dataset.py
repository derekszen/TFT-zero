"""Generate compressed rollout data from heuristic bots."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

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
    shard = _generate_shard_for_bots(
        start_episode=0,
        episodes=episodes,
        seed=seed,
        bots=bots,
        config_overrides={},
    )
    _write_dataset(output, shard)


def generate_dataset_parallel(
    episodes: int,
    output: Path,
    seed: int = 0,
    workers: int | None = None,
    chunk_size: int | None = None,
) -> dict[str, float | int | str]:
    """Generate rollout data in worker processes and write one `.npz` file."""

    actual_workers = resolve_worker_count(workers, episodes)
    started = time.perf_counter()
    if actual_workers <= 1 or episodes <= 1:
        shard = _generate_shard(0, episodes, seed, len(default_bots()), {})
    else:
        chunks = _episode_chunks(episodes, actual_workers, chunk_size)
        shards = []
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = [
                executor.submit(
                    _generate_shard,
                    start_episode,
                    count,
                    seed,
                    len(default_bots()),
                    {},
                )
                for start_episode, count in chunks
            ]
            for future in futures:
                shards.append(future.result())
        shard = _merge_shards(shards)

    _write_dataset(output, shard)
    elapsed = time.perf_counter() - started
    transitions = int(len(shard["actions"]))
    return {
        "episodes": episodes,
        "transitions": transitions,
        "workers": actual_workers,
        "elapsed_sec": elapsed,
        "transitions_per_sec": transitions / elapsed if elapsed else 0.0,
        "path": str(output),
        "file_mb": output.stat().st_size / 1_000_000,
    }


def resolve_worker_count(workers: int | None, episodes: int) -> int:
    """Resolve worker count. `None` or `0` means hardware-aware auto."""

    if workers is not None and workers > 0:
        return min(workers, max(1, episodes))
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, 32, max(1, episodes))


def _generate_shard(
    start_episode: int,
    episodes: int,
    seed: int,
    bot_count: int,
    config_overrides: dict[str, Any],
) -> dict[str, np.ndarray]:
    bots = default_bots()[:bot_count]
    return _generate_shard_for_bots(start_episode, episodes, seed, bots, config_overrides)


def _generate_shard_for_bots(
    start_episode: int,
    episodes: int,
    seed: int,
    bots: list[BaseBot],
    config_overrides: dict[str, Any],
) -> dict[str, np.ndarray]:
    obs_rows = []
    action_rows = []
    reward_rows = []
    next_obs_rows = []
    done_rows = []
    mask_rows = []
    episode_id_rows = []
    step_idx_rows = []
    return_rows = []

    for episode_id in range(start_episode, start_episode + episodes):
        bot = bots[episode_id % len(bots)]
        episode_seed = seed + episode_id
        env = MiniTFTGymEnv(config=EnvConfig(seed=episode_seed, **config_overrides))
        bot_rng = np.random.default_rng(_bot_seed(seed, episode_id))
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

    return {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "actions": np.asarray(action_rows, dtype=np.int64),
        "rewards": np.asarray(reward_rows, dtype=np.float32),
        "next_obs": np.asarray(next_obs_rows, dtype=np.float32),
        "dones": np.asarray(done_rows, dtype=np.bool_),
        "masks": np.asarray(mask_rows, dtype=np.bool_),
        "episode_ids": np.asarray(episode_id_rows, dtype=np.int64),
        "step_idxs": np.asarray(step_idx_rows, dtype=np.int64),
        "returns": np.asarray(return_rows, dtype=np.float32),
    }


def _write_dataset(output: Path, shard: dict[str, np.ndarray]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **shard)


def _merge_shards(shards: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = shards[0].keys()
    return {key: np.concatenate([shard[key] for shard in shards]) for key in keys}


def _episode_chunks(
    episodes: int,
    workers: int,
    chunk_size: int | None = None,
) -> list[tuple[int, int]]:
    if chunk_size is None or chunk_size <= 0:
        chunk_size = max(1, (episodes + workers - 1) // workers)
    chunks = []
    start = 0
    while start < episodes:
        count = min(chunk_size, episodes - start)
        chunks.append((start, count))
        start += count
    return chunks


def _bot_seed(seed: int, episode_id: int) -> int:
    return (seed * 1_000_003 + episode_id * 97 + 17) % (2**32 - 1)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. Use 0 for hardware-aware auto.",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()
    if args.workers == 1:
        started = time.perf_counter()
        generate_dataset(args.episodes, args.out, args.seed)
        elapsed = time.perf_counter() - started
        print(f"wrote {args.out} in {elapsed:.3f}s with 1 worker")
    else:
        metrics = generate_dataset_parallel(
            episodes=args.episodes,
            output=args.out,
            seed=args.seed,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
        print(
            f"wrote {args.out} in {metrics['elapsed_sec']:.3f}s with "
            f"{metrics['workers']} workers "
            f"({metrics['transitions_per_sec']:.1f} transitions/sec)"
        )


if __name__ == "__main__":
    main()
