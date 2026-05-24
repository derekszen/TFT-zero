"""Tiny environment throughput benchmark."""

from __future__ import annotations

import argparse
import time

import numpy as np

from mini_tft.bots import RandomBot
from mini_tft.core.config import EnvConfig
from mini_tft.rl.gym_env import MiniTFTGymEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1_000)
    args = parser.parse_args()

    started = time.perf_counter()
    bot = RandomBot()
    steps = 0
    for seed in range(args.episodes):
        env = MiniTFTGymEnv(config=EnvConfig(seed=seed))
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = bot.act(env, obs, rng)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
    elapsed = time.perf_counter() - started
    print(
        f"episodes={args.episodes} steps={steps} "
        f"elapsed={elapsed:.3f}s steps_per_sec={steps / elapsed:.1f}"
    )


if __name__ == "__main__":
    main()
