"""Run and render one bot episode."""

from __future__ import annotations

import argparse

import numpy as np

from mini_tft.bots import GreedyBoardBot, RandomBot
from mini_tft.core.config import EnvConfig
from mini_tft.rl.gym_env import MiniTFTGymEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bot", choices=["random", "greedy"], default="greedy")
    args = parser.parse_args()

    bot = RandomBot() if args.bot == "random" else GreedyBoardBot()
    env = MiniTFTGymEnv(config=EnvConfig(seed=args.seed))
    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)
    terminated = truncated = False
    while not (terminated or truncated):
        action = bot.act(env, obs, rng)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(env.render())
        print(f"action={action} reward={reward:.3f}")
        print("-" * 72)


if __name__ == "__main__":
    main()
