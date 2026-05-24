"""Evaluate heuristic bots or a MaskablePPO checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

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

EVAL_SEEDS = tuple(range(1000, 1100))


def evaluate_bot(bot: BaseBot, seeds: Iterable[int] = EVAL_SEEDS) -> dict[str, float]:
    summaries = []
    for seed in seeds:
        env = MiniTFTGymEnv(config=EnvConfig(seed=seed))
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = bot.act(env, obs, rng)
            obs, _, terminated, truncated, _ = env.step(action)
        summaries.append(env.episode_summary())

    return {
        "mean_final_hp": float(np.mean([summary["final_hp"] for summary in summaries])),
        "survival_rate": float(
            np.mean([summary["final_reason"] == "max_round" for summary in summaries])
        ),
        "mean_survived_round": float(np.mean([summary["survived_round"] for summary in summaries])),
        "mean_final_board_strength": float(
            np.mean([summary["final_board_strength"] for summary in summaries])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    seeds = range(1000, 1000 + args.episodes)
    bots = [
        RandomBot(),
        GreedyBoardBot(),
        EconBot(),
        RerollBot(),
        FastLevelBot(),
        TraitCommitBot("ranger", "TraitCommitBot[ranger]"),
    ]
    for bot in bots:
        metrics = evaluate_bot(bot, seeds)
        print(bot.name, metrics)


if __name__ == "__main__":
    main()
