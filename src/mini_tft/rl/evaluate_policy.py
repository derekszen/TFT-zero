"""Evaluate heuristic bots or a MaskablePPO checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
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


def evaluate_checkpoint(checkpoint: Path, seeds: Iterable[int] = EVAL_SEEDS) -> dict[str, float]:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    model = MaskablePPO.load(checkpoint)
    summaries = []
    for seed in seeds:
        env = MiniTFTGymEnv(config=EnvConfig(seed=seed))
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            obs, _, terminated, truncated, _ = env.step(int(action))
        summaries.append(env.episode_summary())
        env.close()

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
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    seeds = range(1000, 1000 + args.episodes)
    bots: list[BaseBot] = [
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
    if args.checkpoint is not None:
        metrics = evaluate_checkpoint(args.checkpoint, seeds)
        print(f"MaskablePPO[{args.checkpoint}]", metrics)


if __name__ == "__main__":
    main()
