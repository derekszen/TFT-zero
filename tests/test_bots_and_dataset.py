from __future__ import annotations

import numpy as np

from mini_tft.bots import (
    EconBot,
    FastLevelBot,
    GreedyBoardBot,
    RandomBot,
    RerollBot,
    TraitCommitBot,
)
from mini_tft.core.config import EnvConfig
from mini_tft.rl.dataset import load_dataset
from mini_tft.rl.gym_env import MiniTFTGymEnv
from mini_tft.tools.generate_bot_dataset import generate_dataset


def test_each_bot_runs_one_episode_without_illegal_actions() -> None:
    bots = [
        RandomBot(),
        GreedyBoardBot(),
        EconBot(),
        RerollBot(),
        FastLevelBot(),
        TraitCommitBot("ranger"),
    ]

    for idx, bot in enumerate(bots):
        env = MiniTFTGymEnv(config=EnvConfig(seed=idx, max_steps_per_episode=300))
        rng = np.random.default_rng(idx)
        obs, _ = env.reset(seed=idx)
        terminated = truncated = False
        while not (terminated or truncated):
            action = bot.act(env, obs, rng)
            obs, _, terminated, truncated, info = env.step(action)
            assert info["legal_action"] is True


def test_generate_and_load_small_dataset(tmp_path) -> None:
    path = tmp_path / "dataset.npz"

    generate_dataset(episodes=2, output=path, seed=123, bots=[GreedyBoardBot()])
    dataset = load_dataset(path)

    assert dataset.obs.shape == dataset.next_obs.shape
    assert dataset.actions.shape[0] == dataset.obs.shape[0]
    assert dataset.masks.shape[0] == dataset.obs.shape[0]
