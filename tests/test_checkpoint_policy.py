from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mini_tft.core.actions import NUM_ACTIONS
from mini_tft.core.config import EnvConfig
from mini_tft.core.featurize import observation_dim
from mini_tft.core.lobby import new_lobby_state
from mini_tft.core.set_data import load_set
from mini_tft.rl.checkpoint_policy import load_lobby_checkpoint_policy
from mini_tft.rl.train_puffer_ppo import MaskedActorCritic
from mini_tft.tools.evaluate_lobby_policy import run_lobby_evaluation


def test_puffer_checkpoint_policy_uses_tail_action_mask(tmp_path: Path) -> None:
    config = EnvConfig(seed=3)
    data = load_set(config.dataset)
    checkpoint = _write_puffer_checkpoint(
        tmp_path / "policy.pt",
        data_dim=observation_dim(data, config),
    )
    policy = load_lobby_checkpoint_policy(checkpoint, checkpoint_format="auto", device="cpu")
    state = new_lobby_state(config, data, seed=3, player_count=4)
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[1] = True

    action = policy(0, state, mask, data, config, np.random.default_rng(4))

    assert action == 1


def test_lobby_evaluation_accepts_puffer_checkpoint(tmp_path: Path) -> None:
    config = EnvConfig(seed=5, starting_gold=10, max_actions_per_round=4, max_round=2)
    data = load_set(config.dataset)
    checkpoint = _write_puffer_checkpoint(
        tmp_path / "policy.pt",
        data_dim=observation_dim(data, config),
    )

    report = run_lobby_evaluation(
        episodes=1,
        seed=5,
        checkpoint=checkpoint,
        checkpoint_format="puffer",
        opponent_policy_name="tempo",
        player_count=4,
        max_actions_per_player=2,
        device="cpu",
        config=config,
    )

    assert report["status"] == "pass"
    assert report["checkpoint_format"] == "puffer"
    assert 1.0 <= report["mean_placement"] <= 4.0


def _write_puffer_checkpoint(path: Path, *, data_dim: int) -> Path:
    model = MaskedActorCritic(
        observation_dim=data_dim + NUM_ACTIONS,
        action_dim=NUM_ACTIONS,
        hidden_sizes=[8],
    )
    with torch.no_grad():
        model.actor.bias.zero_()
        model.actor.bias[0] = 10.0
    torch.save(
        {
            "kind": "puffer_ppo",
            "model_state_dict": model.state_dict(),
            "resolved": {
                "action_dim": NUM_ACTIONS,
                "observation_dim": data_dim + NUM_ACTIONS,
                "hidden_sizes": [8],
            },
            "args": {"hidden_sizes": "8"},
        },
        path,
    )
    return path
