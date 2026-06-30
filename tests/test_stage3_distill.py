from __future__ import annotations

import numpy as np

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import lobby_action_mask, new_lobby_state
from mini_tft.core.lobby_step import fast_level_lobby_policy, tempo_lobby_policy
from mini_tft.core.set_data import load_set
from mini_tft.search.distill import (
    generate_teacher_dataset,
    load_search_dataset,
    load_student_policy,
    policy_target_from_trace,
    train_student,
    validate_dataset_schema,
)
from mini_tft.search.mcts import MCTSConfig


def test_policy_target_from_trace_normalizes_visit_counts() -> None:
    target = policy_target_from_trace(
        {
            "selected_action": 0,
            "action_stats": [
                {"action": 0, "visits": 1},
                {"action": 2, "visits": 3},
            ],
        }
    )

    assert np.isclose(target.sum(), 1.0)
    assert np.isclose(target[0], 0.25)
    assert np.isclose(target[2], 0.75)


def test_teacher_dataset_schema_has_legal_masked_targets(tmp_path) -> None:
    config = EnvConfig(seed=3, starting_gold=10, max_actions_per_round=4, max_round=4)
    dataset_path = tmp_path / "teacher.npz"

    generate_teacher_dataset(
        out_path=dataset_path,
        suite_name="mixed",
        episodes=1,
        seed=3,
        player_count=4,
        max_actions_per_player=4,
        env_config=config,
        mcts_config=MCTSConfig(
            simulations=2,
            rollout_rounds=1,
            max_root_actions=2,
            max_actions_per_player=4,
            prior_improvement_margin=0.25,
        ),
        opponent_policy=tempo_lobby_policy,
        prior_policy=fast_level_lobby_policy,
        rollout_policy=fast_level_lobby_policy,
        player_order_mode="fixed",
    )

    dataset = load_search_dataset(dataset_path)
    validate_dataset_schema(dataset, env_config=config)
    assert dataset.size > 0
    assert dataset.obs.shape[0] == dataset.policy_targets.shape[0]
    assert np.allclose(dataset.policy_targets.sum(axis=1), 1.0)
    assert np.all(
        (dataset.policy_targets * dataset.masks.astype(np.float32)).sum(axis=1) > 0.999
    )


def test_student_trainer_checkpoint_load_and_legal_policy(tmp_path) -> None:
    config = EnvConfig(seed=4, starting_gold=10, max_actions_per_round=4, max_round=4)
    dataset_path = tmp_path / "teacher.npz"
    checkpoint_path = tmp_path / "student.pt"

    generate_teacher_dataset(
        out_path=dataset_path,
        suite_name="mixed",
        episodes=1,
        seed=4,
        player_count=4,
        max_actions_per_player=4,
        env_config=config,
        mcts_config=MCTSConfig(
            simulations=2,
            rollout_rounds=1,
            max_root_actions=2,
            max_actions_per_player=4,
            prior_improvement_margin=0.25,
        ),
        opponent_policy=tempo_lobby_policy,
        prior_policy=fast_level_lobby_policy,
        rollout_policy=fast_level_lobby_policy,
        player_order_mode="fixed",
    )
    report = train_student(
        dataset_paths=[dataset_path],
        out_path=checkpoint_path,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        selected_action_coef=3.0,
        hidden_sizes=(128,),
        seed=4,
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert report["status"] == "pass"
    assert report["final_loss"] is not None
    assert report["final_selected_action_acc"] > 0.5

    policy = load_student_policy(checkpoint_path, device="cpu")
    data = load_set()
    lobby = new_lobby_state(config, data, seed=4, player_count=4)
    mask = lobby_action_mask(lobby, 0, data, config)
    action = policy(0, lobby, mask, data, config, np.random.default_rng(4))

    assert mask[action]
