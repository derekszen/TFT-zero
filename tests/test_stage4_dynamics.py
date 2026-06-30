from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import lobby_action_mask, new_lobby_state
from mini_tft.core.lobby_step import fast_level_lobby_policy, tempo_lobby_policy
from mini_tft.core.set_data import load_set
from mini_tft.search.distill import generate_teacher_dataset, train_student
from mini_tft.search.dynamics import (
    ModelBackedRootPlanner,
    ModelSearchConfig,
    generate_transition_dataset,
    load_transition_dataset,
    train_dynamics_model,
    validate_transition_dataset_schema,
)
from mini_tft.search.mcts import RAW_PRIMITIVE_LEGACY_ACTION_SURFACE, MCTSConfig
from mini_tft.tools.stage4_dynamics import run_stage4_loop


def test_transition_dataset_schema_has_legal_masked_targets(tmp_path: Path) -> None:
    config = EnvConfig(seed=5, starting_gold=10, max_actions_per_round=4, max_round=4)
    dataset_path = tmp_path / "transition.npz"

    generate_transition_dataset(
        out_path=dataset_path,
        suite_name="mixed",
        episodes=1,
        seed=5,
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

    dataset = load_transition_dataset(dataset_path)
    validate_transition_dataset_schema(dataset, env_config=config)
    assert dataset.size > 0
    assert dataset.obs.shape == dataset.next_obs.shape
    assert np.all(dataset.masks[np.arange(dataset.size), dataset.actions])
    assert np.allclose(dataset.policy_targets.sum(axis=1), 1.0)
    assert np.all(
        (dataset.policy_targets * dataset.masks.astype(np.float32)).sum(axis=1) > 0.999
    )


def test_dynamics_trainer_checkpoint_load_and_model_search_legal_action(
    tmp_path: Path,
) -> None:
    config = EnvConfig(seed=6, starting_gold=10, max_actions_per_round=4, max_round=4)
    dataset_path = tmp_path / "transition.npz"
    checkpoint_path = tmp_path / "dynamics.pt"

    generate_transition_dataset(
        out_path=dataset_path,
        suite_name="mixed",
        episodes=1,
        seed=6,
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
    report = train_dynamics_model(
        dataset_paths=[dataset_path],
        out_path=checkpoint_path,
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        hidden_sizes=(64,),
        latent_dim=32,
        seed=6,
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert report["status"] == "pass"
    assert report["final_next_obs_loss"] is not None
    assert report["final_next_obs_loss"] < report["persistence_next_obs_mse"]

    data = load_set()
    lobby = new_lobby_state(config, data, seed=6, player_count=4)
    mask = lobby_action_mask(lobby, 0, data, config)
    planner = ModelBackedRootPlanner(
        checkpoint_path=checkpoint_path,
        config=ModelSearchConfig(
            simulations=2,
            max_root_actions=2,
            simulator_value_guard=True,
            simulator_value_threshold=0.02,
            simulator_value_max_actions=6,
        ),
        device="cpu",
        prior_policy=fast_level_lobby_policy,
    )
    decision = planner.plan(
        state=lobby,
        mask=mask,
        data=data,
        env_config=config,
        rng=np.random.default_rng(6),
    )

    assert mask[decision.selected_action]
    assert decision.simulations_run == 2
    assert decision.trace["action_surface"] == RAW_PRIMITIVE_LEGACY_ACTION_SURFACE
    assert decision.trace["considered_actions"]
    assert decision.trace["simulator_value_guard"] is not None


def test_stage4_loop_smoke_writes_required_artifacts(tmp_path: Path) -> None:
    config = EnvConfig(seed=7, starting_gold=10, max_actions_per_round=4, max_round=4)
    stage3_dir = tmp_path / "stage3"
    stage3_dataset = stage3_dir / "student_dataset.npz"
    stage3_checkpoint = stage3_dir / "promoted" / "student_stage3.pt"

    generate_teacher_dataset(
        out_path=stage3_dataset,
        suite_name="mixed",
        episodes=1,
        seed=7,
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
    train_student(
        dataset_paths=[stage3_dataset],
        out_path=stage3_checkpoint,
        epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        selected_action_coef=3.0,
        hidden_sizes=(64,),
        seed=7,
        device="cpu",
    )
    stage3_decision = stage3_dir / "decision.md"
    stage3_metrics = stage3_dir / "metrics.json"
    stage3_decision.write_text("**Decision:** `PASS_STAGE_3`\n", encoding="utf-8")
    stage3_metrics.write_text(
        json.dumps({"classification": "PASS_STAGE_3", "best_checkpoint": str(stage3_checkpoint)}),
        encoding="utf-8",
    )
    stage2_metrics = tmp_path / "stage2_metrics.json"
    stage2_metrics.write_text(
        json.dumps(
            {
                "classification": "PASS_TO_STAGE_3",
                "mcts_mixed_eval": {"mcts_decisions_per_sec": 1.0},
                "mcts_frozen_eval": {"mcts_decisions_per_sec": 1.0},
            }
        ),
        encoding="utf-8",
    )

    report = run_stage4_loop(
        out_dir=tmp_path / "stage4",
        stage2_metrics=stage2_metrics,
        stage3_decision=stage3_decision,
        stage3_metrics=stage3_metrics,
        stage3_checkpoint=stage3_checkpoint,
        ppo_checkpoint=Path("checkpoints/ppo_lobby_frozenpool_probe10k2_lr1e5_20260625T1052_h256.zip"),
        frozen_pool=(),
        attempts=1,
        max_wall_clock_sec=120.0,
        teacher_episodes=1,
        eval_episodes=1,
        seed=700,
        eval_seed=800,
        player_count=4,
        max_actions_per_player=4,
        teacher_simulations=2,
        teacher_max_root_actions=2,
        model_simulations=2,
        model_max_root_actions=2,
        train_epochs=2,
        batch_size=64,
        hidden_sizes=(32,),
        latent_dim=16,
        device="cpu",
        player_order_mode="fixed",
    )

    assert (tmp_path / "stage4" / "datasets").is_dir()
    assert (tmp_path / "stage4" / "checkpoints").is_dir()
    assert (tmp_path / "stage4" / "evals").is_dir()
    assert (tmp_path / "stage4" / "loops").is_dir()
    assert (tmp_path / "stage4" / "metrics.json").is_file()
    assert (tmp_path / "stage4" / "decision.md").is_file()
    assert (tmp_path / "stage4" / "loop-run-log.md").is_file()
    assert report["attempts"]
