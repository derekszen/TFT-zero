from __future__ import annotations

from copy import deepcopy

import numpy as np

from mini_tft.core.actions import Action
from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import lobby_action_mask, lobby_signature, new_lobby_state
from mini_tft.core.lobby_step import fast_level_lobby_policy, tempo_lobby_policy
from mini_tft.core.set_data import load_set
from mini_tft.rl.lobby_env import MiniTFTLobbyEnv
from mini_tft.search.mcts import (
    RAW_PRIMITIVE_LEGACY_ACTION_SURFACE,
    MCTSConfig,
    RootMCTSPlanner,
    forced_action_rollout_value,
)


def test_lobby_clone_restore_replays_same_policy_branch() -> None:
    config = EnvConfig(seed=0, starting_gold=10, max_actions_per_round=4, max_round=4)
    env = MiniTFTLobbyEnv(
        seed=0,
        config=config,
        player_count=4,
        policy=fast_level_lobby_policy,
        max_actions_per_player=4,
        player_order_mode="fixed",
    )
    env.reset(seed=123)
    snapshot = env.clone_state()

    env.step(None)
    signature_after_first_step = env.state_signature()
    env.restore_state(snapshot)
    env.step(None)

    assert env.state_signature() == signature_after_first_step


def test_mcts_selects_legal_action_and_exports_visit_values() -> None:
    config = EnvConfig(seed=7, starting_gold=10, max_actions_per_round=4, max_round=4)
    data = load_set()
    state = new_lobby_state(config, data, seed=7, player_count=4)
    mask = lobby_action_mask(state, 0, data, config)
    planner = RootMCTSPlanner(
        config=MCTSConfig(
            simulations=4,
            rollout_rounds=1,
            max_root_actions=4,
            max_actions_per_player=4,
        ),
        opponent_policy=tempo_lobby_policy,
        rollout_policy=fast_level_lobby_policy,
    )
    planner.set_episode_context(episode=0, seed=7)

    decision = planner.plan(
        state=state,
        mask=mask,
        data=data,
        env_config=config,
        rng=np.random.default_rng(99),
    )

    assert mask[decision.selected_action]
    assert decision.simulations_run == 4
    assert decision.illegal_actions == 0
    assert decision.trace["selected_action"] == decision.selected_action
    assert decision.trace["action_surface"] == RAW_PRIMITIVE_LEGACY_ACTION_SURFACE
    assert decision.trace["legal_actions"]
    assert all(row["visits"] > 0 for row in decision.trace["action_stats"])
    assert all(row["mean_value"] is not None for row in decision.trace["action_stats"])


def test_mcts_prior_guard_keeps_legal_prior_action_when_margin_is_high() -> None:
    config = EnvConfig(seed=7, starting_gold=10, max_actions_per_round=4, max_round=4)
    data = load_set()
    state = new_lobby_state(config, data, seed=7, player_count=4)
    mask = lobby_action_mask(state, 0, data, config)

    def end_turn_prior(*_args: object) -> int:
        return int(Action.END_TURN)

    planner = RootMCTSPlanner(
        config=MCTSConfig(
            simulations=4,
            rollout_rounds=1,
            max_root_actions=4,
            max_actions_per_player=4,
            prior_improvement_margin=999.0,
        ),
        opponent_policy=tempo_lobby_policy,
        rollout_policy=fast_level_lobby_policy,
        prior_policy=end_turn_prior,
    )

    decision = planner.plan(
        state=state,
        mask=mask,
        data=data,
        env_config=config,
        rng=np.random.default_rng(99),
    )

    assert decision.selected_action == int(Action.END_TURN)
    assert decision.trace["prior_action"] == int(Action.END_TURN)
    assert decision.trace["selected_from_prior"] is True


def test_mcts_planning_does_not_mutate_live_state_or_rng() -> None:
    config = EnvConfig(seed=8, starting_gold=10, max_actions_per_round=4, max_round=4)
    data = load_set()
    state = new_lobby_state(config, data, seed=8, player_count=4)
    mask = lobby_action_mask(state, 0, data, config)
    signature_before = lobby_signature(state)
    rng = np.random.default_rng(123)
    rng_state_before = deepcopy(rng.bit_generator.state)
    planner = RootMCTSPlanner(
        config=MCTSConfig(
            simulations=4,
            rollout_rounds=1,
            max_root_actions=4,
            max_actions_per_player=4,
        ),
        opponent_policy=tempo_lobby_policy,
        rollout_policy=fast_level_lobby_policy,
    )

    planner.plan(
        state=state,
        mask=mask,
        data=data,
        env_config=config,
        rng=rng,
    )

    assert lobby_signature(state) == signature_before
    assert rng.bit_generator.state == rng_state_before


def test_forced_action_rollout_uses_legal_branch_without_illegal_actions() -> None:
    config = EnvConfig(seed=9, starting_gold=10, max_actions_per_round=4, max_round=4)
    data = load_set()
    state = new_lobby_state(config, data, seed=9, player_count=4)
    mask = lobby_action_mask(state, 0, data, config)
    action = int(Action.BUY_XP if mask[int(Action.BUY_XP)] else Action.END_TURN)

    result = forced_action_rollout_value(
        state=state,
        action=action,
        data=data,
        env_config=config,
        rng=np.random.default_rng(9),
        opponent_policy=tempo_lobby_policy,
        rollout_policy=fast_level_lobby_policy,
        rollout_rounds=1,
        max_actions_per_player=4,
    )

    assert result.illegal_actions == 0
    assert result.rollout_depth == 1
