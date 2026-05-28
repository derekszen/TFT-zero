from __future__ import annotations

import numpy as np

from mini_tft import EnvConfig, MiniTFTEnv
from mini_tft.core.actions import NUM_ACTIONS, Action, move_bench_to_board_action
from mini_tft.core.state import state_signature


def test_env_reset_and_end_turn_step() -> None:
    env = MiniTFTEnv(EnvConfig(seed=7))

    obs, info = env.reset(seed=7)

    assert env.observation_space.contains(obs)
    assert info["action_mask"].dtype == np.bool_
    assert info["action_mask"].shape == (NUM_ACTIONS,)
    assert info["action_mask"][Action.END_TURN]

    next_obs, reward, terminated, truncated, next_info = env.step(Action.END_TURN)

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert next_info["legal_action"] is True


def test_same_seed_and_actions_are_deterministic() -> None:
    actions = [Action.END_TURN, Action.END_TURN, Action.END_TURN, Action.END_TURN]

    signatures = []
    for _ in range(2):
        env = MiniTFTEnv(EnvConfig(seed=42))
        env.reset(seed=42)
        for action in actions:
            env.step(action)
        assert env.state is not None
        signatures.append(state_signature(env.state))

    assert signatures[0] == signatures[1]


def test_episode_terminates_at_max_round() -> None:
    env = MiniTFTEnv(EnvConfig(seed=3, max_round=2))
    env.reset(seed=3)

    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(Action.END_TURN)

    assert terminated
    assert env.state is not None
    assert env.state.final_reason in {"hp_zero", "max_round"}


def test_explicit_placement_action_moves_unit_from_bench_to_board() -> None:
    env = MiniTFTEnv(EnvConfig(seed=11, starting_gold=10))
    env.reset(seed=11)

    buy_action = next(
        action
        for action, legal in enumerate(env.action_masks())
        if legal and Action.BUY_SHOP_0 <= action <= Action.BUY_SHOP_4
    )
    env.step(buy_action)
    assert env.state is not None
    unit_id = env.state.bench[0].unit_id if env.state.bench[0] is not None else None

    action = move_bench_to_board_action(0, 0)
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert reward > 0
    assert not terminated
    assert not truncated
    assert info["legal_action"] is True
    assert env.state.board[0] is not None
    assert env.state.board[0].unit_id == unit_id
    assert env.state.bench[0] is None
    assert env.state.step_count == 2


def test_max_actions_per_round_auto_ends_turn() -> None:
    env = MiniTFTEnv(EnvConfig(seed=12, starting_gold=20, max_actions_per_round=2))
    env.reset(seed=12)

    _, _, terminated, truncated, info = env.step(Action.ROLL)

    assert not terminated
    assert not truncated
    assert info["round"] == 1
    assert info["round_action_count"] == 1
    assert info["auto_end_turn"] is False

    _, _, terminated, truncated, info = env.step(Action.ROLL)

    assert not terminated
    assert not truncated
    assert info["round"] == 2
    assert info["round_action_count"] == 0
    assert info["auto_end_turn"] is True
