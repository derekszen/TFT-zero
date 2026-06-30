from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from gymnasium import spaces

from mini_tft.core.actions import Action
from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import tempo_lobby_policy
from mini_tft.rl.gym_env import MiniTFTGymEnv
from mini_tft.rl.lobby_env import MiniTFTLobbyEnv
from mini_tft.rl.puffer_env import (
    MiniTFTLobbyFlatEnv,
    MiniTFTLobbyHeroMaskInObsEnv,
    MiniTFTLobbyHeroNativePufferEnv,
    MiniTFTMaskInObsEnv,
    StrategicMaskInObsEnv,
    StrategicNativePufferEnv,
    flatten_lobby_observation,
    make_lobby_hero_native_puffer_env,
    make_lobby_hero_puffer_env,
    make_lobby_puffer_env,
    make_puffer_env,
    make_strategic_native_puffer_env,
    make_strategic_puffer_env,
)
from mini_tft.strategic.core import StrategicAction, StrategicConfig


def test_single_player_mask_in_obs_matches_base_env_reset_and_step() -> None:
    config = EnvConfig(seed=7, starting_gold=10, max_round=3)
    base = MiniTFTGymEnv(config=config)
    wrapped = MiniTFTMaskInObsEnv(config=config)

    base_obs, base_info = base.reset(seed=123)
    wrapped_obs, wrapped_info = wrapped.reset(seed=123)

    assert wrapped.observation_space.contains(wrapped_obs)
    np.testing.assert_allclose(wrapped_obs[: wrapped.base_observation_dim], base_obs)
    np.testing.assert_array_equal(
        wrapped_obs[wrapped.base_observation_dim :],
        base_info["action_mask"].astype(np.float32),
    )
    np.testing.assert_array_equal(wrapped_info["action_mask"], base_info["action_mask"])

    base_step = base.step(int(Action.END_TURN))
    wrapped_step = wrapped.step(int(Action.END_TURN))

    np.testing.assert_allclose(wrapped_step[0][: wrapped.base_observation_dim], base_step[0])
    np.testing.assert_array_equal(
        wrapped_step[0][wrapped.base_observation_dim :],
        base_step[4]["action_mask"].astype(np.float32),
    )
    assert wrapped_step[1:4] == base_step[1:4]


def test_lobby_flat_env_matches_lobby_env_reset_and_step() -> None:
    config = EnvConfig(seed=8, starting_gold=10, max_round=3, max_actions_per_round=2)
    base = MiniTFTLobbyEnv(
        seed=8,
        config=config,
        player_count=4,
        max_actions_per_player=2,
        player_order_mode="fixed",
    )
    wrapped = MiniTFTLobbyFlatEnv(
        seed=8,
        config=config,
        player_count=4,
        max_actions_per_player=2,
        player_order_mode="fixed",
    )

    base_obs, base_info = base.reset(seed=321)
    wrapped_obs, wrapped_info = wrapped.reset(seed=321)

    assert wrapped.observation_space.contains(wrapped_obs)
    np.testing.assert_allclose(wrapped_obs, flatten_lobby_observation(base_obs))
    assert wrapped_info["signature"] == base_info["signature"]

    action_plan = np.full((4, 2), int(Action.END_TURN), dtype=np.int64)
    base_step = base.step(action_plan)
    wrapped_step = wrapped.step(action_plan.reshape(-1))

    np.testing.assert_allclose(wrapped_step[0], flatten_lobby_observation(base_step[0]))
    assert wrapped_step[1:4] == base_step[1:4]
    assert wrapped_step[4]["signature"] == base_step[4]["signature"]


def test_lobby_hero_mask_in_obs_matches_base_env_reset_and_step() -> None:
    from mini_tft.rl.lobby_env import MiniTFTLobbyHeroEnv

    config = EnvConfig(seed=9, starting_gold=10, max_round=3, max_actions_per_round=2)
    base = MiniTFTLobbyHeroEnv(
        seed=9,
        config=config,
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=2,
        allow_oracle_macro_actions=False,
    )
    wrapped = MiniTFTLobbyHeroMaskInObsEnv(
        seed=9,
        config=config,
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=2,
        allow_oracle_macro_actions=False,
    )

    base_obs, base_info = base.reset(seed=987)
    wrapped_obs, wrapped_info = wrapped.reset(seed=987)

    assert wrapped.observation_space.contains(wrapped_obs)
    np.testing.assert_allclose(wrapped_obs[: wrapped.base_observation_dim], base_obs)
    np.testing.assert_array_equal(
        wrapped_obs[wrapped.base_observation_dim :],
        base_info["action_mask"].astype(np.float32),
    )
    np.testing.assert_array_equal(wrapped_info["action_mask"], base_info["action_mask"])

    action = int(Action.END_TURN)
    base_step = base.step(action)
    wrapped_step = wrapped.step(action)

    np.testing.assert_allclose(wrapped_step[0][: wrapped.base_observation_dim], base_step[0])
    np.testing.assert_array_equal(
        wrapped_step[0][wrapped.base_observation_dim :],
        base_step[4]["action_mask"].astype(np.float32),
    )
    assert wrapped_step[1:4] == base_step[1:4]


def test_strategic_mask_in_obs_reset_and_step() -> None:
    config = StrategicConfig(max_round=3, max_actions_per_round=2)
    env = StrategicMaskInObsEnv(seed=10, config=config)

    obs, info = env.reset(seed=432)
    action_space = cast(spaces.Discrete, env.action_space)

    assert env.observation_space.contains(obs)
    assert obs.shape[0] == env.base_observation_dim + action_space.n
    np.testing.assert_array_equal(
        obs[env.base_observation_dim :],
        info["action_mask"].astype(np.float32),
    )
    assert bool(info["action_mask"][StrategicAction.HOLD])

    next_obs, reward, terminated, truncated, next_info = env.step(int(StrategicAction.HOLD))

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert not (terminated and truncated)
    np.testing.assert_array_equal(
        next_obs[env.base_observation_dim :],
        next_info["action_mask"].astype(np.float32),
    )


def test_pufferlib_emulated_envs_reset_and_step() -> None:
    pytest.importorskip("pufferlib.emulation")

    single = make_puffer_env(seed=9, config=EnvConfig(seed=9, max_round=3))
    lobby = make_lobby_puffer_env(
        seed=10,
        config=EnvConfig(seed=10, max_round=3, max_actions_per_round=2),
        player_count=4,
    )
    lobby_hero = make_lobby_hero_puffer_env(
        seed=11,
        config=EnvConfig(seed=11, max_round=3, max_actions_per_round=2),
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=2,
        allow_oracle_macro_actions=False,
    )
    strategic = make_strategic_puffer_env(
        seed=12,
        config=StrategicConfig(max_round=3, max_actions_per_round=2),
    )
    try:
        single_obs, single_info = single.reset(seed=456)
        assert single_obs.shape == (1, single.observation_space.shape[0])
        assert single_info["action_mask"].shape[0] == cast(spaces.Discrete, single.action_space).n

        single_next, single_reward, single_terminated, single_truncated, _ = single.step(
            int(Action.END_TURN)
        )
        assert single_next.shape == single_obs.shape
        assert isinstance(single_reward, float)
        assert not (single_terminated and single_truncated)

        lobby_obs, lobby_info = lobby.reset(seed=654)
        lobby_action_space = cast(spaces.MultiDiscrete, lobby.action_space)
        assert lobby_obs.shape == (1, lobby.observation_space.shape[0])
        assert lobby_info["agents"] == ["player_0", "player_1", "player_2", "player_3"]

        lobby_action = np.full(lobby_action_space.nvec.shape, int(Action.END_TURN), dtype=np.int64)
        lobby_next, lobby_reward, lobby_terminated, lobby_truncated, _ = lobby.step(lobby_action)
        assert lobby_next.shape == lobby_obs.shape
        assert isinstance(lobby_reward, float)
        assert not (lobby_terminated and lobby_truncated)

        lobby_hero_obs, lobby_hero_info = lobby_hero.reset(seed=765)
        assert lobby_hero_obs.shape == (1, lobby_hero.observation_space.shape[0])
        assert lobby_hero_info["action_mask"].shape[0] == cast(
            spaces.Discrete,
            lobby_hero.action_space,
        ).n

        lobby_hero_next, lobby_hero_reward, hero_terminated, hero_truncated, _ = (
            lobby_hero.step(int(Action.END_TURN))
        )
        assert lobby_hero_next.shape == lobby_hero_obs.shape
        assert isinstance(lobby_hero_reward, float)
        assert not (hero_terminated and hero_truncated)

        strategic_obs, strategic_info = strategic.reset(seed=876)
        assert strategic_obs.shape == (1, strategic.observation_space.shape[0])
        assert strategic_info["action_mask"].shape[0] == cast(
            spaces.Discrete,
            strategic.action_space,
        ).n

        strategic_next, strategic_reward, strategic_terminated, strategic_truncated, _ = (
            strategic.step(int(StrategicAction.HOLD))
        )
        assert strategic_next.shape == strategic_obs.shape
        assert isinstance(strategic_reward, float)
        assert not (strategic_terminated and strategic_truncated)
    finally:
        single.close()
        lobby.close()
        lobby_hero.close()
        strategic.close()


def test_lobby_hero_native_puffer_env_reset_shape_and_mask_tail() -> None:
    pytest.importorskip("pufferlib.emulation")

    config = EnvConfig(seed=12, starting_gold=10, max_round=3, max_actions_per_round=2)
    env = MiniTFTLobbyHeroNativePufferEnv(
        num_envs=3,
        seed=12,
        config=config,
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=2,
        allow_oracle_macro_actions=False,
    )
    try:
        obs, infos = env.reset(seed=222)
        single_obs_shape = env.single_observation_space.shape
        assert single_obs_shape is not None

        assert obs.shape == (3, single_obs_shape[0])
        assert obs.dtype == np.float32
        assert len(infos) == 3
        assert env.action_masks().shape == (3, cast(spaces.Discrete, env.single_action_space).n)
        for env_index in range(3):
            np.testing.assert_array_equal(
                obs[env_index, env.base_observation_dim :],
                env.envs[env_index].action_masks().astype(np.float32),
            )
    finally:
        env.close()


def test_lobby_hero_native_puffer_env_matches_single_env_parity() -> None:
    pytest.importorskip("pufferlib.emulation")

    config = EnvConfig(seed=13, starting_gold=10, max_round=3, max_actions_per_round=2)
    base = MiniTFTLobbyHeroMaskInObsEnv(
        seed=13,
        config=config,
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=2,
        allow_oracle_macro_actions=False,
    )
    native = make_lobby_hero_native_puffer_env(
        num_envs=1,
        seed=13,
        config=config,
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=2,
        allow_oracle_macro_actions=False,
    )
    try:
        base_obs, base_info = base.reset(seed=333)
        native_obs, native_infos = native.reset(seed=333)

        np.testing.assert_allclose(native_obs[0], base_obs)
        np.testing.assert_array_equal(native_infos[0]["action_mask"], base_info["action_mask"])

        action = int(Action.END_TURN)
        base_step = base.step(action)
        native_step = native.step(np.array([action], dtype=np.int64))

        np.testing.assert_allclose(native_step[0][0], base_step[0])
        assert float(native_step[1][0]) == pytest.approx(base_step[1])
        assert bool(native_step[2][0]) == base_step[2]
        assert bool(native_step[3][0]) == base_step[3]
    finally:
        base.close()
        native.close()


def test_lobby_hero_native_puffer_env_steps_independent_agents() -> None:
    pytest.importorskip("pufferlib.emulation")

    config = EnvConfig(seed=14, starting_gold=10, max_round=3, max_actions_per_round=2)
    env = make_lobby_hero_native_puffer_env(
        num_envs=2,
        seed=14,
        config=config,
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=2,
        allow_oracle_macro_actions=False,
    )
    try:
        first_obs, _ = env.reset(seed=444)
        next_obs, rewards, terminals, truncations, infos = env.step(
            np.array([int(Action.END_TURN), int(Action.ROLL)], dtype=np.int64)
        )

        assert next_obs.shape == first_obs.shape
        assert rewards.shape == (2,)
        assert terminals.shape == (2,)
        assert truncations.shape == (2,)
        assert len(infos) == 2
        assert not np.shares_memory(next_obs[0], next_obs[1])
    finally:
        env.close()


def test_lobby_hero_native_puffer_env_auto_resets_completed_agents() -> None:
    pytest.importorskip("pufferlib.emulation")

    config = EnvConfig(seed=15, starting_gold=10, max_round=1, max_actions_per_round=1)
    env = make_lobby_hero_native_puffer_env(
        num_envs=2,
        seed=15,
        config=config,
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=1,
        allow_oracle_macro_actions=False,
    )
    try:
        env.reset(seed=555)
        saw_done = False
        for _ in range(20):
            obs, _rewards, terminals, truncations, infos = env.step(
                np.array([int(Action.END_TURN), int(Action.END_TURN)], dtype=np.int64)
            )
            done = np.logical_or(terminals, truncations)
            if bool(done.any()):
                saw_done = True
                done_index = int(np.flatnonzero(done)[0])
                assert "terminal_observation" in infos[done_index]
                assert "reset_info" in infos[done_index]
                assert env.single_observation_space.contains(obs[done_index])
                break

        assert saw_done
    finally:
        env.close()


def test_strategic_native_puffer_env_reset_shape_and_mask_tail() -> None:
    pytest.importorskip("pufferlib.emulation")

    config = StrategicConfig(max_round=3, max_actions_per_round=2)
    env = StrategicNativePufferEnv(num_envs=4, seed=16, config=config)
    try:
        obs, infos = env.reset(seed=666)
        single_obs_shape = env.single_observation_space.shape
        assert single_obs_shape is not None

        assert obs.shape == (4, single_obs_shape[0])
        assert obs.dtype == np.float32
        assert len(infos) == 4
        assert env.action_masks().shape == (4, cast(spaces.Discrete, env.single_action_space).n)
        for env_index in range(4):
            np.testing.assert_array_equal(
                obs[env_index, env.base_observation_dim :],
                env.action_masks()[env_index].astype(np.float32),
            )
    finally:
        env.close()


def test_strategic_native_puffer_env_matches_single_env_parity() -> None:
    pytest.importorskip("pufferlib.emulation")

    config = StrategicConfig(max_round=6, max_actions_per_round=3)
    base = StrategicMaskInObsEnv(seed=17, config=config)
    native = make_strategic_native_puffer_env(num_envs=1, seed=17, config=config)
    try:
        base_obs, base_info = base.reset(seed=777)
        native_obs, native_infos = native.reset(seed=777)

        np.testing.assert_allclose(native_obs[0], base_obs)
        np.testing.assert_array_equal(native_infos[0]["action_mask"], base_info["action_mask"])

        for action in (
            StrategicAction.BUY_HIGHEST_COST,
            StrategicAction.FIELD_STRONGEST,
            StrategicAction.HOLD,
            StrategicAction.ROLL,
            StrategicAction.GREED_ECON,
        ):
            base_step = base.step(int(action))
            native_step = native.step(np.array([int(action)], dtype=np.int64))
            np.testing.assert_allclose(native_step[0][0], base_step[0], rtol=1e-6, atol=1e-6)
            assert float(native_step[1][0]) == pytest.approx(base_step[1])
            assert bool(native_step[2][0]) == base_step[2]
            assert bool(native_step[3][0]) == base_step[3]
            if base_step[2] or base_step[3]:
                break
    finally:
        base.close()
        native.close()


def test_strategic_native_puffer_env_auto_resets_completed_agents() -> None:
    pytest.importorskip("pufferlib.emulation")

    config = StrategicConfig(max_round=1, max_actions_per_round=1)
    env = make_strategic_native_puffer_env(num_envs=2, seed=18, config=config)
    try:
        env.reset(seed=888)
        obs, _rewards, terminals, truncations, infos = env.step(
            np.array([int(StrategicAction.HOLD), int(StrategicAction.HOLD)], dtype=np.int64)
        )
        done = np.logical_or(terminals, truncations)

        assert bool(done.all())
        assert "terminal_observation" in infos[0]
        assert "reset_info" in infos[0]
        assert env.single_observation_space.contains(obs[0])
    finally:
        env.close()
