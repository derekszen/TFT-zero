from __future__ import annotations

import numpy as np

from mini_tft.core.actions import BUY_SHOP_OFFSET, NUM_ACTIONS, Action
from mini_tft.core.combat import base_damage_by_round
from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby import lobby_signature, new_lobby_state, resolve_lobby_round
from mini_tft.core.lobby_step import (
    apply_lobby_action,
    fast_level_lobby_policy,
    lobby_legal_action_mask,
    mixed_lobby_policy,
    ordered_active_players,
    step_lobby_round_with_policies,
    tempo_lobby_policy,
)
from mini_tft.core.set_data import load_set
from mini_tft.core.state import UnitInstance
from mini_tft.rl.lobby_env import MiniTFTLobbyEnv, MiniTFTLobbyHeroEnv, MiniTFTLobbySnapshot
from mini_tft.tools.evaluate_lobby_policy import run_lobby_evaluation
from mini_tft.tools.set1_lobby_step_smoke import run_lobby_step_smoke


def test_lobby_action_mask_respects_shared_pool_availability() -> None:
    config = EnvConfig(seed=0, starting_gold=10)
    data = load_set()
    lobby = new_lobby_state(config, data, seed=0, player_count=2)
    player = lobby.players[0]
    player.shop[0] = 1
    lobby.pool.counts[1] = 0

    mask = lobby_legal_action_mask(lobby, 0, data, config)

    assert not mask[BUY_SHOP_OFFSET]


def test_apply_lobby_buy_sell_round_trips_pool_copy() -> None:
    config = EnvConfig(seed=0, starting_gold=10)
    data = load_set()
    lobby = new_lobby_state(config, data, seed=0, player_count=2)
    player = lobby.players[0]
    player.shop[0] = 1
    before = lobby.pool.available(1)

    assert apply_lobby_action(lobby, 0, Action.BUY_SHOP_0, data, config, np.random.default_rng(1))
    assert lobby.pool.available(1) == before - 1
    assert player.bench[0] is not None

    assert apply_lobby_action(
        lobby,
        0,
        Action.SELL_BENCH_0,
        data,
        config,
        np.random.default_rng(1),
    )
    assert lobby.pool.available(1) == before
    assert player.bench[0] is None


def test_step_all_players_takes_policy_actions_then_resolves_combat() -> None:
    config = EnvConfig(seed=0, starting_gold=10, max_actions_per_round=8)
    data = load_set()
    lobby = new_lobby_state(config, data, seed=10, player_count=8)
    rng = np.random.default_rng(99)

    result = step_lobby_round_with_policies(
        lobby,
        data,
        config,
        rng,
        tempo_lobby_policy,
        max_actions_per_player=8,
    )

    assert result.round == 1
    assert result.player_order == tuple(range(8))
    assert result.action_count > 8
    assert result.resolved_fights == 4
    assert result.illegal_actions == 0
    assert lobby.round == 2
    assert sum(player.total_units_bought for player in lobby.players) > 0
    assert all(any(unit is not None for unit in player.board) for player in lobby.players)


def test_step_all_players_is_deterministic_for_same_seed_and_policy() -> None:
    config = EnvConfig(seed=0, starting_gold=10, max_actions_per_round=8)
    data = load_set()
    lobby_a = new_lobby_state(config, data, seed=50, player_count=8)
    lobby_b = new_lobby_state(config, data, seed=50, player_count=8)

    step_lobby_round_with_policies(
        lobby_a,
        data,
        config,
        np.random.default_rng(123),
        tempo_lobby_policy,
        max_actions_per_player=8,
    )
    step_lobby_round_with_policies(
        lobby_b,
        data,
        config,
        np.random.default_rng(123),
        tempo_lobby_policy,
        max_actions_per_player=8,
    )

    assert lobby_signature(lobby_a) == lobby_signature(lobby_b)


def test_lobby_upset_damage_uses_winner_margin_not_absolute_gap() -> None:
    config = EnvConfig(seed=0, combat_sigmoid_scale=1_000_000.0)
    data = load_set()
    lobby = new_lobby_state(config, data, seed=52, player_count=2)
    lobby.players[0].board = [UnitInstance(1), *([None] * 8)]
    lobby.players[1].board = [
        UnitInstance(20, stars=2),
        UnitInstance(21, stars=2),
        None,
        None,
        None,
        None,
        UnitInstance(18, stars=2, items=[1, 2]),
        None,
        None,
    ]

    results = resolve_lobby_round(
        lobby,
        data,
        config,
        np.random.default_rng(2),
        matchups=[(0, 1)],
    )

    result = results[0]
    assert result.strength_a < result.strength_b
    assert result.winner == 0
    assert result.damage == base_damage_by_round(result.round)
    assert lobby.players[1].hp == config.starting_hp - result.damage


def test_mixed_lobby_policy_is_deterministic_for_same_seed() -> None:
    config = EnvConfig(seed=0, starting_gold=10, max_actions_per_round=8)
    data = load_set()
    lobby_a = new_lobby_state(config, data, seed=51, player_count=8)
    lobby_b = new_lobby_state(config, data, seed=51, player_count=8)

    step_lobby_round_with_policies(
        lobby_a,
        data,
        config,
        np.random.default_rng(124),
        mixed_lobby_policy,
        max_actions_per_player=8,
    )
    step_lobby_round_with_policies(
        lobby_b,
        data,
        config,
        np.random.default_rng(124),
        mixed_lobby_policy,
        max_actions_per_player=8,
    )

    assert lobby_signature(lobby_a) == lobby_signature(lobby_b)


def test_lobby_env_reset_step_and_restore_exposes_metrics() -> None:
    config = EnvConfig(seed=0, starting_gold=10, max_actions_per_round=4, max_round=4)
    env = MiniTFTLobbyEnv(
        seed=0,
        config=config,
        player_count=4,
        policy=fast_level_lobby_policy,
        max_actions_per_player=4,
        player_order_mode="random",
    )
    obs, info = env.reset(seed=123)
    snapshot = env.clone_state()

    assert obs["players"].shape == (4, 76)
    assert obs["action_masks"].shape == (4, NUM_ACTIONS)
    assert obs["active"].tolist() == [True, True, True, True]
    assert obs["hp"].tolist() == [100.0, 100.0, 100.0, 100.0]
    assert obs["placements"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert info["agents"] == ["player_0", "player_1", "player_2", "player_3"]

    env.step(None)
    first_signature = env.state_signature()
    env.restore_state(snapshot)
    env.step(None)

    assert isinstance(snapshot, MiniTFTLobbySnapshot)
    assert env.state_signature() == first_signature
    assert env.action_mask("player_0").shape == (NUM_ACTIONS,)


def test_lobby_hero_env_steps_player_zero_against_scripted_opponents() -> None:
    env = MiniTFTLobbyHeroEnv(
        seed=0,
        config=EnvConfig(seed=0, starting_gold=10, max_actions_per_round=4, max_round=4),
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        max_actions_per_player=4,
    )
    obs, info = env.reset(seed=123)

    assert obs.ndim == 1
    assert env.action_masks().shape == (NUM_ACTIONS,)
    assert info["hp"] == 100

    obs, reward, terminated, truncated, info = env.step(int(Action.END_TURN))

    assert obs.ndim == 1
    assert isinstance(reward, float)
    assert not (terminated and truncated)
    assert info["round"] >= 1


def test_lobby_hero_env_can_mask_oracle_macro_actions() -> None:
    config = EnvConfig(seed=0)
    env = MiniTFTLobbyHeroEnv(
        seed=0,
        config=config,
        player_count=4,
        opponent_policy=tempo_lobby_policy,
        allow_oracle_macro_actions=False,
    )
    _obs, _info = env.reset(seed=123)
    assert env.state is not None
    player = env.state.players[0]
    weak_unit_id = min(env.data.units, key=lambda unit_id: env.data.units[unit_id].base_power)
    strong_unit_id = max(env.data.units, key=lambda unit_id: env.data.units[unit_id].base_power)
    player.board[0] = UnitInstance(unit_id=weak_unit_id)
    player.bench[0] = UnitInstance(unit_id=strong_unit_id)
    player.item_bench.append(env.data.completed_item_ids[0])

    raw_mask = lobby_legal_action_mask(env.state, 0, env.data, config)
    filtered_mask = env.action_masks()

    assert raw_mask[Action.FIELD_BEST_BOARD]
    assert raw_mask[Action.SLAM_BEST_ITEM]
    assert not filtered_mask[Action.FIELD_BEST_BOARD]
    assert not filtered_mask[Action.SLAM_BEST_ITEM]


def test_player_order_modes_cover_fixed_random_and_rotating() -> None:
    config = EnvConfig(seed=0)
    data = load_set()
    lobby = new_lobby_state(config, data, seed=50, player_count=4)

    assert ordered_active_players(lobby, np.random.default_rng(1), "fixed") == (0, 1, 2, 3)
    assert ordered_active_players(lobby, np.random.default_rng(1), "rotating", 1) == (
        1,
        2,
        3,
        0,
    )
    random_order = ordered_active_players(lobby, np.random.default_rng(1), "random")
    assert sorted(random_order) == [0, 1, 2, 3]
    assert random_order == ordered_active_players(lobby, np.random.default_rng(1), "random")


def test_lobby_policy_evaluation_reports_placement_hp_and_top_rates() -> None:
    report = run_lobby_evaluation(
        episodes=2,
        seed=7,
        hero_policy_name="fast_level",
        opponent_policy_name="tempo",
        player_count=4,
        max_actions_per_player=4,
        config=EnvConfig(seed=7, starting_gold=10, max_actions_per_round=4, max_round=4),
    )

    assert report["status"] == "pass"
    assert 1.0 <= report["mean_placement"] <= 4.0
    assert 0.0 <= report["top1_rate"] <= 1.0
    assert 0.0 <= report["top4_rate"] <= 1.0
    assert report["mean_final_hp"] >= 0.0
    assert sum(report["placement_histogram"].values()) == 2
    assert report["hero_actions"] > 0
    assert report["total_macro_actions"] == (
        report["total_field_best_board_actions"] + report["total_slam_best_item_actions"]
    )
    assert report["hero_macro_actions"] == (
        report["hero_field_best_board_actions"] + report["hero_slam_best_item_actions"]
    )
    assert 0.0 <= report["hero_macro_action_rate"] <= 1.0


def test_lobby_policy_evaluation_can_disallow_and_gate_hero_macro_actions() -> None:
    config = EnvConfig(seed=7, starting_gold=10, max_actions_per_round=4, max_round=4)
    gated_report = run_lobby_evaluation(
        episodes=1,
        seed=7,
        hero_policy_name="fast_level",
        opponent_policy_name="tempo",
        player_count=4,
        max_actions_per_player=4,
        config=config,
        allow_hero_macro_actions=True,
        max_hero_macro_action_rate=0.0,
    )
    disallowed_report = run_lobby_evaluation(
        episodes=1,
        seed=7,
        hero_policy_name="fast_level",
        opponent_policy_name="tempo",
        player_count=4,
        max_actions_per_player=4,
        config=config,
        allow_hero_macro_actions=False,
        max_hero_macro_action_rate=0.0,
    )

    assert gated_report["hero_macro_actions"] > 0
    assert gated_report["status"] == "fail"
    assert disallowed_report["allow_hero_macro_actions"] is False
    assert disallowed_report["hero_macro_actions"] == 0
    assert disallowed_report["hero_macro_action_rate"] == 0.0
    assert disallowed_report["status"] == "pass"


def test_lobby_step_smoke_report_has_throughput_and_hp() -> None:
    report = run_lobby_step_smoke(lobbies=2, max_rounds=2, seed=0)

    assert report["status"] == "pass"
    assert report["resolved_fights"] > 0
    assert report["actions"] > 0
    assert report["rounds_per_sec"] > 0
    assert report["mean_final_hp"] >= 0.0
    assert report["mean_final_board_strength"] >= 0.0
