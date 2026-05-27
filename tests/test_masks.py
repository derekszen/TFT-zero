from __future__ import annotations

from mini_tft import EnvConfig, MiniTFTEnv
from mini_tft.core.actions import Action, move_bench_to_board_action, move_board_to_bench_action
from mini_tft.core.state import UnitInstance


def test_mask_blocks_unaffordable_economy_actions() -> None:
    env = MiniTFTEnv(EnvConfig(seed=1, starting_gold=0))
    env.reset(seed=1)

    mask = env.action_masks()

    assert mask[Action.END_TURN]
    assert not mask[Action.ROLL]
    assert not mask[Action.BUY_XP]
    assert not mask[Action.SELL_BENCH_0]


def test_buying_unit_enables_sell_action() -> None:
    env = MiniTFTEnv(EnvConfig(seed=1, starting_gold=10))
    env.reset(seed=1)

    buy_actions = [
        action
        for action, legal in enumerate(env.action_masks())
        if legal and 3 <= action <= 7
    ]
    assert buy_actions

    env.step(buy_actions[0])

    assert env.action_masks()[Action.SELL_BENCH_0]


def test_mask_allows_explicit_bench_to_board_when_under_cap() -> None:
    env = MiniTFTEnv(EnvConfig(seed=2))
    env.reset(seed=2)
    assert env.state is not None
    env.state.bench[0] = UnitInstance(unit_id=1)

    mask = env.action_masks()

    assert mask[move_bench_to_board_action(0, 0)]
    assert not mask[move_bench_to_board_action(1, 0)]
    assert not mask[move_board_to_bench_action(0, 0)]


def test_mask_blocks_empty_board_target_when_board_is_full_but_allows_swaps() -> None:
    env = MiniTFTEnv(EnvConfig(seed=3))
    env.reset(seed=3)
    assert env.state is not None
    env.state.level = 1
    env.state.board[0] = UnitInstance(unit_id=1)
    env.state.bench[0] = UnitInstance(unit_id=2)

    mask = env.action_masks()

    assert mask[move_bench_to_board_action(0, 0)]
    assert not mask[move_bench_to_board_action(0, 1)]
    assert mask[move_board_to_bench_action(0, 0)]
