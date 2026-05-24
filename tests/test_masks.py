from __future__ import annotations

from mini_tft import EnvConfig, MiniTFTEnv
from mini_tft.core.actions import Action


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
