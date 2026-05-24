"""Discrete V0 action contract."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    END_TURN = 0
    ROLL = 1
    BUY_XP = 2
    BUY_SHOP_0 = 3
    BUY_SHOP_1 = 4
    BUY_SHOP_2 = 5
    BUY_SHOP_3 = 6
    BUY_SHOP_4 = 7
    SELL_BENCH_0 = 8
    SELL_BENCH_1 = 9
    SELL_BENCH_2 = 10
    SELL_BENCH_3 = 11
    SELL_BENCH_4 = 12
    SELL_BENCH_5 = 13
    SELL_BENCH_6 = 14
    SELL_BENCH_7 = 15
    SELL_BENCH_8 = 16
    FIELD_BEST_BOARD = 17
    SLAM_BEST_ITEM = 18


BUY_SHOP_OFFSET = Action.BUY_SHOP_0.value
SELL_BENCH_OFFSET = Action.SELL_BENCH_0.value
FIELD_BEST_BOARD = Action.FIELD_BEST_BOARD.value
SLAM_BEST_ITEM = Action.SLAM_BEST_ITEM.value
NUM_ACTIONS = 19


def action_name(action: int) -> str:
    """Return a stable debug name for an action."""

    try:
        return Action(action).name.lower()
    except ValueError:
        return "unknown"


def is_buy_shop_action(action: int) -> bool:
    return Action.BUY_SHOP_0 <= action <= Action.BUY_SHOP_4


def is_sell_bench_action(action: int) -> bool:
    return Action.SELL_BENCH_0 <= action <= Action.SELL_BENCH_8
