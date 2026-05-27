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
SHOP_SIZE = 5
BENCH_SIZE = 9
BOARD_SIZE = 9

MOVE_BENCH_TO_BOARD_OFFSET = SLAM_BEST_ITEM + 1
MOVE_BENCH_TO_BOARD_COUNT = BENCH_SIZE * BOARD_SIZE
MOVE_BOARD_TO_BENCH_OFFSET = MOVE_BENCH_TO_BOARD_OFFSET + MOVE_BENCH_TO_BOARD_COUNT
MOVE_BOARD_TO_BENCH_COUNT = BOARD_SIZE * BENCH_SIZE
NUM_ACTIONS = MOVE_BOARD_TO_BENCH_OFFSET + MOVE_BOARD_TO_BENCH_COUNT


def action_name(action: int) -> str:
    """Return a stable debug name for an action."""

    if is_move_bench_to_board_action(action):
        bench_index, board_index = decode_move_bench_to_board_action(action)
        return f"move_bench_{bench_index + 1}_to_board_{board_index + 1}"
    if is_move_board_to_bench_action(action):
        board_index, bench_index = decode_move_board_to_bench_action(action)
        return f"move_board_{board_index + 1}_to_bench_{bench_index + 1}"
    try:
        return Action(action).name.lower()
    except ValueError:
        return "unknown"


def is_buy_shop_action(action: int) -> bool:
    return Action.BUY_SHOP_0 <= action <= Action.BUY_SHOP_4


def is_sell_bench_action(action: int) -> bool:
    return Action.SELL_BENCH_0 <= action <= Action.SELL_BENCH_8


def move_bench_to_board_action(bench_index: int, board_index: int) -> int:
    _validate_slot(bench_index, BENCH_SIZE, "bench")
    _validate_slot(board_index, BOARD_SIZE, "board")
    return MOVE_BENCH_TO_BOARD_OFFSET + bench_index * BOARD_SIZE + board_index


def move_board_to_bench_action(board_index: int, bench_index: int) -> int:
    _validate_slot(board_index, BOARD_SIZE, "board")
    _validate_slot(bench_index, BENCH_SIZE, "bench")
    return MOVE_BOARD_TO_BENCH_OFFSET + board_index * BENCH_SIZE + bench_index


def is_move_bench_to_board_action(action: int) -> bool:
    return MOVE_BENCH_TO_BOARD_OFFSET <= action < MOVE_BOARD_TO_BENCH_OFFSET


def is_move_board_to_bench_action(action: int) -> bool:
    return MOVE_BOARD_TO_BENCH_OFFSET <= action < NUM_ACTIONS


def decode_move_bench_to_board_action(action: int) -> tuple[int, int]:
    if not is_move_bench_to_board_action(action):
        raise ValueError(f"Not a bench-to-board move action: {action}")
    offset = action - MOVE_BENCH_TO_BOARD_OFFSET
    return divmod(offset, BOARD_SIZE)


def decode_move_board_to_bench_action(action: int) -> tuple[int, int]:
    if not is_move_board_to_bench_action(action):
        raise ValueError(f"Not a board-to-bench move action: {action}")
    offset = action - MOVE_BOARD_TO_BENCH_OFFSET
    return divmod(offset, BENCH_SIZE)


def _validate_slot(index: int, size: int, label: str) -> None:
    if not 0 <= index < size:
        raise ValueError(f"{label} index out of range: {index}")
