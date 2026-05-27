"""Compatibility exports for the core actions module."""

from mini_tft.core.actions import (
    BUY_SHOP_OFFSET,
    MOVE_BENCH_TO_BOARD_OFFSET,
    MOVE_BOARD_TO_BENCH_OFFSET,
    NUM_ACTIONS,
    SELL_BENCH_OFFSET,
    Action,
    action_name,
    decode_move_bench_to_board_action,
    decode_move_board_to_bench_action,
    is_move_bench_to_board_action,
    is_move_board_to_bench_action,
    move_bench_to_board_action,
    move_board_to_bench_action,
)
from mini_tft.core.masks import legal_action_mask

__all__ = [
    "BUY_SHOP_OFFSET",
    "MOVE_BENCH_TO_BOARD_OFFSET",
    "MOVE_BOARD_TO_BENCH_OFFSET",
    "NUM_ACTIONS",
    "SELL_BENCH_OFFSET",
    "Action",
    "action_name",
    "decode_move_bench_to_board_action",
    "decode_move_board_to_bench_action",
    "is_move_bench_to_board_action",
    "is_move_board_to_bench_action",
    "legal_action_mask",
    "move_bench_to_board_action",
    "move_board_to_bench_action",
]
