"""Compatibility exports for the core actions module."""

from mini_tft.core.actions import (
    BUY_SHOP_OFFSET,
    NUM_ACTIONS,
    SELL_BENCH_OFFSET,
    Action,
    action_name,
)
from mini_tft.core.masks import legal_action_mask

__all__ = [
    "BUY_SHOP_OFFSET",
    "NUM_ACTIONS",
    "SELL_BENCH_OFFSET",
    "Action",
    "action_name",
    "legal_action_mask",
]
