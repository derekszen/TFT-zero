"""Compatibility exports for the core economy module."""

from mini_tft.core.economy import (
    LEVEL_XP,
    apply_xp,
    income_after_combat,
    interest,
    sell_value,
    xp_needed,
)

__all__ = [
    "LEVEL_XP",
    "apply_xp",
    "income_after_combat",
    "interest",
    "sell_value",
    "xp_needed",
]
