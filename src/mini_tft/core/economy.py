"""Simplified TFT economy helpers."""

from __future__ import annotations

from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData
from mini_tft.core.state import UnitInstance

LEVEL_XP = {
    3: 6,
    4: 10,
    5: 20,
    6: 36,
    7: 56,
    8: 80,
}


def interest(gold: int, cap: int = 5) -> int:
    return min(cap, max(0, gold) // 10)


def income_after_combat(gold: int, won: bool, config: EnvConfig) -> int:
    win_gold = config.win_gold if won else 0
    return config.base_income + win_gold + interest(gold + win_gold, config.max_interest)


def xp_needed(level: int) -> int:
    return LEVEL_XP.get(level, 999_999)


def apply_xp(level: int, xp: int, amount: int, max_level: int) -> tuple[int, int]:
    next_level = level
    next_xp = xp + amount
    while next_level < max_level and next_xp >= xp_needed(next_level):
        next_xp -= xp_needed(next_level)
        next_level += 1
    if next_level >= max_level:
        next_xp = 0
    return next_level, next_xp


def sell_value(unit: UnitInstance, data: GameData) -> int:
    return data.units[unit.unit_id].cost * (3 ** (unit.stars - 1))
