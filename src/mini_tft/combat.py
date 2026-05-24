"""Compatibility exports for the core combat module."""

from mini_tft.core.combat import (
    CombatResult,
    CombatStats,
    base_damage_by_round,
    board_strength,
    resolve_combat,
    unit_standalone_power,
)

__all__ = [
    "CombatResult",
    "CombatStats",
    "base_damage_by_round",
    "board_strength",
    "resolve_combat",
    "unit_standalone_power",
]
