"""Trait counting and abstract effect activation."""

from __future__ import annotations

from mini_tft.core.set_data import GameData
from mini_tft.core.state import UnitInstance


def trait_counts(board: list[UnitInstance | None], data: GameData) -> dict[str, int]:
    """Count traits by unique unit ID on board."""

    unit_ids = {unit.unit_id for unit in board if unit is not None}
    counts = {trait: 0 for trait in data.trait_ids}
    for unit_id in unit_ids:
        for trait in data.units[unit_id].traits:
            counts[trait] += 1
    return counts


def active_trait_effects(
    board: list[UnitInstance | None],
    data: GameData,
) -> dict[str, dict[str, float]]:
    counts = trait_counts(board, data)
    active: dict[str, dict[str, float]] = {}
    for trait_id, count in counts.items():
        trait = data.traits[trait_id]
        reached = [breakpoint for breakpoint in trait.breakpoints if breakpoint <= count]
        if reached:
            active[trait_id] = trait.breakpoints[max(reached)]
    return active


def trait_count_vector(board: list[UnitInstance | None], data: GameData) -> list[float]:
    counts = trait_counts(board, data)
    return [counts[trait] / 6.0 for trait in data.trait_ids]
