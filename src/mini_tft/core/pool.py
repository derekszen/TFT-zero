"""Shared champion-pool helpers for lobby simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mini_tft.core.ids import EMPTY
from mini_tft.core.set_data import GameData
from mini_tft.core.state import UnitInstance

DEFAULT_COPIES_BY_COST = {
    1: 29,
    2: 22,
    3: 18,
    4: 12,
    5: 10,
}


@dataclass
class SharedUnitPool:
    """Finite champion copy pool shared by all players in a lobby."""

    counts: dict[int, int]

    @classmethod
    def from_data(
        cls,
        data: GameData,
        copies_by_cost: dict[int, int] | None = None,
    ) -> SharedUnitPool:
        copies = copies_by_cost or DEFAULT_COPIES_BY_COST
        return cls({unit_id: copies[data.units[unit_id].cost] for unit_id in data.units})

    def clone(self) -> SharedUnitPool:
        return SharedUnitPool(dict(self.counts))

    def available(self, unit_id: int) -> int:
        return self.counts.get(unit_id, 0)

    def can_take(self, unit_id: int, copies: int = 1) -> bool:
        return copies >= 0 and self.available(unit_id) >= copies

    def take(self, unit_id: int, copies: int = 1) -> bool:
        if not self.can_take(unit_id, copies):
            return False
        self.counts[unit_id] -= copies
        return True

    def return_copies(self, unit_id: int, copies: int = 1) -> None:
        if copies < 0:
            raise ValueError("copies must be non-negative")
        self.counts[unit_id] = self.available(unit_id) + copies

    def return_unit(self, unit: UnitInstance) -> None:
        self.return_copies(unit.unit_id, copies_for_stars(unit.stars))

    def total_remaining(self) -> int:
        return sum(self.counts.values())

    def signature(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self.counts.items()))


def copies_for_stars(stars: int) -> int:
    """Return underlying one-star copies represented by a unit instance."""

    if stars < 1:
        raise ValueError("stars must be positive")
    return 3 ** (stars - 1)


def sample_shop_from_pool(
    data: GameData,
    pool: SharedUnitPool,
    level: int,
    size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample shop offers from currently available shared-pool copies.

    Offers do not reserve copies. A local duplicate guard only prevents one
    generated shop from showing more copies of a unit than the pool has.
    """

    available_level = max(key for key in data.shop_odds if key <= level)
    base_weights = data.shop_odds[available_level]
    costs = np.arange(1, 6)
    local_counts: dict[int, int] = {}
    shop: list[int] = []

    for _ in range(size):
        cost_candidates = [
            cost
            for cost in costs
            if _available_units_for_cost(data, pool, int(cost), local_counts)
        ]
        if not cost_candidates:
            shop.append(EMPTY)
            continue

        weights = np.asarray([base_weights[cost - 1] for cost in cost_candidates], dtype=float)
        weights = weights / weights.sum()
        cost = int(rng.choice(np.asarray(cost_candidates), p=weights))
        units = _available_units_for_cost(data, pool, cost, local_counts)
        unit_id = int(rng.choice(np.asarray(units)))
        local_counts[unit_id] = local_counts.get(unit_id, 0) + 1
        shop.append(unit_id)

    return shop


def _available_units_for_cost(
    data: GameData,
    pool: SharedUnitPool,
    cost: int,
    local_counts: dict[int, int],
) -> list[int]:
    return [
        unit_id
        for unit_id in data.units_by_cost.get(cost, ())
        if pool.available(unit_id) > local_counts.get(unit_id, 0)
    ]
