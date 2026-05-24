"""Shop sampling."""

from __future__ import annotations

import numpy as np

from mini_tft.core.ids import EMPTY
from mini_tft.core.set_data import GameData


def sample_shop(
    data: GameData,
    level: int,
    size: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample a shop with replacement from level cost odds."""

    available_level = max(key for key in data.shop_odds if key <= level)
    cost_weights = data.shop_odds[available_level]
    costs = np.arange(1, 6)

    shop: list[int] = []
    for _ in range(size):
        cost = int(rng.choice(costs, p=cost_weights))
        candidates = data.units_by_cost.get(cost, ())
        shop.append(int(rng.choice(candidates)) if candidates else EMPTY)
    return shop
