from __future__ import annotations

import numpy as np

from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import load_set
from mini_tft.core.shop import sample_shop
from mini_tft.core.state import UnitInstance, new_game_state
from mini_tft.core.upgrades import auto_combine


def test_three_one_star_copies_combine_to_two_star() -> None:
    data = load_set("set1")
    config = EnvConfig(seed=0)
    rng = np.random.default_rng(0)
    state = new_game_state(config, seed=0, shop=sample_shop(data, 3, 5, rng))
    state.bench[0] = UnitInstance(1)
    state.bench[1] = UnitInstance(1)
    state.bench[2] = UnitInstance(1)

    auto_combine(state)

    units = [unit for unit in state.bench if unit is not None]
    assert len(units) == 1
    assert units[0].unit_id == 1
    assert units[0].stars == 2
