from __future__ import annotations

import numpy as np

from mini_tft.core.set_data import load_set


def test_set1_data_loads_and_references_are_valid() -> None:
    data = load_set("set1")

    assert len(data.units) == 24
    assert len(data.completed_item_ids) == 10
    assert len(data.component_item_ids) == 7
    assert len(data.items) == 17
    assert len(data.item_recipes) == 10
    assert len(data.enemy_curve) == 36

    for unit in data.units.values():
        assert unit.traits
        assert all(trait in data.traits for trait in unit.traits)

    for weights in data.shop_odds.values():
        assert weights.shape == (5,)
        assert np.isclose(weights.sum(), 1.0)
