from __future__ import annotations

from mini_tft import EnvConfig, MiniTFTEnv
from mini_tft.core.actions import Action
from mini_tft.core.items import combine_first_components, first_combinable_recipe
from mini_tft.core.set_data import load_set
from mini_tft.core.state import UnitInstance


def test_item_data_defines_components_and_completed_recipes() -> None:
    data = load_set()

    assert 101 in data.component_item_ids
    assert 1 in data.completed_item_ids
    assert data.item_recipes[(101, 102)] == 1
    assert data.items[101].kind == "component"
    assert data.items[1].kind == "completed"


def test_components_combine_deterministically_without_mutating_unrelated_items() -> None:
    data = load_set()
    env = MiniTFTEnv(EnvConfig(seed=20))
    env.reset(seed=20)
    assert env.state is not None
    env.state.item_bench = [107, 101, 102]

    recipe = first_combinable_recipe(env.state.item_bench, data)
    completed = combine_first_components(env.state, data)

    assert recipe == (1, 2, 1)
    assert completed == 1
    assert env.state.item_bench == [107, 1]


def test_slam_best_item_combines_components_then_slams_completed_item() -> None:
    env = MiniTFTEnv(EnvConfig(seed=21))
    env.reset(seed=21)
    assert env.state is not None
    env.state.board[0] = UnitInstance(unit_id=2)
    env.state.item_bench = [101, 102]

    assert env.action_masks()[Action.SLAM_BEST_ITEM]
    env.step(Action.SLAM_BEST_ITEM)
    assert env.state.item_bench == [1]
    assert env.state.board[0].items == []

    assert env.action_masks()[Action.SLAM_BEST_ITEM]
    env.step(Action.SLAM_BEST_ITEM)
    assert env.state.item_bench == []
    assert env.state.board[0].items == [1]


def test_pve_drop_awards_components_not_completed_items() -> None:
    env = MiniTFTEnv(EnvConfig(seed=22))
    env.reset(seed=22)
    assert env.state is not None

    env.step(Action.END_TURN)

    assert len(env.state.item_bench) == 1
    dropped = env.state.item_bench[0]
    assert env.data.items[dropped].kind == "component"
