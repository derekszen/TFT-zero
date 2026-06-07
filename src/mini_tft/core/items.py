"""Abstract item helpers."""

from __future__ import annotations

from collections import Counter

from mini_tft.core.config import EnvConfig
from mini_tft.core.rounds import is_item_drop_round
from mini_tft.core.set_data import GameData
from mini_tft.core.state import GameState, UnitInstance


def is_component_item(item_id: int, data: GameData) -> bool:
    return data.items[item_id].kind == "component"


def is_completed_item(item_id: int, data: GameData) -> bool:
    return data.items[item_id].kind == "completed"


def first_combinable_recipe(item_ids: list[int], data: GameData) -> tuple[int, int, int] | None:
    """Return indexes plus completed item id for the first recipe available on bench."""

    counts = Counter(item_ids)
    for components, completed_id in sorted(data.item_recipes.items(), key=lambda row: row[1]):
        first_id, second_id = components
        required = Counter(components)
        if any(counts[component_id] < amount for component_id, amount in required.items()):
            continue
        first_index = item_ids.index(first_id)
        second_index = next(
            index
            for index, item_id in enumerate(item_ids)
            if item_id == second_id and index != first_index
        )
        return first_index, second_index, completed_id
    return None


def combine_first_components(state: GameState, data: GameData) -> int | None:
    recipe = first_combinable_recipe(state.item_bench, data)
    if recipe is None:
        return None
    first_index, second_index, completed_id = recipe
    for index in sorted((first_index, second_index), reverse=True):
        del state.item_bench[index]
    state.item_bench.append(completed_id)
    return completed_id


def best_item_target(
    board: list[UnitInstance | None],
    item_id: int,
    data: GameData,
    config: EnvConfig,
) -> int | None:
    item = data.items[item_id]
    if item.kind != "completed":
        return None
    candidates: list[tuple[float, int]] = []
    for index, unit in enumerate(board):
        if unit is None or len(unit.items) >= config.max_items_on_unit:
            continue
        unit_def = data.units[unit.unit_id]
        role_match = 1.0 if unit_def.role in item.tags else 0.0
        candidates.append((role_match * 100.0 + unit_def.base_power * unit.stars, index))
    if not candidates:
        return None
    return max(candidates)[1]


def slam_best_item(state: GameState, data: GameData, config: EnvConfig) -> bool:
    """Attach a completed item, or combine components if no completed item is ready."""

    if not state.item_bench:
        return False

    completed_index = next(
        (
            index
            for index, item_id in enumerate(state.item_bench)
            if is_completed_item(item_id, data)
        ),
        None,
    )
    if completed_index is None:
        return combine_first_components(state, data) is not None

    item_id = state.item_bench[completed_index]
    target = best_item_target(state.board, item_id, data, config)
    if target is None:
        return combine_first_components(state, data) is not None
    unit = state.board[target]
    if unit is None:
        return False
    unit.items.append(item_id)
    del state.item_bench[completed_index]
    state.total_item_slams += 1
    return True


def maybe_drop_item(state: GameState, data: GameData, config: EnvConfig, rng) -> None:
    if not is_item_drop_round(state.round):
        return
    if len(state.item_bench) >= config.item_bench_size:
        return
    drop_pool = data.component_item_ids or data.completed_item_ids or tuple(data.items)
    state.item_bench.append(int(rng.choice(drop_pool)))
