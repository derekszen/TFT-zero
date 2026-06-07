"""Abstract item helpers."""

from __future__ import annotations

from mini_tft.core.config import EnvConfig
from mini_tft.core.rounds import is_item_drop_round
from mini_tft.core.set_data import GameData
from mini_tft.core.state import GameState, UnitInstance


def best_item_target(
    board: list[UnitInstance | None],
    item_id: int,
    data: GameData,
    config: EnvConfig,
) -> int | None:
    item = data.items[item_id]
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
    """Attach the first item in the bench to its best role match."""

    if not state.item_bench:
        return False
    item_id = state.item_bench[0]
    target = best_item_target(state.board, item_id, data, config)
    if target is None:
        return False
    unit = state.board[target]
    if unit is None:
        return False
    unit.items.append(item_id)
    del state.item_bench[0]
    state.total_item_slams += 1
    return True


def maybe_drop_item(state: GameState, data: GameData, config: EnvConfig, rng) -> None:
    if not is_item_drop_round(state.round):
        return
    if len(state.item_bench) >= config.item_bench_size:
        return
    state.item_bench.append(int(rng.choice(tuple(data.items))))
