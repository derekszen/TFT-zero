"""Board selection helpers."""

from __future__ import annotations

from mini_tft.core.combat import board_strength
from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData
from mini_tft.core.state import GameState, UnitInstance


def field_best_board(state: GameState, data: GameData, config: EnvConfig) -> bool:
    """Move owned units into the best greedy board up to current level."""

    current_signature = _unit_list_signature(state.board)
    board, bench = select_best_board(state, data, config)
    next_signature = _unit_list_signature(board)
    state.board = board
    state.bench = bench
    return current_signature != next_signature


def would_change_best_board(state: GameState, data: GameData, config: EnvConfig) -> bool:
    board, _ = select_best_board(state, data, config)
    return _unit_list_signature(board) != _unit_list_signature(state.board)


def select_best_board(
    state: GameState,
    data: GameData,
    config: EnvConfig,
) -> tuple[list[UnitInstance | None], list[UnitInstance | None]]:
    owned = [unit.clone() for unit in [*state.board, *state.bench] if unit is not None]
    remaining = owned
    chosen: list[UnitInstance] = []

    slots = min(state.level, config.max_level, len(remaining))
    for _ in range(slots):
        best_index = max(
            range(len(remaining)),
            key=lambda index: board_strength([*chosen, remaining[index]], data).strength,
        )
        chosen.append(remaining.pop(best_index))

    board: list[UnitInstance | None] = [unit.clone() for unit in chosen]
    board.extend([None] * (config.max_level - len(board)))

    bench: list[UnitInstance | None] = [unit.clone() for unit in remaining[: config.bench_size]]
    bench.extend([None] * (config.bench_size - len(bench)))
    return board, bench


def _unit_list_signature(units: list[UnitInstance | None]) -> tuple[object, ...]:
    return tuple(
        None if unit is None else (unit.unit_id, unit.stars, tuple(unit.items))
        for unit in units
    )
