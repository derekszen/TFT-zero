"""Board selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from mini_tft.core.combat import board_strength
from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData
from mini_tft.core.state import GameState, UnitInstance


@dataclass(frozen=True)
class CandidateBoard:
    """A legal board/bench layout ranked by abstract board strength."""

    board: tuple[UnitInstance | None, ...]
    bench: tuple[UnitInstance | None, ...]
    strength: float


@dataclass(frozen=True)
class _OwnedEntry:
    unit: UnitInstance
    board_index: int | None


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


def generate_candidate_boards(
    state: GameState,
    data: GameData,
    config: EnvConfig,
    top_k: int = 5,
) -> tuple[CandidateBoard, ...]:
    """Generate top-k legal board candidates from currently owned units.

    Candidates are pure clones of the state units. Each returned layout fields no
    more than the current level cap and preserves every owned unit across the
    candidate board plus bench. Results are ranked by the current abstract
    ``board_strength`` score.
    """

    if top_k <= 0:
        return ()

    owned = _owned_entries(state)
    max_board_units = min(max(state.level, 0), config.max_level, len(owned))
    min_board_units = max(0, len(owned) - config.bench_size)
    if min_board_units > max_board_units:
        raise ValueError(
            "owned units exceed candidate board/bench capacity at the current level"
        )

    candidates: list[CandidateBoard] = []
    seen: set[tuple[object, ...]] = set()
    owned_indexes = range(len(owned))
    for board_units in range(min_board_units, max_board_units + 1):
        for selected_indexes in combinations(owned_indexes, board_units):
            for assignments in _candidate_slot_assignments(
                selected_indexes,
                owned,
                data,
                config,
            ):
                candidate = _make_candidate(selected_indexes, assignments, owned, data, config)
                signature = _candidate_signature(candidate.board, candidate.bench)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            -candidate.strength,
            _unit_list_sort_signature(candidate.board),
            _unit_list_sort_signature(candidate.bench),
        )
    )
    return tuple(candidates[:top_k])


def _owned_entries(state: GameState) -> list[_OwnedEntry]:
    entries: list[_OwnedEntry] = []
    for board_index, unit in enumerate(state.board):
        if unit is not None:
            entries.append(_OwnedEntry(unit=unit, board_index=board_index))
    for unit in state.bench:
        if unit is not None:
            entries.append(_OwnedEntry(unit=unit, board_index=None))
    return entries


def _candidate_slot_assignments(
    selected_indexes: tuple[int, ...],
    owned: list[_OwnedEntry],
    data: GameData,
    config: EnvConfig,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not selected_indexes:
        return ((),)

    assignments = [
        _preferred_slot_assignment(selected_indexes, owned, data, config, support_back=True),
        _preferred_slot_assignment(selected_indexes, owned, data, config, support_back=False),
        _front_loaded_slot_assignment(selected_indexes),
        _current_slot_assignment(selected_indexes, owned, data, config),
    ]
    unique: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for assignment in assignments:
        normalized = tuple(sorted(assignment, key=lambda pair: pair[1]))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)


def _preferred_slot_assignment(
    selected_indexes: tuple[int, ...],
    owned: list[_OwnedEntry],
    data: GameData,
    config: EnvConfig,
    *,
    support_back: bool,
) -> tuple[tuple[int, int], ...]:
    available_slots = set(range(config.max_level))
    assignment: list[tuple[int, int]] = []
    for owned_index in sorted(
        selected_indexes,
        key=lambda index: _assignment_order(owned[index].unit, data),
    ):
        unit = owned[owned_index].unit
        slot = next(
            slot
            for slot in _preferred_slots(unit, data, support_back=support_back)
            if slot in available_slots
        )
        available_slots.remove(slot)
        assignment.append((owned_index, slot))
    return tuple(assignment)


def _front_loaded_slot_assignment(
    selected_indexes: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    return tuple((owned_index, slot) for slot, owned_index in enumerate(selected_indexes))


def _current_slot_assignment(
    selected_indexes: tuple[int, ...],
    owned: list[_OwnedEntry],
    data: GameData,
    config: EnvConfig,
) -> tuple[tuple[int, int], ...]:
    selected = set(selected_indexes)
    available_slots = set(range(config.max_level))
    assignment: list[tuple[int, int]] = []
    unassigned: list[int] = []

    for owned_index in selected_indexes:
        board_index = owned[owned_index].board_index
        if board_index is None or board_index not in available_slots:
            unassigned.append(owned_index)
            continue
        available_slots.remove(board_index)
        assignment.append((owned_index, board_index))

    for owned_index in sorted(
        unassigned,
        key=lambda index: _assignment_order(owned[index].unit, data),
    ):
        unit = owned[owned_index].unit
        slot = next(slot for slot in _preferred_slots(unit, data) if slot in available_slots)
        available_slots.remove(slot)
        assignment.append((owned_index, slot))

    unused_selected = selected - {owned_index for owned_index, _ in assignment}
    if unused_selected:
        raise AssertionError(f"unassigned selected units: {unused_selected}")
    return tuple(assignment)


def _assignment_order(unit: UnitInstance, data: GameData) -> tuple[int, float, int, int]:
    unit_def = data.units[unit.unit_id]
    role_priority = {"tank": 0, "carry": 1, "support": 2}.get(unit_def.role, 3)
    return (role_priority, -_unit_sort_power(unit, data), unit.unit_id, unit.stars)


def _unit_sort_power(unit: UnitInstance, data: GameData) -> float:
    unit_def = data.units[unit.unit_id]
    return unit_def.base_power * unit.stars + len(unit.items) * 2.0


def _preferred_slots(
    unit: UnitInstance,
    data: GameData,
    *,
    support_back: bool = True,
) -> tuple[int, ...]:
    role = data.units[unit.unit_id].role
    if role == "tank":
        return (0, 1, 2, 3, 4, 5, 6, 7, 8)
    if role == "carry":
        return (6, 7, 8, 3, 4, 5, 0, 1, 2)
    if role == "support" and not support_back:
        return (3, 4, 5, 6, 7, 8, 0, 1, 2)
    return (6, 7, 8, 3, 4, 5, 0, 1, 2)


def _make_candidate(
    selected_indexes: tuple[int, ...],
    assignments: tuple[tuple[int, int], ...],
    owned: list[_OwnedEntry],
    data: GameData,
    config: EnvConfig,
) -> CandidateBoard:
    selected = set(selected_indexes)
    board: list[UnitInstance | None] = [None for _ in range(config.max_level)]
    for owned_index, board_index in assignments:
        board[board_index] = owned[owned_index].unit.clone()

    bench_units = [
        owned[index].unit.clone()
        for index in range(len(owned))
        if index not in selected
    ]
    bench: list[UnitInstance | None] = bench_units
    bench.extend([None] * (config.bench_size - len(bench)))
    strength = board_strength(board, data).strength
    return CandidateBoard(board=tuple(board), bench=tuple(bench), strength=strength)


def _candidate_signature(
    board: tuple[UnitInstance | None, ...],
    bench: tuple[UnitInstance | None, ...],
) -> tuple[object, ...]:
    return (_unit_list_signature(list(board)), _unit_list_signature(list(bench)))


def _unit_list_sort_signature(units: tuple[UnitInstance | None, ...]) -> tuple[object, ...]:
    return tuple(
        (-1, 0, ())
        if unit is None
        else (unit.unit_id, unit.stars, tuple(unit.items))
        for unit in units
    )


def _unit_list_signature(units: list[UnitInstance | None]) -> tuple[object, ...]:
    return tuple(
        None if unit is None else (unit.unit_id, unit.stars, tuple(unit.items))
        for unit in units
    )
