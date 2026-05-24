"""Automatic unit upgrade logic."""

from __future__ import annotations

from mini_tft.core.state import GameState, UnitInstance

MAX_STARS = 3


def auto_combine(state: GameState) -> None:
    """Combine three matching copies across bench and board until stable."""

    changed = True
    while changed:
        changed = False
        for stars in range(1, MAX_STARS):
            groups: dict[int, list[tuple[str, int, UnitInstance]]] = {}
            for zone, units in (("board", state.board), ("bench", state.bench)):
                for index, unit in enumerate(units):
                    if unit is not None and unit.stars == stars:
                        groups.setdefault(unit.unit_id, []).append((zone, index, unit))

            for unit_id, slots in groups.items():
                if len(slots) < 3:
                    continue
                _combine_slots(state, unit_id, stars, slots[:3])
                changed = True
                break
            if changed:
                break


def _combine_slots(
    state: GameState,
    unit_id: int,
    stars: int,
    slots: list[tuple[str, int, UnitInstance]],
) -> None:
    items: list[int] = []
    target_zone, target_index, _ = slots[0]
    for zone, index, unit in slots:
        items.extend(unit.items)
        if zone == "board":
            target_zone, target_index = zone, index
        if zone == "board":
            state.board[index] = None
        else:
            state.bench[index] = None

    upgraded = UnitInstance(unit_id=unit_id, stars=stars + 1, items=items[:3])
    if target_zone == "board":
        state.board[target_index] = upgraded
    else:
        state.bench[target_index] = upgraded
