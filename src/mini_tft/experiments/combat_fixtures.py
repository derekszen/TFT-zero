"""Sanity fixtures for lightweight combat model comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from mini_tft.core.state import UnitInstance

BOARD_SIZE = 9


@dataclass(frozen=True)
class CombatFixture:
    """A relative combat expectation.

    The fixture passes when `better_board` scores higher than `worse_board` by at
    least `min_margin`. These are sanity cases for comparing lightweight combat
    models, not claims of exact TFT fidelity.
    """

    name: str
    better_board: list[UnitInstance | None]
    worse_board: list[UnitInstance | None]
    min_margin: float
    rationale: str


def combat_fixtures() -> list[CombatFixture]:
    """Return fixed boards used to compare combat model variants."""

    return [
        CombatFixture(
            name="balanced_frontline_beats_extra_backline",
            better_board=_board([_unit(1), _unit(2), None, None, None, None, _unit(13)]),
            worse_board=_board([_unit(2), _unit(7), None, None, None, None, _unit(13)]),
            min_margin=4.0,
            rationale="A tank plus carries should beat a pure backline pile at similar tier.",
        ),
        CombatFixture(
            name="proper_positioning_beats_reversed_positioning",
            better_board=_board([_unit(1), None, None, None, None, None, _unit(2), _unit(13)]),
            worse_board=_board([_unit(2), _unit(13), None, None, None, None, _unit(1)]),
            min_margin=3.0,
            rationale="Frontline units should be better in front slots than back slots.",
        ),
        CombatFixture(
            name="item_on_matching_role_beats_wrong_role",
            better_board=_board([_unit(1), None, None, None, None, None, _unit(18, items=[1])]),
            worse_board=_board([_unit(1, items=[1]), None, None, None, None, None, _unit(18)]),
            min_margin=2.0,
            rationale="Carry items should matter more on carries than tanks.",
        ),
        CombatFixture(
            name="trait_synergy_beats_unsynergistic_board",
            better_board=_board([_unit(1), _unit(22), None, None, None, None, _unit(2)]),
            worse_board=_board([_unit(1), _unit(10), None, None, None, None, _unit(8)]),
            min_margin=3.0,
            rationale="Noble 3 should beat a similarly sized board without active synergy.",
        ),
        CombatFixture(
            name="upgraded_low_cost_beats_unupgraded_pair",
            better_board=_board([_unit(1, stars=2), None, None, None, None, None, _unit(2)]),
            worse_board=_board([_unit(1), None, None, None, None, None, _unit(2), _unit(4)]),
            min_margin=2.0,
            rationale="A useful 2-star frontline should beat extra unupgraded filler.",
        ),
        CombatFixture(
            name="late_capped_board_beats_early_board",
            better_board=_board(
                [
                    _unit(20),
                    _unit(21),
                    _unit(22),
                    _unit(16),
                    None,
                    None,
                    _unit(18, items=[1, 2]),
                    _unit(23),
                    _unit(24),
                ]
            ),
            worse_board=_board([_unit(1), _unit(3), None, None, None, None, _unit(2), _unit(7)]),
            min_margin=80.0,
            rationale="A late-game capped board should dominate a cheap early board.",
        ),
    ]


def _unit(unit_id: int, *, stars: int = 1, items: list[int] | None = None) -> UnitInstance:
    return UnitInstance(unit_id=unit_id, stars=stars, items=list(items or []))


def _board(units: list[UnitInstance | None]) -> list[UnitInstance | None]:
    if len(units) > BOARD_SIZE:
        raise ValueError(f"Board fixture has too many slots: {len(units)}")
    return [unit.clone() if unit is not None else None for unit in units] + [None] * (
        BOARD_SIZE - len(units)
    )
