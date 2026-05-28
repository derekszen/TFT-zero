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
    category: str = "general"
    weight: float = 1.0


def combat_fixtures() -> list[CombatFixture]:
    """Return fixed boards used to compare combat model variants."""

    return [
        CombatFixture(
            name="balanced_frontline_beats_extra_backline",
            better_board=_board([_unit(1), _unit(2), None, None, None, None, _unit(13)]),
            worse_board=_board([_unit(2), _unit(7), None, None, None, None, _unit(13)]),
            min_margin=4.0,
            rationale="A tank plus carries should beat a pure backline pile at similar tier.",
            category="positioning",
        ),
        CombatFixture(
            name="proper_positioning_beats_reversed_positioning",
            better_board=_board([_unit(1), None, None, None, None, None, _unit(2), _unit(13)]),
            worse_board=_board([_unit(2), _unit(13), None, None, None, None, _unit(1)]),
            min_margin=3.0,
            rationale="Frontline units should be better in front slots than back slots.",
            category="positioning",
        ),
        CombatFixture(
            name="item_on_matching_role_beats_wrong_role",
            better_board=_board([_unit(1), None, None, None, None, None, _unit(18, items=[1])]),
            worse_board=_board([_unit(1, items=[1]), None, None, None, None, None, _unit(18)]),
            min_margin=2.0,
            rationale="Carry items should matter more on carries than tanks.",
            category="item_fit",
        ),
        CombatFixture(
            name="trait_synergy_beats_unsynergistic_board",
            better_board=_board([_unit(1), _unit(22), None, None, None, None, _unit(2)]),
            worse_board=_board([_unit(1), _unit(10), None, None, None, None, _unit(8)]),
            min_margin=3.0,
            rationale="Noble 3 should beat a similarly sized board without active synergy.",
            category="trait_role",
        ),
        CombatFixture(
            name="upgraded_low_cost_beats_unupgraded_pair",
            better_board=_board([_unit(1, stars=2), None, None, None, None, None, _unit(2)]),
            worse_board=_board([_unit(1), None, None, None, None, None, _unit(2), _unit(4)]),
            min_margin=2.0,
            rationale="A useful 2-star frontline should beat extra unupgraded filler.",
            category="upgrade_tempo",
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
            category="late_game_scaling",
        ),
        CombatFixture(
            name="tank_frontline_beats_same_tanks_backline",
            better_board=_board(
                [_unit(20), _unit(21), None, None, None, None, _unit(18), _unit(13)]
            ),
            worse_board=_board(
                [_unit(18), _unit(13), None, None, None, None, _unit(20), _unit(21)]
            ),
            min_margin=5.0,
            rationale="High-value tanks should gain clear value from occupying front slots.",
            category="positioning",
        ),
        CombatFixture(
            name="carry_backline_beats_same_carries_frontline",
            better_board=_board([_unit(20), None, None, None, None, None, _unit(18), _unit(23)]),
            worse_board=_board([_unit(18), _unit(23), None, None, None, None, _unit(20)]),
            min_margin=3.0,
            rationale=(
                "Premium carries should be safer and more valuable in back slots "
                "than front slots."
            ),
            category="positioning",
        ),
        CombatFixture(
            name="support_backline_beats_support_frontline",
            better_board=_board([_unit(20), None, None, None, None, None, _unit(24), _unit(18)]),
            worse_board=_board([_unit(24), _unit(18), None, None, None, None, _unit(20)]),
            min_margin=2.0,
            rationale="Support units should lose value when forced into early frontline contact.",
            category="positioning",
        ),
        CombatFixture(
            name="mixed_rows_beat_clumped_backline",
            better_board=_board(
                [_unit(20), None, None, _unit(24), None, None, _unit(18), _unit(13)]
            ),
            worse_board=_board(
                [None, None, None, None, None, None, _unit(20), _unit(24), _unit(18)]
            ),
            min_margin=4.0,
            rationale=(
                "A board with frontline, utility, and backline should beat a "
                "clumped no-frontline board."
            ),
            category="positioning",
        ),
        CombatFixture(
            name="two_star_three_cost_tank_beats_one_star_fillers",
            better_board=_board([_unit(12, stars=2), None, None, None, None, None, _unit(13)]),
            worse_board=_board([_unit(12), _unit(10), _unit(5), None, None, None, _unit(13)]),
            min_margin=3.0,
            rationale=(
                "A real upgraded frontline should beat a wider board of weak "
                "one-star filler tanks."
            ),
            category="upgrade_tempo",
        ),
        CombatFixture(
            name="three_star_low_cost_core_beats_two_star_pair",
            better_board=_board([_unit(1, stars=3), None, None, None, None, None, _unit(2)]),
            worse_board=_board(
                [_unit(1, stars=2), _unit(3, stars=2), None, None, None, None, _unit(2)]
            ),
            min_margin=8.0,
            rationale=(
                "A three-star core should represent a meaningful capstone over "
                "several modest upgrades."
            ),
            category="upgrade_tempo",
        ),
        CombatFixture(
            name="tank_items_on_tank_beat_tank_items_on_carry",
            better_board=_board([_unit(21, items=[6, 7]), None, None, None, None, None, _unit(18)]),
            worse_board=_board([_unit(21), None, None, None, None, None, _unit(18, items=[6, 7])]),
            min_margin=6.0,
            rationale="Durability items should create more board value on tanks than on carries.",
            category="item_fit",
        ),
        CombatFixture(
            name="carry_items_on_carry_beat_carry_items_on_support",
            better_board=_board(
                [_unit(20), None, None, None, None, None, _unit(18, items=[1, 2]), _unit(24)]
            ),
            worse_board=_board(
                [_unit(20), None, None, None, None, None, _unit(18), _unit(24, items=[1, 2])]
            ),
            min_margin=4.0,
            rationale="Damage items should be prioritized on carries over support units.",
            category="item_fit",
        ),
        CombatFixture(
            name="support_items_on_support_beat_support_items_on_carry",
            better_board=_board(
                [_unit(20), None, None, None, None, None, _unit(18), _unit(24, items=[4])]
            ),
            worse_board=_board(
                [_unit(20), None, None, None, None, None, _unit(18, items=[4]), _unit(24)]
            ),
            min_margin=2.0,
            rationale="Mana/support items should create more value on supports than carries.",
            category="item_fit",
        ),
        CombatFixture(
            name="disruption_items_on_support_beat_disruption_items_on_carry",
            better_board=_board(
                [_unit(20), None, None, None, None, None, _unit(18), _unit(24, items=[10])]
            ),
            worse_board=_board(
                [_unit(20), None, None, None, None, None, _unit(18, items=[10]), _unit(24)]
            ),
            min_margin=1.5,
            rationale=(
                "Disruption utility should be better represented on supports "
                "than main carries."
            ),
            category="item_fit",
        ),
        CombatFixture(
            name="ranger_pair_beats_same_cost_no_trait_carries",
            better_board=_board([_unit(20), None, None, None, None, None, _unit(13), _unit(8)]),
            worse_board=_board([_unit(20), None, None, None, None, None, _unit(15), _unit(17)]),
            min_margin=4.0,
            rationale=(
                "An active ranger pair should beat similar-cost carries without "
                "a relevant shared trait."
            ),
            category="trait_role",
        ),
        CombatFixture(
            name="knight_brawler_frontline_beats_roleless_frontline",
            better_board=_board([_unit(20), _unit(21), _unit(5), None, None, None, _unit(18)]),
            worse_board=_board([_unit(22), _unit(12), _unit(10), None, None, None, _unit(18)]),
            min_margin=3.0,
            rationale="Active frontline traits should matter beyond raw tank count.",
            category="trait_role",
        ),
        CombatFixture(
            name="assassin_pair_challenges_ranger_backline",
            better_board=_board(
                [_unit(20), None, None, None, _unit(11), None, _unit(14), _unit(18)]
            ),
            worse_board=_board(
                [_unit(20), None, None, None, _unit(13), None, _unit(8), _unit(18)]
            ),
            min_margin=2.0,
            rationale=(
                "A Pyke/Katarina pair should add enough backline pressure to "
                "challenge a conventional ranger backline shell."
            ),
            category="assassin_pressure",
        ),
        CombatFixture(
            name="itemized_assassin_carry_beats_front_to_back_item_stack",
            better_board=_board(
                [_unit(20), None, None, None, _unit(11), None, _unit(14, items=[1, 2]), _unit(18)]
            ),
            worse_board=_board(
                [_unit(20), None, None, None, _unit(16), None, _unit(15), _unit(18, items=[1, 2])]
            ),
            min_margin=3.0,
            rationale=(
                "An itemized assassin carry should gain extra value from backline "
                "access, not score like a generic front-to-back carry."
            ),
            category="assassin_pressure",
        ),
        CombatFixture(
            name="assassin_pressure_shell_beats_same_roles_without_access",
            better_board=_board(
                [_unit(20), None, None, None, _unit(11), None, _unit(14), _unit(18)]
            ),
            worse_board=_board(
                [_unit(20), None, None, None, _unit(16), None, _unit(15), _unit(18)]
            ),
            min_margin=6.0,
            rationale=(
                "A balanced shell with assassin access should beat the same "
                "frontline/carry/support shape without backline access."
            ),
            category="assassin_pressure",
        ),
        CombatFixture(
            name="six_noble_capped_board_beats_high_cost_goodstuff",
            better_board=_board(
                [
                    _unit(1),
                    _unit(22),
                    _unit(24),
                    _unit(5),
                    None,
                    None,
                    _unit(2),
                    _unit(7),
                    _unit(18),
                ]
            ),
            worse_board=_board(
                [
                    _unit(20),
                    _unit(21),
                    _unit(12),
                    _unit(16),
                    None,
                    None,
                    _unit(18),
                    _unit(19),
                    _unit(23),
                ]
            ),
            min_margin=4.0,
            rationale=(
                "A completed vertical trait should compete with and beat loose "
                "high-cost goodstuff."
            ),
            category="late_game_scaling",
        ),
        CombatFixture(
            name="full_capped_itemized_board_beats_partial_lategame_board",
            better_board=_board(
                [
                    _unit(20, items=[6]),
                    _unit(21, items=[7]),
                    _unit(22),
                    _unit(16, items=[4]),
                    None,
                    None,
                    _unit(18, items=[1, 2]),
                    _unit(23, items=[9]),
                    _unit(24, items=[10]),
                ]
            ),
            worse_board=_board(
                [_unit(20), _unit(21), None, _unit(16), None, None, _unit(18), _unit(23)]
            ),
            min_margin=35.0,
            rationale=(
                "A fully capped and itemized board should clearly outscale a "
                "partial late-game board."
            ),
            category="late_game_scaling",
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
