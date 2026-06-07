"""Focused Set-1 MiniTFT abstract combat fixture gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from math import exp
from typing import Any

from mini_tft.core.combat import CombatStats, board_strength
from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData, load_set
from mini_tft.core.state import UnitInstance

BOARD_SIZE = 9


@dataclass(frozen=True)
class CombatFixtureGate:
    name: str
    category: str
    better_label: str
    better_board: list[UnitInstance | None]
    worse_label: str
    worse_board: list[UnitInstance | None]
    round_num: int
    min_strength_delta: float
    min_win_probability_delta: float
    rationale: str


def combat_fixture_gates() -> tuple[CombatFixtureGate, ...]:
    """Return the branch-scoped abstract combat fixture expectations."""

    return (
        CombatFixtureGate(
            name="frontline_backline_positioning",
            category="positioning",
            better_label="frontline tank with backline carries",
            better_board=_board([_unit(20), None, None, None, None, None, _unit(18), _unit(13)]),
            worse_label="carries fronted with tank stranded backline",
            worse_board=_board([_unit(18), _unit(13), None, None, None, None, _unit(20)]),
            round_num=18,
            min_strength_delta=8.0,
            min_win_probability_delta=0.10,
            rationale="Proper tank-front/carry-back positioning should clearly beat reversed rows.",
        ),
        CombatFixtureGate(
            name="itemized_carry",
            category="item_fit",
            better_label="Draven carry with carry items",
            better_board=_board(
                [_unit(20), None, None, None, None, None, _unit(18, items=[1, 2]), _unit(24)]
            ),
            worse_label="frontline tank holding carry items",
            worse_board=_board(
                [_unit(20, items=[1, 2]), None, None, None, None, None, _unit(18), _unit(24)]
            ),
            round_num=22,
            min_strength_delta=12.0,
            min_win_probability_delta=0.15,
            rationale="Carry-tagged completed items should create more value on the main carry.",
        ),
        CombatFixtureGate(
            name="two_star_upgrade",
            category="upgrade_tempo",
            better_label="two-star Braum frontline",
            better_board=_board([_unit(12, stars=2), None, None, None, None, None, _unit(13)]),
            worse_label="one-star filler frontline pair",
            worse_board=_board([_unit(12), _unit(10), None, None, None, None, _unit(13)]),
            round_num=15,
            min_strength_delta=6.0,
            min_win_probability_delta=0.10,
            rationale="A meaningful two-star frontline should beat extra one-star filler.",
        ),
        CombatFixtureGate(
            name="assassin_pressure",
            category="assassin_pressure",
            better_label="Pyke/Katarina access shell",
            better_board=_board(
                [_unit(20), None, None, None, _unit(11), None, _unit(14), _unit(18)]
            ),
            worse_label="same role shell without assassin access",
            worse_board=_board(
                [_unit(20), None, None, None, _unit(16), None, _unit(15), _unit(18)]
            ),
            round_num=22,
            min_strength_delta=6.0,
            min_win_probability_delta=0.08,
            rationale="Assassin backline access should improve a comparable tank/carry shell.",
        ),
        CombatFixtureGate(
            name="trait_breakpoint",
            category="trait_breakpoint",
            better_label="active two-ranger backline",
            better_board=_board([_unit(20), None, None, None, None, None, _unit(13), _unit(8)]),
            worse_label="same-cost carries without a breakpoint",
            worse_board=_board([_unit(20), None, None, None, None, None, _unit(15), _unit(17)]),
            round_num=18,
            min_strength_delta=4.0,
            min_win_probability_delta=0.05,
            rationale="An active ranger breakpoint should beat similar-cost loose carries.",
        ),
        CombatFixtureGate(
            name="no_frontline_penalty",
            category="positioning",
            better_label="balanced frontline plus backline",
            better_board=_board([_unit(20), None, None, None, None, None, _unit(18), _unit(23)]),
            worse_label="same-count carry pile with no frontline",
            worse_board=_board(
                [None, None, None, None, None, None, _unit(18), _unit(19), _unit(23)]
            ),
            round_num=20,
            min_strength_delta=10.0,
            min_win_probability_delta=0.12,
            rationale="A no-frontline carry pile should be penalized versus a balanced board.",
        ),
    )


def run_combat_fixture_report(
    *,
    data: GameData | None = None,
    config: EnvConfig | None = None,
) -> dict[str, Any]:
    """Evaluate focused fixtures and return scalar report data."""

    actual_data = data or load_set()
    actual_config = config or EnvConfig()
    rows = [
        evaluate_combat_fixture(fixture, actual_data, actual_config)
        for fixture in combat_fixture_gates()
    ]
    passed = sum(1 for row in rows if row["status"] == "pass")
    return {
        "status": "pass" if passed == len(rows) else "fail",
        "passed": passed,
        "total": len(rows),
        "fixtures": rows,
    }


def evaluate_combat_fixture(
    fixture: CombatFixtureGate,
    data: GameData,
    config: EnvConfig,
) -> dict[str, Any]:
    better_stats = board_strength(fixture.better_board, data)
    worse_stats = board_strength(fixture.worse_board, data)
    better_win_probability = _deterministic_win_probability(
        better_stats,
        fixture.round_num,
        data,
        config,
    )
    worse_win_probability = _deterministic_win_probability(
        worse_stats,
        fixture.round_num,
        data,
        config,
    )
    strength_delta = better_stats.strength - worse_stats.strength
    win_probability_delta = better_win_probability - worse_win_probability
    ok = (
        strength_delta >= fixture.min_strength_delta
        and win_probability_delta >= fixture.min_win_probability_delta
    )
    return {
        "name": fixture.name,
        "category": fixture.category,
        "status": "pass" if ok else "fail",
        "round": fixture.round_num,
        "better": fixture.better_label,
        "worse": fixture.worse_label,
        "better_strength": round(better_stats.strength, 3),
        "worse_strength": round(worse_stats.strength, 3),
        "strength_delta": round(strength_delta, 3),
        "min_strength_delta": fixture.min_strength_delta,
        "better_win_probability": round(better_win_probability, 6),
        "worse_win_probability": round(worse_win_probability, 6),
        "win_probability_delta": round(win_probability_delta, 6),
        "min_win_probability_delta": fixture.min_win_probability_delta,
        "rationale": fixture.rationale,
    }


def format_text(report: dict[str, Any]) -> str:
    lines = [
        f"MiniTFT combat fixture report: {report['status']} "
        f"({report['passed']}/{report['total']})",
        "",
        (
            "fixture                         status  round  str_delta  str_min  "
            "pwin_delta  pwin_min"
        ),
        "-" * 88,
    ]
    for row in report["fixtures"]:
        lines.append(
            f"{row['name']:<31} {row['status']:<6} "
            f"{row['round']:>5} "
            f"{row['strength_delta']:>10.3f} "
            f"{row['min_strength_delta']:>8.3f} "
            f"{row['win_probability_delta']:>10.3f} "
            f"{row['min_win_probability_delta']:>8.3f}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report focused MiniTFT combat fixture gates.")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero unless all fixtures pass.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_combat_fixture_report()
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text(report), end="")
    return 1 if args.strict and report["status"] != "pass" else 0


def _deterministic_win_probability(
    stats: CombatStats,
    round_num: int,
    data: GameData,
    config: EnvConfig,
) -> float:
    enemy_index = min(max(0, round_num - 1), len(data.enemy_curve) - 1)
    enemy_strength = max(0.0, data.enemy_curve[enemy_index] - stats.enemy_power_penalty)
    diff = stats.strength - enemy_strength
    return 1.0 / (1.0 + exp(-(diff / config.combat_sigmoid_scale)))


def _unit(unit_id: int, *, stars: int = 1, items: list[int] | None = None) -> UnitInstance:
    return UnitInstance(unit_id=unit_id, stars=stars, items=list(items or []))


def _board(units: list[UnitInstance | None]) -> list[UnitInstance | None]:
    if len(units) > BOARD_SIZE:
        raise ValueError(f"Board fixture has too many slots: {len(units)}")
    return [unit.clone() if unit is not None else None for unit in units] + [None] * (
        BOARD_SIZE - len(units)
    )


if __name__ == "__main__":
    raise SystemExit(main())
