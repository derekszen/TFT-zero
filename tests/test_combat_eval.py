from __future__ import annotations

from mini_tft.experiments.combat_fixtures import combat_fixtures
from mini_tft.tools.combat_eval import run_combat_eval


def test_combat_eval_reports_all_fixtures() -> None:
    report = run_combat_eval(benchmark_iters=1)

    assert 14 <= report["total"] <= 20
    assert len(report["fixtures"]) == report["total"]
    assert report["benchmark"]["calls"] == report["total"] * 2
    assert report["passed"] <= report["total"]
    assert report["status"] in {"pass", "fail"}


def test_combat_fixture_suite_covers_core_categories() -> None:
    fixtures = combat_fixtures()
    categories = {fixture.category for fixture in fixtures}
    names = {fixture.name for fixture in fixtures}

    assert categories >= {
        "positioning",
        "upgrade_tempo",
        "item_fit",
        "trait_role",
        "late_game_scaling",
    }
    assert all(fixture.weight > 0 for fixture in fixtures)
    assert len(names) == len(fixtures)


def test_combat_eval_star_tempo_variant_covers_previous_gaps() -> None:
    report = run_combat_eval(benchmark_iters=0)
    fixtures = {row["name"]: row for row in report["fixtures"]}

    positioning = fixtures["proper_positioning_beats_reversed_positioning"]
    assert positioning["status"] == "pass"
    assert positioning["margin"] >= positioning["min_margin"]

    upgraded = fixtures["upgraded_low_cost_beats_unupgraded_pair"]
    assert upgraded["status"] == "pass"
    assert upgraded["margin"] >= upgraded["min_margin"]
