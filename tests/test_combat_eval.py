from __future__ import annotations

from mini_tft.tools.combat_eval import run_combat_eval


def test_combat_eval_runs_all_fixtures_without_strict_failure() -> None:
    report = run_combat_eval(benchmark_iters=1)

    assert report["total"] >= 5
    assert report["passed"] == report["total"]
    assert len(report["fixtures"]) == report["total"]
    assert report["benchmark"]["calls"] == report["total"] * 2


def test_combat_eval_star_tempo_variant_covers_previous_gaps() -> None:
    report = run_combat_eval(benchmark_iters=0)
    fixtures = {row["name"]: row for row in report["fixtures"]}

    positioning = fixtures["proper_positioning_beats_reversed_positioning"]
    assert positioning["status"] == "pass"
    assert positioning["margin"] >= positioning["min_margin"]

    upgraded = fixtures["upgraded_low_cost_beats_unupgraded_pair"]
    assert upgraded["status"] == "pass"
    assert upgraded["margin"] >= upgraded["min_margin"]
