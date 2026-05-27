from __future__ import annotations

from mini_tft.experiments.combat_fixtures import combat_fixtures
from mini_tft.tools.combat_eval import format_markdown, run_combat_eval
from mini_tft.tools.compare_combat_variants import format_markdown as format_variant_markdown


def test_combat_eval_reports_all_fixtures() -> None:
    report = run_combat_eval(benchmark_iters=1)

    assert 14 <= report["total"] <= 20
    assert len(report["fixtures"]) == report["total"]
    assert report["benchmark"]["calls"] == report["total"] * 2
    assert report["passed"] <= report["total"]
    assert report["status"] in {"pass", "fail"}
    assert "decision_score" in report
    assert "margin_score" in report
    assert "mean_normalized_surplus" in report
    assert report["category_summaries"]


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


def test_combat_eval_reports_surplus_metrics() -> None:
    report = run_combat_eval(benchmark_iters=0)

    for row in report["fixtures"]:
        assert row["category"]
        assert row["weight"] > 0
        assert row["surplus"] == round(row["margin"] - row["min_margin"], 3)
        assert "normalized_surplus" in row

    markdown = format_markdown(report)
    assert "Decision score" in markdown
    assert "Category Summary" in markdown
    assert "Norm Surplus" in markdown


def test_variant_markdown_sorts_by_decision_score_before_throughput() -> None:
    rows = [
        {
            "name": "faster",
            "commit": "1111111",
            "passed": 20,
            "total": 20,
            "decision_score": 101.0,
            "margin_score": 1.0,
            "mean_surplus": 1.0,
            "mean_normalized_surplus": 0.1,
            "calls_per_sec": 999_999.0,
            "failures": [],
            "path": "/tmp/faster",
        },
        {
            "name": "better",
            "commit": "2222222",
            "passed": 20,
            "total": 20,
            "decision_score": 102.0,
            "margin_score": 2.0,
            "mean_surplus": 2.0,
            "mean_normalized_surplus": 0.2,
            "calls_per_sec": 100.0,
            "failures": [],
            "path": "/tmp/better",
        },
    ]

    markdown = format_variant_markdown(rows)

    assert markdown.index("| better ") < markdown.index("| faster ")
