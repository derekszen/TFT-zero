from __future__ import annotations

import pytest

from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData, load_set
from mini_tft.tools.combat_fixture_report import (
    combat_fixture_gates,
    evaluate_combat_fixture,
    format_text,
    run_combat_fixture_report,
)

REQUIRED_FIXTURES = {
    "frontline_backline_positioning",
    "itemized_carry",
    "two_star_upgrade",
    "assassin_pressure",
    "trait_breakpoint",
    "no_frontline_penalty",
}


@pytest.fixture(scope="module")
def data() -> GameData:
    return load_set()


@pytest.fixture(scope="module")
def config() -> EnvConfig:
    return EnvConfig()


def test_combat_fixture_gate_covers_required_scenarios() -> None:
    fixtures = combat_fixture_gates()
    names = {fixture.name for fixture in fixtures}

    assert names == REQUIRED_FIXTURES
    assert len(names) == len(fixtures)
    assert all(fixture.min_strength_delta > 0 for fixture in fixtures)
    assert all(fixture.min_win_probability_delta > 0 for fixture in fixtures)


@pytest.mark.parametrize("fixture", combat_fixture_gates(), ids=lambda fixture: fixture.name)
def test_each_combat_fixture_passes_meaningful_quality_gate(
    fixture,
    data: GameData,
    config: EnvConfig,
) -> None:
    row = evaluate_combat_fixture(fixture, data, config)

    assert row["status"] == "pass", row
    assert row["strength_delta"] >= row["min_strength_delta"]
    assert row["win_probability_delta"] >= row["min_win_probability_delta"]


def test_combat_fixture_report_summarizes_pass_fail_and_deltas(
    data: GameData,
    config: EnvConfig,
) -> None:
    report = run_combat_fixture_report(data=data, config=config)

    assert report["status"] == "pass"
    assert report["passed"] == report["total"] == len(REQUIRED_FIXTURES)
    for row in report["fixtures"]:
        assert row["name"] in REQUIRED_FIXTURES
        assert row["status"] == "pass"
        assert row["strength_delta"] > 0
        assert row["win_probability_delta"] > 0

    text = format_text(report)
    assert "MiniTFT combat fixture report: pass" in text
    assert "str_delta" in text
    assert "pwin_delta" in text
    assert all(name in text for name in REQUIRED_FIXTURES)
