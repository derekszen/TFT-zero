from __future__ import annotations

from mini_tft.tools.web_ui_regression_gate import (
    WebUiGateConfig,
    format_markdown,
    run_web_ui_regression_gate,
)


def test_web_ui_regression_gate_passes_playability_checks() -> None:
    report = run_web_ui_regression_gate(WebUiGateConfig(seed=0))

    assert report["status"] == "pass"
    names = {check["name"] for check in report["checks"]}
    assert {
        "opening_state",
        "enemy_preview_schedule",
        "item_action_flow",
        "manual_move",
    } <= names
    assert not report["failures"]


def test_web_ui_regression_gate_markdown_is_agent_readable() -> None:
    report = run_web_ui_regression_gate(WebUiGateConfig(seed=0))

    markdown = format_markdown(report)

    assert "# MiniTFT Web UI Regression Gate" in markdown
    assert "| Check | Status | Key Metrics |" in markdown
    assert "`enemy_preview_schedule`" in markdown
    assert "`item_action_flow`" in markdown
