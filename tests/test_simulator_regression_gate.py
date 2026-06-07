from __future__ import annotations

from mini_tft.tools.simulator_regression_gate import (
    RegressionGateConfig,
    format_markdown,
    run_simulator_regression_gate,
)


def test_simulator_regression_gate_passes_minimum_quality_checks() -> None:
    report = run_simulator_regression_gate(
        RegressionGateConfig(
            benchmark_episodes=2,
            dataset_episodes=2,
            parallel_dataset_episodes=2,
            parallel_workers=1,
            eval_episodes=1,
            trace_steps=2,
            level_episodes=2,
        )
    )

    assert report["status"] == "pass"
    names = {check["name"] for check in report["checks"]}
    assert {
        "sim_smoke",
        "determinism",
        "round_schedule",
        "item_flow",
        "candidate_boards",
        "combat_fixtures",
        "level_pacing",
    } <= names
    assert not report["failures"]


def test_simulator_regression_gate_fails_on_bad_level_threshold() -> None:
    report = run_simulator_regression_gate(
        RegressionGateConfig(
            level_episodes=2,
            min_mean_final_level=99.0,
            run_smoke_report=False,
        )
    )

    assert report["status"] == "fail"
    assert "level_pacing" in report["failures"]


def test_simulator_regression_gate_markdown_is_agent_readable() -> None:
    report = run_simulator_regression_gate(
        RegressionGateConfig(level_episodes=2, run_smoke_report=False)
    )

    markdown = format_markdown(report)

    assert "# MiniTFT Simulator Regression Gate" in markdown
    assert "| Check | Status | Key Metrics |" in markdown
    assert "`combat_fixtures`" in markdown
    assert "`candidate_boards`" in markdown
