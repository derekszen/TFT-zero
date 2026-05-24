from __future__ import annotations

from mini_tft.tools.sim_smoke import SmokeConfig, format_markdown, run_smoke


def test_sim_smoke_report_has_agent_friendly_sections() -> None:
    report = run_smoke(
        SmokeConfig(
            benchmark_episodes=1,
            dataset_episodes=1,
            eval_episodes=1,
            trace_steps=2,
            seed=0,
            run_checks=False,
        )
    )
    markdown = format_markdown(report)

    assert report["status"] == "pass"
    assert "# MiniTFT Simulator Smoke Report" in markdown
    assert "## Throughput" in markdown
    assert "## Dataset Shape" in markdown
    assert "## Bot Evaluation" in markdown
    assert "## Trace Preview" in markdown
    assert report["dataset"]["transitions"] > 0
    assert report["benchmark"]["steps"] > 0
