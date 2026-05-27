from __future__ import annotations

from mini_tft.tools.combat_eval import run_combat_eval


def test_combat_eval_runs_all_fixtures_without_strict_failure() -> None:
    report = run_combat_eval(benchmark_iters=1)

    assert report["total"] >= 5
    assert report["passed"] >= 1
    assert len(report["fixtures"]) == report["total"]
    assert report["benchmark"]["calls"] == report["total"] * 2
