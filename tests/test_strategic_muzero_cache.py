from __future__ import annotations

import json

import pytest

from mini_tft.strategic.adapters.mcts import StrategicMCTSConfig
from mini_tft.strategic.adapters.muzero_cache import cache_metrics, generate_mcts_cache
from mini_tft.tools.generate_strategic_muzero_cache import (
    StrategicMuZeroCacheRunConfig,
    run_strategic_muzero_cache_run,
)
from mini_tft.tools.strategic_muzero_loop import (
    StrategicMuZeroLoopConfig,
    build_loop_report,
    verify_muzero_loop_metrics,
)


def test_generate_mcts_cache_uses_visit_policy_targets() -> None:
    rows = generate_mcts_cache(
        episodes=1,
        max_rows=8,
        seed=3,
        mcts_config=StrategicMCTSConfig(simulations=4, max_depth=4, rollout_steps=2),
    )
    metrics = cache_metrics(rows)

    assert len(rows) == 8
    assert metrics["mcts_target_rows"] == 8
    assert metrics["mcts_target_rate"] == 1.0
    assert metrics["legal_action_rate"] == 1.0
    assert metrics["policy_target_valid_rate"] == 1.0
    assert metrics["illegal_policy_mass_max"] == 0.0
    assert metrics["value_target_finite_rate"] == 1.0
    assert rows[0].metadata["policy_target_source"] == "mcts"
    assert rows[0].metadata["mcts_simulations"] == 4
    assert float(rows[0].policy_target.sum()) == pytest.approx(1.0)


def test_mcts_cache_run_feeds_loop_cache_gates(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    report = run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=cache_dir,
            episodes=2,
            max_rows=16,
            seed=7,
            simulations=4,
            max_depth=4,
            rollout_steps=2,
        )
    )

    metrics_path = cache_dir / "metrics.json"
    rows_path = cache_dir / "rows.jsonl"
    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    loop_report = build_loop_report(
        StrategicMuZeroLoopConfig(
            out_dir=tmp_path / "loop",
            min_cache_rows=16,
            cache_metrics_path=metrics_path,
            cache_rows_path=rows_path,
        )
    )
    verifier = verify_muzero_loop_metrics(loop_report, require_queue_ready=False)
    failed_checks = {finding["check"] for finding in verifier["findings"]}

    assert persisted == report
    assert report["status"] == "smoke_only"
    assert loop_report["metrics"]["cache"]["rows"] == 16
    assert loop_report["metrics"]["cache"]["mcts_target_rows"] == 16
    assert loop_report["metrics"]["determinism"]["fixed_seed_reproducible"] is True
    assert "cache_rows" not in failed_checks
    assert "mcts_target_rows" not in failed_checks
    assert "mcts_target_rate" not in failed_checks
    assert "determinism" not in failed_checks
    assert "train_smoke_ran" in failed_checks
