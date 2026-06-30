from __future__ import annotations

import json
from math import isfinite

from mini_tft.tools.generate_strategic_muzero_cache import (
    StrategicMuZeroCacheRunConfig,
    run_strategic_muzero_cache_run,
)
from mini_tft.tools.strategic_muzero_loop import (
    StrategicMuZeroLoopConfig,
    build_loop_report,
    verify_muzero_loop_metrics,
)
from mini_tft.tools.train_strategic_muzero_smoke import (
    StrategicMuZeroTrainSmokeConfig,
    run_strategic_muzero_train_smoke,
)


def test_strategic_muzero_train_smoke_writes_metrics_and_checkpoint(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    train_dir = tmp_path / "train"
    run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=cache_dir,
            episodes=2,
            max_rows=16,
            seed=11,
            simulations=4,
            max_depth=4,
            rollout_steps=2,
        )
    )

    report = run_strategic_muzero_train_smoke(
        StrategicMuZeroTrainSmokeConfig(
            rows_path=cache_dir / "rows.jsonl",
            out_dir=train_dir,
            epochs=3,
            learning_rate=0.02,
            seed=5,
        )
    )
    metrics = json.loads((train_dir / "metrics.json").read_text(encoding="utf-8"))
    train = metrics["metrics"]["train_smoke"]

    assert metrics == report
    assert report["status"] == "smoke_only"
    assert train["ran"] is True
    assert train["losses_finite"] is True
    assert train["checkpoint_exists"] is True
    assert (train_dir / "train_smoke.npz").exists()
    assert isfinite(train["policy_loss_final"])
    assert isfinite(train["value_loss_final"])
    assert isfinite(train["dynamics_loss_final"])


def test_train_smoke_metrics_complete_loop_verifier(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    train_dir = tmp_path / "train"
    baseline_path = tmp_path / "baseline.json"
    parity_path = tmp_path / "parity.json"
    run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=cache_dir,
            episodes=2,
            max_rows=16,
            seed=13,
            simulations=4,
            max_depth=4,
            rollout_steps=2,
        )
    )
    run_strategic_muzero_train_smoke(
        StrategicMuZeroTrainSmokeConfig(
            rows_path=cache_dir / "rows.jsonl",
            out_dir=train_dir,
            epochs=3,
            learning_rate=0.02,
            seed=7,
        )
    )
    baseline_path.write_text(
        json.dumps(
            {
                "policies": ["heuristic", "random"],
                "comparison": {
                    "best_by_mean_placement": "heuristic",
                    "best_by_mean_scenario_score": "heuristic",
                },
            }
        ),
        encoding="utf-8",
    )
    parity_path.write_text(
        json.dumps(
            {
                "schema": "strategic-parity-matrix/v1",
                "status": "pass",
                "oracle_backend": "python",
                "backends": ["python", "native_cpp", "ocean_c"],
                "summary": {
                    "total_checks": 4,
                    "passed": 4,
                    "failed": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    loop_report = build_loop_report(
        StrategicMuZeroLoopConfig(
            out_dir=tmp_path / "loop",
            min_cache_rows=16,
            cache_metrics_path=cache_dir / "metrics.json",
            cache_rows_path=cache_dir / "rows.jsonl",
            train_metrics_path=train_dir / "metrics.json",
            baseline_metrics_path=baseline_path,
            parity_metrics_path=parity_path,
        )
    )
    verifier = verify_muzero_loop_metrics(loop_report, require_queue_ready=True)

    assert verifier["verdict"] == "ACCEPT"
    assert verifier["status"] == "pass"
    assert verifier["summary"]["failed"] == 0
