from __future__ import annotations

import json
import shutil

import pytest

from mini_tft.tools.strategic_muzero_run_loop import (
    StrategicMuZeroRunLoopConfig,
    run_strategic_muzero_run_loop,
)


def test_strategic_muzero_run_loop_runs_full_programmatic_gate(tmp_path) -> None:
    if shutil.which("cc") is None:
        pytest.skip("C compiler is not available")

    report = run_strategic_muzero_run_loop(
        StrategicMuZeroRunLoopConfig(
            out_dir=tmp_path,
            seed=23,
            cache_episodes=2,
            cache_rows=16,
            mcts_simulations=4,
            mcts_max_depth=4,
            mcts_rollout_steps=2,
            train_epochs=3,
            train_learning_rate=0.02,
            baseline_episodes=2,
            parity_seeds=(0,),
            parity_scenarios=("reset_only",),
        )
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    verifier = json.loads(
        (tmp_path / "gate" / "verifier" / "metrics.json").read_text(encoding="utf-8")
    )
    criteria = metrics["metrics"]["programmatic_criteria"]

    assert report["status"] == "pass"
    assert metrics["status"] == "pass"
    assert verifier["verdict"] == "ACCEPT"
    assert verifier["summary"]["failed"] == 0
    assert criteria["failed"] == 0
    assert all(check["passed"] for check in criteria["checks"])
    state = json.loads((tmp_path / "loop-state.json").read_text(encoding="utf-8"))
    gate_state = json.loads(
        (tmp_path / "gate" / "loop-state.json").read_text(encoding="utf-8")
    )
    assert metrics["metrics"]["cache"]["rows"] == 16
    assert metrics["metrics"]["baselines"]["policies"] == ["heuristic", "random"]
    assert state["blocked_condition"]
    assert state["pause_criteria"]
    assert state["kill_criteria"]
    assert state["codex_allowance_check"]["source"] == "user"
    assert state["codex_allowance_check"]["weekly_usage"] == "ample"
    assert state["codex_allowance_check"]["decision"] == "continue"
    assert gate_state["codex_allowance_check"]["source"] == "user"
    assert gate_state["codex_allowance_check"]["weekly_usage"] == "ample"
    assert gate_state["codex_allowance_check"]["decision"] == "continue"
    assert (tmp_path / "parity_matrix" / "metrics.json").exists()
    assert (tmp_path / "policy_eval" / "metrics.json").exists()
    assert (tmp_path / "cache" / "rows.jsonl").exists()
    assert (tmp_path / "train_smoke" / "train_smoke.npz").exists()
    assert (tmp_path / "gate" / "final_report.md").exists()
    assert (tmp_path / "verifier" / "metrics.json").exists()
    assert (tmp_path / "verifier" / "decision.md").exists()
    assert (tmp_path / "loop-state.json").exists()
    assert (tmp_path / "loop-run-log.md").exists()
