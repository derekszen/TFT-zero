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
            judge_packet_name="strategic-muzero-run-loop-test",
            judge_out_root=tmp_path / "judge",
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
    assert metrics["metrics"]["claim_scope"]["name"] == "cache_supervised_muzero_style_v0"
    assert metrics["metrics"]["antigravity_judge"]["status"] == "pending_verdict"
    assert metrics["metrics"]["antigravity_judge"]["preferred_model"] == "gemini-3.5-flash-low"
    assert (
        metrics["metrics"]["antigravity_judge"]["packet"]["out_dir"]
        == str(tmp_path / "judge" / "strategic-muzero-run-loop-test")
    )
    assert "antigravity_ai_router_command.txt" in (
        metrics["metrics"]["antigravity_judge"]["packet"]["artifacts"]
    )
    assert state["blocked_condition"]
    assert state["pause_criteria"]
    assert state["kill_criteria"]
    assert state["config"]["automation_level"] == "L1"
    assert state["config"]["wall_clock_limit_minutes"] == 360
    assert state["codex_allowance_check"]["source"] == "unknown"
    assert state["codex_allowance_check"]["weekly_usage"] == "unknown"
    assert state["codex_allowance_check"]["decision"] == "soft-pause"
    assert gate_state["codex_allowance_check"]["source"] == "unknown"
    assert gate_state["codex_allowance_check"]["weekly_usage"] == "unknown"
    assert gate_state["codex_allowance_check"]["decision"] == "soft-pause"
    assert "provide Codex allowance status" in state["current_next_action"]
    assert (tmp_path / "parity_matrix" / "metrics.json").exists()
    assert (tmp_path / "policy_eval" / "metrics.json").exists()
    assert (tmp_path / "cache" / "rows.jsonl").exists()
    assert (tmp_path / "train_smoke" / "train_smoke.npz").exists()
    assert (tmp_path / "gate" / "final_report.md").exists()
    assert (tmp_path / "verifier" / "metrics.json").exists()
    assert (tmp_path / "verifier" / "decision.md").exists()
    assert (tmp_path / "loop-state.json").exists()
    assert (tmp_path / "loop-run-log.md").exists()
    assert (tmp_path / "overnight_goal.md").exists()
    assert (tmp_path / "judge" / "strategic-muzero-run-loop-test" / "prompt.md").exists()


def test_strategic_muzero_run_loop_blocks_after_attempt_cap(tmp_path) -> None:
    if shutil.which("cc") is None:
        pytest.skip("C compiler is not available")

    config = StrategicMuZeroRunLoopConfig(
        out_dir=tmp_path,
        seed=23,
        attempt_cap=1,
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

    first = run_strategic_muzero_run_loop(config)
    second = run_strategic_muzero_run_loop(config)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "loop-state.json").read_text(encoding="utf-8"))
    attempt_guard = metrics["metrics"]["attempt_guard"]

    assert first["status"] == "pass"
    assert second["status"] == "blocked"
    assert attempt_guard["attempt"] == 2
    assert attempt_guard["attempt_cap"] == 1
    assert attempt_guard["exceeded"] is True
    assert state["status"] == "blocked"
    assert state["attempt"] == 2
    assert "attempt cap exceeded" in state["current_next_action"]


def test_strategic_muzero_run_loop_allows_user_waived_launch_metadata(tmp_path) -> None:
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
            codex_allowance_source="user-waived",
            codex_five_hour_window_remaining="waived",
            codex_weekly_usage="waived",
            codex_allowance_decision="continue",
        )
    )

    state = json.loads((tmp_path / "loop-state.json").read_text(encoding="utf-8"))

    assert report["status"] == "pass"
    assert state["codex_allowance_check"]["source"] == "user-waived"
    assert state["codex_allowance_check"]["decision"] == "continue"
    assert "queue the longer MuZero-style run" in state["current_next_action"]
