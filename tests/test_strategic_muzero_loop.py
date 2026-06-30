from __future__ import annotations

import json

from mini_tft.tools.strategic_muzero_loop import (
    StrategicMuZeroLoopConfig,
    build_loop_report,
    run_strategic_muzero_loop,
    verify_muzero_loop_metrics,
)


def test_strategic_muzero_loop_writes_fail_closed_scaffold(tmp_path) -> None:
    report = run_strategic_muzero_loop(
        StrategicMuZeroLoopConfig(out_dir=tmp_path, min_cache_rows=8)
    )

    verifier = json.loads((tmp_path / "verifier" / "metrics.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "loop-state.json").read_text(encoding="utf-8"))

    assert report["deliverable"] == "muzero_cache"
    assert report["status"] == "blocked"
    assert verifier["verdict"] == "REJECT"
    assert state["status"] == "blocked"
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "decision.md").exists()
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "loop-run-log.md").exists()
    assert (tmp_path / "verifier" / "decision.md").exists()
    assert "cache_rows" in {finding["check"] for finding in verifier["findings"]}


def test_strategic_muzero_loop_verifier_accepts_queue_ready_metrics() -> None:
    verifier = verify_muzero_loop_metrics(_ready_report(), require_queue_ready=True)

    assert verifier["verdict"] == "ACCEPT"
    assert verifier["status"] == "pass"
    assert verifier["summary"]["failed"] == 0


def test_strategic_muzero_loop_verifier_rejects_missing_mcts_targets() -> None:
    report = _ready_report()
    report["metrics"]["cache"]["mcts_target_rows"] = 0
    report["metrics"]["cache"]["mcts_target_rate"] = 0.0

    verifier = verify_muzero_loop_metrics(report, require_queue_ready=True)

    assert verifier["verdict"] == "REJECT"
    failed = {finding["check"] for finding in verifier["findings"]}
    assert {"mcts_target_rows", "mcts_target_rate"} <= failed


def test_strategic_muzero_loop_verifier_rejects_missing_parity_gate() -> None:
    report = _ready_report()
    report["metrics"]["parity"] = {
        "present": False,
        "status": "not_run",
        "total_checks": 0,
        "failed": 0,
    }

    verifier = verify_muzero_loop_metrics(report, require_queue_ready=True)

    assert verifier["verdict"] == "REJECT"
    failed = {finding["check"] for finding in verifier["findings"]}
    assert "parity_matrix" in failed


def test_strategic_muzero_loop_validates_cache_rows_jsonl(tmp_path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps(_cache_row()) + "\n", encoding="utf-8")

    report = build_loop_report(
        StrategicMuZeroLoopConfig(
            out_dir=tmp_path,
            min_cache_rows=1,
            cache_rows_path=rows_path,
        )
    )
    cache = report["metrics"]["cache"]

    assert cache["rows"] == 1
    assert cache["legal_action_rate"] == 1.0
    assert cache["policy_target_valid_rate"] == 1.0
    assert cache["illegal_policy_mass_max"] == 0.0
    assert cache["value_target_finite_rate"] == 1.0
    assert cache["mcts_target_rows"] == 0


def test_strategic_muzero_loop_allows_verified_smoke_without_baseline_or_checkpoint() -> None:
    report = _ready_report()
    report["metrics"]["baselines"] = {
        "present": False,
        "comparison_present": False,
        "policies": [],
    }
    report["metrics"]["train_smoke"]["checkpoint_exists"] = False

    verifier = verify_muzero_loop_metrics(report, require_queue_ready=False)

    assert verifier["verdict"] == "ACCEPT"
    assert verifier["status"] == "smoke_only"


def _ready_report() -> dict:
    rows = 16
    return {
        "schema": "quality-gate/v1",
        "deliverable": "muzero_cache",
        "status": "blocked",
        "seed": 0,
        "commands": [],
        "metrics": {
            "acceptance": {
                "min_cache_rows": rows,
                "policy_sum_tolerance": 1e-5,
                "require_queue_ready": True,
            },
            "action_surface": {
                "name": "strategic_11_macro",
                "observation_dim": 38,
                "action_dim": 11,
            },
            "cache": {
                "rows": rows,
                "mcts_target_rows": rows,
                "mcts_target_rate": 1.0,
                "legal_action_rate": 1.0,
                "policy_target_valid_rate": 1.0,
                "policy_target_sum_error_max": 0.0,
                "illegal_policy_mass_max": 0.0,
                "value_target_finite_rate": 1.0,
            },
            "search_smoke": {
                "ran": True,
                "total_decisions": rows,
                "illegal_action_count": 0,
            },
            "train_smoke": {
                "ran": True,
                "losses_finite": True,
                "checkpoint_exists": True,
            },
            "baselines": {
                "present": True,
                "comparison_present": True,
                "policies": ["random", "heuristic"],
            },
            "parity": {
                "present": True,
                "schema": "strategic-parity-matrix/v1",
                "status": "pass",
                "oracle_backend": "python",
                "backends": ["python", "native_cpp", "ocean_c"],
                "total_checks": 48,
                "passed": 48,
                "failed": 0,
            },
            "determinism": {
                "fixed_seed_reproducible": True,
            },
        },
        "artifacts": [],
        "known_limits": [],
    }


def _cache_row() -> dict:
    legal_mask = [True] * 11
    policy_target = [0.0] * 11
    policy_target[3] = 1.0
    return {
        "observation": [0.0] * 38,
        "legal_mask": legal_mask,
        "action": 3,
        "reward": 0.0,
        "next_observation": [0.0] * 38,
        "done": False,
        "policy_target": policy_target,
        "value_target": 0.25,
        "metadata": {"policy_target_source": "heuristic"},
    }
