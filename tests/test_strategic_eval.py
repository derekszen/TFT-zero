from __future__ import annotations

import json

import pytest

from mini_tft.strategic.core import NUM_ACTIONS
from mini_tft.strategic.core.actions import action_name
from mini_tft.tools.evaluate_strategic_policy import (
    StrategicPolicyEvaluationConfig,
    build_policy_specs,
    run_strategic_policy_evaluation,
)


def test_strategic_policy_evaluation_writes_auditable_artifacts(tmp_path) -> None:
    report = run_strategic_policy_evaluation(
        StrategicPolicyEvaluationConfig(
            out_dir=tmp_path,
            episodes=3,
            seed=19,
        )
    )

    metrics_path = tmp_path / "metrics.json"
    decision_path = tmp_path / "decision.md"
    episodes_path = tmp_path / "episodes.jsonl"
    actions_path = tmp_path / "actions.jsonl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    episode_rows = [
        json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()
    ]
    action_rows = [
        json.loads(line) for line in actions_path.read_text(encoding="utf-8").splitlines()
    ]

    assert metrics == report
    assert decision_path.exists()
    assert metrics["schema"] == "strategic-policy-eval/v1"
    assert metrics["status"] == "smoke_only"
    assert metrics["policies"] == ["heuristic", "random"]
    assert metrics["episodes_per_policy"] == 3
    assert len(episode_rows) == 6
    assert action_rows
    assert metrics["overall"]["illegal_action_count"] == 0
    assert metrics["overall"]["total_actions"] == len(action_rows)
    assert set(metrics["comparison"]) == {
        "best_by_mean_placement",
        "best_by_mean_scenario_score",
    }

    expected_actions = {action_name(action_id) for action_id in range(NUM_ACTIONS)}
    for policy in metrics["policies"]:
        summary = metrics["policy_summaries"][policy]
        assert summary["episodes"] == 3
        assert sum(summary["placement_counts"].values()) == 3
        assert 0.0 <= summary["survivor_rate"] <= 1.0
        assert 0.0 <= summary["mean_scenario_score"] <= 1.0
        assert summary["illegal_action_count"] == 0
        assert set(summary["action_counts"]) >= expected_actions
        assert sum(summary["action_counts"].values()) == summary["total_actions"]
        assert sum(summary["action_rates"].values()) == pytest.approx(1.0)


def test_strategic_policy_evaluation_is_deterministic(tmp_path) -> None:
    left = run_strategic_policy_evaluation(
        StrategicPolicyEvaluationConfig(
            out_dir=tmp_path / "left",
            policies=build_policy_specs(["heuristic"]),
            episodes=4,
            seed=7,
        )
    )
    right = run_strategic_policy_evaluation(
        StrategicPolicyEvaluationConfig(
            out_dir=tmp_path / "right",
            policies=build_policy_specs(["heuristic"]),
            episodes=4,
            seed=7,
        )
    )

    assert left == right


def test_weakest_legal_policy_is_explicit_legal_baseline(tmp_path) -> None:
    report = run_strategic_policy_evaluation(
        StrategicPolicyEvaluationConfig(
            out_dir=tmp_path,
            policies=build_policy_specs(["weakest_legal", "worst_first"]),
            episodes=2,
            seed=11,
        )
    )

    assert report["policies"] == ["weakest_legal", "worst_first"]
    for policy in report["policies"]:
        summary = report["policy_summaries"][policy]
        assert summary["illegal_action_count"] == 0
        assert summary["action_counts"]["hold"] == summary["total_actions"]


def test_strategic_policy_evaluation_rejects_unknown_policy(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown strategic policy"):
        run_strategic_policy_evaluation(
            StrategicPolicyEvaluationConfig(
                out_dir=tmp_path,
                policies=build_policy_specs(["missing"]),
                episodes=1,
            )
        )
