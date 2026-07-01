from __future__ import annotations

import json

from mini_tft.tools.generate_strategic_muzero_cache import (
    StrategicMuZeroCacheRunConfig,
    run_strategic_muzero_cache_run,
)
from mini_tft.tools.strategic_policy_improvement_gate import (
    REQUIRED_POLICIES,
    StrategicPolicyImprovementGateConfig,
    run_strategic_policy_improvement_gate,
)
from mini_tft.tools.train_strategic_muzero_torch import (
    StrategicMuZeroTorchTrainConfig,
    run_strategic_muzero_torch_train,
)


def test_policy_improvement_gate_writes_repeated_seed_artifacts(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    train_dir = tmp_path / "train"
    gate_dir = tmp_path / "gate"
    run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=cache_dir,
            episodes=2,
            max_rows=16,
            seed=41,
            simulations=4,
            max_depth=4,
            rollout_steps=2,
        )
    )
    run_strategic_muzero_torch_train(
        StrategicMuZeroTorchTrainConfig(
            rows_path=cache_dir / "rows.jsonl",
            out_dir=train_dir,
            epochs=1,
            learning_rate=1e-3,
            batch_size=8,
            hidden_size=32,
            seed=43,
            device="cpu",
        )
    )

    report = run_strategic_policy_improvement_gate(
        StrategicPolicyImprovementGateConfig(
            checkpoint_path=train_dir / "strategic_muzero_torch.pt",
            out_dir=gate_dir,
            seeds=(47, 53),
            episodes_per_seed=2,
            device="cpu",
        )
    )
    metrics = json.loads((gate_dir / "metrics.json").read_text(encoding="utf-8"))
    loop_state = json.loads((gate_dir / "loop-state.json").read_text(encoding="utf-8"))

    assert metrics == report
    assert report["schema"] == "strategic-policy-improvement-gate/v1"
    assert report["verdict"] in {"ACCEPT", "REJECT"}
    assert report["status"] in {"pass", "reject"}
    assert report["policies"] == list(REQUIRED_POLICIES)
    assert report["promotion_decision"]["seed_count"] == 2
    assert report["promotion_decision"]["required_seed_wins"] == 2
    assert set(report["promotion_decision"]["criteria"]) == {
        "required_policies_present",
        "torch_zero_illegal_actions",
        "beats_random_most_seeds",
        "beats_weakest_legal_most_seeds",
        "beats_worst_first_most_seeds",
        "aggregate_action_distribution_present",
    }
    assert set(report["aggregate_policy_summaries"]) == set(REQUIRED_POLICIES)
    for policy in REQUIRED_POLICIES:
        summary = report["aggregate_policy_summaries"][policy]
        assert summary["episodes"] == 4
        assert "mean_placement" in summary
        assert "mean_scenario_score" in summary
        assert "mean_final_round" in summary
        assert "mean_reward" in summary
        assert "illegal_action_count" in summary
        assert "survivor_rate" in summary
        assert sum(summary["action_counts"].values()) == summary["total_actions"]

    assert loop_state["current_stage"] == "final_report"
    assert loop_state["verdict"] == report["verdict"]
    assert (gate_dir / "decision.md").exists()
    assert (gate_dir / "final_report.md").exists()
    assert (gate_dir / "loop-run-log.md").exists()
    assert (gate_dir / "seed_evals" / "seed_47" / "metrics.json").exists()
    assert (gate_dir / "seed_evals" / "seed_53" / "metrics.json").exists()
