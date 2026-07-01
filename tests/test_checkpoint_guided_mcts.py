from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_tft.strategic.adapters.mcts import StrategicMCTSConfig, StrategicMCTSPlanner
from mini_tft.strategic.core import StrategicConfig, legal_action_mask, reset
from mini_tft.tools.checkpoint_guided_mcts_smoke import (
    CheckpointGuidedMCTSSmokeConfig,
    run_checkpoint_guided_mcts_smoke,
)
from mini_tft.tools.generate_strategic_muzero_cache import (
    StrategicMuZeroCacheRunConfig,
    run_strategic_muzero_cache_run,
)
from mini_tft.tools.train_strategic_muzero_torch import (
    StrategicMuZeroTorchTrainConfig,
    load_torch_muzero_policy_value,
    run_strategic_muzero_torch_train,
)


def test_checkpoint_policy_value_masks_illegal_actions(tmp_path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    evaluator = load_torch_muzero_policy_value(checkpoint, device="cpu")
    state = reset(seed=5)
    mask = legal_action_mask(state)

    priors, value = evaluator(state, mask, StrategicConfig())

    assert priors.shape == mask.shape
    assert float(priors.sum()) == pytest.approx(1.0)
    assert float(priors[~mask].sum()) == pytest.approx(0.0)
    assert value == pytest.approx(float(value))


def test_checkpoint_policy_value_allows_empty_mask_for_value_only(tmp_path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    evaluator = load_torch_muzero_policy_value(checkpoint, device="cpu")
    state = reset(seed=5)
    mask = legal_action_mask(state)
    empty_mask = mask & False

    priors, value = evaluator(state, empty_mask, StrategicConfig())

    assert priors.shape == mask.shape
    assert float(priors.sum()) == pytest.approx(0.0)
    assert value == pytest.approx(float(value))


def test_checkpoint_guided_mcts_records_checkpoint_metadata(tmp_path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    evaluator = load_torch_muzero_policy_value(checkpoint, device="cpu")
    state = reset(seed=7)
    planner = StrategicMCTSPlanner(
        config=StrategicMCTSConfig(
            simulations=2,
            max_depth=2,
            rollout_steps=1,
            prior_mode="checkpoint",
            value_mode="checkpoint",
            checkpoint_path=str(checkpoint),
            checkpoint_evaluator=evaluator,
        )
    )

    decision = planner.plan(state)

    assert legal_action_mask(state)[decision.selected_action]
    assert float(decision.visit_policy.sum()) == pytest.approx(1.0)
    assert decision.trace["prior_mode"] == "checkpoint"
    assert decision.trace["value_mode"] == "checkpoint"
    assert decision.trace["checkpoint_path"] == str(checkpoint)


def test_checkpoint_guided_mcts_smoke_writes_comparison_artifacts(tmp_path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)
    out_dir = tmp_path / "checkpoint_guided"

    report = run_checkpoint_guided_mcts_smoke(
        CheckpointGuidedMCTSSmokeConfig(
            out_dir=out_dir,
            checkpoint_path=checkpoint,
            episodes=1,
            max_rows=4,
            seed=11,
            simulations=2,
            max_depth=2,
            rollout_steps=1,
            skip_determinism_check=True,
        )
    )

    persisted = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (out_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loop_state = json.loads((out_dir / "loop-state.json").read_text(encoding="utf-8"))
    loop_log = (out_dir / "loop-run-log.md").read_text(encoding="utf-8")

    assert persisted == report
    assert report["status"] == "pass"
    assert report["metrics"]["programmatic_checks_passed"] == report["metrics"][
        "programmatic_checks_total"
    ]
    assert report["metrics"]["quality_checks"]["checkpoint_prior_mode_recorded"] is True
    assert report["metrics"]["quality_checks"]["checkpoint_value_mode_recorded"] is True
    assert report["metrics"]["comparison"]["rows_match"] is True
    assert rows
    assert rows[0]["metadata"]["policy_target_source"] == "checkpoint_guided_mcts"
    assert rows[0]["metadata"]["mcts_prior_mode"] == "checkpoint"
    assert rows[0]["metadata"]["mcts_value_mode"] == "checkpoint"
    assert loop_state["owner"] == "codex"
    assert loop_state["verifier"] == "pending_post_run_judge"
    assert loop_state["state_prune_rules"]
    assert loop_state["pause_criteria"]
    assert loop_state["kill_criteria"]
    assert "pending post-run judge" in loop_log
    assert (out_dir / "checkpoint_guided_cache" / "metrics.json").exists()
    assert (out_dir / "heuristic_cache" / "metrics.json").exists()


def _tiny_checkpoint(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    train_dir = tmp_path / "train"
    run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=cache_dir,
            episodes=1,
            max_rows=8,
            seed=3,
            simulations=2,
            max_depth=2,
            rollout_steps=1,
            determinism_check=False,
        )
    )
    run_strategic_muzero_torch_train(
        StrategicMuZeroTorchTrainConfig(
            rows_path=cache_dir / "rows.jsonl",
            out_dir=train_dir,
            epochs=1,
            learning_rate=1e-3,
            batch_size=4,
            hidden_size=16,
            seed=5,
            device="cpu",
        )
    )
    return train_dir / "strategic_muzero_torch.pt"
