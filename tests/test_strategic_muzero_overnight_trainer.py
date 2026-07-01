from __future__ import annotations

import json
import shutil

import pytest

from mini_tft.tools.generate_strategic_muzero_cache import (
    StrategicMuZeroCacheRunConfig,
    run_strategic_muzero_cache_run,
)
from mini_tft.tools.run_strategic_muzero_overnight import (
    StrategicMuZeroOvernightConfig,
    run_strategic_muzero_overnight,
)
from mini_tft.tools.train_strategic_muzero_torch import (
    StrategicMuZeroTorchTrainConfig,
    StrategicTorchPolicyEvalConfig,
    run_strategic_muzero_torch_train,
    run_torch_checkpoint_policy_evaluation,
)


def test_torch_train_and_eval_write_checkpoint_and_metrics(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=cache_dir,
            episodes=2,
            max_rows=16,
            seed=17,
            simulations=4,
            max_depth=4,
            rollout_steps=2,
        )
    )

    train_report = run_strategic_muzero_torch_train(
        StrategicMuZeroTorchTrainConfig(
            rows_path=cache_dir / "rows.jsonl",
            out_dir=train_dir,
            epochs=2,
            learning_rate=1e-3,
            batch_size=8,
            hidden_size=32,
            seed=19,
            device="cpu",
        )
    )
    checkpoint = train_dir / "strategic_muzero_torch.pt"
    eval_report = run_torch_checkpoint_policy_evaluation(
        StrategicTorchPolicyEvalConfig(
            checkpoint_path=checkpoint,
            out_dir=eval_dir,
            episodes=2,
            seed=23,
            device="cpu",
        )
    )

    train = train_report["metrics"]["train_torch"]
    assert train_report["status"] == "smoke_only"
    assert train["losses_finite"] is True
    assert train["checkpoint_exists"] is True
    assert train["legal_argmax_rate"] == pytest.approx(1.0)
    assert checkpoint.exists()
    assert eval_report["status"] == "smoke_only"
    assert eval_report["policies"] == [
        "random",
        "weakest_legal",
        "worst_first",
        "heuristic",
        "torch_muzero",
    ]
    assert (eval_dir / "actions.jsonl").exists()


def test_overnight_wrapper_writes_incremental_loop_artifacts(tmp_path) -> None:
    if shutil.which("cc") is None:
        pytest.skip("C compiler is not available")

    report = run_strategic_muzero_overnight(
        StrategicMuZeroOvernightConfig(
            out_dir=tmp_path,
            seed=31,
            cache_episodes=2,
            cache_rows=16,
            mcts_simulations=4,
            mcts_max_depth=4,
            mcts_rollout_steps=2,
            train_epochs=1,
            train_learning_rate=1e-3,
            train_batch_size=8,
            hidden_size=32,
            device="cpu",
            eval_episodes=2,
            preflight_rows=8,
            preflight_episodes=2,
            parity_seeds=(0,),
            parity_scenarios=("reset_only",),
            codex_allowance_source="unknown",
            codex_allowance_decision="continue",
        )
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    heartbeat_rows = [
        json.loads(line)
        for line in (tmp_path / "heartbeat.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loop_log = (tmp_path / "loop-run-log.md").read_text(encoding="utf-8")
    loop_state = json.loads((tmp_path / "loop-state.json").read_text(encoding="utf-8"))

    assert report == metrics
    assert metrics["status"] == "smoke_only"
    assert metrics["claim_label"] == "smoke_only"
    assert metrics["failed_checks"] == []
    if metrics["native_search"]["native_available"]:
        assert metrics["metrics"]["search_smoke"]["backend"] == "native"
    else:
        assert metrics["metrics"]["search_smoke"]["backend"] == "python"
        assert metrics["metrics"]["search_smoke"]["fallback_reason"]
    assert len(heartbeat_rows) >= 10
    assert heartbeat_rows[0]["stage"] == "start"
    assert any(row["stage"] == "cache_generation" for row in heartbeat_rows)
    assert any(row["stage"] == "torch_training" for row in heartbeat_rows)
    assert loop_state["current_stage"] == "final_report"
    assert loop_state["attempt"] == 1
    assert loop_state["codex_allowance_check"]["source"] == "unknown"
    assert metrics["config"]["codex_allowance_check"]["source"] == "unknown"
    assert metrics["validation"]["failed"] is False
    assert "cache_generation" in loop_log
    assert "Codex allowance" in loop_log
    assert "torch_training" in loop_log
    assert (tmp_path / "command.json").exists()
    assert (tmp_path / "trainer_command.txt").exists()
    assert (tmp_path / "gpu_samples.jsonl").exists()
    assert (tmp_path / "nvidia_smi_before.txt").exists()
    assert (tmp_path / "nvidia_smi_after.txt").exists()
    assert (tmp_path / "cache" / "rows.jsonl").exists()
    assert (tmp_path / "train_torch" / "strategic_muzero_torch.pt").exists()
    assert (tmp_path / "torch_policy_eval" / "metrics.json").exists()
    assert (tmp_path / "gate" / "verifier" / "metrics.json").exists()
