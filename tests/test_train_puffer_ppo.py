from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mini_tft.rl.train_puffer_ppo import (
    MaskedActorCritic,
    action_mask_from_observation,
    make_training_env_batch,
    parse_args,
    puffer_checkpoint_path,
    resolve_rollout_and_batch_size,
    train_puffer_ppo,
)


def _require_legacy_pufferlib() -> None:
    pytest.importorskip("pufferlib.emulation")
    pufferlib = pytest.importorskip("pufferlib")
    if not hasattr(pufferlib, "PufferEnv"):
        pytest.skip("legacy PufferEnv wrapper API is not available")


def test_action_mask_from_observation_uses_tail_mask() -> None:
    obs = torch.zeros((2, 6), dtype=torch.float32)
    obs[0, -3:] = torch.tensor([0.0, 1.0, 0.0])
    obs[1, -3:] = torch.tensor([0.0, 0.0, 0.0])

    mask = action_mask_from_observation(obs, 3)

    assert mask.tolist() == [[False, True, False], [True, False, False]]


def test_masked_actor_critic_never_samples_illegal_actions() -> None:
    policy = MaskedActorCritic(observation_dim=6, action_dim=3, hidden_sizes=[8])
    obs = torch.zeros((64, 6), dtype=torch.float32)
    obs[:, -2] = 1.0

    actions, _logprobs, _entropy, values = policy.get_action_and_value(obs)

    assert actions.unique().tolist() == [1]
    assert values.shape == (64,)


def test_resolve_rollout_and_batch_size_clamps_to_rollout() -> None:
    args = parse_args(["--num-envs", "2", "--n-steps", "4", "--batch-size", "64"])

    assert resolve_rollout_and_batch_size(args) == (8, 8)


def test_parse_args_defaults_to_emulated_puffer_backend() -> None:
    args = parse_args([])

    assert args.puffer_backend == "emulated"


def test_make_training_env_batch_selects_native_backend() -> None:
    _require_legacy_pufferlib()
    from mini_tft.core.config import EnvConfig

    batch = make_training_env_batch(
        env_kind="lobby",
        puffer_backend="native",
        seed=123,
        config=EnvConfig(seed=123, max_round=2, max_actions_per_round=2),
        num_envs=2,
        players=4,
        max_actions_per_player=2,
        lobby_opponent_policy="tempo",
        allow_oracle_macro_actions=False,
    )
    try:
        obs, infos = batch.reset(123)

        assert batch.auto_resets
        assert batch.num_envs == 2
        assert obs.shape[0] == 2
        assert len(infos) == 2
    finally:
        batch.close()


def test_make_training_env_batch_selects_native_strategic_backend() -> None:
    _require_legacy_pufferlib()
    from mini_tft.strategic.core import StrategicConfig

    batch = make_training_env_batch(
        env_kind="strategic",
        puffer_backend="native",
        seed=126,
        config=StrategicConfig(max_round=3, max_actions_per_round=2),
        num_envs=3,
        players=4,
        max_actions_per_player=2,
        lobby_opponent_policy="tempo",
        allow_oracle_macro_actions=False,
    )
    try:
        obs, infos = batch.reset(126)

        assert batch.auto_resets
        assert batch.num_envs == 3
        assert obs.shape == (3, 49)
        assert len(infos) == 3
    finally:
        batch.close()


def test_puffer_checkpoint_path_uses_pt_suffix() -> None:
    assert puffer_checkpoint_path(Path("checkpoints/run")).suffix == ".pt"
    assert puffer_checkpoint_path(Path("checkpoints/run.pt")) == Path("checkpoints/run.pt")


def test_puffer_ppo_smoke_writes_checkpoint_and_manifest(tmp_path: Path) -> None:
    _require_legacy_pufferlib()
    output = tmp_path / "puffer_lobby_smoke.pt"
    args = parse_args(
        [
            "--timesteps",
            "8",
            "--seed",
            "123",
            "--out",
            str(output),
            "--num-envs",
            "2",
            "--n-steps",
            "2",
            "--batch-size",
            "4",
            "--update-epochs",
            "1",
            "--device",
            "cpu",
            "--env-kind",
            "lobby",
            "--players",
            "4",
            "--max-round",
            "2",
            "--max-actions-per-round",
            "2",
            "--max-actions-per-player",
            "2",
            "--hidden-sizes",
            "16",
            "--disallow-oracle-macro-actions",
        ]
    )

    result = train_puffer_ppo(args)

    checkpoint = Path(result["checkpoint_path"])
    manifest = Path(result["manifest_path"])
    assert checkpoint == output
    assert checkpoint.exists()
    assert manifest.exists()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["kind"] == "puffer_ppo"
    assert manifest_data["args"]["env_kind"] == "lobby"
    assert manifest_data["args"]["puffer_backend"] == "emulated"
    assert manifest_data["summary"]["total_steps"] >= 8


def test_strategic_puffer_ppo_smoke_writes_checkpoint_and_manifest(tmp_path: Path) -> None:
    _require_legacy_pufferlib()
    output = tmp_path / "strategic_puffer_smoke.pt"
    args = parse_args(
        [
            "--timesteps",
            "8",
            "--seed",
            "125",
            "--out",
            str(output),
            "--num-envs",
            "2",
            "--n-steps",
            "2",
            "--batch-size",
            "4",
            "--update-epochs",
            "1",
            "--device",
            "cpu",
            "--env-kind",
            "strategic",
            "--max-round",
            "3",
            "--max-actions-per-round",
            "2",
            "--hidden-sizes",
            "16",
        ]
    )

    result = train_puffer_ppo(args)

    checkpoint = Path(result["checkpoint_path"])
    manifest = Path(result["manifest_path"])
    assert checkpoint == output
    assert checkpoint.exists()
    assert manifest.exists()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["kind"] == "puffer_ppo"
    assert manifest_data["args"]["env_kind"] == "strategic"
    assert manifest_data["args"]["puffer_backend"] == "emulated"
    assert manifest_data["resolved"]["action_dim"] == 11
    assert manifest_data["summary"]["total_steps"] >= 8


def test_native_puffer_ppo_smoke_writes_checkpoint_and_manifest(tmp_path: Path) -> None:
    _require_legacy_pufferlib()

    output = tmp_path / "native_puffer_lobby_smoke.pt"
    args = parse_args(
        [
            "--timesteps",
            "8",
            "--seed",
            "124",
            "--out",
            str(output),
            "--num-envs",
            "2",
            "--n-steps",
            "2",
            "--batch-size",
            "4",
            "--update-epochs",
            "1",
            "--device",
            "cpu",
            "--env-kind",
            "lobby",
            "--puffer-backend",
            "native",
            "--players",
            "4",
            "--max-round",
            "2",
            "--max-actions-per-round",
            "2",
            "--max-actions-per-player",
            "2",
            "--hidden-sizes",
            "16",
            "--disallow-oracle-macro-actions",
        ]
    )

    result = train_puffer_ppo(args)

    checkpoint = Path(result["checkpoint_path"])
    manifest = Path(result["manifest_path"])
    assert checkpoint == output
    assert checkpoint.exists()
    assert manifest.exists()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["kind"] == "puffer_ppo"
    assert manifest_data["args"]["env_kind"] == "lobby"
    assert manifest_data["args"]["puffer_backend"] == "native"
    assert manifest_data["resolved"]["puffer_backend"] == "native"
    assert manifest_data["summary"]["total_steps"] >= 8


def test_native_strategic_puffer_ppo_smoke_writes_checkpoint_and_manifest(
    tmp_path: Path,
) -> None:
    _require_legacy_pufferlib()

    output = tmp_path / "native_strategic_puffer_smoke.pt"
    args = parse_args(
        [
            "--timesteps",
            "8",
            "--seed",
            "127",
            "--out",
            str(output),
            "--num-envs",
            "2",
            "--n-steps",
            "2",
            "--batch-size",
            "4",
            "--update-epochs",
            "1",
            "--device",
            "cpu",
            "--env-kind",
            "strategic",
            "--puffer-backend",
            "native",
            "--max-round",
            "3",
            "--max-actions-per-round",
            "2",
            "--hidden-sizes",
            "16",
        ]
    )

    result = train_puffer_ppo(args)

    checkpoint = Path(result["checkpoint_path"])
    manifest = Path(result["manifest_path"])
    assert checkpoint == output
    assert checkpoint.exists()
    assert manifest.exists()
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["kind"] == "puffer_ppo"
    assert manifest_data["args"]["env_kind"] == "strategic"
    assert manifest_data["args"]["puffer_backend"] == "native"
    assert manifest_data["resolved"]["puffer_backend"] == "native"
    assert manifest_data["resolved"]["action_dim"] == 11
    assert manifest_data["summary"]["total_steps"] >= 8
