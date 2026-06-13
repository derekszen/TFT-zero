from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from mini_tft.rl.train_ppo import (
    create_or_load_model,
    make_env,
    model_archive_path,
    resolve_rollout_and_batch_size,
    resume_custom_objects,
    write_experiment_manifest,
)


def _args(**overrides: object) -> Namespace:
    values = {
        "timesteps": 128,
        "seed": 0,
        "out": Path("checkpoints/test_ppo"),
        "init": None,
        "num_envs": 4,
        "n_steps": 64,
        "batch_size": None,
        "learning_rate": 1e-4,
        "device": "cpu",
        "verbose": 0,
        "hidden_sizes": "32,32",
    }
    values.update(overrides)
    return Namespace(**values)


def test_resume_custom_objects_include_cli_overrides() -> None:
    assert resume_custom_objects(learning_rate=1e-4, n_steps=256, batch_size=128) == {
        "learning_rate": 1e-4,
        "n_steps": 256,
        "batch_size": 128,
    }


def test_resume_model_load_receives_custom_objects() -> None:
    class FakeModel:
        verbose = 0

    class FakePPO:
        load_kwargs: dict[str, object] | None = None

        @classmethod
        def load(cls, path: Path, **kwargs: object) -> FakeModel:
            cls.load_kwargs = {"path": path, **kwargs}
            return FakeModel()

    args = _args(init=Path("checkpoints/init.zip"), n_steps=512, batch_size=2048)
    _, reset_num_timesteps = create_or_load_model(
        FakePPO,
        env=object(),
        args=args,
        batch_size=2048,
        policy_kwargs={"net_arch": [32, 32]},
    )

    assert reset_num_timesteps is False
    assert FakePPO.load_kwargs is not None
    assert FakePPO.load_kwargs["path"] == args.init
    assert FakePPO.load_kwargs["custom_objects"] == {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": 2048,
    }


def test_resolve_rollout_and_batch_size_clamps_to_rollout() -> None:
    assert resolve_rollout_and_batch_size(_args(num_envs=2, n_steps=8, batch_size=64)) == (16, 16)
    assert resolve_rollout_and_batch_size(_args(num_envs=2, n_steps=8, batch_size=None)) == (16, 16)


def test_make_env_supports_lobby_training_wrapper() -> None:
    env = make_env(
        123,
        env_kind="lobby",
        lobby_opponent_policy="tempo",
        players=4,
        max_actions_per_player=4,
        allow_oracle_macro_actions=False,
    )()
    try:
        obs, _info = env.reset(seed=123)
        assert obs.ndim == 1
        assert env.action_space.n > 0
        assert env.action_masks().shape == (env.action_space.n,)
        assert not env.allow_oracle_macro_actions
    finally:
        env.close()


def test_write_experiment_manifest_records_model_path_and_args(tmp_path: Path) -> None:
    output = tmp_path / "ppo_smoke"
    manifest_path = write_experiment_manifest(
        kind="ppo",
        output=output,
        args=_args(out=output),
        resolved={"batch_size": 16, "policy_kwargs": {"net_arch": [32, 32]}},
        elapsed_sec=1.25,
    )

    text = manifest_path.read_text(encoding="utf-8")
    assert manifest_path == tmp_path / "ppo_smoke.manifest.json"
    assert f'"model_path": "{model_archive_path(output)}"' in text
    assert '"kind": "ppo"' in text
    assert '"batch_size": 16' in text
