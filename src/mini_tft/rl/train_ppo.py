"""Train a first MaskablePPO policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import gymnasium as gym

from mini_tft.core.config import EnvConfig
from mini_tft.rl.gym_env import MiniTFTGymEnv
from mini_tft.rl.lobby_env import LOBBY_POLICY_BY_NAME, MiniTFTLobbyHeroEnv


def make_env(
    seed: int,
    *,
    env_kind: str = "single_player",
    lobby_opponent_policy: str = "tempo",
    players: int = 8,
    max_actions_per_player: int | None = None,
) -> Callable[[], gym.Env[Any, Any]]:
    def _factory() -> gym.Env[Any, Any]:
        config = EnvConfig(seed=seed)
        if env_kind == "single_player":
            return MiniTFTGymEnv(config=config)
        if env_kind == "lobby":
            return MiniTFTLobbyHeroEnv(
                config=config,
                player_count=players,
                opponent_policy=_lobby_policy_by_name(lobby_opponent_policy),
                max_actions_per_player=max_actions_per_player,
            )
        raise ValueError(f"unsupported env kind: {env_kind}")

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ppo_v0"))
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="Optional MaskablePPO checkpoint to continue from.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--env-kind", choices=["single_player", "lobby"], default="single_player")
    parser.add_argument(
        "--lobby-opponent-policy",
        choices=sorted(LOBBY_POLICY_BY_NAME),
        default="tempo",
    )
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=None)
    parser.add_argument(
        "--hidden-sizes",
        default="64,64",
        help="Comma-separated MLP hidden sizes used for new policies.",
    )
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    if args.num_envs < 1:
        raise SystemExit("--num-envs must be at least 1")

    env_fns = [
        make_env(
            args.seed + index,
            env_kind=args.env_kind,
            lobby_opponent_policy=args.lobby_opponent_policy,
            players=args.players,
            max_actions_per_player=args.max_actions_per_player,
        )
        for index in range(args.num_envs)
    ]
    if args.num_envs == 1:
        env: Any = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="fork")

    rollout_size, batch_size = resolve_rollout_and_batch_size(args)
    policy_kwargs = {"net_arch": parse_hidden_sizes(args.hidden_sizes)}

    started = time.perf_counter()
    try:
        model, reset_num_timesteps = create_or_load_model(
            MaskablePPO,
            env,
            args,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
        )
        print_resolved_config(args, rollout_size=rollout_size, batch_size=batch_size)
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=reset_num_timesteps)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
        manifest_path = write_experiment_manifest(
            kind="ppo",
            output=args.out,
            args=args,
            resolved={
                "rollout_size": rollout_size,
                "batch_size": batch_size,
                "policy_kwargs": policy_kwargs,
                "reset_num_timesteps": reset_num_timesteps,
            },
            elapsed_sec=time.perf_counter() - started,
        )
        print(f"manifest: `{manifest_path}`")
    finally:
        env.close()


def resolve_rollout_and_batch_size(args: argparse.Namespace) -> tuple[int, int]:
    rollout_size = args.n_steps * args.num_envs
    batch_size = args.batch_size if args.batch_size is not None else min(4096, rollout_size)
    return rollout_size, max(1, min(batch_size, rollout_size))


def create_or_load_model(
    maskable_ppo: Any,
    env: Any,
    args: argparse.Namespace,
    *,
    batch_size: int,
    policy_kwargs: dict[str, Any],
) -> tuple[Any, bool]:
    if args.init is None:
        model = maskable_ppo(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=batch_size,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
            seed=args.seed,
            device=args.device,
        )
        return model, True

    model = maskable_ppo.load(
        args.init,
        env=env,
        device=args.device,
        custom_objects=resume_custom_objects(
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=batch_size,
        ),
    )
    model.verbose = args.verbose
    return model, False


def resume_custom_objects(
    *,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
) -> dict[str, float | int]:
    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
    }


def print_resolved_config(args: argparse.Namespace, *, rollout_size: int, batch_size: int) -> None:
    mode = "new" if args.init is None else f"resume from {args.init}"
    print("# PPO Training")
    print(f"- mode: {mode}")
    print(f"- env_kind: {args.env_kind}")
    if args.env_kind == "lobby":
        print(f"- lobby_opponent_policy: {args.lobby_opponent_policy}")
        print(f"- players: {args.players}")
        print(f"- max_actions_per_player: {args.max_actions_per_player}")
    print(f"- learning_rate: {args.learning_rate}")
    print(f"- n_steps: {args.n_steps}")
    print(f"- num_envs: {args.num_envs}")
    print(f"- rollout_size: {rollout_size}")
    print(f"- batch_size: {batch_size}")
    print()


def write_experiment_manifest(
    *,
    kind: str,
    output: Path,
    args: argparse.Namespace,
    resolved: dict[str, Any],
    elapsed_sec: float,
) -> Path:
    archive_path = model_archive_path(output)
    manifest_path = archive_path.with_suffix(".manifest.json")
    manifest = {
        "kind": kind,
        "created_at_unix": time.time(),
        "elapsed_sec": elapsed_sec,
        "model_path": str(archive_path),
        "args": _json_ready(vars(args)),
        "resolved": _json_ready(resolved),
        "git": _git_metadata(),
        "packages": _package_versions(),
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_json, encoding="utf-8")
    return manifest_path


def model_archive_path(output: Path) -> Path:
    if output.suffix == ".zip":
        return output
    return Path(f"{output}.zip")


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--hidden-sizes must include at least one integer")
    if any(size <= 0 for size in sizes):
        raise ValueError("--hidden-sizes must contain positive integers")
    return sizes


def _lobby_policy_by_name(name: str):
    try:
        return LOBBY_POLICY_BY_NAME[name]
    except KeyError as exc:
        choices = ", ".join(sorted(LOBBY_POLICY_BY_NAME))
        raise ValueError(
            f"unknown lobby opponent policy {name!r}; choose one of: {choices}"
        ) from exc


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    return value


def _git_metadata() -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output("status", "--porcelain=v1")
    return {
        "commit": commit,
        "branch": branch,
        "status_porcelain": status,
        "dirty": bool(status),
    }


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    packages = ["tft-zero", "gymnasium", "numpy", "sb3-contrib", "stable-baselines3", "torch"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


if __name__ == "__main__":
    main()
