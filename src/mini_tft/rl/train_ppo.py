"""Train a first MaskablePPO policy."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mini_tft.core.config import EnvConfig
from mini_tft.rl.gym_env import MiniTFTGymEnv


def make_env(seed: int) -> Callable[[], MiniTFTGymEnv]:
    def _factory() -> MiniTFTGymEnv:
        return MiniTFTGymEnv(config=EnvConfig(seed=seed))

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

    env_fns = [make_env(args.seed + index) for index in range(args.num_envs)]
    if args.num_envs == 1:
        env: Any = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="fork")

    rollout_size = args.n_steps * args.num_envs
    batch_size = args.batch_size if args.batch_size is not None else min(4096, rollout_size)
    batch_size = max(1, min(batch_size, rollout_size))
    policy_kwargs = {"net_arch": parse_hidden_sizes(args.hidden_sizes)}

    try:
        if args.init is None:
            model = MaskablePPO(
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
            reset_num_timesteps = True
        else:
            model = MaskablePPO.load(args.init, env=env, device=args.device)
            model.verbose = args.verbose
            reset_num_timesteps = False
        model.learn(total_timesteps=args.timesteps, reset_num_timesteps=reset_num_timesteps)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
    finally:
        env.close()


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--hidden-sizes must include at least one integer")
    if any(size <= 0 for size in sizes):
        raise ValueError("--hidden-sizes must contain positive integers")
    return sizes


if __name__ == "__main__":
    main()
