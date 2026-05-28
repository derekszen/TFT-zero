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
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
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

    try:
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
            verbose=args.verbose,
            seed=args.seed,
            device=args.device,
        )
        model.learn(total_timesteps=args.timesteps)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
