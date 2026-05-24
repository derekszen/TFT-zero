"""Train a first MaskablePPO policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_tft.core.config import EnvConfig
from mini_tft.rl.gym_env import MiniTFTGymEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/ppo_v0"))
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    env = MiniTFTGymEnv(config=EnvConfig(seed=args.seed))
    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=4096,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)


if __name__ == "__main__":
    main()
