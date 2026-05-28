# TFT-zero

Python-first prototype for a small TFT-like reinforcement-learning environment.

The project starts with a data-driven simulator, Gymnasium-compatible env, simple
heuristic bots, logged rollout data, PPO baselines, and later world-model or
MuZero-style experiments. Runtime speed is not the first bottleneck here;
iteration speed and clear simulator boundaries are.

## Setup

This repo uses `uv` and a local `.venv`.

```bash
uv sync
source .venv/bin/activate
uv run pytest
```

Useful variants:

```bash
uv sync --extra train
uv sync --extra assets
uv run ruff check
uv run python sim_smoke.py
uv run python -m mini_tft.tools.tune_parallel_rollout --episodes 1000
uv run python -m mini_tft.tools.inspect_episode --seed 0
uv run python -m mini_tft.tools.inspect_episode --mode interactive --seed 0
uv run python -m mini_tft.tools.generate_bot_dataset --episodes 100 --out rollouts/bot_dataset_v0.npz
uv run python -m mini_tft.tools.generate_bot_dataset --episodes 1000 --workers 0 --out rollouts/bot_dataset_parallel_v0.npz
uv run python -m mini_tft.tools.generate_bot_dataset --suite fastlevel --episodes 5000 --workers 0 --out rollouts/fastlevel_bc_5k.npz
uv run python -m mini_tft.rl.pretrain_bc --dataset rollouts/fastlevel_bc_5k.npz --epochs 80 --hidden-sizes 256,256 --out checkpoints/bc_fastlevel_5k_e80_h256
uv run python -m mini_tft.rl.train_ppo --init checkpoints/bc_fastlevel_5k_e80_h256.zip --timesteps 250000 --num-envs 8 --device cpu --out checkpoints/ppo_from_bc_fastlevel_250k_h256
uv run python -m mini_tft.rl.evaluate_policy --episodes 100 --checkpoint checkpoints/ppo_from_bc_fastlevel_250k_h256.zip
uv run python -m mini_tft.tools.scrape_assets
```

Pipeline-friendly smoke output:

```bash
uv run python sim_smoke.py --format json > /tmp/tft_zero_sim_smoke.json
```

The initial sync installs only the simulator/runtime dependencies plus the
default dev group. Training and asset-scraping dependencies are opt-in so the
base environment stays light.

## Current Scope

v0 is intentionally small but playable:

- simulator internals under `src/mini_tft/core/`
- heuristic bots under `src/mini_tft/bots/`
- RL and dataset helpers under `src/mini_tft/rl/`
- CLI/debug tools under `src/mini_tft/tools/`
- data-driven 24-unit Set 1-labeled pack under `src/mini_tft/data/`
- Gymnasium-style flat observation environment with bool action masks
- smoke tests under `tests/`
- markdown specs under `docs/`

The env is still not exact TFT. Combat, traits, and items are abstract power
models designed to support fast RL iteration.

## Documentation

- [Project Brief](docs/PROJECT_BRIEF.md)
- [Architecture](docs/ARCHITECTURE.md)
- [V0 Build Plan](docs/V0_BUILD_PLAN.md)
- [Data Generation](docs/DATA_GENERATION.md)
- [Training Plan](docs/TRAINING.md)
- [Assets](docs/ASSETS.md)

## Current Baseline

Fixed-seed eval over seeds `1000..1099` now has a warm-started PPO policy above
the strongest heuristic:

| Policy | Mean final HP | Survival rate | Mean final strength |
| --- | ---: | ---: | ---: |
| FastLevelBot | 69.05 | 0.99 | 304.52 |
| BC FastLevel 5k/e80/h256 | 66.30 | 1.00 | 295.88 |
| PPO from BC 250k/h256 | 70.87 | 1.00 | 312.49 |

## Development Rules

- Keep game content data-driven.
- Keep rendering/debug UI separate from RL state.
- Prefer clear Python first; port hot paths only after profiling.
- Add the smallest useful smoke test for every behavior change.
