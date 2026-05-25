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

## Development Rules

- Keep game content data-driven.
- Keep rendering/debug UI separate from RL state.
- Prefer clear Python first; port hot paths only after profiling.
- Add the smallest useful smoke test for every behavior change.
