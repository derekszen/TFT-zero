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

## Current Architecture

The repo has three tracks. Keep their claims separate:

| Track | What exists | What it proves |
| --- | --- | --- |
| Set-1-like toy simulator | `MiniTFTEnv`, bots, dataset generation, BC/PPO training | RL can learn inside the fast abstract simulator. |
| Current-patch MetaTFT value/planner | rich MetaTFT ingest, current-patch encoder/value model, target-guided shop/econ planner | Aggregate data can score and complete symbolic current-patch boards. |
| Fight-value teacher path | Set 4 teacher labels and `FightValueNet` experiments | Detailed combat can generate labels for a faster learned evaluator, but this is not current-patch validated. |

The toy env is a complete executable `reset/step` RL environment, but it is not
exact TFT. The current-patch MetaTFT path uses real aggregate data, but it is not
yet a full turn-by-turn RL environment because it does not own shops, augments,
items, combat, and episode transitions end to end.

## Documentation

- [Project Brief](docs/PROJECT_BRIEF.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Branch Architecture](docs/BRANCH_ARCHITECTURE.md)
- [Current-Patch MetaTFT](docs/CURRENT_PATCH_METATFT.md)
- [Fight Value Model](docs/FIGHT_VALUE_MODEL.md)
- [V0 Build Plan](docs/V0_BUILD_PLAN.md)
- [Data Generation](docs/DATA_GENERATION.md)
- [Training Plan](docs/TRAINING.md)
- [Assets](docs/ASSETS.md)

## Current Baseline

Toy Set-1-like simulator eval over seeds `1000..1499`:

| Policy | Mean final HP | Survival rate | Mean final strength |
| --- | ---: | ---: | ---: |
| FastLevelBot | 68.812 | 0.986 | 300.75 |
| PPO from BC 5M/h256 | 77.100 | 0.984 | 336.66 |

This is a toy-simulator result. The board-strength number comes from the
handcrafted abstract combat model, not from MetaTFT or real player data.

Current-patch planner gate on the 2026-05-31 rich MetaTFT snapshot:

```text
level 8 exact top-comp board match: 1.0
level 9 exact top-comp board match: 1.0
```

That gate validates target-board completion for fixed planner traces. It does
not prove organic RL policy learning.

## Development Rules

- Keep game content data-driven.
- Keep rendering/debug UI separate from RL state.
- Prefer clear Python first; port hot paths only after profiling.
- Add the smallest useful smoke test for every behavior change.
