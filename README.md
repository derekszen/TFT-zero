# TFT-zero

Python-first research prototype for learning TFT-like macro play.

The project is not trying to clone live TFT first. It is trying to build a fast,
inspectable simulator where shop, economy, bench, board, item, and combat-value
decisions can be trained and audited before moving toward current-patch planning.

## 1. Status

### Tracks

| Track | What exists | What it proves |
| --- | --- | --- |
| Toy Set-1-like simulator | `MiniTFTEnv`, masks, bots, datasets, BC/PPO | RL can learn in the fast abstract simulator. |
| Toy Set-1-like lobby | `MiniTFTLobbyEnv`, shared pool, policy stepping, placement metrics | Survival can be separated from actual lobby placement. |
| Current-patch MetaTFT value/planner | rich catalog ingest, encoders, value model, target-guided planner | Aggregate data can score and assemble symbolic current-patch boards. |
| Fight-value teacher path | Set 4 teacher labels and `FightValueNet` experiments | Detailed combat can label faster value models, but is not current-patch validated. |

### Current Claims

- The toy env is a complete executable `reset/step` RL environment.
- The toy lobby can run 8 synchronized players with shared-pool shops,
  board-vs-board scalar combat, placement assignment, and top-1/top-4 metrics.
- The toy 5M PPO-from-BC checkpoint beats `FastLevelBot` on mean HP and scalar
  board strength under the toy combat model.
- The current-patch planner gates prove target-board assembly under controlled
  traces, not organic full-game RL.
- The simulator now has deterministic clone/restore state snapshots for future
  simulator-backed search.
- CI-quality local gates are `pytest`, `ruff`, and full-repo Pyright.

### Not Proven

- No real TFT player-rank claim.
- No complete current-patch `reset/step` environment yet.
- No current-patch combat oracle yet.
- No proof that MetaTFT aggregates alone are enough for RL rewards.
- No claim that toy-lobby placement is calibrated to real TFT lobby outcomes yet.

## 2. Setup

```bash
uv sync
source .venv/bin/activate
uv run pytest
uv run ruff check
uv run --all-extras pyright
```

Optional extras:

```bash
uv sync --extra train
uv sync --extra assets
uv sync --extra fight
```

## 3. Common Workflows

### Inspect The Simulator

```bash
uv run python -m mini_tft.tools.inspect_episode --seed 0
uv run python -m mini_tft.tools.inspect_episode --mode interactive --seed 0
uv run python sim_smoke.py --format json > /tmp/tft_zero_sim_smoke.json
```

### Generate Bot Data

```bash
uv run python -m mini_tft.tools.generate_bot_dataset \
  --suite fastlevel \
  --episodes 5000 \
  --workers 0 \
  --out rollouts/fastlevel_bc_5k.npz
```

### Pretrain And Continue PPO

```bash
uv run python -m mini_tft.rl.pretrain_bc \
  --dataset rollouts/fastlevel_bc_5k.npz \
  --epochs 80 \
  --hidden-sizes 256,256 \
  --out checkpoints/bc_fastlevel_5k_e80_h256

uv run python -m mini_tft.rl.train_ppo \
  --init checkpoints/bc_fastlevel_5k_e80_h256.zip \
  --timesteps 250000 \
  --num-envs 8 \
  --n-steps 256 \
  --batch-size 2048 \
  --device cpu \
  --out checkpoints/ppo_from_bc_fastlevel_250k_h256
```

Training commands write a sidecar `*.manifest.json` next to each checkpoint with
CLI args, resolved training settings, git metadata, package versions, and elapsed
time. PPO resume passes `learning_rate`, `n_steps`, and `batch_size` through
`MaskablePPO.load(custom_objects=...)`.

### Evaluate

```bash
uv run python -m mini_tft.rl.evaluate_policy \
  --episodes 100 \
  --checkpoint checkpoints/ppo_from_bc_fastlevel_250k_h256.zip
```

### Evaluate Lobby Placement

```bash
uv run python -m mini_tft.tools.evaluate_lobby_policy \
  --episodes 100 \
  --hero-policy fast_level \
  --opponent-policy tempo \
  --format markdown

uv run python -m mini_tft.tools.set1_lobby_step_smoke \
  --lobbies 100 \
  --max-rounds 8 \
  --format json
```

Lobby evaluation reports mean placement, top-1/top-4 rate, final HP, survived
round, board strength, placement histogram, actions, and fights. Use this when
survival-rate claims are too coarse.

## 4. Regression Gates

### Always Run Before Simulator, Reward, Search, Or Planner Changes

```bash
uv run pytest
uv run ruff check
uv run --all-extras pyright
uv run python -m mini_tft.tools.set1_lobby_step_smoke --strict
```

### Simulator And UI Gates

```bash
uv run python -m mini_tft.tools.simulator_regression_gate --strict
uv run python -m mini_tft.tools.web_ui_regression_gate --strict
```

### Current-Patch Planner Gates

```bash
uv run python -m mini_tft.tools.run_current_patch_planner_gates \
  --catalog data/metatft/current_rich_catalog_2026-05-31.json \
  --checkpoint checkpoints/fight_value/current_patch_board_value_2026-05-31.pt \
  --device cpu \
  --suite minimum
```

Use `--suite strict` when changing target completion pressure, action pacing, or
search behavior.

## 5. Baseline Result

Toy Set-1-like simulator eval over seeds `1000..1499`:

| Policy | Mean final HP | Survival rate | Mean final strength |
| --- | ---: | ---: | ---: |
| FastLevelBot | 68.812 | 0.986 | 300.75 |
| PPO from BC 5M/h256 | 77.100 | 0.984 | 336.66 |

This is a toy-simulator result. The board-strength number comes from the
handcrafted abstract combat model, not MetaTFT or real player data.

## 6. Documentation Map

### Start Here

- [Project Brief](docs/PROJECT_BRIEF.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)

### Simulator And RL

- [Simulator Roadmap](docs/SIMULATOR_ROADMAP.md)
- [Data Generation](docs/DATA_GENERATION.md)
- [V0 Build Plan](docs/V0_BUILD_PLAN.md)
- [Autoresearch Scaffold](docs/AUTORESEARCH.md)

### Current Patch And Combat Value

- [Current-Patch MetaTFT](docs/CURRENT_PATCH_METATFT.md)
- [Fight Value Model](docs/FIGHT_VALUE_MODEL.md)
- [Branch Architecture](docs/BRANCH_ARCHITECTURE.md)

### Local Assets

- [Assets](docs/ASSETS.md)

## 7. Development Rules

- Keep game content data-driven.
- Keep rendering/debug UI separate from RL state.
- Keep state explicit, serializable, and cloneable.
- Prefer clear Python first; port hot paths only after profiling.
- Add or run the smallest useful smoke test for every behavior change.
