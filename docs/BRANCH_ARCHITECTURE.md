# Branch Architecture

This repo should stay split by research question. The branches do not need to
be mutually exclusive forever, but each branch should have one MVP owner.

## Branch Map

| Branch | Role | Scope | Not Scope |
| --- | --- | --- | --- |
| `main` | integration | Stable docs and code after an MVP branch is validated | active experiments |
| `mvp/set1-mini-env` | toy env MVP | Fast Set-1-like single-player simulator, abstract combat, action masks, bots as executable baselines | PPO training reports, MetaTFT current-patch value modeling |
| `mvp/set1-ppo-baseline` | toy RL MVP | BC/PPO pipeline on top of the Set-1-like simulator, fixed-seed heuristic comparisons, training reports | real-TFT skill claims, current-patch board value |
| `mvp/current-metatft-value-model` | current-patch value/planner MVP | MetaTFT rich catalog, current-patch encoders, value model, target-guided planner, batch top-comp match gates | full turn-by-turn simulator until shop/econ/augments/combat are owned by a `reset/step` env |

Existing `exp/*` and `combat/*` branches are research worktrees. Treat them as
candidate sources for cherry-picks, not as the canonical MVP branches.

## Parallel Development Setup

Use Git worktrees, not submodules.

Submodules are a poor fit here because the MVPs still share one Python package,
one lockfile, one test suite, and many schemas. Splitting them into nested repos
would make cross-branch changes harder to test and merge. Worktrees give each
MVP its own checkout while keeping one repo history.

Current canonical worktrees:

```text
/mnt/ssd2/Projects/TFT-zero
  branch: mvp/current-metatft-value-model

/mnt/ssd2/Projects/TFT-zero-mvp-set1-mini-env
  branch: mvp/set1-mini-env

/mnt/ssd2/Projects/TFT-zero-mvp-set1-ppo-baseline
  branch: mvp/set1-ppo-baseline
```

Useful commands:

```bash
git worktree list

cd /mnt/ssd2/Projects/TFT-zero-mvp-set1-mini-env
uv sync
uv run python sim_smoke.py

cd /mnt/ssd2/Projects/TFT-zero-mvp-set1-ppo-baseline
uv sync --extra train
uv run python -m mini_tft.rl.evaluate_policy --episodes 100

cd /mnt/ssd2/Projects/TFT-zero
uv sync --extra train --extra fight
uv run pytest -q tests/test_metatft_catalog_encoders.py
uv run python -m mini_tft.tools.evaluate_current_patch_planner \
  --catalog data/metatft/current_rich_catalog_2026-05-31.json \
  --checkpoint checkpoints/fight_value/current_patch_board_value_2026-05-31.pt \
  --device cpu \
  --trace-mode shop-planning \
  --comp-limit 8 \
  --demo-levels 8,9 \
  --match-levels 8,9 \
  --top-k 10 \
  --min-recall 0.75 \
  --max-actions 8 \
  --require-exact-match-rate 8:1.0 \
  --require-exact-match-rate 9:1.0
```

Only use submodules if a component becomes a genuinely reusable external
library with its own release cycle, independent tests, and no need to import the
main package internals. None of the current MVPs meet that bar.

## MVP Definitions

### Set-1 Mini Env

Purpose:

- provide a fast executable RL environment
- test whether agents can learn TFT-like shop/econ/upgrade behavior
- keep combat abstract but deterministic under seed

Core validation:

```bash
uv run python sim_smoke.py
uv run pytest -q
uv run python -m mini_tft.rl.evaluate_policy --episodes 100
```

Expected reports:

- simulator throughput
- deterministic same-seed/action-sequence test
- heuristic bot baseline table

### Set-1 PPO Baseline

Purpose:

- establish BC and PPO baselines against scripted policies
- keep FastLevelBot, GreedyBoardBot, and other bots as baselines, not separate
  simulator tracks
- measure learning inside the toy simulator

Core validation:

```bash
uv run python -m mini_tft.tools.generate_bot_dataset \
  --suite fastlevel \
  --episodes 5000 \
  --workers 0 \
  --out rollouts/fastlevel_bc_5k.npz

uv run python -m mini_tft.rl.pretrain_bc \
  --dataset rollouts/fastlevel_bc_5k.npz \
  --epochs 80 \
  --hidden-sizes 256,256 \
  --out checkpoints/bc_fastlevel_5k_e80_h256

uv run python -m mini_tft.rl.train_ppo \
  --init checkpoints/bc_fastlevel_5k_e80_h256.zip \
  --timesteps 250000 \
  --num-envs 8 \
  --out checkpoints/ppo_from_bc_fastlevel_250k_h256

uv run python -m mini_tft.rl.evaluate_policy \
  --episodes 500 \
  --checkpoint checkpoints/ppo_from_bc_fastlevel_250k_h256.zip
```

Current strongest toy-sim checkpoint:

```text
checkpoints/ppo_from_bc_fastlevel_5m_h256.zip
```

Latest 500-seed result:

| Policy | Mean final HP | Survival rate | Mean final strength |
| --- | ---: | ---: | ---: |
| FastLevelBot | 68.812 | 0.986 | 300.75 |
| PPO from BC 5M/h256 | 77.100 | 0.984 | 336.66 |

Interpretation:

- PPO builds stronger boards than FastLevelBot under the toy scalar combat
  model.
- Survival is effectively tied on this horizon.
- This is not a real-TFT player-rank benchmark.

### Current MetaTFT Value Model

Purpose:

- learn current-patch board/comp value from MetaTFT aggregate data
- score final boards and candidate shop/bench/board transitions
- become the board-value component for a future current-patch symbolic or
  learned world model

Core validation:

```bash
uv run python -m mini_tft.tools.fetch_metatft_comp_strength \
  --rich \
  --comp-detail-limit 999 \
  --out data/metatft/current_rich_catalog.json \
  --min-count 3000

uv run python -m mini_tft.rl.train_current_patch_value_model \
  --catalog data/metatft/current_rich_catalog.json \
  --device cuda \
  --blind-target-metadata \
  --epochs 100 \
  --embed-dim 16 \
  --hidden-dim 32 \
  --out checkpoints/fight_value/current_patch_board_value.pt
```

Required before RL use:

- heldout comp-ranking benchmark against MetaTFT rankings
- planner trace reports that show level 8/9 final boards matching top comps
- explicit state/action schema for shop, econ, bench, board, augments, and items
- a `reset/step` env wrapper if it is going to train with RL

Current checkpoint quality is moderate, not production-grade:

```text
heldout pairwise accuracy: ~0.643
heldout Spearman:          ~0.407
heldout top-k overlap:     ~0.50
```

Current planner trace gate on the 2026-05-31 rich snapshot:

```text
level 8 exact_match_rate: 1.0
level 9 exact_match_rate: 1.0
hard shop-planning action mix over 16 traces:
  64 buy_to_board, 16 roll, 16 end_turn
```

Interpretation:

- The planner can complete target top-comp boards in the hard gate when a fixed
  trace exposes missing target units across two shops.
- This is a regression gate for symbolic board completion, not proof that an RL
  agent has learned scouting, rolling, augments, items, or combat.

## Auto-Research Harness Target

The next cleanup step is a harness that runs experiments uniformly across these
branches and emits comparable artifacts:

```text
config.yaml
metrics.json
report.md
trace samples
checkpoint path
data snapshot hash/path
git commit and branch
```

The harness should answer:

- Which environment or value source produced this result?
- Which fixed seeds or heldout comps were used?
- Which baseline was beaten?
- Is this a toy-simulator claim, a MetaTFT ranking claim, or a fight-teacher
  calibration claim?

This matters because the project now has three different notions of strength:

- toy Set-1 scalar board strength
- learned current-patch MetaTFT board value
- teacher/detailed-fight combat labels

Reports must label which one they use.
