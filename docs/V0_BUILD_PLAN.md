# V0 Build Plan

v0 is the smallest useful simulator that can generate RL transitions.

## Phase 1: Repo And Env Skeleton

- `uv` project metadata
- local `.venv` workflow
- package under `src/mini_tft`
- JSON data pack
- Gymnasium-style flat-observation `MiniTFTEnv`
- legal bool action mask
- smoke tests

Done when:

- `uv sync` succeeds
- `uv run pytest` succeeds
- `uv run python sim_smoke.py` succeeds
- a reset and one legal step work

## Phase 2: Playable Abstract Simulator

- shop rolling from cost odds
- buying units into bench
- selling units
- XP buys and leveling
- interest and income
- auto-combine upgrades
- field best board
- slam abstract items
- abstract combat against enemy curve
- reward shaping
- deterministic seeding

Done when:

- fixed-seed rollouts are reproducible
- illegal actions are masked and penalized
- basic economy invariants have tests

## Phase 3: Bots And Debugging

- text renderer
- `RandomBot`
- `GreedyBoardBot`
- `EconBot`
- `RerollBot`
- `FastLevelBot`
- `TraitCommitBot`

Done when:

- every bot runs fixed-seed episodes without illegal actions
- episode summaries expose final HP, survived round, strength, and action counts

## Phase 4: Dataset Generation

- compressed `.npz` transition writer
- dataset loader and shape validator

Done when:

- a command can generate `rollouts/bot_dataset_v0.npz`
- stored transitions reload cleanly
- masks, rewards, and episode returns are included

## Phase 5: Baseline Learning

- behavioral cloning from bot data
- MaskablePPO with `sb3-contrib`
- evaluation harness against scripted bots
- metrics for survival round, final HP, economy, and traits

Done when:

- PPO learns above random on the simplified environment
- training logs are reproducible enough to compare changes
