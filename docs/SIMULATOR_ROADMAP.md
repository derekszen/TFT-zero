# Simulator Roadmap

This document tracks the missing pieces for making the Set-1-like simulator
more useful for RL while keeping it fast enough to generate training data.

The target is not exact Riot TFT. The target is a fast executable environment
that produces strategically meaningful data: shops, economy, bench pressure,
items, board strength, opponent pressure, and final-board commitments.

## Difficulty Scale

| Level | Meaning |
| --- | --- |
| Easy | Can be implemented and merged as a focused worktree with local tests. |
| Medium | Still branch-sized, but it touches env state, observations, masks, and bots. |
| Hard | Needs design review or should be split into multiple branches before merge. |

## Current Gap List

| # | Gap | Current state | Useful next version | Difficulty | Branch-sized? |
| ---: | --- | --- | --- | --- | --- |
| 1 | Real round/stage/PvE structure | `MiniTFTEnv` has a simple round counter; the web UI derives a stage label from it. | Add canonical Set-1-like stage/round metadata, PvE rounds, round-specific gold/item drops, and stage-aware smoke metrics. | Easy to Medium | Yes |
| 2 | 8-player lobby pressure | Main env is single-player against an enemy curve. | Add a lobby shell with 8 player states, bot-controlled opponents, HP standings, pairings, and later a shared pool. | Medium to Hard | Yes, if split |
| 3 | Better combat value model | Abstract scalar combat already includes roles, stars, items, traits, position multipliers, and assassin pressure. | Add stronger combat fixtures and calibration gates; tune symbolic combat from fixture outcomes before using it for RL claims. | Easy to Medium | Yes |
| 4 | Real board placement / candidate boards | Move actions already exist; `FIELD_BEST_BOARD` still hides too much for policy learning. | Add a candidate-board generator/top-k board action path, and keep manual placement/debug UI working. | Easy to Medium | Yes |
| 5 | Realistic item flow | Completed items drop periodically; `SLAM_BEST_ITEM` attaches the first item to the best target. | Add Set-1-like components, PvE item drops, simple item combining, and explicit slam choices. | Medium | Yes |
| 6 | Opponent policy distribution | Heuristic bots exist, but the main env does not use them as live lobby opponents. | TODO: connect bot archetypes into lobby pressure and enemy-board sampling. | Medium | Yes, after #2 shell |
| 7 | Calibration/regression gates | `sim_smoke.py`, PPO evals, and planner gates exist; simulator realism gates are still thin. | TODO: add recurring gates for round timing, level timing, final-board archetype match, combat fixtures, item timing, and throughput. | Easy | Yes |

## Quick Worktree Tasks

These are bounded tasks that a coding agent can implement in a separate
worktree and merge quickly if the smoke tests pass.

### 1. Stage/Round/PvE Metadata

Suggested branch:

```text
sim/set1-stage-rounds-pve
```

Scope:

- Add a round metadata helper that maps env round to `stage`, `stage_round`,
  `stage_label`, and `round_type`.
- Mark PvE rounds separately from PvP rounds.
- Move item drops from a fixed modulo into a round schedule.
- Add the metadata to `info`, text render output, and smoke reports.

Acceptance:

```bash
uv run pytest -q tests/test_env.py tests/test_web_ui.py
uv run python sim_smoke.py --format json
```

### 2. Combat Fixture Gate

Suggested branch:

```text
sim/combat-fixture-gates
```

Scope:

- Add fixtures for frontline/backline, itemized carry, two-star upgrades,
  assassin pressure, trait breakpoints, and no-frontline penalties.
- Add a command that prints pass/fail plus scalar strength deltas.
- Do not tune reward functions in this branch.

Acceptance:

```bash
uv run pytest -q tests/test_combat*.py
uv run python -m mini_tft.tools.sim_smoke --format json
```

### 3. Candidate-Board Actions

Suggested branch:

```text
sim/candidate-board-actions
```

Scope:

- Keep existing slot-level move actions.
- Add a helper that generates top-k legal candidate boards from owned units.
- Add either a debug command or optional action wrapper that lets policies pick
  a candidate board instead of relying only on `FIELD_BEST_BOARD`.
- Add tests that candidate boards respect level caps and preserve owned units.

Acceptance:

```bash
uv run pytest -q tests/test_masks.py tests/test_env.py
uv run python sim_smoke.py
```

### 4. Item Components V0

Suggested branch:

```text
sim/set1-item-components-v0
```

Scope:

- Add component item IDs and a small recipe table.
- Make PvE drops produce components.
- Add explicit combine/slam behavior for completed items.
- Keep a compatibility mode for old completed-item-only data if needed.

Acceptance:

```bash
uv run pytest -q tests/test_items.py tests/test_env.py
uv run python sim_smoke.py
```

### 5. Lobby Shell V0

Suggested branch:

```text
sim/set1-lobby-shell-v0
```

Scope:

- Add `Set1LobbyState` with 8 player states.
- Step one planning/combat round for all players using supplied policies.
- Track player HP, alive status, pairings, and standings.
- Start without shared pool if needed; add shared pool in a separate branch.

Acceptance:

```bash
uv run pytest -q tests/test_lobby*.py
uv run python -m mini_tft.tools.benchmark_env
```

## Harder Work

These should not be assigned as one large branch.

### Full 8-Player Lobby With Shared Pool

Why it is hard:

- Shared pool changes shop odds and roll EV.
- Player action order can bias contested units.
- Opponent pairings and simultaneous rounds need deterministic clone/restore
  behavior for search and reproducibility.

Split into:

- lobby state and round stepping
- shared pool buy/sell/upgrade accounting
- rotating/randomized action order
- Gymnasium/PettingZoo-style wrapper
- throughput and determinism gates

### Accurate Combat Engine

Why it is hard:

- Targeting, pathing, attack speed, mana, casts, CC, assassin jumps, Zephyr-like
  effects, and item interactions are a large engine project.
- It is likely slower than the rest of the env and can dominate RL data cost.

Preferred approach:

- Keep fast symbolic combat for RL.
- Use fixture-calibrated combat gates.
- Use detailed combat only as a teacher/label generator if needed.

### Current-Patch Full Environment

Why it is hard:

- Current TFT requires augments, portals/encounters depending on set, modern item
  systems, patch-specific trait logic, and current unit namespaces.
- MetaTFT aggregate data is not turn-by-turn action data.

Preferred approach:

- Keep the current-patch path as a value/planner track until the Set-1-like
  symbolic simulator is more reliable.
- Only build a current-patch `reset/step` env after the state/action schemas and
  value gates are stable.

## Recommended Near-Term Order

1. Stage/round/PvE metadata.
2. Combat fixture gate.
3. Item components V0.
4. Candidate-board actions.
5. Lobby shell V0.
6. Opponent policy distribution.
7. Shared unit pool.

This order keeps each branch testable and avoids turning the simulator into a
slow exact-combat project before the RL loop needs it.
