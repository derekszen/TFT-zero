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
| 1 | Real round/stage/PvE structure | Implemented: canonical Set-1-like stage labels, PvE rounds, stage-aware web payloads, and PvE component drops. | Remaining: tune exact PvE rewards/drops if the learning loop needs closer Set-1 fidelity. | Easy | Done for V0 |
| 2 | 8-player lobby pressure | Implemented V0: `MiniTFTLobbyEnv`, synchronized players, shared pool, board-vs-board scalar fights, HP standings, placement assignment, clone/restore, and placement metrics. | Remaining: ghost-army handling for odd player counts, stronger opponent distributions, and calibration against known lobby curves. | Medium to Hard | V0 done |
| 3 | Better combat value model | Abstract scalar combat already includes roles, stars, items, traits, position multipliers, and assassin pressure. | Add stronger combat fixtures and calibration gates; tune symbolic combat from fixture outcomes before using it for RL claims. | Easy to Medium | Yes |
| 4 | Real board placement / candidate boards | Implemented: slot-level move actions, manual browser moves, and top-k candidate-board generation. `FIELD_BEST_BOARD` still exists as a debug convenience. | Remaining: expose candidate-board choices as explicit policy actions or wrappers for training. | Easy to Medium | Yes |
| 5 | Realistic item flow | Implemented: Set-1-like components, PvE component drops, deterministic component combine, best-target slam, and browser labels for combine/slam. | Remaining: explicit item-choice actions such as choose recipe, choose item, and choose target unit. | Medium | Yes |
| 6 | Opponent policy distribution | V0 lobby policy stepping supports fast-level, tempo, random, action plans, and checkpoint-as-hero evaluation. | Remaining: define a stable opponent-policy mixture and regression thresholds for placement/top-4, not only survival. | Medium | Yes |
| 7 | Calibration/regression gates | Implemented: full pytest/ruff/Pyright gates plus simulator gate coverage for throughput, determinism, round timing, item flow, combat fixtures, level pacing, opponent-pressure HP caps, candidate boards, and web UI payload behavior. | Remaining: add historical threshold tracking if CI starts collecting long-run metrics. | Easy | Done for V0 |
| 8 | Search snapshot boundary | Implemented: `GameState.clone()` plus `MiniTFTEnv.clone_state()` / `restore_state()` capture state and RNG for deterministic branch replay. | Remaining: add search-specific throughput benchmarks and MCTS budget gates. | Easy to Medium | Yes |

## Current Recurring Gates

Run these before simulator, planner, reward, search, or browser-playability changes:

```bash
uv run pytest
uv run ruff check
uv run --all-extras pyright
uv run python -m mini_tft.tools.web_ui_regression_gate --strict
uv run python -m mini_tft.tools.simulator_regression_gate --strict
uv run python -m mini_tft.tools.set1_lobby_step_smoke --strict
```

Use the web UI gate when changing browser payloads, enemy previews, item
affordances, or manual move behavior. Use the simulator gate when changing core
state transitions, combat, items, candidate boards, pacing, or throughput.

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

### 5. Simulator Regression Gate

Suggested branch:

```text
sim/simulator-regression-gates
```

Scope:

- Add one recurring simulator realism gate command.
- Report throughput, determinism, round timing, item timing, combat fixture pass
  rate, level pacing, opponent-pressure HP caps, and candidate-board quality.
- Emit compact Markdown by default and JSON for pipelines.

Acceptance:

```bash
uv run pytest -q tests/test_simulator_regression_gate.py
uv run python -m mini_tft.tools.simulator_regression_gate --strict
```

## Harder Work

These should not be assigned as one large branch.

### Lobby V1 Fidelity

Why it is hard:

- The V0 shared pool changes shop odds and roll EV, but it still uses simple
  pairings and scalar combat.
- Player action order can bias contested units, so placement comparisons should
  use random or rotating order.
- Odd-player lobbies need ghost-board handling instead of byes.
- Opponent policy distributions need fixed suites before placement can become a
  stable regression metric.

Split into:

- ghost-board fights for odd active counts
- opponent-policy mixture suites
- placement/top-4 regression thresholds
- native PufferEnv/C++ parity for the hot loop after Python behavior is stable

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
5. Simulator regression gate.
6. Opponent policy distribution.
7. Lobby placement/top-4 regression thresholds.
8. Native Puffer/C++ parity for lobby stepping.

This order keeps each branch testable and avoids turning the simulator into a
slow exact-combat project before the RL loop needs it.
