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
| 8 | Search snapshot boundary | Implemented: `GameState.clone()` plus `MiniTFTEnv.clone_state()` / `restore_state()` capture state and RNG for deterministic branch replay. Stage 2 MCTS, Stage 3 distillation, and Stage 4 dynamics-search gates have passed. | Remaining: Stage 5 self-play/search-target feedback loop and larger eval gates. | Medium | Yes |

## Tracked RL Failure: PPO Policy Degradation

Historical note: long blind PPO continuation regressed badly on 2026-06-25 even
with zero illegal actions. Treat
`checkpoints/ppo_lobby_frozenpool_probe10k2_lr1e5_20260625T1052_h256.zip` as a
reference baseline, not as a checkpoint to keep extending by default.

Current action: use Stage 5 search-target feedback from
`docs/MUZERO_STAGE_GATE.md`. Do not run long PPO continuation as a substitute.

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

## Open Simulator Gaps

These are still relevant after the Stage 1-4 search scaffold:

1. Ghost-board handling for odd active-player counts.
2. Stable opponent-mixture suites and historical placement/top-4 thresholds.
3. Explicit item-choice actions: choose recipe, choose item, choose target.
4. Candidate-board policy actions that reduce dependence on `FIELD_BEST_BOARD`.
5. Native Puffer/C++ parity only after Python Stage 5 behavior is stable.
6. More combat calibration fixtures before using simulator results for stronger
   skill claims.

Do not turn these into one large branch. Each should have focused tests and a
clear before/after gate.

## Deferred 10x Throughput Track

The order-of-magnitude speed path is a batched compiled lobby core, but it
should stay behind Python simulator quality work until the game loop is worth
porting. The Python simulator remains the reference implementation and the web
UI should keep using it until compiled parity is proven.

Target throughput gate:

- Current matched lobby PPO benchmark: about `950-966` steps/sec after Python
  hot-path cleanup.
- 10x target: `>=9,500` steps/sec on the same benchmark settings.
- Minimum useful compiled-core milestone: `>=3x` over the current Python
  backend with exact legal-mask, reward, terminal, placement, HP, and
  observation parity.

### Python Logic Prerequisites

Patch these before porting more logic into C++/Numba:

1. Replace oracle macro dependence in training:
   - keep `FIELD_BEST_BOARD` and `SLAM_BEST_ITEM` available for debug/UI gates;
   - add primitive or candidate-board training modes that do not require
     best-board oracle masking every decision;
   - benchmark PPO with oracle macros disabled and report placement/top-4, not
     only throughput.
   - follow [Non-Oracle Action Surface Runbook](NON_ORACLE_ACTION_SURFACE_RUNBOOK.md)
     before treating oracle-disabled training as promotion evidence.
2. Stabilize board-placement actions:
   - expose candidate-board choices as explicit policy actions or wrappers;
   - add parity tests that candidate-board actions preserve owned-unit multisets,
     level caps, item ownership, and bench capacity;
   - make browser/manual play still use the same move/candidate semantics.
3. Improve item-choice realism:
   - split `SLAM_BEST_ITEM` into choose recipe, choose item/component, and
     choose target where practical;
   - add tests for component ownership, completed-item recipes, target capacity,
     and deterministic PvE drops.
4. Calibrate lobby difficulty:
   - define fixed opponent suites for tempo, fast-level, mixed, and frozen PPO;
   - track mean placement, top-1/top-4, HP, board strength, action mix, and
     illegal actions under fixed seed ranges;
   - do not use survival-only claims for promotion.
5. Repair search/evaluator gates first:
   - keep outcome metrics episode-level;
   - rerun Stage 4/Stage 5 gates before claiming MuZero improvement;
   - reject candidates that only copy the prior or rely entirely on guard/macro
     shortcuts.

### Compiled Core Patch Surfaces

Once the Python rules above are stable, port in layers:

1. `board_strength_batch`
   - Inputs: board unit ids, stars, item ids, slot indexes, trait tables, item
     effect tables.
   - Output: strength scores and optional trait/effect summaries.
   - Reference: `mini_tft.core.combat.board_strength`.
   - Required tests: exact parity on fixed board fixtures, item fixtures, trait
     breakpoint fixtures, empty boards, duplicate units, and max-item units.
2. `best_board_batch`
   - Inputs: flattened board+bench owned units, level, board size, bench size.
   - Output: best board layout, bench layout, and `would_change_best_board`.
   - Reference: `mini_tft.core.board.select_best_board`.
   - Required tests: clone isolation, owned-unit multiset preservation, level
     caps, stable tie-breaking, and current `FIELD_BEST_BOARD` mask parity.
3. `legal_mask_batch`
   - Inputs: player state arrays, shop ids, gold, level, XP, item bench, shared
     pool counts, config constants.
   - Output: `NUM_ACTIONS` boolean masks for each player/env.
   - Reference: `mini_tft.core.masks.legal_action_mask` plus lobby shared-pool
     buy filtering.
   - Required tests: exact action-id parity for roll, XP, buy, sell, bench-board
     moves, board-bench moves, item actions, oracle macro actions, done players,
     and pool-exhausted shop slots.
4. `apply_action_batch`
   - Inputs: state arrays, legal actions, shared pool, RNG streams.
   - Output: mutated state arrays plus action-applied flags and counters.
   - Reference: `mini_tft.core.lobby_step.apply_lobby_action`.
   - Required tests: buy/sell pool round trips, roll/shop determinism,
     XP/leveling, board moves, bench moves, item slams, illegal-action rejection
     through the Python safe wrapper, and action counter behavior.
5. `combat_round_batch`
   - Inputs: active player boards, matchups, RNG streams, HP/placement state.
   - Output: combat results, damage, HP, placements, final reasons.
   - Reference: `mini_tft.core.lobby.resolve_lobby_round`.
   - Required tests: matchup determinism, odd-player policy/ghost handling once
     implemented, damage margin, placement ordering, top-1/top-4 summaries, and
     terminal/max-round behavior.
6. `observe_batch`
   - Inputs: state arrays and config/data tables.
   - Output: preallocated observation arrays and optional action-mask tails for
     Puffer.
   - Reference: `mini_tft.core.featurize.featurize_state` and
     `mini_tft.rl.puffer_env` flattening.
   - Required tests: shape/dtype parity, mask-tail placement, clipping bounds,
     reset/auto-reset behavior, and checkpoint policy compatibility.
7. `step_lobby_batch`
   - Inputs: batched lobby states, hero actions, opponent policy hooks or
     precomputed opponent actions, RNG streams.
   - Output: observations, rewards, dones, truncations, masks, summaries.
   - Reference: `MiniTFTLobbyHeroEnv.step` and
     `step_lobby_round_with_policies`.
   - Required tests: fixed-seed rollout parity over full episodes, PPO smoke,
     benchmark artifacts, and branch/restore or snapshot parity for MCTS.

### Integration Rules

- Keep the Python env as the authoritative UI/debug path.
- Add compiled core behind an explicit backend flag such as
  `--sim-backend python|compiled`.
- Never remove Python parity tests when adding compiled code.
- Do not require the web UI to use the compiled backend; the UI can adopt it
  later through an adapter after parity is stable.
- Treat RNG mismatches as blockers unless the benchmark explicitly declares a
  distributional, not exact, parity mode.
