# Architecture

The repo has two layers:

1. Existing MiniTFT code that proves the prototype can run, train, evaluate,
   and render a toy TFT-shaped environment.
2. A new strategic lane that should become the small, repeatable proof surface
   for MuZero cache data, Puffer speed, and playable inspection.

## Active Shape

```text
src/mini_tft/strategic/
  core/
    state.py     compact shop/econ/board/HP/round state
    actions.py   small strategic action set and legal-mask contract
    rules.py     economy, shop, board strength, round-based enemy pressure
    obs.py       observation vector and metadata

  adapters/
    puffer        batched rollout surface and benchmark harness
    muzero_cache  obs/action/reward/next_obs/done/policy/value rows
    web_demo      playable surface over the same rules
    baselines     random, econ, tempo, roll-down, and greedy policies
```

The exact module names may change during implementation, but the boundary
should not: rules live once, adapters only translate.

`src/mini_tft/core/` is the existing toy engine. Use it as a reference or bridge
only; new strategic-lane logic belongs under `src/mini_tft/strategic/`.

## Current Code Map

```text
src/mini_tft/
  strategic/    active clean lane: core rules plus adapters
  core/         existing toy simulator, lobby state, rules, masks, combat
  bots/         scripted baselines
  rl/           Gym/Puffer wrappers, BC/PPO, checkpoint loading, evaluation
  search/       historical MCTS/distillation/dynamics scaffolds
  metatft/      historical current-patch catalog/value/planner path
  fight_model/  historical combat-teacher experiments
  tools/        gates, reports, smoke commands, benchmarks
  web/          local playable/debug UI
  data/         packaged toy set JSON
```

## Main Data Flows

### Existing Toy Env

```text
JSON set data
  -> GameData
  -> GameState
  -> legal action mask
  -> MiniTFTEnv.step(action)
  -> observation, reward, done, info
```

Use this as the correctness reference until the strategic lane has its own
focused tests.

### Existing Toy Lobby

```text
GameData + SharedUnitPool
  -> Set1LobbyState players
  -> policy actions
  -> shared shop/econ/item mutations
  -> scalar board-vs-board combat
  -> placement, top-1/top-4, HP, board strength
```

Use this when measuring placement-shaped behavior. Survival alone is not first
place.

### Strategic Lane

```text
fixed seed
  -> strategic state
  -> legal strategic actions
  -> deterministic step
  -> round-based enemy pressure
  -> placement proxy
  -> shared adapters for speed, cache, and demo
  -> quality gate artifacts
```

This is the active simplification path. It intentionally removes raw slot-level
actions, oracle macros, and multi-agent opponent state from the learning/search
surface.

## Decision Diamonds

Use these before adding code:

| Diamond | Question | Pass Signal |
| --- | --- | --- |
| Need | Does this serve MuZero cache, Puffer speed, or playable demo? | One deliverable is named. |
| Boundary | Does it use the shared strategic rules? | No adapter-specific fork of core rules. |
| Markov | Does current state contain every fact needed for transition/reward? | No transition depends on hidden history. |
| Determinism | Can fixed seeds reproduce the same trace? | Trace/checksum test passes. |
| Masking | Are illegal actions impossible for training/search? | Legal-mask test passes. |
| Pressure | Is difficulty round-based and calibrated? | Heuristic mean placement lands around `4` to `7`. |
| Evidence | Does it write auditable artifacts? | `metrics.json`, `decision.md`, and report exist. |

## Historical Lanes

Archived docs under `docs/archive/` preserve details for:

- candidate-choice MuZero reruns;
- raw-action Stage 2-5 gates;
- non-oracle action-surface work;
- Puffer material-speedup attempts;
- current-patch MetaTFT planner/value work;
- fight-value teacher experiments.

Use `docs/LEGACY_CODE.md` before deleting historical modules. Several old paths
are no longer the active research lane but still have tests, checkpoint loaders,
or UI references.

Do not use those as active instructions unless the user explicitly reopens that
track.
