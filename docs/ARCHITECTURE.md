# Architecture

The project asks one research question:

```text
Can a TFT-like agent learn shop/econ/board planning in a fast simulator, then
ground that planning in real current-patch board-strength data?
```

The answer is being built in staged tracks. Keep the tracks separate when making
claims.

## 1. Tracks

| Track | Main paths | Status | Claim it supports |
| --- | --- | --- | --- |
| Toy simulator | `src/mini_tft/core/`, `src/mini_tft/rl/gym_env.py` | Complete single-player Gymnasium env | RL can learn in a fast TFT-like loop. |
| Toy lobby | `src/mini_tft/core/lobby*.py`, `src/mini_tft/rl/lobby_env.py` | V0 shared-pool 8-player lobby | Survival, HP, and placement can be measured separately. |
| Toy policies | `src/mini_tft/bots/`, `src/mini_tft/rl/` | Heuristic bots, BC, PPO | Learned policies can beat scripted baselines under identical seeds. |
| Current-patch value/planner | `src/mini_tft/metatft/` | Rich catalog, encoders, value scorer, target-guided planner | Aggregate data can score and assemble symbolic current-patch boards. |
| Fight-value teacher | `src/mini_tft/fight_model/` | Experimental teacher-label/value-net path | Detailed combat can generate labels for a faster evaluator. |

## 2. System Flow

### Toy RL Loop

```text
set JSON data
  -> GameData
  -> GameState
  -> legal action mask
  -> MiniTFTEnv.step(action)
  -> shop/econ/items/combat/reward mutation
  -> observation + info
```

`MiniTFTEnv` is the single-player executable RL environment.

### Toy Lobby Loop

```text
GameData + SharedUnitPool
  -> Set1LobbyState with N GameState players
  -> policy actions for each active player
  -> shared-pool shop/econ/item mutations
  -> board-vs-board scalar combat
  -> HP standings, placements, top-1/top-4 metrics
```

`MiniTFTLobbyEnv` is the current Python reference for placement-shaped
experiments. It is not yet the native Puffer/C++ hot loop.

### Current-Patch Planner Loop

```text
MetaTFT aggregate endpoints
  -> rich catalog
  -> CurrentBoardState
  -> CurrentBoardEncoder
  -> value model
  -> candidate shop/bench/board transitions
  -> target-guided planner gates
```

This is not a full game env. It can score and assemble symbolic boards, but it
does not own full stochastic shops, augments, item choices, combat, rewards, or
episode termination.

### Fight-Value Loop

```text
teacher simulator or heuristic labels
  -> fight-label shards
  -> FightValueNet
  -> optional fast combat-value adapter
```

The Set 4 teacher path is experimental and not a current-patch combat oracle.

## 3. Core Contracts

### State

Toy simulator state is explicit Python data:

- round, HP, gold, level, XP;
- shop, bench, board, item bench;
- step counters and episode termination fields;
- last combat/value summary fields.

`GameState.clone()` deep-copies mutable unit, board, bench, shop, and item lists.
This keeps search branches isolated.

### Environment

`MiniTFTEnv` owns mutation and reward emission:

- `reset(seed=...)` creates deterministic initial state;
- `step(action)` validates legality through the action mask;
- `action_masks()` is the legal-action contract for policies and search;
- `clone_state()` captures `GameState` plus RNG state;
- `restore_state(snapshot)` restores deterministic branches.

The clone/restore API is the current boundary for simulator-backed MCTS.

`MiniTFTLobbyEnv` exposes the same branch-search idea at lobby scope:

- `clone_state()` captures all players, shared pool, placements, RNG state, and
  player-order rotation;
- `restore_state(snapshot)` replays stochastic lobby branches exactly;
- `episode_summary(player_id)` reports placement, final HP, survived round, and
  final board strength.

### Actions

The toy action space is discrete and mask-driven:

- end turn;
- roll;
- buy XP;
- buy shop slot;
- sell bench slot;
- move bench to board;
- move board to bench;
- debug conveniences such as `FIELD_BEST_BOARD` and `SLAM_BEST_ITEM`.

Future search should consume the same masks instead of inventing a second
legality layer.

### Data

Set content is data-driven:

- units define cost, traits, role, and base power;
- traits define breakpoints;
- items define components, tags, and scalar effects;
- shop odds and enemy curves live in JSON.

Python defines rules. JSON defines content.

## 4. Evaluation Gates

### CI-Quality Gates

```bash
uv run pytest
uv run ruff check
uv run --all-extras pyright
```

CI runs the same three checks.

### Simulator Gates

Use these for simulator, reward, item, board, search, or UI changes:

```bash
uv run python -m mini_tft.tools.simulator_regression_gate --strict
uv run python -m mini_tft.tools.web_ui_regression_gate --strict
```

The simulator gate covers throughput, determinism, round timing, item flow,
candidate boards, combat fixtures, level pacing, opponent-pressure HP caps, and
web payload behavior.

### Current-Patch Planner Gates

Use minimum gates before planner/RL reward/search changes:

```bash
uv run python -m mini_tft.tools.run_current_patch_planner_gates \
  --catalog data/metatft/current_rich_catalog_2026-05-31.json \
  --checkpoint checkpoints/fight_value/current_patch_board_value_2026-05-31.pt \
  --device cpu \
  --suite minimum
```

Use `--suite strict` when changing target-completion pressure, action pacing, or
search behavior.

## 5. Supported Claims

| Claim | Evidence | Limitation |
| --- | --- | --- |
| PPO can improve over scripted play in the toy simulator | 5M PPO report beats `FastLevelBot` on mean HP and board strength | Uses abstract Set-1-like combat. |
| Current-patch value can rank boards moderately | heldout MetaTFT ranking metrics | Not strong enough to be the only reward oracle. |
| Current-patch planner can complete target boards in fixed traces | planner gates require level 8/9 exact match rate `1.0` | Trace seeds expose target units; this is not learned gameplay. |
| The toy simulator can support branch-based search | clone/restore tests replay stochastic branches exactly | Search policy/value targets still need to be implemented. |
| The toy lobby can report placement separately from survival | lobby tests and `evaluate_lobby_policy` cover placements, top-1/top-4, HP, and clone/restore | Opponent policy distribution and real TFT calibration are still open. |

## 6. Not Proven

- The toy PPO checkpoint is not a real TFT player-rank benchmark.
- The toy simulator is not Riot-accurate TFT.
- MetaTFT aggregate data is not turn-by-turn action-trace data.
- The current-patch planner is not an RL policy.
- There is no complete current-patch `reset/step` env yet.
- The fight-value teacher/model path is not a validated current-patch combat
  oracle.

## 7. Package Map

```text
src/mini_tft/
  core/        toy simulator and lobby state, rules, masks, rewards, combat
  bots/        scripted policies
  rl/          Gym envs, lobby env, datasets, BC, PPO, evaluation, value training
  metatft/     current-patch catalog, encoders, value model, planner
  fight_model/ teacher-label pipeline and fight-value network
  tools/       data generation, gates, reports, benchmarks
  web/         local debug UI
  data/        packaged toy set JSON
```

## 8. Design Rules

- Keep game content data-driven.
- Keep state explicit, serializable, and cloneable.
- Keep action masking as part of the env contract.
- Keep debug UI separate from learning state.
- Prefer clear Python first; port hot paths only after profiling.
- Add or run the smallest useful smoke test for every behavior change.
