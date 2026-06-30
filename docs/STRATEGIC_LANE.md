# Strategic Lane

## Purpose

Build the smallest TFT-shaped simulator that is easy to understand, fast to
batch, playable by a human, and sufficient for MuZero-style cache/data smoke
experiments.

This lane is a new clean proof surface. It is not a full TFT clone and not the
legacy Stage 2-5 promotion ladder.

## Deliverables

| Deliverable | Required Proof |
| --- | --- |
| MuZero cache/data | Dataset rows with observation, mask, action, reward, next observation, done, policy target, value target, and metadata. |
| Puffer speed | Matched reference-vs-batched benchmark with repeated runs and a clear decision. |
| Playable demo | Human-playable route or documented blocked state using the same rules as training. |

Write artifacts under:

```text
artifacts/strategic_lane/
  puffer_speed/
  muzero_cache/
  zero_smoke/
  playable_demo/
  metrics.json
  decision.md
  final_report.md
```

## Core Simulator

New strategic-lane code belongs under:

```text
src/mini_tft/strategic/
  core/
  adapters/
```

The existing `src/mini_tft/core/` package is the historical toy engine. Use it as
a correctness reference or compatibility bridge only. Do not add new strategic
rules there unless the task explicitly says to patch the legacy engine.

The core simulator should own one canonical implementation of:

- shop generation and roll/level/econ timing;
- gold, level, XP, HP, round, and terminal state;
- bench and board summaries;
- unit cost/star/trait abstractions;
- board-strength scoring;
- enemy pressure and placement-shaped outcome;
- legal action masks;
- deterministic `reset(seed)` and `step(action)`.

Adapters may translate state into their own format, but they must not fork game
rules.

## Markov Invariant

The simulator state must be Markov-sufficient:

```text
P(next_state, reward | history, action)
= P(next_state, reward | current_state, action)
```

If a future transition depends on a fact, that fact belongs in canonical state.
Do not recover transition behavior from hidden logs, prior observations, global
mutable counters, or adapter-local history.

The agent observation may intentionally hide fields, but hidden fields must be
documented as partial observability. A weak observation is allowed; accidental
history dependence in the simulator is not.

## Enemy Pressure

The strategic lane must not implement multi-agent lobby complexity. Opponents are
modeled as a simple round-based pressure curve:

```text
enemy_strength = f(round, optional_seed_noise)
scenario_score = g(final_hp, round_survived, final_board_strength)
placement_proxy = elimination timing bucket
```

There is no opponent board state, opponent bench, opponent economy, shared pool,
scouting, or self-play in this lane. If a pressure term affects transitions or
reward, it must be derived from canonical state fields such as `round` and RNG
state.

Calibration target: a basic TFT-shaped heuristic should rarely, ideally never,
reach `max_round`, and only a small tail should survive past Stage 5-3. Most
heuristic results should land in weaker elimination buckets such as 6-8. If most
heuristic episodes coast to `max_round`, the pressure curve or damage is too
easy. If nearly every episode dies before Stage 3, the curve is too punishing or
the action surface is broken.

`placement_proxy` is not real lobby placement. It is a stage-aware elimination
bucket:

| Outcome | Proxy |
| --- | ---: |
| Survive to `max_round` | 1 |
| Die at Stage 6-5 or later | 2 |
| Die at Stage 6-1 or later | 3 |
| Die at Stage 5-5 or later | 4 |
| Die at Stage 5-1 or later | 5 |
| Die at Stage 4-1 or later | 6 |
| Die at Stage 3-1 or later | 7 |
| Die earlier | 8 |

Use `scenario_score` for dense learning/debug signal. Use `placement_proxy` for
coarse reporting only.

## Strategic Action Set

Start with a deliberately small action surface:

```text
HOLD
LEVEL
ROLL
BUY_BEST_UPGRADE
BUY_BEST_SYNERGY
BUY_HIGHEST_COST
FIELD_STRONGEST
GREED_ECON
SLAM_CARRY_ITEM
SLAM_TANK_ITEM
SLAM_SUPPORT_ITEM
```

Add actions only when a failure is observable in the playable demo, baseline
evaluation, or MuZero/Puffer artifacts.

Role-item slams are intentionally abstract: the state tracks role item counts
and role item slots instead of individual components. This keeps item decisions
Markov, visible in the web payload, and cheap to batch.

## Removed From Learning/Search

- Raw 181-action primitive board controls.
- Bench-slot and board-slot micro-management as the main task.
- Oracle macro actions as learned/search actions.
- Full multi-agent self-play.
- Opponent boards, opponent economy, shared pools, and scouting.
- Detailed pathing, targeting, spell timing, and exact patch fidelity.
- Current-patch MetaTFT planner assumptions.
- Legacy Stage 2-5 promotion claims.

Debug UI may keep oracle buttons if they are clearly separated from training and
search.

## Adapter Contracts

### Puffer Adapter

- Emits batched observations, rewards, terminals/truncations, and masks.
- Benchmarks against the reference strategic step path.
- Reports steps/sec, speedup, confidence/variance, and pass/fail decision.

Current scaffold status: `mini_tft.tools.strategic_lane_gate` writes semantic
native Puffer vector throughput evidence when PufferLib is available, with
matched scalar replay parity and a repeated speed decision. The current default
gate uses batch size 256 and records the result under
`artifacts/strategic_lane/puffer_speed/`.

### MuZero Cache Adapter

- Generates deterministic cache rows from fixed seeds.
- Validates action legality and target schema.
- Trains at least a tiny policy/value/dynamics smoke model from the cache.
- Labels results as smoke unless they beat baselines under `QUALITY_GATE`.

### Playable Demo Adapter

- Uses the same strategic rules.
- Shows HP, gold, level, round, shop, board summary, enemy pressure, and result.
- Allows a human to play through the strategic action set.

## Minimum Acceptance

- One shared rules implementation feeds all adapters.
- Fixed-seed traces are reproducible.
- Illegal actions are rejected or masked before training/search.
- Baselines include at least random and one TFT-shaped heuristic.
- Enemy pressure is round-based and calibrated against heuristic placement.
- Artifacts include `metrics.json`, `decision.md`, and `final_report.md`.
- `docs/QUALITY_GATE.md` says whether the result is pass, fail, or smoke-only.
