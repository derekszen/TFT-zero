# Architecture

The project is trying to answer one research question:

```text
Can we build a TFT agent that learns shop/econ/board planning, then ground that
planning in real current-patch board strength data?
```

There are two practical routes in the repo:

- **Executable simulator route:** build a fast, simplified TFT-like game where
  RL can actually run millions of transitions.
- **Current-patch value route:** use MetaTFT aggregate data to learn a symbolic
  board/value model and planner for real current-patch units, items, augments,
  traits, and comps.

These routes are complementary, not equivalent. The toy simulator is the
environment where RL is currently executable. The MetaTFT path is the
real-current-patch value/planner layer, but it is not yet a full game
environment.

The repo currently has three related but different tracks:

- a fast executable Set-1-like toy simulator for RL loop research
- a current-patch MetaTFT value/planner path for real-meta board evaluation
- a fight-value teacher/model path for combat-label experiments

Keep these separate when reading results. A PPO checkpoint trained in the toy
simulator proves the RL pipeline can learn inside that simulator. It does not
prove real TFT skill. A current-patch planner trace that matches MetaTFT comps
proves symbolic board completion under fixed trace seeds, not full-game RL.

## Top-Down Shape

Read the system from top to bottom like this:

```text
Research goal
  -> environments / value sources
  -> state encoders
  -> policies and planners
  -> evaluation gates
  -> claims we are allowed to make
```

### 1. Environments / Value Sources

There are three notions of "strength" in the repo:

- `MiniTFTEnv`: a complete Set-1-like toy simulator with abstract combat.
- MetaTFT current-patch aggregate data: real comp/unit/item/augment/stage-line
  statistics used to train and evaluate board value.
- Fight-value teacher experiments: detailed-fight labels and learned fight
  evaluators, currently experimental and not validated as a current-patch oracle.

Only `MiniTFTEnv` is a complete executable RL environment today.

### 2. State Encoders

The toy simulator uses explicit Python state for round, HP, gold, level, shop,
bench, board, items, and traits.

The current-patch path uses `CurrentBoardState` and `CurrentBoardEncoder` to
represent symbolic real-patch boards:

```text
stage, level, gold, HP
board units + items + positions
bench units
traits
augments
target comp metadata
```

This makes current-patch boards trainable by ordinary ML models without needing
vision or a real TFT client.

### 3. Policies / Planners

There are three policy layers:

- heuristic bots for the toy simulator
- BC/PPO policies for the toy simulator
- a current-patch symbolic shop/econ planner that ranks candidate
  shop/bench/board transitions with a learned value scorer plus target-comp
  completion pressure

The current-patch planner is not yet an RL policy. It is a target-guided policy
shell used to test whether candidate board transitions can complete top comps
under controlled traces.

### 4. Evaluation Gates

The main recurring gates are:

- `sim_smoke.py` for toy simulator speed, determinism, and trace sanity
- fixed-seed heuristic/PPO evaluation for toy RL
- heldout MetaTFT ranking metrics for current-patch value quality
- `run_current_patch_planner_gates --suite minimum` before planner/RL
  reward/search changes
- `run_current_patch_planner_gates --suite strict` when changing target
  completion pressure, action pacing, or search

The gate suite is intentionally explicit because otherwise it is too easy to
confuse target-board assembly with actual planning.

## Current System Map

| Layer | Main paths | Status | Answers |
| --- | --- | --- | --- |
| Toy simulator | `src/mini_tft/core/`, `src/mini_tft/rl/gym_env.py` | Full executable single-player env | Can an agent learn shop/econ/board decisions in a fast TFT-like loop? |
| Toy policies | `src/mini_tft/bots/`, `src/mini_tft/rl/train_ppo.py` | Heuristic bots, BC, MaskablePPO | Does PPO beat scripted baselines under identical seeds? |
| Toy strength model | `src/mini_tft/core/combat.py` | Handwritten scalar board strength | Is this board strong according to our abstract Set-1-like rules? |
| Current-patch catalog | `src/mini_tft/metatft/catalog.py`, `fetch.py` | Rich aggregate ingest | What units/items/traits/augments/comps exist in the current MetaTFT snapshot? |
| Current-patch encoder/value | `src/mini_tft/metatft/encoders.py`, `value_training.py` | Learned board-value scorer | Can aggregate data rank current-patch boards/comps? |
| Current-patch planner shell | `src/mini_tft/metatft/planner.py`, `policy.py`, `planner_evaluation.py` | Target-guided symbolic policy shell with batch gates | Can fixed level 8/9 traces complete top MetaTFT boards? |
| Fight-value teacher path | `src/mini_tft/rl/train_fight_value_model.py`, `docs/FIGHT_VALUE_MODEL.md` | Experimental teacher-label/value-net path | Can a detailed teacher produce labels for a faster learned fight evaluator? |

## Current Claims

The strongest claims currently supported by code and reports are:

| Claim | Evidence | Limitation |
| --- | --- | --- |
| PPO can improve over scripted play in the toy simulator | 5M PPO report beats FastLevelBot on mean HP and board strength | Uses abstract Set-1-like combat, not real TFT |
| Current-patch value model can rank boards moderately | heldout MetaTFT ranking smoke: pairwise accuracy ~0.643, Spearman ~0.407 | Not yet strong enough to be the only reward oracle |
| Current-patch planner can complete target boards in fixed traces | batch gate requires level 8 and 9 exact match rate of `1.0` | Trace seeds already expose target units; this is planner correctness, not learned gameplay |

## What Works

- The toy Set-1-like simulator runs as a real `reset/step` environment with
  action masks, bots, dataset generation, text/UI inspection, and smoke tests.
- The toy PPO pipeline has a real trained checkpoint. The best documented
  5M-step PPO-from-BC checkpoint beats FastLevelBot on mean final HP and final
  scalar board strength in the toy simulator, while survival rate is effectively
  tied.
- The current-patch MetaTFT ingest builds a rich catalog of units, items,
  augments, traits, comps, and stage lines from aggregate endpoints.
- The current-patch encoder/value path trains a board-value model and evaluates
  it against heldout MetaTFT comp rankings.
- The current-patch planner can complete fixed level 8/9 target comps in
  `shop-planning`, `distractor-heavy`, and `multi-roll` gate suites with exact
  match rate `1.0` on the pinned 2026-05-31 snapshot.
- The planner gate command now gives a compact CI/agent-friendly report while
  saving full trace JSON when `--out` is provided.

## What Has Not Worked / Is Not Proven

- The Set-1-like toy simulator is not a real TFT simulator. Its combat is a
  handcrafted scalar board-strength model, not Riot-accurate combat.
- The toy PPO result is not evidence of human-rank TFT skill. It only proves
  learning inside that toy environment.
- The MetaTFT value model is useful but moderate. It is not strong enough yet to
  be trusted as the only reward oracle for current-patch RL.
- MetaTFT aggregate data is not action-trace data. It does not tell us exact
  shop offers, roll timing, augment choices, positioning, or combat outcomes for
  each player turn.
- The current-patch planner gates are fixed synthetic traces with known target
  comps. Passing them proves symbolic board completion and action pacing under
  controlled conditions, not organic full-game planning.
- There is no complete current-patch `reset/step` RL environment yet. Shops,
  augments, item choices, economy, combat/value replacement, rewards, and
  termination still need to be owned end to end before current-patch RL is real.
- The fight-value teacher/model path is experimental and not yet connected as a
  validated current-patch combat oracle.

## Practical Interpretation

The architecture right now is best understood as:

```text
Toy simulator:
  proves RL mechanics and simulator throughput.

MetaTFT value/planner:
  builds a real-patch symbolic board-strength and target-board planning layer.

Future current-patch RL env:
  should combine current-patch state/action schemas, MetaTFT value estimates,
  calibrated combat/fight-value models, and stochastic shop/econ/augment
  transitions into one executable reset/step loop.
```

The next big architecture step is not another isolated planner script. It is an
experiment harness and eventually a current-patch environment wrapper that can
run many episodes, save comparable artifacts, and label exactly which source of
strength each result used.

For the Set-1-like simulator specifically, use
[`docs/SIMULATOR_ROADMAP.md`](SIMULATOR_ROADMAP.md) as the branch-sized backlog.
It classifies the simulator gaps by implementation difficulty and gives
worktree-sized acceptance criteria.

## What Is A Full Simulator Here?

A full simulator needs an executable `reset/step` loop that owns the state and
all stochastic transitions:

- shop odds and rolling
- XP, level, gold, interest, and streaks
- bench, board, item, augment, and trait state
- combat or a calibrated fight-value replacement
- episode termination and rewards
- action masks for everything an agent can choose

`MiniTFTEnv` has this for the toy Set-1-like environment. The current-patch
MetaTFT path does not yet have it. It can encode states and score candidate
successor boards, but it does not yet generate full current-patch games by
itself.

The MetaTFT aggregate data is also not action-trace data. It tells us which
final comps, units, items, augments, and stage lines correlate with placements.
It does not directly tell us each player's exact shop, roll, buy, sell, augment,
positioning, and combat decisions. That is why the current-patch path is a
value/planner layer today, not a complete RL environment.

## Package Layout

```text
src/mini_tft/
  data/
    set1_units.json
    set1_traits.json
    set1_items.json
    shop_odds.json
    enemy_curves.json
  core/
    config.py
    state.py
    actions.py
    set_data.py
    shop.py
    economy.py
    upgrades.py
    traits.py
    items.py
    combat.py
    rewards.py
    env.py
    masks.py
    featurize.py
  bots/
  metatft/
  rl/
  tools/
```

## Design Rules

- Data files define set content. Python code defines rules.
- State objects stay explicit and serializable.
- Env stepping owns mutation and reward emission.
- Rendering and assets are debug-only, not part of learning state.
- Action masking is part of the env contract from day one.
- Tests should exercise reset, legal actions, data loading, and one rollout path.

## Data-Driven Content

Each set should provide:

- units with cost, traits, and base stats
- traits with activation tiers
- items with stat/effect metadata
- shop odds by player level

The starter `set1` pack uses real Set 1 champion names as labels, but all rule
effects are abstract. It is not a Riot data fidelity target.

## State Boundary

The core state should include:

- round and step index
- player HP
- gold, level, and XP
- shop slots
- bench slots
- board slots
- unit stars and item IDs
- active traits

Only include information in the observation that the policy is allowed to know.
Keep hidden RNG state and future shops out of observations.

## Action Boundary

v0 should keep a discrete action space that separates macro decisions from
explicit board membership decisions:

- end turn
- roll shop
- buy XP
- buy a shop slot
- sell a bench slot
- field best board
- slam best item
- move bench slot to board slot
- move board slot to bench slot

The simulator keeps `field best board` as a heuristic/debug convenience, but RL
policies can use masked placement actions directly. Board slot coordinates are
currently membership slots only; exact hex positioning does not affect combat in
v0.

Later versions can add:

- combine units
- slam or move items
- choose carousel or augment-like rewards

## Combat Boundary

Do not start with full combat. Start with an abstract board-strength function and
a scripted enemy tempo curve. Replace this later with richer combat only after
the economy/shop loop is learnable.

## Set-1-Like Board Strength

The toy simulator predicts board strength with a hand-built scalar:

```text
unit base power
* star multiplier
* item role multiplier
* role/position multiplier
+ trait effects
+ item fit bonuses
+ board balance bonuses
+ upgrade reliability bonuses
+ assassin pressure bonus
```

Combat compares that scalar against a scripted enemy curve:

```text
p_win = sigmoid((my_strength - enemy_strength) / combat_sigmoid_scale)
```

This is valid as an internal benchmark for PPO versus heuristic bots, because
all policies face the same seeds and rules. It is not a real TFT board-strength
oracle.

## Current-Patch Value Path

The MetaTFT path is intended to become the real-meta board evaluator. It uses
current-patch aggregate data to train a value model over symbolic board states:

```text
current board/state -> predicted placement/value
```

That model can rank final boards and candidate transitions. The current
checkpoint is useful as a research artifact, but it is not yet strong enough to
be treated as the only reward oracle. The latest heldout ranking smoke was
moderate: pairwise accuracy around `0.643`, Spearman around `0.407`, and top-k
overlap around `0.50` on heldout comps.

The planner now adds a symbolic target-completion layer on top of value scoring:

```text
score(candidate) = model_value(candidate)
                 + target completion bonus
                 - missing/off-target/duplicate penalties
```

There are four planner trace modes:

- `completion`: verifies target board assembly when the trace already contains
  target bench support.
- `shop-planning`: starts from a partial board with no target bench and checks
  visible shop buys, rolling, and stopping after exact completion.
- `distractor-heavy`: adds off-target shop units around visible targets.
- `multi-roll`: starts with an off-target shop and splits missing target units
  across later shops, requiring multiple rolls.

Use `shop-planning` as the recurring regression gate before planner or
reward/search changes. Use `distractor-heavy` or `multi-roll` as stricter
follow-up gates when changing target completion pressure, action pacing, or
search behavior:

```bash
uv run python -m mini_tft.tools.run_current_patch_planner_gates \
  --catalog data/metatft/current_rich_catalog_2026-05-31.json \
  --checkpoint checkpoints/fight_value/current_patch_board_value_2026-05-31.pt \
  --device cpu \
  --suite minimum \
  --comp-limit 8 \
  --demo-levels 8,9 \
  --match-levels 8,9 \
  --top-k 10 \
  --min-recall 0.75 \
  --out runs/current_patch_planner_gate.json
```

## Auto-Research Harness Direction

The next architecture layer should be an experiment harness, not another
one-off script. It should run the same loop for each hypothesis:

1. fetch or load a pinned data snapshot
2. build encoded datasets and train a checkpoint
3. run fixed-seed or heldout-comp evaluation
4. compare against baselines in a machine-readable report
5. save artifacts, traces, metrics, and config together

For this project, the useful experiment families are:

- toy-simulator PPO/BC baselines against heuristic bots
- current-patch value-model ranking against heldout MetaTFT comp rankings
- planner-policy traces checked for final top-comp match rate by level 8/9
- fight-value teacher/model speed and calibration tests

The harness should make it hard to confuse these tracks. Reports should label
whether a result comes from the toy simulator, current-patch MetaTFT aggregate
data, a fight teacher, or a learned value model.
