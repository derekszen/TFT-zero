# Architecture

The repo currently has three related but different tracks:

- a fast executable Set-1-like toy simulator for RL loop research
- a current-patch MetaTFT value/planner path for real-meta board evaluation
- a fight-value teacher/model path for combat-label experiments

Keep these separate when reading results. A PPO checkpoint trained in the toy
simulator proves the RL pipeline can learn inside that simulator. It does not
prove real TFT skill. A current-patch planner trace that matches MetaTFT comps
proves symbolic board completion under fixed trace seeds, not full-game RL.

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

There are two planner gates:

- `completion`: verifies target board assembly when the trace already contains
  target bench support.
- `shop-planning`: starts from a partial board with no target bench and checks
  visible shop buys, rolling, and stopping after exact completion.

Use `shop-planning` as the recurring regression gate before planner or
reward/search changes:

```bash
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
