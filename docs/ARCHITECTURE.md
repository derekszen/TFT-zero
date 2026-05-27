# Architecture

The simulator should feel like a small engine even while it remains Python.

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
