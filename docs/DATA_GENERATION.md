# Data Generation

Initial RL data should come from the simulator itself.

## Sources

Use three data sources over time:

- scripted bots for bootstrap data
- on-policy PPO or other RL rollout data
- later teacher/search-generated labels

Do not wait for real game data. It is optional and mostly useful for realism or
human imitation later.

## Bot Suites

`generate_bot_dataset.py` supports named bot suites:

| Suite | Contents | Main use |
| --- | --- | --- |
| `default` | random, greedy, economy, reroll, fast-level, and trait-commit bots | broad offline/debug data |
| `fastlevel` | only `FastLevelBot` | high-signal imitation teacher |
| `expert` | `FastLevelBot`, `TraitCommitBot[glacial]`, `TraitCommitBot[ranger]` | mixed stronger-policy coverage |

Examples:

```bash
uv run python -m mini_tft.tools.generate_bot_dataset \
  --suite default \
  --episodes 1000 \
  --workers 0 \
  --out rollouts/default_1k.npz

uv run python -m mini_tft.tools.generate_bot_dataset \
  --suite fastlevel \
  --episodes 5000 \
  --workers 0 \
  --out rollouts/fastlevel_bc_5k.npz
```

## Transition Schema

Each transition should save:

```json
{
  "episode_id": "uuid-or-int",
  "step_idx": 12,
  "obs": {},
  "action": 4,
  "action_type": "buy_shop_slot",
  "action_mask": [],
  "legal": true,
  "reward": 0.1,
  "next_obs": {},
  "done": false,
  "return_to_go": 3.7
}
```

## State Fields To Preserve

Save enough state for debugging, not just model input:

- round
- step
- gold
- HP
- level
- XP
- shop contents
- bench contents
- board contents
- stars
- item IDs
- active traits
- legal action mask

## Transition Diagnostics

Useful derived fields:

- HP delta
- gold delta
- board strength before and after
- action legality
- policy logits, if policy-generated
- value estimate, if policy-generated

## Episode Summary

Each episode should save:

- final HP
- final score
- survival round
- final comp or trait line
- total rolls
- total XP buys
- total item slams
- total illegal action attempts

## Storage Path

Use local generated directories that are ignored by git:

```text
rollouts/
  bot_bootstrap/
  ppo_mixed/
runs/
checkpoints/
```

V0 writes compressed NumPy `.npz` files because the observation/action arrays are
already fixed-shape. JSONL remains useful later for hand-inspecting selected
episodes.
