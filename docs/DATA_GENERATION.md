# Data Generation

Initial RL data should come from the simulator itself.

## Sources

Use three data sources over time:

- scripted bots for bootstrap data
- on-policy PPO or other RL rollout data
- later teacher/search-generated labels

Do not wait for real game data. It is optional and mostly useful for realism or
human imitation later.

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
