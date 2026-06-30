# Training

Training is currently a support surface for the strategic proof, not the main
product. Use it to check that observations, masks, rewards, and adapters can
learn something under fixed gates.

## Active Training Roles

| Role | Purpose | Claim Limit |
| --- | --- | --- |
| BC | Fast sanity check for observation/mask/action paths | Not final policy quality. |
| PPO | Baseline and smoke learner | Toy-simulator result only. |
| Puffer PPO | Rollout/trainer speed surface | Speed evidence only with matched benchmarks. |
| MuZero-style cache | Search/data/model smoke | Not a promotion claim without `QUALITY_GATE`. |

## PPO Smoke

```bash
env -u UV_PYTHON uv run --extra train python -m mini_tft.rl.train_ppo \
  --timesteps 1024 \
  --num-envs 2 \
  --n-steps 64 \
  --batch-size 128 \
  --device cpu \
  --out artifacts/smoke/ppo
```

Evaluate through the matching evaluator for the environment used. For lobby
questions, report placement, top-1/top-4, HP, survived round, board strength,
and illegal actions. Survival is not placement.

## Puffer Smoke

```bash
env -u UV_PYTHON uv run --extra puffer --extra train \
  python -m mini_tft.rl.train_puffer_ppo \
  --env-kind lobby \
  --timesteps 16 \
  --num-envs 2 \
  --n-steps 4 \
  --batch-size 4 \
  --update-epochs 1 \
  --device cpu \
  --players 4 \
  --max-round 3 \
  --max-actions-per-round 2 \
  --max-actions-per-player 2 \
  --disallow-oracle-macro-actions \
  --out artifacts/smoke/puffer_lobby.pt
```

This proves trainer wiring only. A Puffer speed claim requires repeated matched
benchmarks and semantic parity as defined in `docs/QUALITY_GATE.md`.

Strategic native Puffer smoke:

```bash
env -u UV_PYTHON uv run --extra puffer --extra train \
  python -m mini_tft.rl.train_puffer_ppo \
  --env-kind strategic \
  --puffer-backend native \
  --timesteps 128 \
  --num-envs 16 \
  --n-steps 8 \
  --batch-size 64 \
  --update-epochs 1 \
  --device cpu \
  --max-round 6 \
  --max-actions-per-round 2 \
  --out artifacts/strategic_lane/ppo_smoke/strategic_native_puffer_smoke.pt
```

Strategic throughput evidence:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_lane_gate
```

## MuZero-Style Direction

The next claim should be over the strategic action surface, not raw primitive
slot moves:

```text
strategic simulator
  -> legal masked search/candidate choices
  -> cache rows
  -> tiny policy/value/dynamics smoke
  -> baseline comparison
  -> quality gate decision
```

Run the reusable strategic MuZero-style smoke loop with:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_muzero_run_loop \
  --out-dir artifacts/strategic_lane/muzero_run_loop \
  --seed 1000 \
  --cache-rows 1024 \
  --strict
```

The loop writes fresh parity, baseline, cache, train-smoke, and verifier
artifacts under the output directory. The full trainer should preserve that
artifact contract and replace only the train-smoke stage.

The cache path should write auditable observations, masks, actions, rewards,
next observations, dones, policy targets, value targets, and metadata.

## Baseline Discipline

Every training artifact should name:

- environment and action surface;
- seed range;
- opponent/baseline policy where applicable;
- legal action rate;
- reward and outcome metrics;
- checkpoint format;
- exact command.

Use random plus at least one TFT-shaped heuristic as the first baseline set.

## Historical Baseline

The old toy PPO 5M result beat `FastLevelBot` on mean HP and final board
strength in the abstract simulator, but it is not a real TFT claim and not the
active MuZero path. Details are preserved in archived docs.
