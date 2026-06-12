# Training

Training work is staged. Standard policy learning comes first; search and
MuZero-style planning come after the simulator, logs, and baselines are reliable.

## 1. Current Baseline

Toy Set-1-like simulator results use `mini_tft.core.combat`, not current-patch
MetaTFT value or real combat.

The latest documented 5M PPO continuation was measured on 2026-05-31 over fixed
eval seeds `1000..1499`.

| Policy | Mean final HP | Survival rate | Mean survived round | Mean final strength |
| --- | ---: | ---: | ---: | ---: |
| RandomBot | 0.19 | 0.016 | 30.458 | 123.20 |
| GreedyBoardBot | 0.00 | 0.000 | 31.578 | 119.03 |
| EconBot | 0.00 | 0.00 | 20.46 | 0.00 |
| RerollBot | 0.00 | 0.000 | 28.242 | 73.42 |
| TraitCommitBot[ranger] | 31.542 | 0.948 | 35.984 | 249.00 |
| FastLevelBot | 68.812 | 0.986 | 35.994 | 300.75 |
| PPO from BC 5M/h256 | 77.100 | 0.984 | 35.970 | 336.66 |

Interpretation:

- PPO builds stronger final boards than `FastLevelBot` under the toy scalar
  combat formula.
- Survival is effectively tied: PPO has higher HP/strength, but `0.984`
  survival versus `0.986` for `FastLevelBot`.
- This is not a human-rank or real TFT benchmark.

## 2. Guardrails

Run these before trusting a new training result:

```bash
uv run pytest
uv run ruff check
uv run --all-extras pyright
```

For simulator, reward, item, board, or search changes also run:

```bash
uv run python -m mini_tft.tools.simulator_regression_gate --strict
```

Training scripts write checkpoint sidecar manifests:

```text
checkpoint.zip
checkpoint.manifest.json
```

The manifest records CLI args, resolved batch/rollout settings, git metadata,
package versions, elapsed time, and whether timesteps were reset.

## 3. Behavioral Cloning

Use scripted bot data to quickly debug the observation, mask, and action path.

```bash
uv run python -m mini_tft.tools.generate_bot_dataset \
  --suite fastlevel \
  --episodes 5000 \
  --workers 0 \
  --out rollouts/fastlevel_bc_5k.npz

uv run python -m mini_tft.rl.pretrain_bc \
  --dataset rollouts/fastlevel_bc_5k.npz \
  --epochs 80 \
  --batch-size 8192 \
  --hidden-sizes 256,256 \
  --device cpu \
  --out checkpoints/bc_fastlevel_5k_e80_h256
```

BC is not the end goal. It gives PPO a reasonable starting policy and catches
featurization or mask bugs earlier than a long RL run.

## 4. PPO

Use MaskablePPO after the env can run many short episodes without illegal-action
or termination issues.

Track:

- episode return
- survival round and final HP
- final board strength
- illegal action rate
- roll count and XP buy count
- bench overflow or blocked action patterns

Warm-start PPO:

```bash
uv run python -m mini_tft.rl.train_ppo \
  --init checkpoints/bc_fastlevel_5k_e80_h256.zip \
  --timesteps 250000 \
  --num-envs 8 \
  --n-steps 256 \
  --batch-size 2048 \
  --learning-rate 3e-4 \
  --device cpu \
  --out checkpoints/ppo_from_bc_fastlevel_250k_h256
```

Resume behavior is explicit: `learning_rate`, `n_steps`, and `batch_size` are
passed through `MaskablePPO.load(custom_objects=...)`. The command prints the
resolved config before training and writes it to the manifest.

Evaluate:

```bash
uv run python -m mini_tft.rl.evaluate_policy \
  --episodes 100 \
  --checkpoint checkpoints/ppo_from_bc_fastlevel_250k_h256.zip
```

## 5. Lobby Placement Evaluation

Single-player survival is not the same thing as first place. Use the lobby
evaluator when the question is placement, top-4, or final HP against live
scripted opponents:

```bash
uv run python -m mini_tft.tools.evaluate_lobby_policy \
  --episodes 100 \
  --hero-policy fast_level \
  --opponent-policy tempo \
  --format markdown

uv run python -m mini_tft.tools.evaluate_lobby_policy \
  --episodes 100 \
  --checkpoint checkpoints/ppo_from_bc_fastlevel_250k_h256.zip \
  --opponent-policy tempo \
  --format json
```

Track:

- mean placement and placement histogram
- top-1 and top-4 rate
- final HP and survived round
- final board strength
- total actions, fights, and illegal actions

The lobby is still a Set-1-like toy reference. It is the right benchmark for
placement-shaped learning, but not a real TFT rank claim.

## 6. Search And MuZero Path

MuZero-style or simulator-backed MCTS should build on the toy simulator first,
not the current-patch planner shell.

Current search-ready pieces:

- `GameState.clone()` deep-copies mutable state.
- `MiniTFTEnv.clone_state()` captures game state plus RNG state.
- `MiniTFTEnv.restore_state()` restores deterministic search branches.
- `MiniTFTLobbyEnv.clone_state()` captures lobby state, shared pool, RNG state,
  placements, and player-order rotation.
- Tests verify exact stochastic branch replay and alias safety.

Next requirements before serious search training:

1. Add a search benchmark for clone/sec, step/sec, and rollout/sec.
2. Define a per-decision simulation budget, such as 32, 64, 128, or 256 sims.
3. Keep action masks as the planner's legal-action contract.
4. Log search depth, node count, value target, and wall-clock cost in manifests.

Do not jump to MuZero if PPO cannot beat random or scripted baselines. First
inspect rewards, action masks, episode length, observation quality, and whether
scripted bots show a clear skill gradient.

## 7. PufferLib

PufferLib is a rollout-throughput layer, not a quality fix.

Use it when:

- the env semantics are stable;
- PPO/BC logs are reproducible;
- rollout collection is the bottleneck;
- simulator gates still pass under the wrapper.

Do not use it to hide weak value labels, bad action pacing, missing unit costs,
or a poor board-state abstraction.

Recommended path:

1. Keep the Python `MiniTFTLobbyEnv` as the correctness reference.
2. Wrap the Gymnasium env with PufferLib only after tests and placement metrics
   are stable.
3. Port the hot lobby step loop to native `PufferEnv`/C++ only when profiling
   shows Python rollout throughput is the bottleneck.
4. Require parity tests against the Python lobby before using native results for
   training claims.

## 8. Current-Patch And Fight Value

The current-patch path is a value/planner track today, not a full RL env.

High-level flow:

```text
MetaTFT aggregate snapshot
  -> rich catalog
  -> current-patch encoder/value model
  -> target-guided shop/econ planner
  -> planner gates
```

Fight-value work is separate. The Set 4 teacher path can train `FightValueNet`,
but that checkpoint is not a validated current-patch combat oracle. `MiniTFTEnv`
rejects MetaTFT current-patch value checkpoints until the simulator uses the same
unit namespace.

See:

- [Current-Patch MetaTFT](CURRENT_PATCH_METATFT.md)
- [Fight Value Model](FIGHT_VALUE_MODEL.md)
