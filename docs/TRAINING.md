# Training Plan

Start simple.

## First Baseline

Train PPO or behavioral cloning before trying MuZero-style planning.

This answers the first practical question:

Can the environment produce a signal that a standard policy optimizer can learn?

## Behavioral Cloning

Use bot data to train a weak policy quickly.

Inputs:

- observation tensors
- legal action mask
- scripted bot action

Outputs:

- masked action distribution

This is useful even if the policy is mediocre. It provides a starting policy and
debugs the featurization path.

Current command shape:

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

## PPO

Use MaskablePPO after the env can run many short episodes without crashing.
Install it through the `train` extra, which includes `sb3-contrib`.

Track:

- episode return
- survival round
- final HP
- illegal action rate
- roll count
- XP buy count
- bench overflow attempts
- trait diversity

If PPO cannot beat random, do not jump to MuZero. First inspect rewards, action
masks, episode length, observation quality, and whether scripted bots can show a
clear skill gradient.

Warm-start PPO from the BC checkpoint:

```bash
uv run python -m mini_tft.rl.train_ppo \
  --init checkpoints/bc_fastlevel_5k_e80_h256.zip \
  --timesteps 250000 \
  --num-envs 8 \
  --n-steps 256 \
  --batch-size 2048 \
  --device cpu \
  --out checkpoints/ppo_from_bc_fastlevel_250k_h256

uv run python -m mini_tft.rl.evaluate_policy \
  --episodes 100 \
  --checkpoint checkpoints/ppo_from_bc_fastlevel_250k_h256.zip
```

## Current Baseline

Toy Set-1-like simulator results. These use the handcrafted abstract combat
model in `mini_tft.core.combat`, not the current-patch MetaTFT value path.

The latest 5M PPO continuation was measured on 2026-05-31 with fixed eval seeds
`1000..1499`.

| Policy | Mean final HP | Survival rate | Mean survived round | Mean final strength |
| --- | ---: | ---: | ---: | ---: |
| RandomBot | 0.19 | 0.016 | 30.458 | 123.20 |
| GreedyBoardBot | 0.00 | 0.000 | 31.578 | 119.03 |
| EconBot | 0.00 | 0.00 | 20.46 | 0.00 |
| RerollBot | 0.00 | 0.000 | 28.242 | 73.42 |
| TraitCommitBot[ranger] | 31.542 | 0.948 | 35.984 | 249.00 |
| FastLevelBot | 68.812 | 0.986 | 35.994 | 300.75 |
| PPO from BC 5M/h256 | 77.100 | 0.984 | 35.970 | 336.66 |

Run notes:

- 5k FastLevel data generation: `677147` transitions in `39.854s`
  (`16990.7` transitions/sec) with `--workers 0`.
- 5k/e80/h256 BC pretraining: validation masked-action accuracy peaked around
  `0.923`.
- 250k PPO continuation ran at about `2405` env steps/sec on CPU and beat the
  current strongest heuristic, `FastLevelBot`, on the 100-seed eval.
- 5M GPU PPO continuation from `runs/ppo_gpu_smoke_50k_from_bc_h256.zip` took
  `6502s` wall time. Final training log reported `769` env steps/sec at
  `5,304,320` cumulative timesteps. The checkpoint is
  `checkpoints/ppo_from_bc_fastlevel_5m_h256.zip`.

Interpretation:

- The 5M PPO checkpoint builds stronger final boards than `FastLevelBot` under
  the toy simulator's scalar board-strength formula.
- Survival is effectively tied at this horizon: PPO had higher HP/strength but
  `0.984` survival versus `0.986` for `FastLevelBot` on 500 seeds.
- These numbers are not a human-rank benchmark and should not be compared to
  Bronze/Gold real TFT players.

## World Model

After rollout logs exist, train:

```text
state, action -> next state, reward, done
```

Use it for:

- representation learning
- value pretraining
- rollout sanity checks
- later planning experiments

## Fight Value Model

Detailed combat is now separated into a teacher-label pipeline. Generate labels
with the vendored TFTMuZeroAgent Set 4 teacher, train `FightValueNet`, and
benchmark GPU inference before using it inside RL:

```bash
uv run python -m mini_tft.tools.generate_fight_labels \
  --teacher tft-muzero \
  --target-fights 1000000 \
  --workers 12 \
  --shard-size 10000 \
  --out data/fight_labels/set4_teacher_v1

uv run python -m mini_tft.rl.train_fight_value_model \
  --dataset data/fight_labels/set4_teacher_v1 \
  --device cuda \
  --batch-size 16384 \
  --epochs 10 \
  --out checkpoints/fight_value/set4_teacher_v1.pt
```

See `docs/FIGHT_VALUE_MODEL.md` for smoke commands and benchmark details.

The learned fight path is now wired behind `EnvConfig(combat_model="fight_value")`,
but it should stay out of RL rewards/search until:

```bash
uv run python -m mini_tft.tools.calibrate_fight_value_model \
  --checkpoint checkpoints/fight_value/set4_teacher_v1.pt \
  --fixture tests/fixtures/metatft_set17_comp_strength_2026-05-31.json \
  --device cuda \
  --fail-on-threshold
```

passes against real comp ranking data.

The Set 4 teacher checkpoint is intentionally treated as stale for current-patch
MetaTFT work. To adapt the value model to the current patch, fetch a MetaTFT
aggregate snapshot and train the current-patch ranking adapter:

```bash
uv run python -m mini_tft.tools.fetch_metatft_comp_strength \
  --out data/metatft/current_comp_strength.json \
  --min-count 3000

uv run python -m mini_tft.rl.train_metatft_fight_value_model \
  --train-fixture data/metatft/current_comp_strength.json \
  --eval-fixture tests/fixtures/metatft_set17_comp_strength_2026-05-31.json \
  --device cuda \
  --epochs 4000 \
  --out checkpoints/fight_value/metatft_current_patch.pt
```

That current-patch checkpoint is a MetaTFT comp ranker, not a drop-in MiniTFT
combat model. `MiniTFTEnv` rejects it until the simulator uses the same
current-patch unit namespace.

## PufferLib Usefulness

PufferLib is useful here as a rollout-throughput layer once the environment and
policy semantics are stable. Its main benefit is vectorized custom-env training
with serial or multiprocessing workers and less Python overhead than a plain
single-env loop. For this project, use it after the current-patch symbolic
shop/econ policy produces sane traces and after reward/value labels are
calibrated.

Do not use PufferLib to hide model-quality problems. It will not fix weak
MetaTFT value generalization, bad action pacing, missing unit costs, or an
incorrect board-state abstraction. The current order should be:

```text
1. validate current-patch board/value labels
2. produce sane planner traces
3. add RL around that loop
4. use PufferLib if rollout collection becomes the bottleneck
```

## PPO Overnight Candidate

The strongest checked toy-simulator checkpoint is currently the BC-warmed PPO
line. A short GPU continuation from
`checkpoints/ppo_from_bc_fastlevel_250k_h256.zip` improved the 30-seed mean final
HP from `70.5` to `72.5` and kept survival at `1.0`.

Smoke command:

```bash
uv run python -m mini_tft.rl.train_ppo \
  --timesteps 50000 \
  --seed 42 \
  --num-envs 8 \
  --n-steps 512 \
  --batch-size 2048 \
  --learning-rate 1e-4 \
  --device cuda \
  --init checkpoints/ppo_from_bc_fastlevel_250k_h256.zip \
  --out runs/ppo_gpu_smoke_50k_from_bc_h256
```

Completed 5M continuation command:

```bash
uv run python -m mini_tft.rl.train_ppo \
  --timesteps 5000000 \
  --seed 100 \
  --num-envs 8 \
  --n-steps 512 \
  --batch-size 2048 \
  --learning-rate 1e-4 \
  --device cuda \
  --init runs/ppo_gpu_smoke_50k_from_bc_h256.zip \
  --out checkpoints/ppo_from_bc_fastlevel_5m_h256
```

The resume path now passes `learning_rate`, `n_steps`, and `batch_size` through
`MaskablePPO.load(custom_objects=...)`; verify logs show the requested learning
rate before launching long runs.

Evaluation command:

```bash
uv run python -m mini_tft.rl.evaluate_policy \
  --episodes 500 \
  --checkpoint checkpoints/ppo_from_bc_fastlevel_5m_h256.zip
```

## MuZero Later

MuZero-style latent planning fits TFT-like uncertainty better than AlphaZero,
but it should be a later phase. It needs a reliable env, clean logs, and baseline
policies first.
