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

Measured on 2026-05-29 with fixed eval seeds `1000..1099`.

| Policy | Mean final HP | Survival rate | Mean survived round | Mean final strength |
| --- | ---: | ---: | ---: | ---: |
| RandomBot | 0.25 | 0.01 | 30.43 | 124.29 |
| GreedyBoardBot | 0.00 | 0.00 | 31.69 | 121.58 |
| EconBot | 0.00 | 0.00 | 20.46 | 0.00 |
| RerollBot | 0.00 | 0.00 | 27.89 | 70.45 |
| TraitCommitBot[ranger] | 31.31 | 0.94 | 35.99 | 252.01 |
| FastLevelBot | 69.05 | 0.99 | 36.00 | 304.52 |
| BC FastLevel 1k/e8/h64 | 20.15 | 0.72 | 35.67 | 236.56 |
| BC FastLevel 1k/e60/h64 | 55.41 | 0.98 | 36.00 | 281.99 |
| BC FastLevel 1k/e60/h256 | 59.50 | 0.98 | 36.00 | 288.75 |
| BC FastLevel 5k/e80/h256 | 66.30 | 1.00 | 36.00 | 295.88 |
| PPO from BC 250k/h256 | 70.87 | 1.00 | 36.00 | 312.49 |

Run notes:

- 5k FastLevel data generation: `677147` transitions in `39.854s`
  (`16990.7` transitions/sec) with `--workers 0`.
- 5k/e80/h256 BC pretraining: validation masked-action accuracy peaked around
  `0.923`.
- 250k PPO continuation ran at about `2405` env steps/sec on CPU and beat the
  current strongest heuristic, `FastLevelBot`, on the 100-seed eval.

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

## MuZero Later

MuZero-style latent planning fits TFT-like uncertainty better than AlphaZero,
but it should be a later phase. It needs a reliable env, clean logs, and baseline
policies first.
