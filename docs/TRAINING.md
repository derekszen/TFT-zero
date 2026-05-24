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

## MuZero Later

MuZero-style latent planning fits TFT-like uncertainty better than AlphaZero,
but it should be a later phase. It needs a reliable env, clean logs, and baseline
policies first.
