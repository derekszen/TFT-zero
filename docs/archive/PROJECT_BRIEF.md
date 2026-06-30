# Project Brief

Historical brief. For current status, use `README.md`; for active MuZero-stage
work, use `docs/MUZERO_STAGE_GATE.md`.

TFT-zero is a Python-first research prototype for learning TFT-like macro play.

The first objective is not a perfect clone. The first objective is a simulator
that is easy to inspect, mutate, log, and train against.

## Core Decision

Use Python first.

Reasons:

- fastest iteration with coding agents
- easiest RL integration
- easiest debugging and data analysis
- no Python-to-Rust binding layer during early design
- simple path to Gymnasium, NumPy, PyTorch, and logging tools

Rust, Cython, or Numba can be introduced later for profiled hot paths.

## Initial Stack

- Python 3.12
- `uv` with local `.venv`
- Gymnasium-style environment
- NumPy observation and mask data
- JSON data packs for units, traits, items, and shop odds
- PyTorch and Stable-Baselines3 as optional training dependencies

## First Research Question

Can a policy learn useful long-horizon TFT-like macro behavior from a simplified
simulator with stochastic shops, limited bench space, economy pressure, and
trait/team-building incentives?

## Non-Goals For v0

- faithful live TFT rules
- full combat simulation
- multiplayer lobby realism
- polished visual renderer
- icon or thumbnail dependency
- MuZero-first training

## Preferred Order

1. Build a small, correct simulator loop.
2. Add legal action masks and deterministic smoke tests.
3. Add scripted bots and rollout logging.
4. Train PPO or behavioral cloning baselines.
5. Train a world model from logged transitions.
6. Only then evaluate planning methods.
