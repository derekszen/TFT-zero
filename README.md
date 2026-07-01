# TFT-Zero

<p align="center">
  <img src="https://i.redd.it/7musegquo5kf1.gif" alt="Teamfight Tactics gameplay reference" width="900">
</p>

TFT-Zero is a compact TFT-shaped strategic planning testbed. It keeps the
macro loop - shop, gold, rolling, leveling, board strength, item slams, fight
pressure, HP - while using a simplified Markov simulator with a shared
38-dimensional observation and 11-action legal mask.

This repository is not a Riot-accurate TFT clone and does not claim real TFT
ranked performance. The current evidence is easiest to read as two results:

1. **Speed / infrastructure result:** the strategic environment now runs through
   PufferLib 4.0 Ocean-style C and CUDA trainer smoke paths at multi-million
   step throughput.
2. **Trained-model result:** a CUDA Torch V0 policy/value/dynamics checkpoint
   trains on simulator-backed MCTS cache targets and passes the current
   readiness gate, but it is not full iterative MuZero self-play.

Useful links:

- [Link to the simulator](https://minitft.wahrwelt.world/)
- [Paper/report agent context](artifacts/paper_agent_context/README.md)
- [Paper draft in OpenAI Prism](https://prism.openai.com/?u=db147769-d04d-4a0c-bd51-3123fc427703&pg=1&m=neurips_2026.tex&d=7)

## Result 1: Speed

Main claim-safe speed evidence:

| Measurement | Result | Status |
| --- | --- | --- |
| Python scalar strategic heuristic loop | `25,259.39` steps/sec | reference smoke |
| PufferLib 4.0 Ocean C standalone loop | `3,755,013.36` steps/sec, `148.66x` scalar | `smoke_only` |
| PufferLib 4.0 CUDA trainer path | `9,017,194` agent steps/sec, `356.98x` scalar over `262,144` agent steps | `smoke_only` |
| Checkpoint-guided native-root MCTS cache generation | `8,180.78` decisions/sec and `130,892.43` simulations/sec for `native_root_prior_cuda_s16`; `46.02x` at 16 sims and `151.89x` at 64 sims vs single-root CPU Python checkpoint-guided MCTS | `accepted_l3_speedup` |

The Puffer numbers are rollout/trainer-throughput smoke results on a 32 GB RTX
5090 D workstation. The checkpoint-guided MCTS speed result proves the current
cache-generation path can batch checkpoint root priors and hand them to a
compiled native-root MCTS backend. It does not yet prove native learned
leaf-value search or final policy quality.

## Result 2: Trained Model

Latest trained-checkpoint evidence:

| Measurement | Result |
| --- | --- |
| Main evidence packet | `muzero_overnight_20260630T174428Z` |
| Checkpoint | `train_torch/strategic_muzero_torch.pt` |
| Cache rows | `262,144` |
| Observation / action shape | `38 x 11` |
| Legal action rate / policy-target validity | `1.0 / 1.0` |
| Native MCTS cache target speed | `5,710.97` decisions/sec at 64 simulations |
| Torch V0 train setup | CUDA, hidden size `256`, batch size `2048`, `64` epochs |
| Total loss | `4.620549 -> 0.835981` |
| Policy loss | `1.916569 -> 0.798829` |
| Value loss | `2.674111 -> 0.036199` |
| Dynamics loss | `0.119474 -> 0.003810` |
| Policy target top-1 accuracy | `0.131397 -> 0.970245` |
| Legal argmax rate | `1.0` |
| Gate verdict | `ACCEPT`, `21/21` checks |

Policy-evaluation smoke for the trained checkpoint is deliberately modest:
`torch_muzero` has mean placement proxy `7.0`, mean reward `-2.267`, and mean
scenario score `0.2127` over 512 episodes. The heuristic baseline is still
slightly better by reward and scenario score. This is useful trained-checkpoint
evidence and queue readiness; it is not a solved-policy claim.

## AI Agent Quick Onboarding

Read these first:

1. [AGENTS.md](AGENTS.md)
2. [docs/STRATEGIC_LANE.md](docs/STRATEGIC_LANE.md)
3. [docs/QUALITY_GATE.md](docs/QUALITY_GATE.md)
4. [docs/TRAINING.md](docs/TRAINING.md)
5. [artifacts/paper_agent_context/README.md](artifacts/paper_agent_context/README.md)

Then inspect the core files:

```text
src/mini_tft/strategic/core/actions.py
src/mini_tft/strategic/core/state.py
src/mini_tft/strategic/core/rules.py
src/mini_tft/strategic/core/obs.py
src/mini_tft/strategic/adapters/mcts.py
src/mini_tft/strategic/adapters/muzero_cache/export.py
src/mini_tft/strategic/ocean/strategic_tft.c
src/mini_tft/tools/benchmark_puffer4_ocean.py
src/mini_tft/tools/run_strategic_muzero_overnight.py
src/mini_tft/tools/strategic_muzero_run_loop.py
```

Use this command style:

```bash
env -u UV_PYTHON uv run ...
```

Basic validation:

```bash
uv sync
env -u UV_PYTHON uv run pytest
env -u UV_PYTHON uv run ruff check
env -u UV_PYTHON uv run --all-extras pyright
git diff --check
```

Queue a scaffolded PufferLib 4.0 Ocean C speed run:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/strategic_lane/puffer4_ocean_${RUN_ID}"

env -u UV_PYTHON uv run python -m mini_tft.tools.benchmark_puffer4_ocean \
  --out-dir "$OUT" \
  --envs 4096 \
  --steps 10000000
```

Queue the end-to-end MuZero-readiness scaffold:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/strategic_lane/muzero_run_loop_${RUN_ID}"

env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_muzero_run_loop \
  --out-dir "$OUT" \
  --seed 1000 \
  --cache-episodes 64 \
  --cache-rows 1024 \
  --mcts-simulations 16 \
  --mcts-max-depth 10 \
  --mcts-rollout-steps 6 \
  --mcts-prior-mode heuristic \
  --train-epochs 24 \
  --train-learning-rate 0.03 \
  --baseline-episodes 32 \
  --codex-allowance-source user-waived \
  --codex-5h-window-remaining user-waived \
  --codex-weekly-usage user-waived \
  --codex-allowance-decision continue \
  --parity-seed 0 \
  --parity-seed 1 \
  --parity-seed 7 \
  --parity-seed 19 \
  --strict
```

Queue the larger CUDA Torch V0 overnight-style run only when the machine and
budget are appropriate:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/strategic_lane/muzero_overnight_${RUN_ID}"

env -u UV_PYTHON uv run --extra train \
  python -m mini_tft.tools.run_strategic_muzero_overnight \
  --out-dir "$OUT" \
  --seed 5000 \
  --cache-episodes 8192 \
  --cache-rows 262144 \
  --mcts-simulations 64 \
  --mcts-max-depth 8 \
  --mcts-rollout-steps 4 \
  --mcts-prior-mode heuristic \
  --train-epochs 64 \
  --train-learning-rate 0.001 \
  --train-batch-size 2048 \
  --hidden-size 256 \
  --device auto \
  --eval-episodes 512 \
  --parity-seed 0 \
  --parity-seed 1 \
  --parity-seed 7 \
  --parity-seed 19 \
  --codex-allowance-source user-waived \
  --codex-five-hour-window-remaining user-waived \
  --codex-weekly-usage user-waived \
  --codex-allowance-decision continue \
  --strict
```

## Evidence Paths

Speed:

```text
artifacts/strategic_lane/puffer4_speedup_paper/metrics.json
artifacts/strategic_lane/puffer4_speedup_paper/final_report.md
artifacts/strategic_lane/puffer4_speedup_paper/decision.md
/mnt/ssd2/Projects/TFT-zero-strategic-muzero-overnight/artifacts/strategic_lane/batched_cuda_mcts_20260701T030801Z/metrics.json
/mnt/ssd2/Projects/TFT-zero-strategic-muzero-overnight/artifacts/strategic_lane/batched_cuda_mcts_20260701T030801Z/final_report.md
```

Trained model:

```text
/mnt/ssd2/Projects/TFT-zero-strategic-muzero-overnight/artifacts/strategic_lane/muzero_overnight_20260630T174428Z/metrics.json
/mnt/ssd2/Projects/TFT-zero-strategic-muzero-overnight/artifacts/strategic_lane/muzero_overnight_20260630T174428Z/final_report.md
/mnt/ssd2/Projects/TFT-zero-strategic-muzero-overnight/artifacts/strategic_lane/muzero_overnight_20260630T174428Z/train_torch/metrics.json
/mnt/ssd2/Projects/TFT-zero-strategic-muzero-overnight/artifacts/strategic_lane/muzero_overnight_20260630T174428Z/train_torch/strategic_muzero_torch.pt
```

Readiness and smaller smoke evidence:

```text
artifacts/strategic_lane/muzero_run_loop/metrics.json
artifacts/strategic_lane/muzero_run_loop/final_report.md
artifacts/strategic_lane/muzero_run_loop/train_smoke/metrics.json
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/metrics.json
artifacts/strategic_lane/playable_demo/metrics.json
```

## Paper-Agent Update Rule

When updating this README for a paper-writing agent:

1. Use artifact files as source evidence, not memory.
2. Keep only the two headline result categories near the top: speed and trained
   model.
3. Preserve artifact status labels exactly, especially `smoke_only`,
   `accepted_l3_speedup`, and gate verdicts.
4. State whether a model is tiny NumPy train smoke, CUDA Torch V0 supervised
   training, or full iterative MuZero. The current trained checkpoint is CUDA
   Torch V0 supervised training, not full MuZero.
5. Run:

```bash
git diff --check README.md artifacts/paper_agent_context/README.md
```

## Claim Boundaries

- Toy strategic-lane results are simplified simulator results.
- `placement_proxy` is an elimination-timing bucket, not real TFT placement.
- Puffer speed evidence is throughput evidence, not policy-quality evidence.
- Native MCTS is simulator-backed search over the strategic rules, not learned
  MuZero dynamics.
- The trained Torch V0 checkpoint is supervised on simulator/MCTS cache targets.
  It is not full iterative MuZero self-play and is not model-backed search.
- The L3 native-root cache path injects checkpoint priors at the search root; it
  does not yet call a checkpoint value model at native search leaves.
- Full MuZero claims require model-backed search/reanalysis, legal masks,
  auditable cache rows, deterministic seeds, and baseline comparisons under
  [docs/QUALITY_GATE.md](docs/QUALITY_GATE.md).
