# TFT-Zero

<p align="center">
  <img src="https://i.redd.it/7musegquo5kf1.gif" alt="Teamfight Tactics gameplay reference" width="900">
</p>

TFT-Zero is a compact research scaffold for testing **TFT-shaped strategic
planning** with deterministic simulator rules, legal action masks, fast
rollouts, PufferLib 4.0 Ocean-style C infrastructure, and simulator-backed
MCTS.

This is not a Riot-accurate TFT clone, a ranked-player claim, a current-patch
planner, or a completed MuZero system. The active lane is a simplified Markov
strategic simulator intended for controlled planning and throughput
experiments.

Current handoff material:

- [Paper/report agent context](artifacts/paper_agent_context/README.md)
- [Paper draft in OpenAI Prism](https://prism.openai.com/?u=db147769-d04d-4a0c-bd51-3123fc427703&pg=1&m=neurips_2026.tex&d=7)

## AI Agent Quick Onboarding

Use this section before making claims or queuing new runs.

Read these first:

1. [AGENTS.md](AGENTS.md)
2. [docs/STRATEGIC_LANE.md](docs/STRATEGIC_LANE.md)
3. [docs/QUALITY_GATE.md](docs/QUALITY_GATE.md)
4. [docs/TRAINING.md](docs/TRAINING.md)
5. [artifacts/paper_agent_context/README.md](artifacts/paper_agent_context/README.md)

Current report evidence in this checkout starts here:

- `artifacts/strategic_lane/muzero_run_loop/metrics.json`
- `artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/final_report.md`
- `artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/metrics.json`
- `artifacts/strategic_lane/mcts_native_overnight_20260630T235353/metrics.json`
- `artifacts/strategic_lane/puffer4_speedup_paper/metrics.json`

The requested path
`artifacts/strategic_lane/muzero_overnight_20260630T174428Z` was not present
when this README was refreshed. The available full-MuZero overnight launch
evidence is
`artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked`, which
passed preflight but stopped because no production/full strategic MuZero trainer
entry point exists yet.

Queue a scaffolded MuZero-readiness run:

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
  --codex-allowance-source user \
  --codex-5h-window-remaining ample \
  --codex-weekly-usage ample \
  --codex-allowance-decision continue \
  --parity-seed 0 \
  --parity-seed 1 \
  --parity-seed 7 \
  --parity-seed 19 \
  --strict
```

This regenerates the cache, Python MCTS targets, tiny linear train smoke,
baseline eval, parity matrix, quality gate, and verifier. It is a readiness
run, not production MuZero training.

Queue a scaffolded PufferLib 4.0 Ocean C speed smoke:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="artifacts/strategic_lane/puffer4_ocean_${RUN_ID}"

env -u UV_PYTHON uv run python -m mini_tft.tools.benchmark_puffer4_ocean \
  --out-dir "$OUT" \
  --envs 4096 \
  --steps 10000000
```

The saved CUDA trainer smoke was run from a PufferLib 4.0 checkout, not from
the repo root:

```bash
cd /tmp/PufferLib-4.0
env -u UV_PYTHON uv run --all-extras python -m pufferlib.pufferl train strategic_tft \
  --train.total-timesteps 262144 \
  --checkpoint-dir /mnt/ssd2/Projects/TFT-zero/artifacts/strategic_lane/puffer4_train_smoke/checkpoints \
  --log-dir /mnt/ssd2/Projects/TFT-zero/artifacts/strategic_lane/puffer4_train_smoke/logs \
  --checkpoint-interval 1000000000
```

When updating results for a paper-writing agent:

1. Treat `metrics.json`, `final_report.md`, and `decision.md` as source
   evidence, not memory.
2. Update both this README and
   `artifacts/paper_agent_context/README.md` with exact paths, dates, statuses,
   row counts, speeds, loss deltas, and verifier verdicts.
3. Preserve labels such as `smoke_only`, `blocked`, and `pass`; do not promote
   a result beyond its artifact status.
4. For MuZero evidence, always say whether the model is the tiny train smoke or
   a full recurrent learned MuZero trainer. The current full trainer is
   missing.
5. Run `git diff --check README.md artifacts/paper_agent_context/README.md`
   after documentation edits.

## Latest Evidence

As of 2026-07-01, the strongest current evidence is:

| Lane | Main artifact | Key result | Status |
| --- | --- | --- | --- |
| PufferLib 4.0 C/Ocean and CUDA trainer smoke | `artifacts/strategic_lane/puffer4_speedup_paper/metrics.json` | Python scalar `25,259.39` steps/s; Ocean C standalone `3,755,013.36` steps/s (`148.66x`); CUDA trainer smoke `9,017,194` agent steps/s (`356.98x`) | `smoke_only` |
| MuZero-readiness loop | `artifacts/strategic_lane/muzero_run_loop/metrics.json` | `1024` cache rows, obs/action shape `38 x 11`, legal action rate `1.0`, MCTS target rate `1.0`, verifier `ACCEPT` with `21/21` checks | `pass` for readiness |
| Tiny train smoke over MuZero-style cache | `artifacts/strategic_lane/muzero_run_loop/train_smoke/metrics.json` | total loss `4.892351 -> 2.340075`, policy loss `2.396928 -> 2.043442`, value loss `2.372969 -> 0.200895`, dynamics loss `0.122454 -> 0.095737` | `smoke_only` |
| Full strategic MuZero overnight launch | `artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/metrics.json` | preflight passed with verifier `ACCEPT`, but launch blocked because no production/full strategic MuZero trainer entry point exists | `blocked` |
| Native simulator-backed MCTS overnight | `artifacts/strategic_lane/mcts_native_overnight_20260630T235353/metrics.json` | `65,536` episodes/policy; best reward and scenario score from `mcts_1024`; `mcts_1024` reward `-1.339` vs heuristic `-2.187`; `446,638` simulations/s at 1024 sims | `smoke_only` |
| Python simulator-backed MCTS overnight | `artifacts/strategic_lane/mcts_overnight_20260630T203555/metrics.json` | `1,024` episodes/policy; best placement from `mcts_64`; best reward and scenario score from `mcts_256` | `smoke_only` |

Current claim-safe reading:

- PufferLib 4.0 infrastructure can compile and produce high-throughput smoke
  results for the strategic C/Ocean environment.
- The MuZero-readiness loop can generate deterministic cache/search/train-smoke
  artifacts with legal masks, policy targets, value targets, parity evidence,
  and verifier acceptance.
- Native simulator-backed MCTS improves reward and scenario score in the latest
  overnight run, but not the placement proxy.
- No final production MuZero trainer has been launched.

## Main Artifacts

Paper/report handoff:

```text
artifacts/paper_agent_context/README.md
```

Latest MuZero-readiness loop:

```text
artifacts/strategic_lane/muzero_run_loop/metrics.json
artifacts/strategic_lane/muzero_run_loop/final_report.md
artifacts/strategic_lane/muzero_run_loop/decision.md
artifacts/strategic_lane/muzero_run_loop/cache/rows.jsonl
artifacts/strategic_lane/muzero_run_loop/train_smoke/train_smoke.npz
artifacts/strategic_lane/muzero_run_loop/verifier/metrics.json
```

Blocked full-MuZero overnight launch:

```text
artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/metrics.json
artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/final_report.md
artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/decision.md
artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/verifier/metrics.json
```

Native MCTS overnight:

```text
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/metrics.json
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/paper_table.md
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/episodes.jsonl
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/decisions.jsonl
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/command.txt
```

PufferLib 4.0 speed smoke:

```text
artifacts/strategic_lane/puffer4_speedup_paper/metrics.json
artifacts/strategic_lane/puffer4_speedup_paper/final_report.md
artifacts/strategic_lane/puffer4_speedup_paper/decision.md
artifacts/strategic_lane/puffer4_train_smoke/
artifacts/strategic_lane/puffer4_ocean/
artifacts/strategic_lane/puffer4_ocean_after_rng_fix/
artifacts/strategic_lane/puffer4_ocean_commit_smoke/
```

Playable payload smoke:

```text
artifacts/strategic_lane/playable_demo/initial_payload.json
artifacts/strategic_lane/playable_demo/metrics.json
artifacts/strategic_lane/playable_demo/decision.md
```

Older artifacts under `docs/archive/` and older strategic-lane artifact
directories are historical unless explicitly rerun.

## Code Map

Core strategic simulator:

```text
src/mini_tft/strategic/core/actions.py
src/mini_tft/strategic/core/state.py
src/mini_tft/strategic/core/rules.py
src/mini_tft/strategic/core/obs.py
```

Strategic adapters:

```text
src/mini_tft/strategic/adapters/baselines/policies.py
src/mini_tft/strategic/adapters/mcts.py
src/mini_tft/strategic/adapters/muzero_cache/export.py
src/mini_tft/strategic/adapters/puffer/vector_env.py
src/mini_tft/strategic/adapters/puffer/benchmark.py
src/mini_tft/strategic/adapters/web_demo/payload.py
```

PufferLib 4.0 Ocean-style C scaffold:

```text
src/mini_tft/strategic/ocean/strategic_tft.h
src/mini_tft/strategic/ocean/strategic_tft.c
src/mini_tft/strategic/ocean/binding.c
config/strategic_tft.ini
src/mini_tft/tools/benchmark_puffer4_ocean.py
```

MuZero-readiness, MCTS, and gate tools:

```text
src/mini_tft/tools/generate_strategic_muzero_cache.py
src/mini_tft/tools/train_strategic_muzero_smoke.py
src/mini_tft/tools/strategic_muzero_loop.py
src/mini_tft/tools/strategic_muzero_run_loop.py
src/mini_tft/tools/strategic_mcts_smoke.py
src/mini_tft/tools/strategic_parity_matrix.py
src/mini_tft/tools/strategic_lane_gate.py
src/mini_tft/tools/judge_packet.py
```

Legacy Puffer wrapper/training entry points, for historical comparison only:

```text
src/mini_tft/rl/puffer_env.py
src/mini_tft/rl/train_puffer_ppo.py
```

## Reproduce Current Evidence

Use `env -u UV_PYTHON uv run ...` for repo commands.

Install and basic checks:

```bash
uv sync
env -u UV_PYTHON uv run pytest
env -u UV_PYTHON uv run ruff check
env -u UV_PYTHON uv run --all-extras pyright
git diff --check
```

Strategic lane gate:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_lane_gate
```

Native MCTS overnight command used for the latest artifact:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_mcts_smoke \
  --backend native \
  --out-dir artifacts/strategic_lane/mcts_native_overnight_20260630T235353 \
  --episodes 65536 \
  --simulations 64 128 256 512 1024 \
  --max-depth 10 \
  --rollout-steps 8 \
  --prior-mode heuristic \
  --seed 9500 \
  --strict
```

PufferLib 4.0 Ocean standalone smoke:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.benchmark_puffer4_ocean \
  --out-dir artifacts/strategic_lane/puffer4_ocean_commit_smoke \
  --envs 512 \
  --steps 100000
```

Build the strategic env inside a full PufferLib 4.0 checkout:

```bash
git clone --branch 4.0 https://github.com/PufferAI/PufferLib.git ../TFT-zero-puffer4
cd ../TFT-zero-puffer4
mkdir -p ocean/strategic_tft
cp ../TFT-zero/src/mini_tft/strategic/ocean/strategic_tft.* ocean/strategic_tft/
cp ../TFT-zero/src/mini_tft/strategic/ocean/binding.c ocean/strategic_tft/
cp ../TFT-zero/config/strategic_tft.ini config/strategic_tft.ini
bash build.sh strategic_tft --local
```

Use the Python strategic simulator as the parity oracle before reporting any
new PufferLib 4.0 trainer number.

## Claim Boundaries

- Toy and strategic-lane results are simplified simulator results.
- `placement_proxy` is an elimination-timing bucket, not real TFT placement.
- Puffer speed evidence is rollout/trainer-throughput evidence, not policy
  quality evidence.
- Native MCTS is simulator-backed search over the strategic rules, not learned
  MuZero dynamics.
- The current MuZero train step is a tiny linear policy/value/dynamics smoke,
  not full learned recurrent MuZero plus search training.
- Full MuZero claims require a production trainer entry point, model-backed
  search, legal masks, auditable cache rows, deterministic seeds, and baseline
  comparisons under [docs/QUALITY_GATE.md](docs/QUALITY_GATE.md).
