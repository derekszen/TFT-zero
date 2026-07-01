# TFT-Zero Paper Agent Context

This packet is the handoff for a separate paper/report agent, including an
OpenAI Prism paper-writing pass. It is intentionally a manifest plus claim
guide, not a copy of every artifact. Large checkpoints, JSONL traces, and build
outputs stay in their original artifact directories.

## Safe Current Claim

TFT-Zero currently has a clean strategic lane for fast TFT-shaped planning
experiments:

- a simplified Markov strategic simulator with round-based enemy pressure;
- an 11-action legal-masked strategic action surface;
- Python, native C++, and PufferLib 4.0 Ocean-style C parity evidence;
- PufferLib 4.0 C/Ocean standalone and CUDA trainer throughput smoke evidence;
- a 1024-row MuZero-style cache with MCTS visit-policy targets;
- a tiny policy/value/dynamics train smoke over that cache;
- a read-only verifier accepting the MuZero-readiness loop.

Do not claim full TFT fidelity, real ranked performance, current-patch MetaTFT
validity, full MuZero self-play, or final production MuZero training. The train
piece here is a tiny linear policy/value/dynamics smoke model, not a full
learned recurrent MuZero model with model-backed search training.

## Latest Evidence Addendum, 2026-07-01

Use this addendum before older goal1 sections when writing the paper.

The requested report-evidence path
`artifacts/strategic_lane/muzero_overnight_20260630T174428Z` was not present in
the inspected checkout. The available full strategic MuZero overnight launch
evidence is:

```text
artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/metrics.json
artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/final_report.md
artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/decision.md
artifacts/strategic_lane/muzero_overnight_20260630T155605Z_blocked/verifier/metrics.json
```

Result:

- status: `blocked`;
- objective: prepare and queue the overnight full strategic MuZero-style run
  after preflight;
- preflight status: `pass`;
- preflight verifier: `ACCEPT`, `21/21` checks;
- cache rows: `1024`;
- observation/action shape: `38 x 11`;
- legal action rate: `1.0`;
- MCTS target rate: `1.0`;
- parity matrix: `48/48` checks passed;
- checkpoint exists: `true`;
- blocker: no production/full strategic MuZero trainer entry point exists yet.

This is strong preflight evidence for queue readiness, but it is not a final
training result.

The latest complete MuZero-readiness loop artifact is:

```text
artifacts/strategic_lane/muzero_run_loop/metrics.json
artifacts/strategic_lane/muzero_run_loop/final_report.md
artifacts/strategic_lane/muzero_run_loop/decision.md
artifacts/strategic_lane/muzero_run_loop/cache/rows.jsonl
artifacts/strategic_lane/muzero_run_loop/train_smoke/train_smoke.npz
artifacts/strategic_lane/muzero_run_loop/verifier/metrics.json
```

Key values:

- status: `pass`;
- rows: `1024`;
- observation/action shape: `38 x 11`;
- legal action rate: `1.0`;
- MCTS target rate: `1.0`;
- policy target valid rate: `1.0`;
- illegal policy mass max: `0.0`;
- value target finite rate: `1.0`;
- search speed: `160.59` decisions/sec;
- train total loss: `4.892351` to `2.340075`;
- train policy loss: `2.396928` to `2.043442`;
- train value loss: `2.372969` to `0.200895`;
- train dynamics loss: `0.122454` to `0.095737`;
- verifier: `ACCEPT`, `21/21` checks.

The latest native simulator-backed MCTS overnight artifact is:

```text
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/metrics.json
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/paper_table.md
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/episodes.jsonl
artifacts/strategic_lane/mcts_native_overnight_20260630T235353/decisions.jsonl
```

Key values:

- backend: `native`;
- status: `smoke_only`;
- episodes per policy: `65,536`;
- simulations: `64`, `128`, `256`, `512`, `1024`;
- best by mean placement: `heuristic`;
- best by mean reward: `mcts_1024`;
- best by mean scenario score: `mcts_1024`;
- `mcts_1024` reward: `-1.339` vs heuristic `-2.187`;
- `mcts_1024` scenario score: `0.237` vs heuristic `0.214`;
- `mcts_1024` throughput: `446,638` simulations/sec;
- all policies had illegal action count `0`.

Safe paper wording:

> The latest overnight native simulator-backed MCTS smoke shows reward and
> scenario-score gains at higher simulation counts, with `mcts_1024` giving the
> best reward and scenario score over 65,536 episodes per policy. It does not
> improve the placement proxy over the heuristic baseline and remains
> simulator-backed search, not learned MuZero dynamics.

## Current Worktrees And Provenance

Main repo inspected:

```text
/mnt/ssd2/Projects/TFT-zero
```

Current main repo state at inspection:

- branch/commit: `main` at `d810639`;
- local dirty/untracked MuZero loop files are present;
- Puffer 4 speed report provenance records repo commit `d503835`;
- PufferLib 4 speed report records PufferLib commit
  `9a4eb87e6b58c0aa5f22affefb65c7006d384972`.

PufferLib 4 trainer-build checkout inspected:

```text
/tmp/PufferLib-4.0
```

It is on branch `4.0` at commit
`9a4eb87e6b58c0aa5f22affefb65c7006d384972`, with untracked
`ocean/strategic_tft/`, `config/strategic_tft.ini`, built `strategic_tft`, and
`uv.lock`. The persistent checkout at `/mnt/ssd2/Projects/pufferlib` is also on
branch `4.0` at the same upstream commit, but does not currently contain the
untracked `strategic_tft` environment files.

## Read First

Read these in order:

1. `README.md`
2. `AGENTS.md`
3. `docs/ARCHITECTURE.md`
4. `docs/STRATEGIC_LANE.md`
5. `docs/QUALITY_GATE.md`
6. `docs/TRAINING.md`

Historical MuZero-stage docs are archived under `docs/archive/`. Use them as
history only unless a result has been rerun under the current strategic lane.

## Core Code To Cite

Strategic Python rules:

- `src/mini_tft/strategic/core/actions.py`
- `src/mini_tft/strategic/core/state.py`
- `src/mini_tft/strategic/core/rules.py`
- `src/mini_tft/strategic/core/obs.py`

Strategic adapters and baselines:

- `src/mini_tft/strategic/adapters/baselines/policies.py`
- `src/mini_tft/strategic/adapters/mcts.py`
- `src/mini_tft/strategic/adapters/muzero_cache/export.py`
- `src/mini_tft/strategic/adapters/puffer/vector_env.py`
- `src/mini_tft/strategic/adapters/puffer/benchmark.py`
- `src/mini_tft/strategic/adapters/web_demo/payload.py`

PufferLib 4.0 Ocean-style C scaffold:

- `src/mini_tft/strategic/ocean/strategic_tft.h`
- `src/mini_tft/strategic/ocean/strategic_tft.c`
- `src/mini_tft/strategic/ocean/binding.c`
- `config/strategic_tft.ini`
- `src/mini_tft/tools/benchmark_puffer4_ocean.py`

MuZero-readiness tools:

- `src/mini_tft/tools/generate_strategic_muzero_cache.py`
- `src/mini_tft/tools/train_strategic_muzero_smoke.py`
- `src/mini_tft/tools/strategic_muzero_loop.py`
- `src/mini_tft/tools/strategic_muzero_run_loop.py`
- `src/mini_tft/tools/strategic_mcts_smoke.py`
- `src/mini_tft/tools/strategic_parity_matrix.py`
- `src/mini_tft/tools/strategic_lane_gate.py`

Legacy Puffer wrapper/training entry points, for historical comparison only:

- `src/mini_tft/rl/puffer_env.py`
- `src/mini_tft/rl/train_puffer_ppo.py`

## Headline Results

### PufferLib 4.0 C/Ocean Throughput Smoke

Primary artifact packet:

```text
artifacts/strategic_lane/puffer4_speedup_paper/metrics.json
artifacts/strategic_lane/puffer4_speedup_paper/decision.md
artifacts/strategic_lane/puffer4_speedup_paper/final_report.md
```

Key values:

| Path | Throughput | Baseline | Speedup |
| --- | ---: | --- | ---: |
| Python scalar strategic heuristic loop | `25,259.39` steps/sec | reference | `1.00x` |
| PufferLib 4 Ocean C standalone loop | `3,755,013.36` steps/sec | Python scalar | `148.66x` |
| PufferLib 4 CUDA trainer smoke | `9,017,194.00` agent steps/sec | Python scalar | `356.98x` |

Trainer-smoke details:

- agent steps: `262,144`;
- epoch: `1`;
- policy parameters: `55,600`;
- env count reported by trainer: `6,236`;
- env placement proxy: `6.999518871307373`;
- env final round: `14.109685897827148`;
- loss total: `0.6375824809074402`;
- GPU: NVIDIA GeForce RTX 5090 D;
- VRAM used: `3.07 GiB` of `31.35 GiB`;
- artifact log:
  `artifacts/strategic_lane/puffer4_train_smoke/logs/strategic_tft/1782826826743.json`;
- checkpoint:
  `artifacts/strategic_lane/puffer4_train_smoke/checkpoints/strategic_tft/1782826826743/0000000000262144.bin`.

Standalone C/Ocean build-smoke artifacts:

```text
artifacts/strategic_lane/puffer4_ocean/metrics.json
artifacts/strategic_lane/puffer4_ocean_after_rng_fix/metrics.json
artifacts/strategic_lane/puffer4_ocean_build/metrics.json
artifacts/strategic_lane/puffer4_ocean_commit_smoke/metrics.json
```

Build transcript artifact:

```text
artifacts/strategic_lane/puffer4_ocean_build/puffer_build_stdout.txt
artifacts/strategic_lane/puffer4_ocean_build/puffer_build_stderr.txt
```

That transcript shows raylib download, `strategic_tft` compilation, and
`Built: ./strategic_tft`; stderr is empty. The speed report also records that
local setup needed temporary linker workarounds for `ccache`, `libnccl.so`, and
`libomp5.so` names.

Status and claim boundary:

- `puffer4_speedup_paper` status is `smoke_only`.
- This is valid PufferLib 4.0 C/Ocean and CUDA trainer infrastructure evidence.
- It is not repeated-run variance evidence and not final policy-quality
  evidence.
- The Puffer speed claim still needs matched repeated trainer runs and verifier
  acceptance before promotion.

Safe wording:

> In a smoke benchmark on an RTX 5090 D workstation, the PufferLib 4 CUDA
> trainer path for the strategic_tft Ocean environment reached 9.0M agent
> steps/s for a 262k-step run, approximately 357x the scalar Python strategic
> heuristic loop measured at 25.3k steps/s. The optimized standalone C
> environment loop reached 3.76M steps/s, approximately 149x the scalar Python
> loop.

Unsafe wording:

- Do not call this a final MuZero result.
- Do not claim learned policy improvement from this speed benchmark.
- Do not claim real TFT placement or full lobby performance.
- Do not claim stable throughput without repeated matched runs.

### MuZero MCTS Cache Goal 1

Primary artifacts:

```text
artifacts/strategic_lane/muzero_mcts_cache_goal1/rows.jsonl
artifacts/strategic_lane/muzero_mcts_cache_goal1/metrics.json
artifacts/strategic_lane/muzero_mcts_cache_goal1/decision.md
```

Key values:

- rows: `1024`;
- observation dim: `38`;
- action dim: `11`;
- terminal rows: `28`;
- legal action rate: `1.0`;
- MCTS target rows: `1024`;
- MCTS target rate: `1.0`;
- policy target valid rate: `1.0`;
- policy target sum error max: `0.0`;
- illegal policy mass max: `0.0`;
- value target finite rate: `1.0`;
- mean reward: `-0.060785549375892395`;
- mean value target: `-1.4468147436206058`;
- deterministic checksum:
  `caacce04f696cc831b398f5064cce93ecae0287c7734647d2a9550b4c29bafca`;
- fixed-seed reproducible: `true`;
- search backend: Python simulator-backed MCTS;
- simulations per decision: `16`;
- max depth: `10`;
- rollout steps: `6`;
- prior mode: `heuristic`;
- search speed: `160.10` decisions/sec;
- illegal action count: `0`;
- status: `smoke_only`.

The row schema includes:

```text
index, observation, legal_mask, action, reward, next_observation, done,
policy_target, value_target, metadata
```

This proves cache schema, legal masks, MCTS visit-policy targets, value targets,
and deterministic generation. It does not prove production MuZero quality.

### Tiny MuZero Train Smoke Goal 1

Primary artifacts:

```text
artifacts/strategic_lane/muzero_train_smoke_goal1/metrics.json
artifacts/strategic_lane/muzero_train_smoke_goal1/decision.md
artifacts/strategic_lane/muzero_train_smoke_goal1/train_smoke.npz
```

Key values:

- rows: `1024`;
- observation dim: `38`;
- action dim: `11`;
- epochs: `24`;
- learning rate: `0.03`;
- losses finite: `true`;
- checkpoint exists: `true`;
- checkpoint size: about `20.5 KiB`;
- total loss: `4.881350231137606` to `2.3337804243664397`;
- policy loss: `2.3918509247445505` to `2.0404986329927564`;
- value loss: `2.369190318319924` to `0.1990082376698854`;
- dynamics loss: `0.12030898807313187` to `0.09427355370379796`;
- elapsed: `0.0366` sec;
- status: `smoke_only`.

Important caveat:

- This is a tiny linear policy/value/dynamics smoke model.
- Dynamics predicts one-step observations directly from observation plus action.
- This is not a production recurrent MuZero trainer and not model-backed MCTS.

### MuZero-Readiness Loop Goal 1

Primary artifacts:

```text
artifacts/strategic_lane/muzero_loop_goal1/metrics.json
artifacts/strategic_lane/muzero_loop_goal1/decision.md
artifacts/strategic_lane/muzero_loop_goal1/final_report.md
artifacts/strategic_lane/muzero_loop_goal1/loop-state.json
artifacts/strategic_lane/muzero_loop_goal1/loop-run-log.md
artifacts/strategic_lane/muzero_loop_goal1/verifier/metrics.json
artifacts/strategic_lane/muzero_loop_goal1/verifier/decision.md
```

Verifier result:

- verdict: `ACCEPT`;
- status: `pass`;
- checks: `21/21`;
- failed: `0`.

Loop summary values:

- deliverable: `muzero_cache`;
- action surface: `strategic_11_macro`;
- observation/action dimensions: `38 x 11`;
- min cache rows required: `1024`;
- cache rows: `1024`;
- MCTS target rows: `1024`;
- legal action rate: `1.0`;
- policy target valid rate: `1.0`;
- illegal policy mass max: `0.0`;
- value target finite rate: `1.0`;
- train smoke ran: `true`;
- train losses finite: `true`;
- checkpoint exists: `true`;
- baseline comparison present: `true`;
- baseline policies: `heuristic`, `random`;
- parity matrix status: `pass`;
- parity matrix checks: `48/48`;
- parity backends: Python, native C++, Ocean C.

Meaning:

- This loop is queue-ready/readiness evidence for the next scale-up experiment.
- It verifies cache/search/train/baseline/parity evidence.
- It does not train production MuZero and does not make a final quality claim.

### Baseline And Parity Context

Baseline artifact:

```text
artifacts/strategic_lane/policy_eval/metrics.json
```

Key baseline values:

- policies: `heuristic`, `random`;
- episodes per policy: `64`;
- overall illegal actions: `0`;
- heuristic mean placement proxy: `7.0`;
- heuristic mean scenario score: `0.2157147692462813`;
- random mean placement proxy: `7.375`;
- random mean scenario score: `0.18483028798353365`;
- best by mean placement: `heuristic`;
- best by mean scenario score: `heuristic`;
- status: `smoke_only`.

Parity artifact:

```text
artifacts/strategic_lane/parity_matrix/metrics.json
```

Key parity values:

- status: `pass`;
- summary: `48/48` checks passed;
- backends: `python`, `native_cpp`, `ocean_c`;
- seeds: `0`, `1`, `7`, `19`;
- scenarios: reset, economy rounds, roll/buy/field, level tempo, illegal
  actions, terminal pressure;
- Ocean C compared state, observation, legal mask, reward, terminal,
  placement proxy, and scenario score.

### Historical Legacy Puffer-Compatible Vector Claim

Legacy Puffer-compatible artifacts:

```text
artifacts/strategic_lane/puffer_speed/metrics.json
artifacts/strategic_lane/puffer_speed/decision.md
```

Key values:

- benchmark kind: `strategic_native_puffer_vector`;
- scalar-vs-batched semantic parity: `true`;
- Puffer trainer available: `true`;
- reference throughput: `15,970.403511668637` steps/sec;
- batched throughput: `179,413.90385379703` steps/sec;
- speedup: `11.23414970214809x`;
- repeated runs: `3`;
- decision: `pass`.

This is saved environment/vector rollout throughput from the older
Puffer-compatible wrapper path. It is not the new PufferLib 4.0 Ocean/C trainer
result.

## Procedure To Reproduce Current Evidence

### Build And Benchmark The Standalone Ocean C Env

From the TFT-zero repo:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.benchmark_puffer4_ocean \
  --out-dir artifacts/strategic_lane/puffer4_ocean \
  --envs 4096 \
  --steps 10000000
```

Short post-RNG-fix smoke:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.benchmark_puffer4_ocean \
  --out-dir artifacts/strategic_lane/puffer4_ocean_after_rng_fix \
  --envs 1024 \
  --steps 100000
```

The tool compiles:

```bash
cc -O3 -std=c11 -D_POSIX_C_SOURCE=199309L \
  -I src/mini_tft/strategic/ocean \
  src/mini_tft/strategic/ocean/strategic_tft.c \
  -lm \
  -o artifacts/strategic_lane/<out>/strategic_tft_bench
```

The C benchmark allocates many single-agent `StrategicTFT` envs, initializes
observations/rewards/terminals/action masks, steps a C heuristic policy, and
reports JSON throughput plus placement/score aggregates.

### Build Inside PufferLib 4.0

Inside a PufferLib 4.0 checkout:

```bash
git clone --branch 4.0 https://github.com/PufferAI/PufferLib.git ../TFT-zero-puffer4
cd ../TFT-zero-puffer4
mkdir -p ocean/strategic_tft
cp ../TFT-zero/src/mini_tft/strategic/ocean/strategic_tft.* ocean/strategic_tft/
cp ../TFT-zero/src/mini_tft/strategic/ocean/binding.c ocean/strategic_tft/
cp ../TFT-zero/config/strategic_tft.ini config/strategic_tft.ini
bash build.sh strategic_tft --local
```

The inspected trainer-build checkout used:

```text
/tmp/PufferLib-4.0
```

The Puffer binding exposes:

- `OBS_SIZE = 38`;
- `NUM_ATNS = 1`;
- action size `11`;
- `MY_ACTION_MASK = 11`;
- env struct `StrategicTFT`;
- `my_init`, `my_log`, `c_reset`, `c_step`, `c_render`, `c_close`.

### Run PufferLib 4 CUDA Trainer Smoke

From `/tmp/PufferLib-4.0`:

```bash
env -u UV_PYTHON uv run --all-extras python -m pufferlib.pufferl train strategic_tft \
  --train.total-timesteps 262144 \
  --checkpoint-dir /mnt/ssd2/Projects/TFT-zero/artifacts/strategic_lane/puffer4_train_smoke/checkpoints \
  --log-dir /mnt/ssd2/Projects/TFT-zero/artifacts/strategic_lane/puffer4_train_smoke/logs \
  --checkpoint-interval 1000000000
```

The relevant config file is:

```text
config/strategic_tft.ini
```

Current config highlights:

- `env_name = strategic_tft`;
- `vec.total_agents = 4096`;
- `vec.backend = Serial`;
- policy hidden size `128`;
- policy layers `1`;
- default total timesteps `100_000_000`, overridden to `262144` for smoke;
- `gamma = 0.97`;
- `learning_rate = 0.005`;
- `minibatch_size = 32768`;
- `ent_coef = 0.01`.

### Generate MuZero MCTS Cache

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.generate_strategic_muzero_cache \
  --out-dir artifacts/strategic_lane/muzero_mcts_cache_goal1 \
  --episodes 64 \
  --max-rows 1024 \
  --seed 1000 \
  --simulations 16 \
  --max-depth 10 \
  --rollout-steps 6 \
  --prior-mode heuristic \
  --strict
```

This uses simulator-backed MCTS targets. It writes rows with observation, legal
mask, selected action, reward, next observation, done flag, visit-policy target,
value target, and metadata.

### Train Tiny Policy/Value/Dynamics Smoke

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.train_strategic_muzero_smoke \
  --rows artifacts/strategic_lane/muzero_mcts_cache_goal1/rows.jsonl \
  --out-dir artifacts/strategic_lane/muzero_train_smoke_goal1 \
  --epochs 24 \
  --learning-rate 0.03 \
  --seed 2000 \
  --strict
```

This proves a minimal train pipeline can consume the cache and move finite
losses. It does not implement production MuZero.

### Verify Queue-Ready MuZero Evidence

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_parity_matrix \
  --out-dir artifacts/strategic_lane/parity_matrix \
  --strict

env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_muzero_loop \
  --out-dir artifacts/strategic_lane/muzero_loop_goal1 \
  --cache-metrics artifacts/strategic_lane/muzero_mcts_cache_goal1/metrics.json \
  --cache-rows-jsonl artifacts/strategic_lane/muzero_mcts_cache_goal1/rows.jsonl \
  --train-metrics artifacts/strategic_lane/muzero_train_smoke_goal1/metrics.json \
  --baseline-metrics artifacts/strategic_lane/policy_eval/metrics.json \
  --parity-metrics artifacts/strategic_lane/parity_matrix/metrics.json \
  --min-cache-rows 1024 \
  --strict
```

The verifier accepts only when rows, masks, MCTS targets, finite value targets,
determinism, train smoke, checkpoint, baselines, and parity are present.

## Suggested Paper Framing

Call the system `TFT-Zero`.

Suggested abstract-level wording:

> TFT-Zero is a simplified, Markov, TFT-shaped strategic planning environment
> designed to test whether fast C/Puffer rollouts and simulator-backed search can
> support legal-masked MuZero-style data generation on a compact auto-battler
> abstraction.

Suggested contributions:

1. A Markov strategic simulator that removes full multi-agent lobby complexity
   while preserving TFT-shaped decisions: leveling, rolling, buying upgrades,
   fielding board strength, economy, pressure, and item-slam abstractions.
2. A legal-masked 11-action strategic surface shared by baseline policies,
   Python simulation, native C++/Ocean parity checks, cache export, and MCTS.
3. A PufferLib 4.0 Ocean/C path that compiles inside PufferLib and reaches
   multi-million-step smoke throughput, including a 9.0M agent steps/sec CUDA
   trainer smoke on an RTX 5090 D workstation.
4. A deterministic 1024-row MuZero-style cache with legal masks, MCTS
   visit-policy targets, value targets, and a tiny finite-loss
   policy/value/dynamics train smoke.
5. A read-only verifier gate that accepts the current cache/search/train/
   baseline/parity bundle as queue-ready for the next scale-up experiment.

Suggested limitations:

- Not full TFT.
- Not current-patch or ranked-player evidence.
- Placement is an elimination-timing proxy, not real lobby placement.
- Puffer throughput is smoke-only until repeated matched trainer runs and
  verifier acceptance are added.
- MCTS targets are simulator-backed, not learned-dynamics-backed.
- The current train smoke is tiny linear supervised fitting, not production
  recurrent MuZero training.
- No final learned-policy quality claim exists yet.

## Claim Boundary For The Overnight/Scale-Up Run

The current artifact bundle supports the next scale-up experiment:

```text
PufferLib 4.0 C/Ocean fast stepping
  -> simulator-backed MCTS target generation
  -> deterministic MuZero-style cache rows
  -> tiny finite-loss train smoke
  -> queue-ready verifier acceptance
```

It does not support the final claim:

```text
production recurrent MuZero + model-backed search + learned policy improvement
```

Paper language should say "smoke", "readiness", "infrastructure", and
"queue-ready for scale-up" unless a later artifact adds repeated trainer
benchmarks and learned-policy comparisons under `docs/QUALITY_GATE.md`.
