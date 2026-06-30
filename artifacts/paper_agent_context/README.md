# TFT-Zero Paper Agent Context

This packet is the handoff for a separate coding/report agent writing the
paper/report. It is intentionally a manifest, not a copy of every artifact.
Large checkpoints and JSONL traces stay in their original artifact directories.

## Safe Current Claim

TFT-Zero currently has a clean strategic-lane scaffold:

- a simplified Markov TFT-shaped simulator;
- a compact strategic action surface with legal masks;
- a native Python/NumPy Puffer-compatible vector path with scalar parity;
- smoke MuZero-style cache rows;
- a simulator-backed MCTS smoke baseline.

Do not claim full TFT fidelity, real ranked performance, current-patch MetaTFT
validity, full MuZero self-play, or PufferLib 4.0 Ocean/C training yet.

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

- Strategic rules: `src/mini_tft/strategic/core/actions.py`
- Strategic state: `src/mini_tft/strategic/core/state.py`
- Strategic transitions/reward: `src/mini_tft/strategic/core/rules.py`
- Strategic observations: `src/mini_tft/strategic/core/obs.py`
- Baselines: `src/mini_tft/strategic/adapters/baselines/policies.py`
- Puffer vector env: `src/mini_tft/strategic/adapters/puffer/vector_env.py`
- Puffer benchmark: `src/mini_tft/strategic/adapters/puffer/benchmark.py`
- Puffer wrapper/training entry: `src/mini_tft/rl/puffer_env.py`
- PPO trainer: `src/mini_tft/rl/train_puffer_ppo.py`
- MCTS planner: `src/mini_tft/strategic/adapters/mcts.py`
- MCTS smoke CLI: `src/mini_tft/tools/strategic_mcts_smoke.py`
- Strategic gate CLI: `src/mini_tft/tools/strategic_lane_gate.py`

## Claim-Grade Artifacts

### Strategic Lane Gate

- Main report: `artifacts/strategic_lane/final_report.md`
- Main metrics: `artifacts/strategic_lane/metrics.json`
- Main decision: `artifacts/strategic_lane/decision.md`
- Loop state/log: `artifacts/strategic_lane/loop-state.json`,
  `artifacts/strategic_lane/loop-run-log.md`

Key values from `artifacts/strategic_lane/metrics.json`:

- status: `smoke_only`
- heuristic mean placement proxy: `6.65625`
- heuristic death rate: `1.0`
- random mean placement proxy: `7.0`
- enemy model: `round_plus_rng_noise_pressure`

### Puffer Speed

- Metrics: `artifacts/strategic_lane/puffer_speed/metrics.json`
- Decision: `artifacts/strategic_lane/puffer_speed/decision.md`

Claim-grade values:

- benchmark kind: `strategic_native_puffer_vector`
- scalar-vs-batched semantic parity: `true`
- Puffer trainer available: `true`
- reference throughput: `15970.403511668637` steps/sec
- batched throughput: `179413.90385379703` steps/sec
- speedup: `11.23414970214809x`
- repeated runs: `3`
- decision: `pass`

This is saved environment/vector rollout throughput evidence from the legacy
Puffer-compatible wrapper path. It is not PPO learning quality evidence and it
is not PufferLib 4.0 Ocean/C trainer throughput.

### MuZero-Style Cache Smoke

- Rows: `artifacts/strategic_lane/muzero_cache/rows.jsonl`
- Metrics: `artifacts/strategic_lane/muzero_cache/metrics.json`
- Decision: `artifacts/strategic_lane/muzero_cache/decision.md`

Key values:

- rows: `128`
- observation dim: `38`
- action dim: `11`
- legal action rate: `1.0`
- status: `smoke_only`

This proves cache schema and legality, not trained MuZero quality.

### Playable Demo Payload

- Payload: `artifacts/strategic_lane/playable_demo/initial_payload.json`
- Metrics: `artifacts/strategic_lane/playable_demo/metrics.json`
- Decision: `artifacts/strategic_lane/playable_demo/decision.md`

Key values:

- stage label: `Stage 1-1`
- legal actions: `5`
- includes enemy strength: `true`
- includes placement proxy: `true`
- includes scenario score: `true`
- includes role item slots: `true`

Known limit: the browser route is not yet wired to this strategic payload.

### PPO Smoke

- Checkpoint: `artifacts/strategic_lane/ppo_smoke/strategic_native_puffer_smoke.pt`
- Manifest:
  `artifacts/strategic_lane/ppo_smoke/strategic_native_puffer_smoke.manifest.json`

Key values:

- env kind: `strategic`
- Puffer backend: `native`
- package `pufferlib`: `3.0.0`
- total steps: `128`
- updates: `1`
- completed episodes: `0`

This proves trainer wiring only.

### Simulator-Backed MCTS Smoke

- Metrics: `artifacts/strategic_lane/mcts_smoke/metrics.json`
- Paper table: `artifacts/strategic_lane/mcts_smoke/paper_table.md`
- Episodes: `artifacts/strategic_lane/mcts_smoke/episodes.jsonl`
- Decisions: `artifacts/strategic_lane/mcts_smoke/decisions.jsonl`

Key values:

- status: `smoke_only`
- prior mode: `heuristic`
- simulations tested: `8`, `16`, `32`
- best by mean placement: `heuristic`
- best by mean reward: `mcts_32`
- best by scenario score: `mcts_32`
- `mcts_32` mean placement proxy: `6.875`
- `mcts_32` mean final round: `17.0`
- `mcts_32` mean reward: `-0.4211194802858562`
- `mcts_32` simulations/sec: `2703.8318821742355`

This is simulator-backed MCTS, not learned model-backed MuZero search.

### Overnight MCTS Run

Current pending path:

- `artifacts/strategic_lane/mcts_overnight_20260630T203555/`

Current command was launched with:

```bash
env -u UV_PYTHON PYTHONUNBUFFERED=1 uv run python -m mini_tft.tools.strategic_mcts_smoke \
  --out-dir artifacts/strategic_lane/mcts_overnight_20260630T203555 \
  --episodes 1024 \
  --simulations 16 32 64 128 256 \
  --max-depth 10 \
  --rollout-steps 8 \
  --prior-mode heuristic \
  --seed 9200
```

Use this only after `metrics.json` appears in that directory.

## PufferLib 4.0 Minimal Port Lane

Current repo state:

- `pyproject.toml` pins `pufferlib` to PufferLib branch/rev `4.0`
  (`9a4eb87e6b58c0aa5f22affefb65c7006d384972`).
- Local installed `pufferlib` reports version `4.0`.
- PufferLib 4.0 no longer exposes the older Python
  `pufferlib.PufferEnv`/`pufferlib.emulation` surface used by the legacy PPO
  wrappers.
- The current live 4.0 route is the strategic Ocean-style C scaffold.

Remote 4.0 refs checked:

- PufferLib `4.0`: `9a4eb87e6b58c0aa5f22affefb65c7006d384972`
- PufferTank `4.0`: `4e4ad27bd15a56743325f3e4968b6d9120e0661a`

The repo contains a standalone strategic Ocean-style scaffold:

```text
src/mini_tft/strategic/ocean/strategic_tft.h
src/mini_tft/strategic/ocean/strategic_tft.c
src/mini_tft/strategic/ocean/binding.c
config/strategic_tft.ini
src/mini_tft/tools/benchmark_puffer4_ocean.py
```

Commit-smoke command:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.benchmark_puffer4_ocean \
  --envs 512 --steps 100000 \
  --out-dir artifacts/strategic_lane/puffer4_ocean_commit_smoke
```

Observed commit-smoke result:

- backend: `pufferlib_4_ocean_standalone`
- envs: `512`
- steps: `100000`
- steps/sec: `4070980.81`
- mean placement: `6.763175`

This is a standalone C env-loop benchmark, not full Puffer 4 trainer evidence.

To build inside a full PufferLib/Ocean checkout:

```bash
git clone --branch 4.0 https://github.com/PufferAI/PufferLib.git ../TFT-zero-puffer4
cd ../TFT-zero-puffer4
mkdir -p ocean/strategic_tft
cp ../TFT-zero/src/mini_tft/strategic/ocean/strategic_tft.* ocean/strategic_tft/
cp ../TFT-zero/src/mini_tft/strategic/ocean/binding.c ocean/strategic_tft/
cp ../TFT-zero/config/strategic_tft.ini config/strategic_tft.ini
bash build.sh strategic_tft --local
```

Use the Python strategic simulator as the parity oracle before using 4.0 numbers
in the paper. Minimum parity test: fixed seeds, fixed action prefixes, compare
round, hp, gold, level, reward, terminal, legal mask, and placement proxy.

Do not replace the legacy 11.23x vector-wrapper claim with a full PufferLib 4.0
trainer claim until the Ocean lane produces repeated trainer benchmark artifacts.

## Suggested Paper Framing

Call the system `TFT-Zero`.

Suggested abstract-level wording:

> TFT-Zero is a simplified, Markov, TFT-shaped strategic planning environment
> designed to test whether batched reinforcement-learning infrastructure and
> simulator-backed tree search can produce useful planning evidence on a compact
> auto-battler abstraction.

Suggested contributions:

1. A Markov strategic simulator that removes full multi-agent lobby complexity
   while preserving TFT-shaped decisions: leveling, rolling, buying upgrades,
   fielding board strength, economy, pressure, and item-slam abstractions.
2. A legal-masked strategic action surface shared by baseline policies, PPO,
   Puffer-compatible vectorization, web payloads, cache export, and MCTS.
3. A repeated scalar-vs-batched Puffer speed benchmark showing `11.23x` rollout
   throughput improvement with semantic parity.
4. A smoke MuZero-style data cache and simulator-backed MCTS baseline over the
   same action surface.

Suggested limitations:

- Not full TFT.
- Not current-patch or ranked-player evidence.
- Placement is an elimination-timing proxy.
- MCTS currently uses simulator transitions, not learned dynamics.
- PufferLib 4.0 Ocean/C currently has standalone env-loop evidence, not full
  trainer evidence.
