# TFT-Zero

<p align="center">
  <img src="Screenshots/tft_2019_board_bench_kym.jpg" alt="Teamfight Tactics board reference" width="900">
</p>

TFT-Zero is a compact research scaffold for **MiniTFT strategic planning**. The
active lane is a simplified, Markov, TFT-shaped simulator used for three
deliverables:

- MuZero-style V0 cache/data smoke artifacts.
- PufferLib or batched-simulator throughput evidence.
- A playable simulator/demo surface using the same strategic rules.

This repository is not a Riot-accurate TFT clone, a ranked-player claim, a
current-patch MetaTFT planner, or a completed MuZero system.

Report link:

- [Paper draft](https://prism.openai.com/?u=db147769-d04d-4a0c-bd51-3123fc427703&pg=1&m=neurips_2026.tex&d=7)

## Current Architecture

The active rules live under `src/mini_tft/strategic/`. The historical
`src/mini_tft/core/` package is a reference or adapter bridge only.

Core strategic simulator:

```text
src/mini_tft/strategic/core/actions.py
src/mini_tft/strategic/core/state.py
src/mini_tft/strategic/core/rules.py
src/mini_tft/strategic/core/obs.py
```

Adapters and surfaces wrap the canonical rules rather than forking them:

```text
src/mini_tft/strategic/adapters/baselines/
src/mini_tft/strategic/adapters/muzero_cache/
src/mini_tft/strategic/adapters/puffer/
src/mini_tft/strategic/adapters/web_demo/
src/mini_tft/strategic/adapters/mcts.py
src/mini_tft/strategic/ocean/
```

The strategic action surface is deliberately small and legal-masked:

```text
HOLD
LEVEL
ROLL
BUY_BEST_UPGRADE
BUY_BEST_SYNERGY
BUY_HIGHEST_COST
FIELD_STRONGEST
GREED_ECON
SLAM_CARRY_ITEM
SLAM_TANK_ITEM
SLAM_SUPPORT_ITEM
```

Enemy pressure is round-based. The lane does not model opponent boards,
opponent economy, shared pools, scouting, or self-play. `placement_proxy` is an
elimination-timing bucket, not real lobby placement; use `scenario_score` for
dense learning/debug signal.

## MuZero-Style V0 Harness

The current MuZero work is a **cache-supervised V0 harness**. It produces MCTS
target cache rows and a tiny policy/value/dynamics train smoke, then gates the
result with programmatic checks. It is not full iterative MuZero and should stay
labeled as smoke unless `docs/QUALITY_GATE.md` evidence says otherwise.

Main runner:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.strategic_muzero_run_loop \
  --out-dir artifacts/strategic_lane/muzero_run_loop \
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
  --strict
```

The runner writes:

```text
artifacts/strategic_lane/muzero_run_loop/
  parity_matrix/
  policy_eval/
  cache/
  train_smoke/
  gate/
  verifier/
  metrics.json
  decision.md
  final_report.md
  loop-state.json
  loop-run-log.md
```

Queue-readiness is decided by the named `programmatic_criteria` copied from
`gate/verifier/metrics.json`. Required evidence includes parity, random and
heuristic baselines, legal masks, MCTS visit-policy targets, finite train smoke
losses, reproducibility metadata, and loop state/log artifacts.

Useful stage commands remain available for debugging:

```text
mini_tft.tools.strategic_parity_matrix
mini_tft.tools.generate_strategic_muzero_cache
mini_tft.tools.train_strategic_muzero_smoke
mini_tft.tools.strategic_muzero_loop
```

## Remote Overnight Packet

A remote `dual4090` packet is prepared for a larger cache-supervised V0 run. It
is not queued yet, but the prior allowance pause has been waived by the user:

```text
artifacts/strategic_lane/muzero_v0_dual4090_overnight_packet/
```

The packet records the remote project path, run command, local post-run judge
command, remote smoke status, and file hashes. Launch is still gated by the
programmatic verifier, attempt cap, remote readiness, and local Antigravity
verdicting after remote artifacts are collected; it no longer requires a Codex
allowance check.

## Antigravity Judge

Promotion or loop completion can require an independent read-only judge packet
described in `docs/ANTIGRAVITY_JUDGE.md`.

The preferred judge route on this machine is:

```text
Antigravity via local ai-router / CLIProxyAPI
model: gemini-3.5-flash-low
reasoning: highest / reasoning_effort=xhigh
endpoint: http://127.0.0.1:8317/v1
```

Generate a packet with:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.judge_packet \
  --name strategic-cache-smoke \
  --deliverable muzero_cache \
  --objective "Judge whether the MuZero cache smoke evidence is complete and correctly labeled." \
  --evidence artifacts/strategic_lane/muzero_run_loop/metrics.json \
  --evidence artifacts/strategic_lane/muzero_run_loop/decision.md \
  --command "env -u UV_PYTHON uv run pytest tests/test_strategic_muzero_run_loop.py"
```

Packets live under `artifacts/judge/<name>/` and contain `prompt.md`,
`evidence_manifest.json`, `verdict_template.md`,
`antigravity_ai_router_command.txt`, `decision.md`, and `metrics.json`.

The gate fails closed until a verdict starts with `Verdict: ACCEPT` and includes
non-empty `Evidence checked`, `Findings`, and `Suggested action` sections:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.judge_packet \
  --check-verdict artifacts/judge/<name>/verdict.md
```

## Required Artifacts

Strategic-lane artifacts belong under `artifacts/strategic_lane/` unless a
task-specific runbook names another path.

```text
artifacts/strategic_lane/
  metrics.json
  decision.md
  final_report.md
  puffer_speed/
  muzero_cache/
  muzero_v0_dual4090_overnight_packet/
  zero_smoke/
  playable_demo/
```

Loop-style work also needs:

```text
loop-state.json
loop-run-log.md
verifier/
```

`metrics.json` should be machine-readable. `decision.md` should state status,
evidence, limits, and next action. `final_report.md` should say what is proven,
what is smoke-only, what failed, and what remains.

## Common Commands

Use `env -u UV_PYTHON uv run ...` for repo commands.

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

PufferLib 4 Ocean-style standalone smoke:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.benchmark_puffer4_ocean \
  --envs 512 --steps 100000 \
  --out-dir artifacts/strategic_lane/puffer4_ocean_commit_smoke
```

## Active Docs

- [Agent Notes](AGENTS.md): current rules for coding agents.
- [Strategic Lane](docs/STRATEGIC_LANE.md): clean simulator lane and deliverables.
- [Loop Scaffold](docs/LOOP_SCAFFOLD.md): repeatable autonomous loop contract.
- [Antigravity Judge](docs/ANTIGRAVITY_JUDGE.md): read-only ai-router judge packet.
- [Quality Gate](docs/QUALITY_GATE.md): deterministic verification criteria.
- [Archive Index](docs/ARCHIVE_INDEX.md): where old runbooks and reports moved.

## Claim Limits

- MiniTFT strategic-lane results are simplified simulator results only.
- `placement_proxy` is an elimination-timing bucket, not real TFT placement.
- Puffer results are throughput evidence only when backed by matched benchmark
  artifacts with parity and repeated-run variance.
- MuZero-style cache creation is data/smoke evidence, not policy-quality
  evidence.
- V0 cache-supervised runs are not full iterative MuZero.
- Queue-ready MuZero claims require parity, legal masks, cache rows, model/search
  artifacts, reproducibility metadata, baseline comparison, and quality-gate
  acceptance.
- Missing, malformed, or `REJECT` Antigravity verdicts block promotion unless
  the user explicitly approves a documented fallback.

## Pending Run TODO

- Pending: dual4090 Goal 2 checkpoint-guided strategic MCTS run.
- Superseded: heuristic-prior run `20260701T003039Z` was stopped before cache
  generation because it did not close the checkpoint-guided search loop.
- Target artifact: `artifacts/strategic_lane/checkpoint_guided_mcts_<RUN_ID>/`.
- Expected evidence: accepted Torch V0 checkpoint loading, checkpoint
  policy/value guided MCTS rows, matched heuristic-prior comparison rows,
  machine-readable quality checks, and local post-run Antigravity judge verdict.
