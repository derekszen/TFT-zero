# Puffer Material Speedup Runbook

This file keeps the historical filename because existing goals point here, but
the next loop is no longer "add native Puffer." That integration is done. The
new objective is to make the MiniTFT lobby PPO environment materially faster.

Previous accepted result:

- Native Python Puffer backend works and validates.
- Repeated matched benchmark was effectively tied with emulation:
  `native=431.98` steps/sec, `emulated=433.62` steps/sec, speedup `0.996x`.
- Therefore Gymnasium/Puffer emulation is not the dominant bottleneck. The next
  loop must profile and optimize simulator/lobby stepping internals.

Do not treat a working benchmark, one green test run, or one local speedup as
completion. Completion means profiling evidence, an implemented optimization,
regression tests, repeated matched benchmarks, PPO smoke wiring, full
validation, cleanup, final report, and verifier acceptance.

## Goal Prompt

```text
/goal Achieve a material throughput speedup for the MiniTFT lobby PPO environment, verified by profiler evidence, parity tests, repeated matched benchmarks, PPO smoke/evaluation artifacts, and OpenCode Go plan verifier using opencode-go/deepseek-v4-pro, while preserving simulator semantics, legal action masks, rewards, action ids, checkpoint formats, and existing SB3/Puffer comparison surfaces. Follow docs/PUFFER_NATIVE_VECTOR_RUNBOOK.md. Work in a separate worktree on branch feat/puffer-material-speedup. Do not stop after the first passing result or first benchmark improvement; continue through baseline capture, profiling, implementation, regression tests, similar-issue sweep, repeated benchmark validation, PPO smoke/evaluation, full validation, self-review, cleanup, independent verifier review, and final report. Stop only when all acceptance criteria are met or the blocked condition in this runbook is reached.
```

## Success Definition

The target is a repeated, matched, material speedup of the lobby PPO rollout
path.

Required threshold:

- Minimum acceptable speedup: `>= 1.5x` steps/sec over the captured baseline.
- Target speedup: `>= 2.0x` steps/sec.
- Benchmark must use matched seeds/settings and at least `3` repeats.
- If speedup is below `1.5x`, do not call the loop complete. Either iterate or
  stop blocked with profiler evidence explaining why the target is not reachable
  within the attempt cap.

## Scope

In scope:

- Profiling `MiniTFTLobbyHeroEnv`, lobby stepping, combat/value computation,
  observation/mask construction, reset/auto-reset, and PPO rollout env plumbing.
- Python simulator optimizations when they preserve behavior.
- Data-structure or allocation reductions in hot paths.
- Narrow C++/compiled work only if profiling proves Python simulator internals
  are the bottleneck and Python fixes cannot reach the threshold.
- Benchmark tooling improvements needed to compare before/after fairly.

Out of scope:

- Reward changes.
- Action id or legal-mask semantic changes.
- Easier opponents or changed lobby rules to make the benchmark faster.
- Removing checks only to improve speed.
- Long PPO training before speed/parity gates pass.
- Policy-quality claims from smoke runs.

## Worktree Setup

Work outside the dirty main workspace:

```bash
git worktree add ../TFT-zero-puffer-speedup -b feat/puffer-material-speedup
cd ../TFT-zero-puffer-speedup
```

If required Puffer scaffold files are still uncommitted in the main workspace,
copy only the accepted scaffold/native backend files needed for this loop. Do
not copy unrelated experiment outputs.

## Required Artifacts

Write all loop artifacts under:

```text
artifacts/puffer_material_speedup/
  baseline_metrics.json
  profile_summary.md
  metrics.json
  decision.md
  loop-state.json
  loop-run-log.md
  native_smoke.pt
  native_smoke_vs_sb3.json
  final_report.md
```

`loop-state.json` must include:

- `owner`: `codex`
- `schema`: `loop-state/v1`
- `loop`: `puffer_material_speedup`
- `level`: `L1 report-only` unless explicitly changed by a human
- `attempt`
- `attempt_cap`: `3`
- `current_next_action`
- `artifacts`
- `codex_allowance_check`
- `pause_criteria`
- `kill_criteria`
- `prune_rules`
- verifier statuses

Each run must append this allowance block to `loop-run-log.md`:

```md
## Codex Allowance Check
Source: /status | Codex usage dashboard | unknown
Checked at:
5h window remaining:
Weekly usage:
Decision: continue | soft-pause | hard-pause
```

## Required Phases

### Phase 1: Orientation And Baseline

Read:

- `README.md`
- `AGENTS.md`
- `docs/PUFFER_NATIVE_VECTOR_RUNBOOK.md`
- `src/mini_tft/rl/puffer_env.py`
- `src/mini_tft/rl/train_puffer_ppo.py`
- `src/mini_tft/tools/benchmark_puffer_backends.py`
- `src/mini_tft/rl/lobby_env.py`
- `src/mini_tft/core/lobby_step.py`
- `src/mini_tft/core/combat.py`
- `src/mini_tft/core/featurize.py`
- focused tests under `tests/test_puffer_env.py` and
  `tests/test_train_puffer_ppo.py`

Capture the baseline before optimizing:

```bash
env -u UV_PYTHON uv run --extra puffer --extra train \
  python -m mini_tft.tools.benchmark_puffer_backends \
  --out-dir artifacts/puffer_material_speedup/baseline \
  --seed 33000 \
  --repeats 3 \
  --steps 256 \
  --num-envs 4 \
  --players 8 \
  --max-actions-per-player 8 \
  --opponent-policy mixed
```

Copy or summarize the baseline into
`artifacts/puffer_material_speedup/baseline_metrics.json`.

### Phase 2: Profiling

Profile before editing. Use built-in `cProfile` at minimum:

```bash
env -u UV_PYTHON uv run --extra puffer --extra train \
  python -m cProfile \
  -o artifacts/puffer_material_speedup/baseline.prof \
  -m mini_tft.tools.benchmark_puffer_backends \
  --out-dir artifacts/puffer_material_speedup/profile_smoke \
  --seed 33100 \
  --repeats 1 \
  --steps 128 \
  --num-envs 4 \
  --players 8 \
  --max-actions-per-player 8 \
  --opponent-policy mixed \
  --smoke-only
```

Write `profile_summary.md` with:

- top hot functions by cumulative time
- top hot functions by internal time
- allocation/copy suspects visible from code inspection
- the single highest-leverage optimization target
- expected speedup and risk

Do not optimize before identifying a measured hotspot.

### Phase 3: Implementation Attempts

Make one high-leverage optimization attempt at a time. Prefer changes in this
order:

1. Remove unnecessary observation/mask allocations in the hot path.
2. Cache static or per-round structures that are recomputed every action.
3. Reduce lobby/combat repeated work that does not affect semantics.
4. Tighten reset/auto-reset and benchmark stepping overhead.
5. Consider compiled/C++ work only after Python-level hotspots are exhausted.

Every attempt must preserve:

- action ids
- mask semantics
- reward semantics
- placement/top-4/top-1 metrics
- checkpoint format
- public CLI compatibility unless explicitly documented

### Phase 4: Regression Tests

Add or update tests that would fail if the optimization changed behavior:

- legal mask parity
- observation shape and mask tail placement
- deterministic reset/step parity for fixed seeds
- terminal/auto-reset behavior
- benchmark artifact schema
- PPO smoke checkpoint compatibility

Do not weaken tests to pass.

### Phase 5: Similar-Issue Sweep

Search for duplicate hot-path patterns:

```bash
rg -n "action_masks\\(|featurize_state\\(|np\\.concatenate|np\\.stack|deepcopy|copy\\(|lobby_summaries|episode_summary|resolve_fights|step\\(" src/mini_tft tests
```

Fix only clearly related hot-path issues. Record candidates left alone.

### Phase 6: Repeated Benchmark Gate

Run the final repeated matched benchmark:

```bash
env -u UV_PYTHON uv run --extra puffer --extra train \
  python -m mini_tft.tools.benchmark_puffer_backends \
  --out-dir artifacts/puffer_material_speedup \
  --seed 34000 \
  --repeats 3 \
  --steps 512 \
  --num-envs 4 \
  --players 8 \
  --max-actions-per-player 8 \
  --opponent-policy mixed
```

`decision.md` must state:

- baseline steps/sec
- optimized steps/sec
- speedup ratio
- whether the `1.5x` minimum passed
- whether the `2.0x` target passed
- if not passed, the next measured bottleneck

### Phase 7: PPO Smoke And Evaluation

After speed and parity pass, run a short smoke to prove training/eval wiring:

```bash
env -u UV_PYTHON uv run --extra puffer --extra train \
  python -m mini_tft.rl.train_puffer_ppo \
  --env-kind lobby \
  --puffer-backend native \
  --timesteps 4096 \
  --num-envs 4 \
  --n-steps 256 \
  --batch-size 1024 \
  --update-epochs 1 \
  --device cuda \
  --lobby-opponent-policy mixed \
  --players 8 \
  --max-actions-per-player 8 \
  --hidden-sizes 256,256 \
  --out artifacts/puffer_material_speedup/native_smoke.pt
```

Evaluate the smoke checkpoint:

```bash
env -u UV_PYTHON uv run --extra train \
  python -m mini_tft.tools.compare_puffer_ppo \
  --puffer-checkpoint artifacts/puffer_material_speedup/native_smoke.pt \
  --sb3-checkpoint /mnt/ssd2/Projects/TFT-zero/checkpoints/ppo_lobby_frozenpool_probe10k2_lr1e5_20260625T1052_h256.zip \
  --opponent-policy mixed \
  --episodes 3 \
  --seed 43100 \
  --players 8 \
  --max-actions-per-player 8 \
  --device cpu \
  --format json \
  --out artifacts/puffer_material_speedup/native_smoke_vs_sb3.json
```

This smoke proves wiring only. Do not make policy-quality claims from it.

### Phase 8: Full Validation

Run before final report:

```bash
env -u UV_PYTHON uv run --extra puffer --extra train pytest \
  tests/test_puffer_env.py \
  tests/test_train_puffer_ppo.py \
  tests/test_checkpoint_policy.py \
  tests/test_compare_puffer_ppo.py

env -u UV_PYTHON uv run ruff check \
  src/mini_tft/rl/puffer_env.py \
  src/mini_tft/rl/train_puffer_ppo.py \
  src/mini_tft/tools/benchmark_puffer_backends.py \
  tests/test_puffer_env.py \
  tests/test_train_puffer_ppo.py

env -u UV_PYTHON uv run --extra puffer --extra train pyright \
  src/mini_tft/rl/puffer_env.py \
  src/mini_tft/rl/train_puffer_ppo.py \
  src/mini_tft/tools/benchmark_puffer_backends.py

env -u UV_PYTHON uv run --all-extras pyright
git diff --check
```

If optimized code touches additional files, include them in Ruff and Pyright.

### Phase 9: Self-Review And Cleanup

Review as a PR. Reject your own diff if it:

- changes simulator semantics to gain speed
- weakens masks or reward checks
- changes benchmark settings between baseline and optimized runs
- removes useful validation
- leaves generated symlinks, caches, or scratch files visible in `git status`
- claims speed from smoke-only evidence

### Phase 10: Final Report

Write `artifacts/puffer_material_speedup/final_report.md` with:

- branch/worktree path
- files changed
- baseline benchmark table
- optimized benchmark table
- speedup ratio
- profiler hotspot summary
- optimization implemented
- test commands/results
- PPO smoke/eval artifact paths
- whether `1.5x` and `2.0x` passed
- remaining risks
- whether C++ is justified next

## Acceptance Criteria

All must pass:

- Separate worktree/branch `feat/puffer-material-speedup`.
- Baseline benchmark captured before optimization.
- Profile summary identifies measured hotspots.
- At least one hotspot-driven optimization implemented.
- Repeated matched benchmark shows `>= 1.5x` speedup, or the loop stops blocked
  after three attempts with evidence.
- No reward, action id, mask, placement, or checkpoint-format semantic changes.
- Focused tests and full validation pass.
- PPO smoke checkpoint and eval JSON are written.
- `final_report.md`, `metrics.json`, `decision.md`, `loop-state.json`, and
  `loop-run-log.md` exist.
- OpenCode Go `plan` verifier with `opencode-go/deepseek-v4-pro` returns
  `ACCEPT`.
- If repo-local rules require it, secondary built-in `loop_verifier` returns
  `ACCEPT` or its findings are fixed.

## Verifier Step

Use OpenCode Go plan mode as the verifier of record:

```bash
opencode run \
  "Read-only verifier for the MiniTFT Puffer material speedup loop. Do not modify files. Return Verdict: ACCEPT or REJECT, Findings, Evidence checked, and Suggested action." \
  --dir "$PWD" \
  --model opencode-go/deepseek-v4-pro \
  --agent plan \
  --file docs/PUFFER_NATIVE_VECTOR_RUNBOOK.md \
  --file artifacts/puffer_material_speedup/baseline_metrics.json \
  --file artifacts/puffer_material_speedup/profile_summary.md \
  --file artifacts/puffer_material_speedup/metrics.json \
  --file artifacts/puffer_material_speedup/decision.md \
  --file artifacts/puffer_material_speedup/loop-state.json \
  --file artifacts/puffer_material_speedup/loop-run-log.md \
  --file artifacts/puffer_material_speedup/final_report.md
```

Do not use `--dangerously-skip-permissions`. If the verifier rejects, fix and
rerun until accepted or the blocked condition is reached.

## Blocked Conditions

Stop blocked only after `3` evidence-backed attempts when:

- repeated benchmark speedup remains below `1.5x`
- profiler shows time dominated by unavoidable Python simulator semantics and
  no narrow Python optimization remains
- any optimization that reaches speed target breaks parity or validation
- C++/compiled path is required but too large for this loop
- OpenCode Go verifier is unavailable and no human-approved replacement exists

If blocked, write the exact blocker, profiles, benchmark tables, attempted
fixes, and recommended next action.
