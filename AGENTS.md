# TFT-zero Agent Notes

Project-specific rules for coding agents. Follow these after the global agent
instructions supplied by the user.

## Current Objective

Build the clean MiniTFT strategic lane. New work should support one of three
deliverables:

1. MuZero-style cache/data smoke artifacts.
2. PufferLib or batched-simulator throughput evidence.
3. A playable simulator/demo surface using the same strategic rules.

The active contracts are:

```text
src/mini_tft/strategic/
docs/STRATEGIC_LANE.md
docs/LOOP_SCAFFOLD.md
docs/QUALITY_GATE.md
```

Legacy candidate-choice, raw-action Stage 2-5, non-oracle action-surface, and
Puffer speedup runbooks are archived under `docs/archive/`. Treat them as
historical references unless the user explicitly reactivates one.

## Required Command Style

Use:

```bash
env -u UV_PYTHON uv run ...
```

Common checks:

```bash
env -u UV_PYTHON uv run pytest
env -u UV_PYTHON uv run ruff check
env -u UV_PYTHON uv run --all-extras pyright
git diff --check
```

## Implementation Rules

- Prefer the smallest simulator path that is understandable, deterministic, and
  testable.
- Put new strategic-lane implementation under `src/mini_tft/strategic/`.
  Treat `src/mini_tft/core/` as the existing reference engine unless the task is
  explicitly a legacy fix or adapter bridge.
- Preserve the Markov invariant: simulator transition and reward behavior must
  depend only on current canonical state, action, and simulator randomness.
- Keep strategic-lane opponents as round-based enemy pressure. Do not add
  multi-agent opponent boards, economy, shared pools, scouting, or self-play
  unless the user explicitly reopens that scope.
- Keep one canonical strategic rules implementation. Adapters may wrap it; they
  should not fork rules.
- Keep legal action masks part of the environment contract.
- Keep oracle/debug shortcuts out of learned/search action surfaces.
- Label smoke results as smoke. Do not promote quality, speed, or MuZero claims
  without `docs/QUALITY_GATE.md` evidence.
- Preserve unrelated dirty work.

## Loop Rules

Use `docs/LOOP_SCAFFOLD.md` for autonomous or repeated work.

Required loop artifacts:

```text
artifacts/<loop_name>/
  metrics.json
  decision.md
  final_report.md
  loop-state.json
  loop-run-log.md
```

Default attempt cap is `3`. Stop as blocked after the same failure persists
through the cap, and write the exact blocker, evidence, attempted fixes, and
next human decision.

Before marking loop work done, run a read-only verifier where available. Treat
verifier rejection as fail-closed unless the user explicitly overrides it.

## Claim Discipline

- Toy-lobby results are not real TFT rank claims.
- Current-patch MetaTFT planner work is historical unless explicitly resumed.
- Puffer speed claims require matched benchmark artifacts, not trainer import
  success.
- MuZero-style claims require cache rows, legal masks, model/search artifacts,
  deterministic seeds, and baseline comparison.
