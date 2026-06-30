# TFT-Zero Autoresearch Scaffold

This is a local automated-research harness for MiniTFT. It is inspired by the
autoresearch-at-home loop, but it is project-specific: objectives are PPO,
lobby placement, HP, board strength, illegal actions, and simulator safety, not
language-model `val_bpb`.

## Current Objectives

Single-player PPO vs `FastLevelBot`:

```bash
uv run --extra train python -m mini_tft.autoresearch.run \
  --objective experiments/autoresearch/objectives/ppo_single_player.toml \
  --phase public \
  --auto-holdout
```

Lobby-placement PPO vs scripted opponents:

```bash
uv run --extra train python -m mini_tft.autoresearch.run \
  --objective experiments/autoresearch/objectives/ppo_lobby_placement.toml \
  --phase public \
  --auto-holdout
```

Lobby-placement PPO vs a deterministic mixed opponent pool:

```bash
uv run --extra train python -m mini_tft.autoresearch.run \
  --objective experiments/autoresearch/objectives/ppo_lobby_mixed.toml \
  --phase public \
  --auto-holdout
```

During active simulator work, use `--allow-protected-changes` only for
diagnostic runs. The report still records which protected evaluator/simulator
files were dirty.

## Artifacts

Each run writes:

```text
experiments/autoresearch/runs/<timestamp>_<objective>_<phase>_<name>/
  metrics.json
  manifest.json
  decision.md
```

`metrics.json` is the machine-readable record. `decision.md` is the compact
human review. Keep run artifacts local unless a result is intentionally promoted
into a report.

## Safety Rules

Objective files list protected paths. If protected reward, combat, mask,
lobby/evaluator, or gate files are dirty, the objective fails by default. This
prevents a candidate from "winning" by changing the measurement instead of
improving policy behavior.

## Loop

1. Start from a clean objective branch or a dedicated worktree.
2. Pick one hypothesis and make the smallest candidate change.
3. Run focused tests for the changed path.
4. Run the matching autoresearch objective.
5. Keep the change only if the public phase passes and holdout does not regress.
6. Record the reasoning in the run report or a follow-up note.

## MuZero-Stage Loop

The MuZero-stage work has its own gate contract in
`docs/MUZERO_STAGE_GATE.md`. The current autonomous rerun contract is
`docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md`; use that document instead of the
generic PPO objective runner for candidate-choice MuZero work.

Current status:

- Stage 1 diagnostics remain useful as narrow roll/mask/evaluator probes.
- Goal 1 repaired outcome aggregation and Goal 2 reran the repaired Stage 4
  matrix.
- Legacy raw-action Stage 2-4/4-strong artifacts remain historical baselines, not
  candidate-choice promotion evidence.

Current rerun loop:

1. Verify or smoke-train the candidate-choice PPO baseline.
2. Migrate Stage 2 MCTS to non-oracle candidate-board choices.
3. Rerun Stage 3/4 target generation, distillation, dynamics, and ablations on
   the candidate-choice action dimension.
4. Smoke-test Stage 5 replay/search-target scaffolding under the same surface.
5. Run independent verifier checks before any promotion claim.

Do not replace this rerun with another long PPO continuation.

MuZero promotion claims must use `action_surface = candidate_choice`: candidate
board choices plus economy/shop/end actions, with oracle macros disabled and
primitive board moves masked for the learned/search hero. Legacy raw-action
MCTS/dynamics traces are scaffold/debug evidence only.
