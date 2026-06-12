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

## Next Step Toward MuZero

The same harness should later gain a simulator-backed search objective. That
objective should compare PPO-only action selection against MCTS-improved action
selection on fixed lobby seeds, with placement/top-4/HP metrics and clone/restore
determinism gates.
