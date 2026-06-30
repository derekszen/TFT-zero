# Non-Oracle Action Surface Runbook

This runbook converts MiniTFT lobby training from oracle macro actions toward a
playable TFT-like action surface.

`FIELD_BEST_BOARD` and `SLAM_BEST_ITEM` are useful debug/UI affordances, but
they leak a planner's answer into the learned policy. A policy that can press
`FIELD_BEST_BOARD` is not learning board construction; it is delegating board
construction to the simulator. The same applies to `SLAM_BEST_ITEM` for item
targeting.

The target is not exact Riot TFT. The target is a simple, playable placement
game where a hero policy can place `1-8`, can lose, can top-4, and can improve
without oracle macro shortcuts.

## Primary Goal Prompt

```text
/goal Achieve a playable non-oracle MiniTFT lobby action surface for PPO and future MuZero search, verified by oracle-disabled policy/eval gates, candidate-board and item-action parity tests, placement/top-4 reports, PPO smoke/evaluation artifacts, and independent verifier acceptance, while preserving debug/UI oracle buttons, action-id compatibility where possible, checkpoint loading compatibility, and all unrelated dirty work.

Follow docs/NON_ORACLE_ACTION_SURFACE_RUNBOOK.md.

Work in a separate worktree on branch feat/non-oracle-action-surface. Do not modify the dirty main workspace except for explicitly required loop logs or copied accepted artifacts named by this runbook.

Do not stop after merely disabling oracle macros. Disabling oracle macros without replacement currently collapses heuristic policies to 8th place. Continue through baseline measurement, candidate-board action implementation, explicit item-action design or scoped fallback, oracle-disabled training/eval gates, opponent-placement calibration, PPO smoke/evaluation, documentation cleanup, independent verifier review, and final evidence report.

Success requires: oracle macros remain available for debug/UI but are not required for PPO training; at least one non-oracle board-construction surface exists; legal masks and action ids are tested; fixed-seed lobby reports show weak policies can lose and competent non-oracle policies can place above 8th; PPO smoke/eval artifacts are produced; final_report.md, metrics.json, decision.md, loop-state.json, and loop-run-log.md are written; OpenCode Go plan verifier returns ACCEPT; secondary loop_verifier returns ACCEPT.

Attempt cap: 3 implementation attempts. Stop only when all acceptance criteria are met or the blocked condition in this runbook is reached.
```

## Current Evidence

Quick fixed-seed lobby checks showed:

| Hero | Oracle macros | Mean placement | Top-4 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `fast_level` | on | `2.60` | `0.90` | Macro-assisted heuristic is strong. |
| `tempo` | on | `3.35` | `0.60` | Macro-assisted heuristic is playable. |
| `random` | on | `8.00` | `0.00` | Weak policy loses. |
| `fast_level` | off | `8.00` | `0.00` | Removing macros without replacement breaks the action surface. |
| `tempo` | off | `8.00` | `0.00` | Same failure. |

Conclusion: the lobby already supports placement and losses. The blocker is
that non-oracle policies do not yet have a usable board/item action surface.

## Stage 1: Baseline And Gates

Objective: capture the current macro dependency and make it impossible to hide
future regressions behind survival-only metrics.

Required work:

- Add or update a benchmark/eval script that compares policies with
  `allow_hero_macro_actions=true` and `false`.
- Report mean placement, median placement, top-1, top-4, final HP, survived
  round, board strength, action mix, macro action rate, illegal actions, and
  placement histogram.
- Add a decision artifact that explicitly labels macro-assisted results as
  debug/reference, not promotion evidence.

Acceptance:

- `fast_level` or `tempo` with macros disabled is shown to be currently broken
  or weak, using placement/top-4 evidence.
- The report proves weak policies can place `6-8`.
- Outcome metrics are episode-level.

## Stage 2: Candidate-Board Action Surface

Objective: replace `FIELD_BEST_BOARD` with a non-oracle board-construction
surface.

Preferred implementation:

- Keep primitive slot move actions.
- Add a candidate-board wrapper/action mode:
  - candidate `0..k-1` means "apply this generated legal candidate board";
  - candidates are generated from owned board+bench units;
  - candidate generation may use heuristics to propose choices, but the selected
    action is a real policy choice, not a hidden always-best action.
- Candidate actions may live in a wrapper action space if preserving existing
  raw action ids is cleaner.

Required tests:

- Candidate board preserves owned-unit multiset.
- Candidate board respects level cap.
- Candidate board respects bench capacity.
- Candidate board preserves items and stars.
- Candidate board does not mutate source state while generating candidates.
- Candidate board application has deterministic fixed-seed behavior.
- Legal mask parity is clear for raw primitive actions versus wrapper actions.

Acceptance:

- `FIELD_BEST_BOARD` is no longer needed for a heuristic to field a non-empty,
  legal, level-capped board.
- A non-oracle `tempo` or `fast_level` variant places above 8th on a fixed
  mixed-opponent smoke.

## Stage 3: Item Action Surface

Objective: replace `SLAM_BEST_ITEM` as a training dependency.

Preferred implementation:

- Keep `SLAM_BEST_ITEM` for debug/UI.
- Add explicit item actions or a wrapper:
  - choose completed item or component recipe;
  - choose target unit or target slot;
  - reject impossible recipes/targets through legal masks.

Scoped fallback if this is too large:

- Add a non-oracle scripted item policy used by opponent/heuristic baselines
  while the learned hero initially trains item-free or with simplified target
  actions.
- The fallback must be documented as temporary and cannot be promotion evidence
  for full non-oracle play.

Required tests:

- Component ownership is preserved.
- Completed item recipe consumes the right components.
- Target capacity is respected.
- Item ownership stays with the unit through board/bench moves.
- PvE item drops remain deterministic under fixed seeds.

Acceptance:

- PPO/eval can run with `SLAM_BEST_ITEM` masked for the hero.
- Item action counts and illegal action counts are reported.

## Stage 4: Opponent And Placement Calibration

Objective: make the game sane as a placement game, not just an HP survival
game.

Required opponent suites:

- `random`: should usually place bottom.
- `tempo`: should be playable but beatable.
- `fast_level`: should be stronger but not unbeatable.
- `mixed`: default smoke/eval opponent pool.
- frozen PPO/checkpoint pool where available.

Required reports:

- placement histogram over `1-8`;
- top-1/top-4;
- mean final HP;
- survived round;
- board strength;
- macro action rate;
- illegal actions;
- total fights/actions.

Acceptance targets for smoke gates:

- Weak/random hero has poor top-4 and frequent `6-8` placements.
- Competent non-oracle heuristic has nonzero top-4 and does not always place
  8th.
- Macro-assisted baselines are reported separately and not used as non-oracle
  promotion evidence.

## Stage 5: PPO Smoke And Training Surface

Objective: prove PPO can train/evaluate against the non-oracle surface.

Required work:

- Training defaults should prefer oracle-disabled hero training once Stage 2 is
  playable.
- CLI flags must make macro allowance explicit.
- PPO smoke must write checkpoint and manifest artifacts.
- Evaluation must compare against at least one non-oracle heuristic baseline and
  one macro-assisted reference baseline.

Acceptance:

- PPO smoke runs with hero oracle macros disabled.
- No illegal actions.
- Checkpoint loading/evaluation still works for existing SB3/Puffer surfaces.
- Reports label smoke as smoke unless promotion thresholds are met.

## Stage 6: Documentation And Verifiers

Required artifacts under the loop artifact directory:

```text
metrics.json
decision.md
loop-state.json
loop-run-log.md
final_report.md
```

Final report must include:

- what remains debug/UI-only;
- what is valid for training;
- action-surface differences versus the old macro setup;
- placement/top-4 evidence;
- PPO smoke artifacts;
- remaining risks.

Verification:

- OpenCode Go plan verifier using `opencode-go/deepseek-v4-pro` returns
  `ACCEPT`.
- Secondary read-only `loop_verifier` returns `ACCEPT`.

## Blocked Conditions

Stop blocked only after 3 evidence-backed attempts if:

- candidate-board actions cannot preserve unit/item/state parity;
- non-oracle heuristic still always places 8th after candidate-board support;
- item-action scope is too large and no documented fallback is acceptable;
- disabling macros breaks checkpoint/eval compatibility in a way that cannot be
  isolated behind flags;
- placement metrics are not episode-level;
- verifier rejects the same issue after three attempts.
