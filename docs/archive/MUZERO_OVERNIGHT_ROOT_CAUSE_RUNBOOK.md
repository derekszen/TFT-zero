# MuZero Overnight Root-Cause Runbook

This runbook is for an unattended, bounded overnight loop whose job is to find
and repair the highest-probability blocker before starting real Stage 5
candidate-choice MuZero candidate training.

It is not a promotion run. It is a root-cause and readiness loop.

## Objective

Identify and fix the blocker that prevents candidate-choice MuZero training from
having a credible learning signal, then stop at a clear go/no-go decision:

- `READY_FOR_STAGE5_CANDIDATE_TRAINING`
- `BLOCKED_TARGET_QUALITY`
- `BLOCKED_ACTION_SURFACE`
- `BLOCKED_FEATURE_SIGNAL`
- `BLOCKED_SEARCH_TARGETS`
- `BLOCKED_SIMULATOR_OR_EVALUATOR`
- `BLOCKED_VERIFIER_REJECTED`

The loop may run for up to `9` wall-clock hours, but it must stop earlier if the
readiness gate passes or a blocked condition is reached.

## Scope

Work in a separate worktree:

```text
/mnt/ssd2/Projects/TFT-zero-muzero-overnight-rootcause
```

Suggested branch:

```text
feat/muzero-overnight-rootcause
```

Write artifacts under:

```text
artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/
```

Do not modify the main dirty workspace except for explicitly required loop logs
or accepted documentation updates.

## Operating Constraints

- Use candidate-choice action surface only for learned/search hero artifacts.
- Do not use oracle macro actions or primitive board moves for learned/search
  hero policies.
- Keep episode-level outcome metrics separate from decision-level diagnostics.
- Do not run blind PPO continuation as a substitute for root-cause evidence.
- Do not run full Stage 5 candidate training until the readiness gate passes.
- Preserve existing checkpoint compatibility and unrelated dirty work.
- Use `env -u UV_PYTHON uv run ...` for project commands.

## Hypothesis Queue

Test these hypotheses in order. Continue only when the current highest-signal
probe has been run and recorded.

1. **Artifact/evaluator integrity**: target rows, episode ids, action surfaces,
   masks, and outcome aggregation are correct.
2. **Target-quality failure**: branch-value labels are too noisy, too terminal,
   too circular, or not predictive enough for candidate ranking.
3. **Feature-signal failure**: candidate features omit the information needed to
   choose boards, shops, rerolls, economy, or tempo decisions.
4. **Search-target failure**: search targets are mostly uniform, prior-copying,
   guard-only, or insensitive to branch depth/exploration.
5. **Training/eval mismatch**: action dimensions, masks, candidate generation,
   or checkpoint metadata differ between train, search, and eval.

## Work Loop

Repeat until the readiness gate passes, the blocked condition is reached, or the
9-hour budget expires:

1. Record the current hypothesis, attempt number, and next command in
   `loop-state.json`.
2. Run the smallest probe that can falsify or support the hypothesis.
3. If the probe exposes a concrete implementation defect, patch the smallest
   fix.
4. Add or update a focused regression test for the repaired behavior.
5. Rerun the focused test/probe and update metrics.
6. If the same blocker persists after two implementation attempts, escalate to
   Oracle/GPT-5.5 Pro before the third attempt.
7. After Oracle returns, write a concise plan delta, amend this runbook only if
   the amendment tightens or clarifies the gate, then continue with attempt 3.
8. Stop after three evidence-backed attempts on the same blocker.

One implementation attempt is:

```text
probe -> patch -> focused validation -> rerun probe -> self-review
```

## Required Probes

At minimum, run or implement equivalents of these probes:

- target artifact integrity:
  - row count;
  - decision count;
  - branch count per decision;
  - candidate-choice action-surface metadata;
  - legal-mask violations;
  - duplicate or inconsistent episode outcomes.
- branch-value target quality:
  - pairwise rank accuracy by decision-root split;
  - comparison against candidate-id and simple board-strength baselines;
  - shuffled-label control;
  - shortcut-control run if shortcut features exist;
  - branch-depth or rollout-horizon sensitivity.
- temporal rollout summary:
  - starting HP;
  - final HP;
  - HP delta;
  - combat win/loss counts when available;
  - board-strength trajectory;
  - survival or placement delta;
  - rollout step count and horizon metadata.
- search-target diagnostics:
  - selected-from-prior rate;
  - guard override rate;
  - policy target entropy;
  - non-prior action rate;
  - action-value spread per decision.
- readiness smoke:
  - one small candidate-choice Stage 5 train/eval smoke only after target quality
    is good enough to justify training.

## Readiness Gate

Write `decision.md` as `READY_FOR_STAGE5_CANDIDATE_TRAINING` only when all are
true:

- candidate-choice action-surface metadata is present in relevant artifacts;
- candidate-board action count is positive;
- oracle macro action rate is `0`;
- primitive board moves are disabled for learned/search hero artifacts;
- illegal action count is `0`;
- outcome metrics used for decisions are episode-level;
- target-quality probe beats trivial candidate-id and board-strength baselines
  by a meaningful margin, or the report explains why the accepted target is
  still non-circular and useful;
- shuffled-label control is clearly worse than the real-label result;
- search targets are not mostly uniform, prior-copying, or guard-only;
- a focused Stage 5 readiness smoke writes train/eval artifacts;
- OpenCode Go verifier returns `ACCEPT`;
- `loop_verifier` returns `ACCEPT`.

If the target-quality probe cannot beat trivial baselines, do not start real
Stage 5 training. Record the blocker instead.

## Oracle Escalation

Escalate to Oracle/GPT-5.5 Pro after two failed implementation attempts on the
same blocker, before the third attempt.

Use browser mode only:

```bash
oracle --dry-run summary --files-report \
  -p "Review the MiniTFT candidate-choice MuZero root-cause blocker. Identify the most likely root cause, propose the smallest repair plan, and list falsifiable acceptance checks. Do not suggest full training until target-quality/readiness gates pass." \
  --file docs/MUZERO_STAGE_GATE.md \
  --file docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md \
  --file docs/MUZERO_OVERNIGHT_ROOT_CAUSE_RUNBOOK.md \
  --file artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/metrics.json \
  --file artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/decision.md \
  --file artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/root_cause_report.md
```

Then run:

```bash
oracle --engine browser --model gpt-5.5-pro \
  -p "Review the MiniTFT candidate-choice MuZero root-cause blocker using the attached runbooks, metrics, decision, and report. Return: Verdict, Root cause, Minimal repair plan, Required tests/probes, and Stop/go recommendation for Stage 5 candidate training." \
  --file docs/MUZERO_STAGE_GATE.md \
  --file docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md \
  --file docs/MUZERO_OVERNIGHT_ROOT_CAUSE_RUNBOOK.md \
  --file artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/metrics.json \
  --file artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/decision.md \
  --file artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/root_cause_report.md
```

Save Oracle output under:

```text
artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/oracle/
```

Treat Oracle as advisory. Verify every accepted recommendation with code,
tests, probes, or artifacts before changing the readiness decision.

## Required Artifacts

The loop must write:

```text
artifacts/muzero_stage/candidate_choice_rerun/overnight_root_cause/
  loop-state.json
  loop-run-log.md
  commands.md
  metrics.json
  decision.md
  root_cause_report.md
  final_report.md
  probes/
  oracle/
  verifier/
```

`loop-state.json` must include:

- current hypothesis;
- attempt number;
- attempt cap;
- wall-clock start time;
- 9-hour deadline;
- current blocker;
- next command;
- Oracle escalation status;
- verifier status.

`commands.md` must record exact commands, exit status, and artifact paths.

`root_cause_report.md` must summarize:

- probes run;
- evidence for and against each hypothesis;
- root cause selected;
- fixes made;
- remaining uncertainty;
- go/no-go decision for Stage 5 candidate training.

## Validation Commands

Run focused tests for changed files. At minimum, run the nearest applicable
checks:

```bash
env -u UV_PYTHON uv run --extra train pytest -q tests/test_stage2_mcts.py tests/test_stage3_distill.py tests/test_stage4_dynamics.py
env -u UV_PYTHON uv run ruff check src/mini_tft/search src/mini_tft/tools tests
env -u UV_PYTHON uv run --all-extras pyright
git diff --check
```

If these are too broad for the current patch, run the focused subset first,
then record why broader validation was deferred.

## Verifiers

Before claiming readiness:

1. Run OpenCode Go plan verifier with `opencode-go/deepseek-v4-pro`.
2. Run a read-only `loop_verifier` subagent.

The OpenCode prompt must reject readiness if:

- action surface is not candidate-choice;
- oracle macros or primitive board actions are available to learned/search hero
  artifacts;
- outcome metrics are decision-weighted;
- target-quality probes only beat trivial baselines by leakage or shortcuts;
- shuffled-label controls are not worse;
- search targets are uniform, prior-copying, or guard-only;
- required artifacts are missing.

## Blocked Conditions

Stop as blocked if any of these persist after three evidence-backed attempts:

- target-quality remains at or below trivial baselines;
- branch outcomes are too noisy to produce reliable labels at the current
  simulator horizon;
- candidate features are insufficient and require a broader simulator/data
  redesign;
- action-surface or mask invariants fail;
- search targets remain prior-copying or guard-only;
- the same verifier rejection persists;
- required tooling, dependencies, credentials, or hardware are unavailable.

When blocked, write the exact blocker, command evidence, attempted fixes,
Oracle advice if used, and the next human decision.
