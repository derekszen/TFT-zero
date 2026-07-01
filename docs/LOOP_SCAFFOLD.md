# Loop Scaffold

Use this file for repeatable autonomous work. A loop is not done because the
first command passes. It is done only when the objective, verification surface,
artifacts, and review criteria are all satisfied.

## Goal Shape

```text
/goal Achieve [objective], verified by [commands/artifacts], while preserving
[constraints]. Follow docs/LOOP_SCAFFOLD.md and any task-specific runbook. Do
not stop after the first passing result; continue through implementation,
regression tests, similar-issue sweep, validation, self-review, cleanup,
verifier review, and final report. Stop only when all acceptance criteria are
met or the blocked condition is reached.
```

## Default Loop

1. Orient: read `README.md`, `AGENTS.md`, `docs/ARCHITECTURE.md`,
   `docs/STRATEGIC_LANE.md`, `docs/QUALITY_GATE.md`, and relevant code.
2. Define the smallest measurable objective.
3. Write or update `loop-state.json` with plan, attempt cap, and stop rules.
4. Make the smallest safe change.
5. Run focused validation.
6. Run broader validation when focused checks pass.
7. Write artifacts and update `loop-run-log.md`.
8. Search for the same issue pattern in adjacent code/docs.
9. Self-review the diff as if reviewing a PR.
10. Run verifier review where available.
11. Stop only on acceptance or explicit blocked state.

## Required Artifacts

Each loop writes under `artifacts/<loop_name>/` unless a task-specific runbook
names another path:

```text
metrics.json
decision.md
final_report.md
loop-state.json
loop-run-log.md
verifier/
```

`loop-state.json` should include:

```json
{
  "schema": "loop-state/v1",
  "objective": "...",
  "deliverable": "muzero_cache|puffer_speed|playable_demo|docs|other",
  "attempt_cap": 3,
  "attempt": 1,
  "status": "running|accepted|blocked",
  "acceptance_criteria": [],
  "blocked_condition": [],
  "validation_commands": [],
  "artifacts": [],
  "verifier": "not_run|accept|reject|unavailable"
}
```

Append every run to `loop-run-log.md` with timestamp, attempt, action,
validation, verifier verdict, and next action.

## Verification Levels

| Level | Meaning | External Effects |
| --- | --- | --- |
| L1 report-only | Inspect, classify, write report/artifacts | None |
| L2 patch | Modify local repo files | Local only |
| L3 PR/ticket | Create external PR/ticket/update | Requires explicit user scope |
| L4 merge/deploy | Merge, deploy, or mutate production | Disabled unless explicitly allowed |

Default to L1 for new loop patterns. Use L2 only when the user asked for code
or doc changes.

## Attempt Cap

Default cap is `3` evidence-backed attempts for the same blocker. After the cap,
stop and write:

- exact blocker;
- commands/logs proving it;
- attempted fixes;
- current artifacts;
- next human decision needed.

Do not keep looping with no new evidence.

## Verifier Contract

Before completion, run a read-only verifier when available. The verifier may
inspect artifacts and code but must not patch files.

Verifier output is fail-closed:

- `ACCEPT`: loop may complete if all other criteria pass.
- `REJECT`: fix the issue or escalate as blocked.
- unavailable verifier: record the command, failure, and fallback decision.

For external model verification, use the user-approved verifier for the task.
Current preferred verifier for long loops is the reusable Antigravity judge
packet in `docs/ANTIGRAVITY_JUDGE.md`, using local ai-router Antigravity with
the live `gemini-3.5-flash-low` alias and highest thinking in read-only mode. If ai-router
Antigravity is unavailable, record the missing verifier and keep the gate
blocked unless the user explicitly approves a fallback.

## Done Means

A loop is complete only when:

- implementation or report matches the objective;
- focused checks pass;
- broader practical validation passes or is explicitly not applicable;
- artifacts are written;
- the similar-issue sweep is complete;
- docs/config are updated if behavior changed;
- verifier accepts or the user accepts a documented fallback;
- final report states what is proven, smoke-only, failed, and next.
