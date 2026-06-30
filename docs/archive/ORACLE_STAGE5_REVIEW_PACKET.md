# Oracle Stage 5 Review Summary

This document records the Oracle/GPT-5.5 Pro review result for the MiniTFT
MuZero-stage track. It is historical rationale, not the current execution
contract. The active contract is
`docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md`; use
`docs/MUZERO_STAGE_GATE.md` for stage-gate context.

## Verdict

Pause serious Stage 5 work until evaluation is repaired.

Oracle found evidence that recent outcome metrics may be decision-weighted
instead of episode-weighted. If true, placement and HP deltas from Stage 2
through Stage 4 strong are contaminated by episode length and decision count.

Small Stage 5 smoke scaffolding is acceptable only after the evaluator repair
and ablation matrix are complete. Long Stage 5 training or promotion is not
acceptable yet.

## Primary Finding

Outcome metrics must be episode-level:

- mean placement
- top-1/top-4
- final HP
- final board strength
- placement histogram

Decision metrics may remain decision-level:

- action mix
- selected-from-prior
- guard override rate
- decisions/sec
- search source rates

The report must label every metric as `episode_level` or `decision_level`.

## Suspect Historical Artifacts

Treat these as scaffolding inputs, not trusted performance evidence, until the
repaired evaluator reruns them:

```text
artifacts/muzero_stage/stage2/decision.md
artifacts/muzero_stage/stage3/decision.md
artifacts/muzero_stage/stage4/decision.md
artifacts/muzero_stage/stage4_strong/decision.md
artifacts/muzero_stage/stage4_strong/metrics.json
```

Stage 1 remains useful as a narrow diagnostic, but it does not prove the whole
reward/evaluator path is correct.

## Other Oracle Concerns

- Stage 4 strong was compared mainly against Stage 3, while the tiny eval showed
  worse placement than PPO.
- Stage 4 strong improvement may come from a simulator one-step guard rather
  than learned dynamics or learned policy.
- Selected-from-prior remained near `0.995-0.999`, so search may mostly copy the
  prior.
- Dynamics losses were small-scale and need heldout seed checks.
- Replay metadata and suite labels need validation before Stage 5 targets are
  trusted.

## Required Reset

This original four-goal reset has been superseded by the candidate-choice rerun
contract in `docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md`. The original reset was:

1. **Goal 1 evaluator repair**: prove outcome metrics are episode-level and add
   an unequal-decision-count regression test.
2. **Goal 2 Stage 4 ablation/rerun matrix**: compare PPO, Stage 3, Stage 4
   strong, prior-only, no-guard, guard-only, and simulator MCTS if feasible.
3. **Goal 3 Stage 5 replay/scaffold smoke**: generate replay/search-target data
   and run one small smoke cycle with repaired metrics.
4. **Goal 4 guarded Stage 5 promotion**: promote only if repaired public and
   heldout metrics beat or non-regress against PPO and Stage 4 baselines, with
   guard attribution.

## Falsifiers

Stop or demote the current direction if any of these happen on repaired metrics:

- Stage 4 strong no longer beats Stage 3 on heldout episode-level outcomes.
- Stage 4 strong remains clearly worse than PPO on placement.
- Guard-only reproduces nearly all improvement.
- No-guard model search regresses badly versus PPO or Stage 3.
- Stage 5 targets are mostly uniform, prior-copying, or macro shortcut-driven.
- Value predictions do not correlate with paired seed outcomes.

## Minimum Next Experiment

After Goal 1, rerun a repaired matrix with:

- mixed and frozen suites
- public and heldout seeds
- at least `20` public and `20` heldout episodes per suite, unless marked
  smoke-only
- episode-level placement, top-1/top-4, HP, board strength, and histogram
- decision-level action mix, decisions per episode, selected-from-prior, guard
  override rate, and source attribution

Do not run long PPO or long Stage 5 training before this matrix.
