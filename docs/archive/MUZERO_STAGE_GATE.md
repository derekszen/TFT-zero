# MuZero Stage Gate And Reset Contract

This is the current go/no-go contract for the MiniTFT MuZero-stage track. It is
written for coding agents first: follow the goals in order, keep outcome
metrics episode-level, and do not promote Stage 5 from suspect historical
numbers.

## Current Status

Status: **Goal 2 repaired matrix complete and verified; the current target is
the candidate-choice MuZero rerun in `docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md`.**

Oracle/GPT-5.5 Pro found strong evidence that recent trace comparisons may have
averaged terminal outcomes once per decision row instead of once per episode.
That made Stage 2 through Stage 4 strong placement and HP deltas
suspect/historical. Goal 1 repaired trace aggregation so terminal outcomes are
episode-level, and Goal 2 reran the Stage 4 ablation matrix on repaired public
and heldout mixed/frozen suites.

The Stage 4 strong artifact remains useful as a baseline input:

```text
artifacts/muzero_stage/stage4_strong/promoted/dynamics_stage4.pt
```

It is not Stage 5 promotion evidence yet. The repaired Goal 2 matrix shows
Stage 4 strong is hybrid model-backed root search with a simulator one-step
value guard, not full MuZero self-play. Stage 4 no-guard/model-only exactly
matched prior-only on placement and HP in the repaired matrix, while
guard-only/prior+guard reproduced Stage 4 strong within tolerance. The selected
action still comes from the prior policy almost all the time.

Additional reset from the non-oracle PPO work: future PPO and MuZero claims are
scoped to non-oracle candidate-board choices, not raw primitive slot moves.
Primitive board moves remain UI/debug/scripted-policy infrastructure. The
legacy Stage 2-4 search code still uses `raw_primitive_legacy` actions and now
marks traces with that action surface; those traces are scaffold/debug evidence,
not candidate-choice MuZero promotion evidence.

Current repaired artifacts:

```text
artifacts/muzero_stage/reset_evaluator/decision.md
artifacts/muzero_stage/stage4_repaired_matrix/decision.md
artifacts/muzero_stage/stage4_repaired_matrix/metrics.json
```

The next autonomous runbook and one-goal contract is:

```text
docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md
```

## Historical Evidence

Treat this table as scaffolding history, not current performance proof. Use the
Goal 2 repaired matrix for current Stage 4 attribution.

| Stage | Artifact | Historical decision | What it built | Why it is not enough now |
| --- | --- | --- | --- | --- |
| 1. Diagnostics | `artifacts/muzero_stage/stage1/decision.md` | `PASS_TO_STAGE_2` | Roll/mask/value probes for the PPO plateau. | Narrow diagnostic only; it does not prove all evaluator metrics are correct. |
| 2. Simulator MCTS | `artifacts/muzero_stage/stage2/decision.md` | `PASS_TO_STAGE_3` | Clone/restore and legal simulator-backed MCTS. | Placement/HP deltas may be decision-weighted. |
| 3. Search Distillation | `artifacts/muzero_stage/stage3/decision.md` | `PASS_STAGE_3` | Distilled MCTS traces into a policy/value student. | Promotion relied on possibly suspect trace comparisons. |
| 4. Dynamics Search | `artifacts/muzero_stage/stage4/decision.md` | `PASS_STAGE_4` | Trained MuZero-lite representation/dynamics/prediction model. | Mostly prior-copying and small evals. |
| 4 strong. Hybrid Dynamics Search | `artifacts/muzero_stage/stage4_strong/decision.md` | `PASS_STAGE_4_STRONG` | Added model-backed root search plus simulator one-step guard. | Tiny eval, near-prior action selection, and comparison mainly against Stage 3 instead of PPO/guard ablations. |

## Metric Semantics

Outcome metrics are episode-level and must be counted once per lobby:

- mean placement
- top-1/top-4
- final HP
- final board strength
- placement histogram

Decision metrics are decision-level and may be counted once per hero decision:

- action mix
- legal action rates
- selected-from-prior
- guard override rate
- decisions/sec
- search source rates

Reports must make this distinction explicit. A trace comparison that computes
placement or final HP over action rows is invalid.

Action-surface metrics are also required:

- `action_surface`;
- candidate-board action count;
- oracle macro action rate;
- primitive board action allowance;
- candidate action rate;
- illegal action count.

Promotion evidence must use `action_surface = candidate_choice`, hero oracle
macros disabled, and hero primitive board moves disabled.

## Reset Goal Loop

Run these goals in order. Each goal has an attempt cap of `3`; after that,
record the blocker and stop. Every loop writes or updates `loop-state.json`,
`loop-run-log.md`, `metrics.json`, and `decision.md` where applicable.

### Goal 1: Evaluator Repair

Status: `pass`.

Objective: make trace outcome metrics episode-level and prove it with a
synthetic unequal-decision-count regression test.

Acceptance criteria:

- `summarize_trace_rows` reports `outcome_aggregation: episode`.
- Placement, final HP, and final board strength are averaged over unique
  episode outcomes, not decision rows.
- A test with equal episode outcomes but unequal decision counts produces zero
  placement and HP delta.
- `env -u UV_PYTHON uv run pytest tests/test_lobby_diagnostics.py` passes.
- A read-only `loop_verifier` accepts the fix.

Pause/kill criteria:

- Trace rows do not contain enough episode identity to recover episode-level
  outcomes.
- Conflicting outcomes appear for the same episode.
- Existing Stage tools require decision-weighted placement semantics.

### Goal 2: Repaired Stage 4 Ablation Matrix

Status: `pass`. Stage 5 remains locked until candidate-choice migration and
rerun artifacts pass the current runbook.

Objective: rerun the policy/planner comparisons on repaired metrics before
using any old Stage 2-4 claim.

Required policies:

- PPO reference
- Stage 3 student
- Stage 4 strong
- Stage 4 prior-only
- Stage 4 no-guard/model-only
- Stage 4 guard-only or prior-plus-guard
- Simulator MCTS subset if runtime allows

Required suites:

- mixed opponents
- frozen PPO snapshot pool
- public seeds
- heldout seeds

Acceptance criteria:

- Outcome metrics are episode-level.
- Decision metrics include action mix, macro action rates, selected-from-prior,
  guard override rate, and decisions per episode.
- At least `20` public and `20` heldout episodes per suite are run for any
  promotion claim. Smaller runs must be labeled smoke-only.
- Stage 4 strong is compared directly against PPO, Stage 3, prior-only,
  no-guard, and guard-only baselines.
- The report identifies whether any uplift comes from learned dynamics, the
  simulator guard, prior copying, or macro shortcuts.

Pause/kill criteria:

- Stage 4 strong is worse than PPO on repaired placement beyond tolerance.
- Guard-only reproduces most of the improvement.
- No-guard model search regresses sharply.
- Illegal actions or mask violations appear.

### Goal 3: Candidate-Choice Search Surface

Objective: migrate the search/replay scaffold from `raw_primitive_legacy` to the
same candidate-choice action surface used by claim-grade PPO.

Required work:

- merge or recreate candidate-board wrapper actions in the search worktree;
- expose candidate-board tails in legal masks consumed by search;
- disable oracle macros and primitive board moves for learned/search hero
  policies;
- resize policy targets and model action heads to the candidate-choice action
  dimension;
- encode recurrent model actions over candidate-choice ids, not only
  `NUM_ACTIONS`;
- write action-surface metadata into traces, datasets, checkpoints, manifests,
  and reports.

Acceptance criteria:

- Search traces report `action_surface: candidate_choice`.
- Candidate-board action count is positive.
- Oracle macro actions and primitive board actions are unavailable to the
  learned/search hero.
- Policy targets put all mass on legal candidate-choice actions.
- Legacy `raw_primitive_legacy` traces are rejected for MuZero promotion gates.
- Focused tests cover candidate action masks, trace metadata, dataset action
  dimensions, and dynamics recurrent action encoding.

Pause/kill criteria:

- Candidate wrapper actions cannot be made deterministic under clone/restore.
- Candidate policy targets cannot preserve legal-mask semantics.
- Existing checkpoint compatibility would require breaking old raw-action
  artifact loading instead of adding explicit action-surface metadata.

### Goal 4: Stage 5 Replay/Scaffold Smoke

Objective: implement the smallest Stage 5 feedback loop only after Goals 1-3
pass.

Required loop:

```text
generate search-improved lobby traces
-> append replay/search-target buffer
-> train or fine-tune a candidate
-> evaluate repaired public and heldout suites
-> reject or promote by gate
```

Replay rows must include observation, legal mask, behavior prior, model logits
or policy target, value target, reward, done, episode id, decision index, seed,
suite name, opponent policy, player-order mode, checkpoint id, and search
config.

Acceptance criteria:

- One smoke iteration writes replay data, train metrics, eval metrics, and
  decision artifacts under
  `artifacts/muzero_stage/candidate_choice_rerun/stage5/`.
- Evaluation includes PPO, Stage 3, Stage 4 strong, and the Stage 5 candidate.
- Reports separate pre-guard and post-guard selected actions.
- Illegal actions are `0`.

Pause/kill criteria:

- Replay fields are missing or ambiguous.
- Targets are mostly uniform, prior-copying, or guard-only.
- Training improves loss but worsens repaired evaluation.

### Goal 5: Guarded Stage 5 Promotion

Objective: promote a Stage 5 candidate only if repaired metrics show useful
improvement that is not explained entirely by guard shortcuts.

Promotion criteria:

- Mixed placement and HP do not meaningfully regress versus both PPO and Stage
  4 strong.
- Frozen placement improves by at least `0.10` or a predeclared composite
  improves without hiding placement regression.
- Final board strength does not regress by more than `10` versus PPO or Stage 4
  strong.
- Selected-from-prior drops below `0.95`, or at least `5%` non-prior actions
  have nonnegative paired outcome contribution.
- No-guard Stage 5 does not collapse versus PPO/Stage 3.
- Guard-only does not explain all improvement.
- A read-only `loop_verifier` returns `ACCEPT`.

Additional candidate-choice requirements:

- `action_surface = candidate_choice` in every promoted trace/report.
- Candidate action rate is reported and nonzero unless the policy genuinely
  does not need board changes in that episode.
- Oracle macro action rate is zero.
- Primitive board actions are masked for the learned/search hero.

## Immediate Goal Prompt

Use the more detailed prompt in
`docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md#goal-prompt` as the current Codex
goal. The compact form is:

```text
/goal Achieve the candidate-choice MuZero rerun path for MiniTFT. Follow
docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md exactly, work in
/mnt/ssd2/Projects/TFT-zero-candidate-choice-muzero-rerun on branch
feat/candidate-choice-muzero-rerun, and stop only when every runbook acceptance
criterion is met or the runbook blocked condition is reached.
```

## Agent Run Rules

- Use `env -u UV_PYTHON uv run ...` for project commands.
- Keep Stage 5 artifacts under
  `artifacts/muzero_stage/candidate_choice_rerun/stage5/`.
- Keep attempt cap explicit. Default cap is `3`.
- Record `loop-state.json`, `loop-run-log.md`, `metrics.json`, and
  `decision.md` for unattended or repeated loops.
- Do not run long PPO continuation as a substitute for Stage 5.
- Do not change protected simulator/evaluator behavior to make a candidate pass
  unless a failing test proves the evaluator itself is wrong.
- Stop immediately on illegal actions, broken masks, reward-sign suspicion,
  missing replay fields, decision-weighted outcomes, or regression against both
  PPO and the prior promoted baseline.
- Before calling loop work done, run a read-only `loop_verifier` subagent.
