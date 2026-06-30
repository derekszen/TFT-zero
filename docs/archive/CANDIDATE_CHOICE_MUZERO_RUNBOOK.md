# Candidate-Choice MuZero Rerun Runbook

This runbook is the autonomous contract for the next MuZero-stage rerun. It
replaces the old raw-action Stage 2-5 path with the same non-oracle
candidate-board action surface that produced the current claim-grade PPO result.

Do not use this runbook to promote legacy `raw_primitive_legacy` search traces.
Those traces remain useful for clone/restore, mask, and evaluator debugging
only.

Use this as one autonomous goal, not as five separate manual tickets. A passing
test, successful import, first Stage 2 trace, or smoke training run is only a
checkpoint. Completion requires the full rerun path, validation, verifier
acceptance, and a final evidence report, or an explicit blocked stop.

## Objective

Build and validate a candidate-choice MuZero rerun path:

```text
candidate-choice PPO baseline
-> candidate-choice simulator-backed MCTS
-> candidate-choice search-target dataset
-> candidate-choice student/dynamics reruns
-> Stage 5 replay smoke
-> repaired public/heldout evaluation
```

The target claim is narrow: MuZero-style planning over candidate-board choices
in the MiniTFT toy lobby. It is not raw primitive slot-control learning and not
real TFT rank.

If candidate-choice Stage 5 reaches a target-quality, feature-signal, or
search-target blocker, use `docs/MUZERO_OVERNIGHT_ROOT_CAUSE_RUNBOOK.md` for
the overnight root-cause loop before starting any full candidate training run.
That loop is the go/no-go gate for whether the next real Stage 5 candidate
training attempt is justified.

## Goal Prompt

Use this exact prompt for the next long-running Codex goal:

```text
/goal Achieve the candidate-choice MuZero rerun path for MiniTFT, verified by candidate-choice PPO baseline/eval artifacts, candidate-choice Stage 2 MCTS traces, Stage 3 search-target dataset/student artifacts, Stage 4 dynamics/ablation artifacts, Stage 5 replay-smoke artifacts where feasible, focused tests, full practical validation, and independent verifier acceptance, while preserving raw-action/debug/UI compatibility, legal action masks, reward semantics, checkpoint compatibility, and unrelated dirty work.

Follow docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md exactly. Work in a separate worktree at /mnt/ssd2/Projects/TFT-zero-candidate-choice-muzero-rerun on branch feat/candidate-choice-muzero-rerun. Do not modify the main dirty workspace except for explicitly required loop logs or accepted artifacts named by the runbook.

Do not stop after the first passing test, first successful candidate-action import, first benchmark, first Stage 2 trace, or first smoke PPO/eval run. Treat each green result as a checkpoint only. Continue through orientation, worktree setup, candidate-action surface merge, candidate-choice PPO baseline verification, Stage 2 candidate-choice MCTS, Stage 3 candidate-choice distillation, Stage 4 candidate-choice dynamics/ablation, Stage 5 replay smoke where feasible, regression tests, similar-issue sweep, full validation, self-review, documentation cleanup, OpenCode Go plan verifier using opencode-go/deepseek-v4-pro, secondary loop_verifier review, and final evidence report.

Success requires every acceptance criterion in docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md to be met, including action_surface=candidate_choice in promotion artifacts, candidate_board_actions > 0, oracle macro action rate 0, primitive board moves disabled for learned/search hero policies, candidate/macro/primitive action metrics reported, episode-level outcome metrics, required artifacts under artifacts/muzero_stage/candidate_choice_rerun/, final_report.md written, loop-state and loop-run-log updated, OpenCode Go verifier ACCEPT, and secondary loop_verifier ACCEPT.

Stop only when every acceptance criterion is met or the blocked condition in the runbook is reached. If blocked, write the exact blocker, evidence, attempted fixes, and the next required human decision.
```

## Action-Surface Contract

Promotion evidence must satisfy all of these:

- `action_surface = "candidate_choice"` in traces, datasets, checkpoints,
  manifests, metrics, and decisions.
- Candidate-board action count is positive.
- Hero oracle macro actions are disabled:
  - `FIELD_BEST_BOARD` unavailable to learned/search hero;
  - `SLAM_BEST_ITEM` unavailable to learned/search hero.
- Primitive bench/board slot moves are disabled for learned/search hero.
- Economy/shop/end actions remain available through legal masks.
- Primitive slot moves may remain available for UI/debug/scripted policies.
- Legacy `raw_primitive_legacy` artifacts cannot be promotion evidence.

## Known Current Blockers

These are expected at the start of this runbook and should be fixed in order:

- `RootMCTSPlanner` currently supports only `raw_primitive_legacy` and must stop
  hard-rejecting candidate-choice after the adapter is implemented.
- Stage 2 currently consumes raw `NUM_ACTIONS` masks/actions; it must consume
  candidate-tail masks/actions for claim-grade reruns.
- Stage 3 policy targets currently assume `NUM_ACTIONS`; they must use the
  candidate-choice action dimension and store action-surface metadata.
- Stage 4 transition datasets and recurrent dynamics currently assume raw
  action ids; they must support candidate-choice ids and checkpoint metadata.
- Guard and ablation paths must not consider oracle macros or primitive
  bench/board moves when the target surface is candidate-choice.
- Candidate-choice PPO eval tooling may need to be copied or merged from the
  non-oracle worktree if missing from the search worktree.
- There is no full Stage 5 runner yet; the first Stage 5 target is a minimal
  replay-smoke loop, not a promotion run.

## Candidate APIs To Reuse

Prefer copying or merging these surfaces from the non-oracle worktree instead
of inventing new action ids:

- `CANDIDATE_BOARD_ACTION_OFFSET = NUM_ACTIONS`
- `DEFAULT_CANDIDATE_BOARD_ACTIONS`
- `candidate_board_action(candidate_index)`
- `is_candidate_board_action(action, candidate_count=...)`
- `decode_candidate_board_action(action, candidate_count=...)`
- `action_name()` rendering `candidate_board_N`
- `generate_candidate_boards(state, data, config, top_k=...)`
- `candidate_board_mask(state, data, config, top_k=...)`
- `would_apply_candidate_board(...)`
- `apply_candidate_board(...)`

For training/eval parity, reuse the lobby hero settings:

```text
allow_oracle_macro_actions = false
candidate_board_actions = 3
allow_primitive_board_actions = false
```

The Puffer PPO/eval commands expose these as:

```text
--candidate-board-actions 3
--disallow-oracle-macro-actions
--disallow-primitive-board-actions
```

For `evaluate_puffer_lobby_hero`, oracle macro actions are disabled by default.
Do not pass `--allow-oracle-macro-actions` during claim-grade candidate-choice
evals.

## Worktree And Branch

Work in a separate worktree:

```text
/mnt/ssd2/Projects/TFT-zero-candidate-choice-muzero-rerun
```

Suggested branch:

```text
feat/candidate-choice-muzero-rerun
```

Use the current main repo as the search scaffold source and the current
non-oracle worktree as the candidate-action reference:

```text
/mnt/ssd2/Projects/TFT-zero
/mnt/ssd2/Projects/TFT-zero-non-oracle-action-surface
```

Do not modify unrelated dirty work in either source worktree. If code must be
copied or merged from the non-oracle worktree, inspect the diff first and bring
over only the candidate-action surfaces and tests needed for this runbook.

## Autonomous Execution Contract

Run phases in order and keep moving after each green checkpoint:

1. Capture orientation, allowance, and current blocker state.
2. Patch the smallest missing surface for the current phase.
3. Run focused validation for that phase.
4. Produce or refresh that phase's artifacts.
5. Update root and stage `metrics.json`, `decision.md`, `loop-state.json`, and
   `loop-run-log.md`.
6. Search for the same failure pattern in adjacent stage code.
7. Continue to the next phase unless the success or blocked condition is met.

Do not start a long PPO run as a substitute for the rerun path. Use an existing
candidate-choice checkpoint only after validating its manifest/action dimension;
otherwise run a smoke PPO sufficient to exercise the candidate-choice pipeline.
Label anything below the promotion episode counts as smoke-only.

If a phase fails, first fix the implementation or test gap, then rerun the
focused command and the affected stage command. The attempt cap applies to the
same evidence-backed blocker, not to unrelated follow-on bugs.

## Attempt Cap

Attempt cap: `3` full implementation attempts.

One attempt means:

```text
plan -> patch -> focused validation -> rerun relevant stage/eval -> review
```

After three attempts with the same blocker, stop as blocked and write the
blocker, evidence, attempted fixes, and required human decision.

## Required Artifacts

Write all rerun artifacts under:

```text
artifacts/muzero_stage/candidate_choice_rerun/
```

Required files:

```text
artifacts/muzero_stage/candidate_choice_rerun/
  loop-state.json
  loop-run-log.md
  metrics.json
  decision.md
  commands.md
  final_report.md
  candidate_ppo/
  stage2/
  stage3/
  stage4/
  stage5/
  verifier/
```

Each stage directory must contain its own `metrics.json`, `decision.md`, and any
trace/dataset/checkpoint artifacts it produced.

Root `metrics.json` must summarize, per stage:

- `status`: `pass`, `smoke`, `blocked`, or `not_run`;
- artifact paths;
- action surface and action dimension;
- candidate-board action count;
- oracle macro action rate;
- primitive board action allowance/rate;
- illegal action count;
- episode count and whether results are smoke-only.

`loop-state.json` must use this shape:

```json
{
  "owner": "codex",
  "schema": "candidate-choice-muzero-loop-state/v1",
  "loop": "candidate_choice_muzero_rerun",
  "level": "L1 report-only",
  "attempt": 1,
  "attempt_cap": 3,
  "current_phase": "phase_0_orientation",
  "current_next_action": "patch_stage2_candidate_choice_adapter",
  "worktree": "/mnt/ssd2/Projects/TFT-zero-candidate-choice-muzero-rerun",
  "branch": "feat/candidate-choice-muzero-rerun",
  "action_surface": "candidate_choice",
  "candidate_board_actions": 3,
  "allow_oracle_macro_actions": false,
  "allow_primitive_board_actions": false,
  "codex_allowance_check": {
    "source": "unknown",
    "checked_at": "",
    "five_hour_window_remaining": "unknown",
    "weekly_usage": "unknown",
    "decision": "continue"
  },
  "stages": {
    "candidate_ppo": {"status": "not_run", "artifacts": []},
    "stage2": {"status": "not_run", "artifacts": []},
    "stage3": {"status": "not_run", "artifacts": []},
    "stage4": {"status": "not_run", "artifacts": []},
    "stage5": {"status": "not_run", "artifacts": []}
  },
  "verifiers": {
    "opencode_go_deepseek_v4_pro": "not_run",
    "loop_verifier": "not_run"
  },
  "pause_criteria": [
    "soft-pause if Codex weekly usage is at or above 70 percent",
    "pause before any long PPO run not required by acceptance"
  ],
  "kill_criteria": [
    "illegal actions in claim-grade candidate-choice traces",
    "oracle macro action rate above zero in learned/search hero artifacts",
    "decision-weighted outcome metrics in promotion reports",
    "same blocker persists after three evidence-backed attempts"
  ],
  "prune_rules": [
    "keep metrics, decisions, final reports, verifier outputs, and promoted checkpoints",
    "prune redundant smoke traces only after metrics and decisions reference them"
  ]
}
```

`commands.md` must record each command exactly as run, the exit status, and the
artifact or log file that proves the result. Do not rely on memory in the final
report.

Append every loop attempt to `loop-run-log.md` with:

- timestamp;
- attempt number;
- action taken;
- validation run;
- artifact paths;
- verifier verdict if available;
- next action.

## Codex Allowance Check

At the start of the run, append this block to `loop-run-log.md`:

```md
## Codex Allowance Check
Source: /status | Codex usage dashboard | unknown
Checked at:
5h window remaining:
Weekly usage:
Decision: continue | soft-pause | hard-pause
```

If live allowance is unavailable, record `unknown` and continue only with the
bounded implementation/eval steps in this runbook. Do not start extra long
experiments outside the acceptance path.

## Phase 0: Orientation

Read before changing code:

- `README.md`
- `AGENTS.md`
- `docs/MUZERO_STAGE_GATE.md`
- this runbook
- `docs/NON_ORACLE_ACTION_SURFACE_RUNBOOK.md` from the non-oracle worktree if
  present
- `src/mini_tft/core/actions.py`
- `src/mini_tft/core/board.py`
- `src/mini_tft/rl/lobby_env.py`
- `src/mini_tft/rl/puffer_env.py`
- `src/mini_tft/search/mcts.py`
- `src/mini_tft/search/distill.py`
- `src/mini_tft/search/dynamics.py`
- `src/mini_tft/tools/stage2_mcts.py`
- `src/mini_tft/tools/stage3_distill.py`
- `src/mini_tft/tools/stage4_dynamics.py`
- `src/mini_tft/tools/stage4_ablation_matrix.py`
- current git diff/status

Write a short implementation plan to `loop-state.json` before patching.

## Phase 1: Candidate-Choice PPO Baseline

Objective: establish or reuse a valid candidate-choice PPO checkpoint/eval
surface.

Required work:

- Ensure candidate-board wrapper actions are available in the working branch.
- Ensure `mini_tft.tools.evaluate_puffer_lobby_hero` is present or merge it from
  the non-oracle worktree.
- Ensure Puffer lobby training/eval can run with:
  - candidate-board actions enabled;
  - hero oracle macros disabled;
  - primitive board moves disabled for hero.
- If a valid local checkpoint already exists, verify its manifest/action surface
  before reusing it.
- If no valid checkpoint exists, run a short smoke PPO only.

Required verification:

```bash
env -u UV_PYTHON uv run --extra train pytest -q tests/test_lobby_env.py tests/test_puffer_env.py tests/test_train_puffer_ppo.py tests/test_checkpoint_policy.py
env -u UV_PYTHON uv run --all-extras pyright
```

Expected smoke PPO command shape if no compatible checkpoint exists:

```bash
env -u UV_PYTHON uv run --extra puffer --extra train python -m mini_tft.rl.train_puffer_ppo \
  --env-kind lobby \
  --puffer-backend native \
  --lobby-opponent-policy non_oracle_mixed \
  --candidate-board-actions 3 \
  --disallow-oracle-macro-actions \
  --disallow-primitive-board-actions \
  --timesteps 4096 \
  --num-envs 4 \
  --n-steps 64 \
  --batch-size 256 \
  --update-epochs 1 \
  --device cpu \
  --out artifacts/muzero_stage/candidate_choice_rerun/candidate_ppo/candidate_choice_smoke.pt
```

Expected candidate-choice eval shape:

```bash
env -u UV_PYTHON uv run --extra train python -m mini_tft.tools.evaluate_puffer_lobby_hero \
  --checkpoint <candidate-choice-ppo.pt> \
  --episodes 20 \
  --opponent-policy non_oracle_mixed \
  --candidate-board-actions 3 \
  --disallow-primitive-board-actions \
  --strict \
  --out artifacts/muzero_stage/candidate_choice_rerun/candidate_ppo/eval.json
```

Acceptance:

- Evaluation reports `candidate_board_actions > 0`.
- Evaluation reports oracle macro action rate `0`.
- Evaluation reports illegal actions `0`.
- Primitive board moves are disabled for the learned hero.
- Mean placement/top-4 are recorded, even if smoke-only.
- The evaluated checkpoint action dimension equals the candidate-choice env
  action dimension.

## Phase 2: Candidate-Choice Search Adapter

Objective: make Stage 2 MCTS search over the candidate-choice action space.

Required work:

- Add a shared candidate-choice action-surface adapter for:
  - action dimension;
  - legal mask;
  - action names;
  - action application;
  - trace metadata.
- Keep legacy raw-action planner compatibility behind
  `action_surface = "raw_primitive_legacy"`.
- Add candidate-choice MCTS config options:
  - `action_surface`;
  - `candidate_board_actions`;
  - `allow_oracle_macro_actions`;
  - `allow_primitive_board_actions`.
- Add matching Stage 2 CLI flags:
  - `--action-surface candidate_choice`;
  - `--candidate-board-actions 3`;
  - `--disallow-oracle-macro-actions`;
  - `--disallow-primitive-board-actions`.
- Make legacy raw-action traces fail promotion checks when the target surface is
  candidate-choice.

Required verification:

```bash
env -u UV_PYTHON uv run --extra train pytest -q tests/test_stage2_mcts.py
env -u UV_PYTHON uv run ruff check src/mini_tft/search/mcts.py src/mini_tft/tools/stage2_mcts.py tests/test_stage2_mcts.py
```

Required artifact:

```text
artifacts/muzero_stage/candidate_choice_rerun/stage2/
```

Expected Stage 2 command shape after the adapter exists:

```bash
env -u UV_PYTHON uv run --extra train python -m mini_tft.tools.stage2_mcts \
  --out-dir artifacts/muzero_stage/candidate_choice_rerun/stage2 \
  --promoted-checkpoint <candidate-choice-ppo.pt> \
  --action-surface candidate_choice \
  --candidate-board-actions 3 \
  --disallow-oracle-macro-actions \
  --disallow-primitive-board-actions \
  --smoke-episodes 1 \
  --eval-episodes 3
```

Acceptance:

- Stage 2 search traces report `action_surface = candidate_choice`.
- Considered/selected actions are legal under the candidate-choice mask.
- Candidate action rate and macro action rate are reported.
- Oracle macro action rate is `0`.
- Illegal actions are `0`.
- Clone/restore replay remains deterministic.

## Phase 3: Candidate-Choice Distillation

Objective: rerun Stage 3 target generation and student training with
candidate-choice policy targets.

Required work:

- Policy target tensors use the candidate-choice action dimension.
- Dataset schema records action surface and candidate count.
- Dataset validation rejects:
  - illegal selected actions;
  - policy mass on masked actions;
  - raw-action traces when candidate-choice is required.
- Student checkpoint metadata records action surface and action dimension.
- Add matching Stage 3 CLI/config plumbing for `action_surface`,
  `candidate_board_actions`, and oracle/primitive action restrictions.

Required verification:

```bash
env -u UV_PYTHON uv run --extra train pytest -q tests/test_stage3_distill.py
env -u UV_PYTHON uv run ruff check src/mini_tft/search/distill.py src/mini_tft/tools/stage3_distill.py tests/test_stage3_distill.py
```

Required artifact:

```text
artifacts/muzero_stage/candidate_choice_rerun/stage3/
```

Expected Stage 3 command shape after candidate-choice schema support exists:

```bash
env -u UV_PYTHON uv run --extra train python -m mini_tft.tools.stage3_distill \
  --out-dir artifacts/muzero_stage/candidate_choice_rerun/stage3 \
  --stage2-decision artifacts/muzero_stage/candidate_choice_rerun/stage2/decision.md \
  --stage2-metrics artifacts/muzero_stage/candidate_choice_rerun/stage2/metrics.json \
  --promoted-checkpoint <candidate-choice-ppo.pt> \
  --teacher-episodes 1 \
  --eval-episodes 3
```

Acceptance:

- Stage 3 dataset validates under candidate-choice schema.
- Student policy loads and selects legal candidate-choice actions.
- Evaluation is episode-level and reports placement/top-4/HP/board strength.
- Smoke-only runs are labeled smoke-only unless they satisfy the promotion
  episode count.

## Phase 4: Candidate-Choice Dynamics And Ablations

Objective: rerun Stage 4 dynamics/model-search on candidate-choice actions and
attribute any improvement.

Required work:

- Dynamics transition dataset records candidate-choice action surface.
- Recurrent action encoding supports the candidate-choice action dimension.
- Model-search config carries action surface and candidate count.
- Stage 4 model checkpoints record action surface, candidate count, and action
  dimension.
- Stage 4 ablation matrix compares:
  - PPO candidate-choice baseline;
  - Stage 3 candidate-choice student;
  - Stage 4 candidate-choice model search;
  - prior-only;
  - no-guard/model-only;
  - guard-only or prior-plus-guard;
  - simulator MCTS subset if runtime allows.

Required verification:

```bash
env -u UV_PYTHON uv run --extra train pytest -q tests/test_stage4_dynamics.py
env -u UV_PYTHON uv run ruff check src/mini_tft/search/dynamics.py src/mini_tft/tools/stage4_dynamics.py src/mini_tft/tools/stage4_ablation_matrix.py tests/test_stage4_dynamics.py
```

Required artifact:

```text
artifacts/muzero_stage/candidate_choice_rerun/stage4/
```

Expected Stage 4 command shape after candidate-choice dynamics support exists:

```bash
env -u UV_PYTHON uv run --extra train python -m mini_tft.tools.stage4_dynamics \
  --out-dir artifacts/muzero_stage/candidate_choice_rerun/stage4 \
  --stage2-metrics artifacts/muzero_stage/candidate_choice_rerun/stage2/metrics.json \
  --stage3-decision artifacts/muzero_stage/candidate_choice_rerun/stage3/decision.md \
  --stage3-metrics artifacts/muzero_stage/candidate_choice_rerun/stage3/metrics.json \
  --stage3-checkpoint artifacts/muzero_stage/candidate_choice_rerun/stage3/promoted/student_stage3.pt \
  --ppo-checkpoint <candidate-choice-ppo.pt> \
  --teacher-episodes 1 \
  --eval-episodes 3
```

Acceptance:

- Dynamics/action dimensions match candidate-choice masks.
- No-guard/model-only and guard-only results are reported separately.
- Selected-from-prior and guard override rates are reported.
- Outcome metrics are episode-level.
- Illegal actions are `0`.
- Any improvement is attributed as model, guard, prior-copying, or inconclusive.

## Phase 5: Stage 5 Replay Smoke

Objective: implement the smallest candidate-choice search-target feedback loop.

Required loop:

```text
generate candidate-choice search traces
-> append replay/search-target buffer
-> train or fine-tune one candidate
-> evaluate repaired public and heldout suites
-> reject or promote by gate
```

Replay rows must include:

- observation;
- legal mask;
- action surface;
- candidate-board action count;
- behavior prior;
- model logits or policy target;
- value target;
- reward;
- done;
- episode id;
- decision index;
- seed;
- suite name;
- opponent policy;
- player-order mode;
- checkpoint id;
- search config.

Required artifact:

```text
artifacts/muzero_stage/candidate_choice_rerun/stage5/
```

If no Stage 5 runner exists, implement the smallest runner that can write the
replay fields above and run one smoke iteration. Do not promote Stage 5 from an
ad hoc notebook/script that does not write `metrics.json`, `decision.md`, and
replay metadata.

Acceptance:

- One smoke iteration writes replay data, train metrics, eval metrics, and
  decision artifacts.
- Evaluation includes PPO baseline, Stage 3 student, Stage 4 candidate, and
  Stage 5 candidate where available.
- Reports separate pre-guard and post-guard selected actions.
- Illegal actions are `0`.
- Stage 5 is labeled smoke-only unless promotion criteria are met.

## Phase 6: Full Validation

Run the broadest practical validation before finalizing:

```bash
env -u UV_PYTHON uv run --extra train pytest -q tests/test_lobby_env.py tests/test_puffer_env.py tests/test_stage2_mcts.py tests/test_stage3_distill.py tests/test_stage4_dynamics.py tests/test_lobby_diagnostics.py
env -u UV_PYTHON uv run ruff check src/mini_tft/core src/mini_tft/rl src/mini_tft/search src/mini_tft/tools tests
env -u UV_PYTHON uv run --all-extras pyright
git diff --check
```

If a full command is too slow or fails for unrelated pre-existing dirt, run the
narrowest command that preserves exact failure evidence, record the blocker in
`loop-run-log.md`, and continue only if the runbook acceptance still has
evidence.

## Phase 7: Verifiers

Before final completion:

- Run a read-only `loop_verifier` subagent.
- Run the independent OpenCode Go plan verifier with
  `opencode-go/deepseek-v4-pro` if the local tooling is available.

Verifier prompt must ask whether:

- all promotion claims use candidate-choice action surface;
- raw primitive traces are rejected for promotion;
- metrics are episode-level;
- macro/primitive/candidate action rates are reported;
- artifacts and validation commands support the decision.

Treat verifier output as fail-closed:

- `ACCEPT`: may complete if all acceptance criteria are met.
- `REJECT`: fix or explicitly escalate; do not complete.

Use this OpenCode Go verifier command when the CLI is available:

```bash
opencode run \
  "Read-only verifier for the MiniTFT candidate-choice MuZero rerun. Do not modify files. Return Verdict: ACCEPT or REJECT, Findings, Evidence checked, and Suggested action. Reject if promotion evidence uses raw_primitive_legacy, oracle macro shortcuts, decision-weighted outcome metrics, missing candidate action metrics, missing required artifacts, or unvalidated action dimensions." \
  --dir "$PWD" \
  --model opencode-go/deepseek-v4-pro \
  --agent plan \
  --file docs/CANDIDATE_CHOICE_MUZERO_RUNBOOK.md \
  --file artifacts/muzero_stage/candidate_choice_rerun/metrics.json \
  --file artifacts/muzero_stage/candidate_choice_rerun/decision.md \
  --file artifacts/muzero_stage/candidate_choice_rerun/loop-state.json \
  --file artifacts/muzero_stage/candidate_choice_rerun/loop-run-log.md \
  --file artifacts/muzero_stage/candidate_choice_rerun/final_report.md
```

Do not use `--dangerously-skip-permissions`. If OpenCode Go is unavailable, stop
blocked unless the user explicitly approves a replacement verifier.

## Final Report

`final_report.md` must include:

- objective and final status;
- files changed;
- exact commands run and whether they passed;
- artifact index by stage;
- PPO baseline metrics;
- Stage 2/3/4/5 metrics and smoke/promotion label;
- action-surface compliance table;
- verifier verdicts;
- what was proven;
- what remains unproven or blocked;
- safest next action.

## Success Criteria

Stop as complete only when:

- candidate-choice PPO baseline is verified or a smoke baseline is clearly
  labeled;
- Stage 2 candidate-choice search traces exist and validate;
- Stage 3 candidate-choice dataset/student path validates;
- Stage 4 candidate-choice dynamics/ablation path validates;
- Stage 5 smoke replay loop either runs or is explicitly blocked after three
  evidence-backed attempts;
- all artifacts under `artifacts/muzero_stage/candidate_choice_rerun/` are
  written;
- `final_report.md` summarizes what was proven and what remains smoke-only;
- verifier returns `ACCEPT`;
- no unrelated changes were introduced.

## Blocked Conditions

Stop as blocked only after three evidence-backed attempts if:

- candidate-board action tails cannot be merged without breaking existing action
  compatibility;
- clone/restore is nondeterministic under candidate-choice action application;
- search policy targets cannot be aligned with candidate-choice masks;
- dynamics recurrent action encoding cannot handle the new action dimension
  without a broader redesign;
- no valid PPO checkpoint can be evaluated and a smoke checkpoint cannot be
  trained within resource limits;
- verifier rejects the same issue three times;
- continuing requires product/design decisions outside this runbook.

When blocked, write:

- exact blocker;
- command/log evidence;
- attempted fixes;
- recommended human decision;
- safest next action.
