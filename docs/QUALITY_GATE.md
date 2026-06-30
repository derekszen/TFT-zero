# Quality Gate

Use this gate before claiming that the strategic lane, Puffer speed path,
MuZero-style cache path, or playable demo is working.

## Gate Summary

| Gate | Question | Required Evidence |
| --- | --- | --- |
| Intent | Which deliverable is this proving? | `decision.md` names MuZero cache, Puffer speed, playable demo, docs, or blocker. |
| Architecture | Does the work use shared rules? | No duplicated simulator rules inside adapters. |
| Markov | Does transition/reward depend only on current state, action, and simulator randomness? | State-schema review plus fixed-seed branch test. |
| Determinism | Are fixed seeds reproducible? | Trace/checksum test or artifact comparison. |
| Masking | Are illegal actions blocked? | Legal-mask test and zero illegal actions in smoke artifacts. |
| Pressure | Is the round-based enemy curve sane? | Heuristic death rate is nonzero and most weak/basic runs do not coast to max round. |
| Baseline | Is there a comparison point? | Random and at least one heuristic or previous artifact. |
| Artifact | Can the result be audited later? | `metrics.json`, `decision.md`, `final_report.md`, and logs. |
| Verifier | Did an independent read-only review accept it? | Verifier `ACCEPT`, or documented unavailable fallback. |

## Required Commands

Run the smallest relevant subset first, then the full practical set before a
claim:

```bash
env -u UV_PYTHON uv run pytest
env -u UV_PYTHON uv run ruff check
env -u UV_PYTHON uv run --all-extras pyright
git diff --check
```

For simulator or UI changes:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.simulator_regression_gate --strict
env -u UV_PYTHON uv run python -m mini_tft.tools.web_ui_regression_gate --strict
```

For strategic simulator changes, include a Markov review:

- list every state field that affects future transitions or rewards;
- confirm no adapter-local history changes simulator outcomes;
- confirm hidden observation fields are documented as partial observability;
- run a fixed-seed branch/restore or equivalent trace test.

For enemy-pressure changes, include a calibration report:

- no opponent board/economy/shared-pool state in the strategic lane;
- enemy pressure derives from round and simulator randomness only;
- random/legal baseline is reported for context;
- basic TFT-shaped heuristic can die in most runs under default pressure;
- max-round heuristic survivors are near zero and few heuristic runs survive
  past Stage 5-3;
- `placement_proxy` is reported as an elimination-timing bucket, not real lobby placement;
- dense `scenario_score` is reported separately from placement buckets;
- placement proxy formula is written to `decision.md`.

For Puffer claims, include a matched benchmark artifact with:

- reference steps/sec;
- batched/Puffer steps/sec;
- speedup ratio;
- repeated-run variance;
- semantic parity result;
- decision: `pass`, `fail`, or `inconclusive`.

For MuZero-style cache claims, include:

- strategic parity matrix status when using native C++ or Ocean C fast paths;
- cache schema and row count;
- legal action mask validation;
- policy/value target validation;
- tiny policy/value/dynamics smoke metrics;
- baseline comparison;
- decision: `pass`, `fail`, or `smoke_only`.

For playable demo claims, include:

- launch or smoke command;
- screenshot/log/manual smoke note when practical;
- shared-rule path confirmation;
- blocked reason if no demo can launch.

## Artifact Schema

`metrics.json` should be machine-readable and contain at least:

```json
{
  "schema": "quality-gate/v1",
  "deliverable": "muzero_cache|puffer_speed|playable_demo|docs|other",
  "status": "pass|fail|inconclusive|smoke_only|blocked",
  "seed": 0,
  "commands": [],
  "metrics": {},
  "artifacts": [],
  "known_limits": []
}
```

`decision.md` should be short:

```md
# Decision

Status: pass|fail|inconclusive|smoke_only|blocked

Evidence:
- ...

Limits:
- ...

Next:
- ...
```

## Promotion Rules

- Do not promote smoke-only artifacts into quality claims.
- Do not treat Puffer import/training success as speed evidence.
- Do not treat MuZero cache creation as policy-quality evidence.
- Do not compare policies on unmatched seeds or incompatible action surfaces.
- Do not claim real lobby placement from `placement_proxy` buckets.
- Do not add multi-agent opponent state to the strategic lane without a new
  active runbook.
- If verifier rejects, the gate fails until fixed or explicitly overridden.
