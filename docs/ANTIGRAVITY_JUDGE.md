# Antigravity Judge

Use this packet when a deliverable needs an independent read-only judge before
promotion or loop completion.

## Contract

- Preferred runner: Antigravity IDE/manual review.
- Preferred judge model: Flash 3.5.
- Thinking: high.
- Mode: read-only. The judge may inspect code, docs, validation output, and
  listed artifacts, but must not edit files or mutate external systems.
- Gate behavior: fail closed. Missing, malformed, or `REJECT` verdicts block
  completion until fixed or explicitly overridden by the user.

The judge output must use exactly this schema:

```md
Verdict: ACCEPT|REJECT
Evidence checked:
- <file, command, or artifact actually inspected>
Findings:
- <issue, risk, or None.>
Suggested action:
- <next action>
```

## Generate A Packet

Create a reusable packet under `artifacts/judge/<name>/`:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.judge_packet \
  --name strategic-cache-smoke \
  --deliverable muzero_cache \
  --objective "Judge whether the MuZero cache smoke evidence is complete and correctly labeled." \
  --changed-file src/mini_tft/strategic/adapters/muzero_cache/export.py \
  --changed-file docs/STRATEGIC_LANE.md \
  --evidence artifacts/strategic_lane/muzero_cache/metrics.json \
  --evidence artifacts/strategic_lane/muzero_cache/decision.md \
  --command "env -u UV_PYTHON uv run pytest tests/test_strategic_core.py"
```

The packet contains:

- `prompt.md`: instructions to paste/open in Antigravity.
- `evidence_manifest.json`: machine-readable scope and evidence list.
- `verdict_template.md`: default fail-closed verdict shape.
- `gemini_fallback_command.txt`: optional fallback command shape.
- `decision.md`: current gate state, blocked until strict `ACCEPT`.
- `metrics.json`: packet metadata.

## Run In Antigravity

1. Open the worktree in Antigravity.
2. Select Flash 3.5 and set thinking to high.
3. Keep the session read-only.
4. Open `artifacts/judge/<name>/prompt.md` and the listed evidence files.
5. Save the judge response to `artifacts/judge/<name>/verdict.md`.

Accept only if the output starts with `Verdict: ACCEPT` and includes non-empty
`Evidence checked`, `Findings`, and `Suggested action` sections. Otherwise the
gate remains blocked.

## Check The Verdict

Run the strict local checker:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.judge_packet \
  --check-verdict artifacts/judge/<name>/verdict.md
```

The checker exits nonzero unless the verdict file is well-formed and says
`Verdict: ACCEPT`.

## CLI Availability

This repo does not assume an Antigravity CLI. If `antigravity` is unavailable on
`PATH`, use Antigravity as an external IDE/manual runner and keep the local gate
blocked until `verdict.md` is supplied.

The Gemini CLI fallback is optional. In this environment, `gemini --help`
verified the read-only `--approval-mode plan`, `--model`, `--prompt`, and
`--output-format` options. Model access and the exact local model id remain
account/config dependent.

Fallback shape:

```bash
gemini --approval-mode plan \
  --model gemini-3.5-flash \
  --output-format text \
  --prompt "Run this read-only MiniTFT judge packet. Return only the strict schema." \
  < artifacts/judge/<name>/prompt.md
```

If your Gemini CLI names Flash 3.5 differently, replace only the model id. Keep
`--approval-mode plan` and do not use auto-edit or yolo modes for judge work.
