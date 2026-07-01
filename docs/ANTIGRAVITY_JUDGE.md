# Antigravity Judge

Use this packet when a deliverable needs an independent read-only judge before
promotion or loop completion. On this machine, Antigravity is reached through
the local ai-router / CLIProxyAPI OpenAI-compatible endpoint, not through a
standalone `antigravity` CLI.

## Contract

- Preferred runner: Antigravity via ai-router.
- Preferred judge model: `gemini-3.5-flash-low`, the live Antigravity alias on
  this machine.
- Thinking: highest / `reasoning_effort=xhigh` where the router/model supports
  it.
- Mode: read-only. The judge may inspect code, docs, validation output, and
  listed artifacts, but must not edit files or mutate external systems.
- Gate behavior: fail closed. Missing, malformed, or `REJECT` verdicts block
  completion until fixed or explicitly overridden by the user.
- Provider path:
  `client -> http://127.0.0.1:8317/v1 -> CLIProxyAPI -> Antigravity OAuth`.

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
- `antigravity_ai_router_command.txt`: read-only command that calls ai-router
  and writes `verdict.md`.
- `decision.md`: current gate state, blocked until strict `ACCEPT`.
- `metrics.json`: packet metadata.

## Run With ai-router Antigravity

1. Confirm ai-router Antigravity auth is available:

   ```bash
   cd /mnt/ssd2/Projects/ai-router
   bin/select-auth current antigravity
   ```

2. If login is missing or expired, refresh OAuth:

   ```bash
   cd /mnt/ssd2/Projects/ai-router
   bin/login-antigravity
   ```

3. If needed, switch accounts:

   ```bash
   cd /mnt/ssd2/Projects/ai-router
   bin/select-auth list antigravity
   bin/select-auth use antigravity derek --restart
   ```

4. Run the generated command:

   ```bash
   bash artifacts/judge/<name>/antigravity_ai_router_command.txt
   ```

The command evaluates `bin/ai-env antigravity`, overrides
`MODEL=gemini-3.5-flash-low`, writes `artifacts/judge/<name>/verdict.md`, and then
runs the strict verdict checker.

Accept only if the output starts directly with `Verdict: ACCEPT`, with no
Markdown code fence or wrapper text, and includes non-empty `Evidence checked`,
`Findings`, and `Suggested action` sections. Otherwise the gate remains blocked.

## Check The Verdict

Run the strict local checker:

```bash
env -u UV_PYTHON uv run python -m mini_tft.tools.judge_packet \
  --check-verdict artifacts/judge/<name>/verdict.md
```

The checker exits nonzero unless the verdict file is well-formed and says
`Verdict: ACCEPT`.

## Smoke Test

Judge packets use the live `gemini-3.5-flash-low` Antigravity alias and still
request highest available reasoning. A quick router smoke test is:

```bash
cd /mnt/ssd2/Projects/ai-router
eval "$(bin/ai-env antigravity)"
MODEL=gemini-3.5-flash-low bin/test-chat
```
