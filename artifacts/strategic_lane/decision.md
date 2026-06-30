# Decision

Status: smoke_only

Evidence:
- Deliverable: other
- Seed: 0
- Artifacts: puffer_speed/metrics.json, puffer_speed/decision.md, muzero_cache/rows.jsonl, muzero_cache/metrics.json, muzero_cache/decision.md, playable_demo/initial_payload.json, playable_demo/metrics.json, playable_demo/decision.md, metrics.json, decision.md, final_report.md, loop-state.json, loop-run-log.md

Placement proxy bucket:
- Survive max_round -> 1
- Die at round >= 36 -> 2
- Die at round >= 32 -> 3
- Die at round >= 29 -> 4
- Die at round >= 25 -> 5
- Die at round >= 18 -> 6
- Die at round >= 11 -> 7
- Die earlier -> 8

Dense quality score:
- scenario_score = 0.45 * round_frac + 0.25 * hp_frac + 0.30 * strength_ratio

Limits:
- strategic lane is a simplified TFT-shaped simulator, not a full TFT clone
- MuZero cache artifact is smoke data; no dynamics/policy model is promoted here
- playable demo artifact is a shared-rules payload, not a launched browser route
- puffer speed artifact is native vector throughput evidence, not PPO quality evidence
- placement_proxy is an elimination-timing bucket, not real lobby placement

Next:
- Use these smoke artifacts as the baseline for real MuZero trainer wiring.
