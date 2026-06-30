# Strategic Lane Final Report

Status: `smoke_only`

## What Is Proven

- The clean strategic rules can emit legal cache rows for MuZero-style data.
- The benchmark path writes native Puffer vector throughput metrics with scalar parity.
- The playable adapter can serialize a shared-rules state payload.
- The heuristic baseline usually dies under the tuned enemy curve.

## Key Metrics

- Cache rows: 128
- Cache legal action rate: 1.000
- Puffer benchmark kind: strategic_native_puffer_vector
- Native strategic Puffer vector speed ratio: 11.23x
- Puffer speed decision: pass
- Semantic parity: True
- Puffer trainer: True
- Playable legal actions: 5
- Playable stage label: Stage 1-1
- Heuristic mean placement: 6.656
- Heuristic placement counts: {'6': 11, '7': 21}
- Heuristic mean scenario score: 0.339
- Heuristic median final round: 17.0
- Heuristic death rate: 1.000
- Heuristic survivor rate: 0.000
- Heuristic mean HP: 0.000

## Known Limits

- strategic lane is a simplified TFT-shaped simulator, not a full TFT clone
- MuZero cache artifact is smoke data; no dynamics/policy model is promoted here
- playable demo artifact is a shared-rules payload, not a launched browser route
- puffer speed artifact is native vector throughput evidence, not PPO quality evidence
- placement_proxy is an elimination-timing bucket, not real lobby placement

## Next

- Add a browser route or mode that uses `state_payload` and strategic `step`.
- Add a tiny policy/value/dynamics smoke trainer over `rows.jsonl`.
- Use `--env-kind strategic --puffer-backend native` for PPO speed experiments.
