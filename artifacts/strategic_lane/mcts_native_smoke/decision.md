# Strategic MCTS Smoke

Status: `smoke_only`

## Evidence

- Seed: 0
- Episodes per policy: 4
- Simulation counts: 8, 16, 32
- Elapsed sec: 0.014

| Policy | Sims | Mean placement | Mean final round | Death rate | Scenario score | Reward | Decisions/sec | Sims/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic | 0 | 7.000 | 12.000 | 1.000 | 0.218 | -2.129 | 0.00 | 0.00 |
| mcts_16 | 16 | 7.000 | 12.000 | 1.000 | 0.218 | -2.119 | 39990.83 | 639853.24 |
| mcts_32 | 32 | 7.000 | 12.000 | 1.000 | 0.218 | -2.152 | 19897.21 | 636710.78 |
| mcts_8 | 8 | 7.000 | 12.000 | 1.000 | 0.218 | -2.119 | 78393.54 | 627148.28 |
| random | 0 | 7.250 | 10.750 | 1.000 | 0.202 | -2.651 | 0.00 | 0.00 |

## Limits

- simulator-backed MCTS uses the real strategic simulator, not learned dynamics
- placement_proxy is an elimination bucket, not real TFT placement
- this is a smoke run unless episode counts and seed ranges are expanded
