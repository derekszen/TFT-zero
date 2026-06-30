# Strategic Parity Matrix

Status: `pass`
Oracle backend: `python`
Backends: python, native_cpp, ocean_c
Seeds: 0, 1, 7, 19
Total checks: 48
Failed checks: 0

| Backend | Scenario | Seed | Status | Rows | Mismatches |
| --- | --- | ---: | --- | ---: | ---: |
| native_cpp | reset_only | 0 | pass | 1 | 0 |
| ocean_c | reset_only | 0 | pass | 1 | 0 |
| native_cpp | reset_only | 1 | pass | 1 | 0 |
| ocean_c | reset_only | 1 | pass | 1 | 0 |
| native_cpp | reset_only | 7 | pass | 1 | 0 |
| ocean_c | reset_only | 7 | pass | 1 | 0 |
| native_cpp | reset_only | 19 | pass | 1 | 0 |
| ocean_c | reset_only | 19 | pass | 1 | 0 |
| native_cpp | economy_rounds | 0 | pass | 4 | 0 |
| ocean_c | economy_rounds | 0 | pass | 4 | 0 |
| native_cpp | economy_rounds | 1 | pass | 4 | 0 |
| ocean_c | economy_rounds | 1 | pass | 4 | 0 |
| native_cpp | economy_rounds | 7 | pass | 4 | 0 |
| ocean_c | economy_rounds | 7 | pass | 4 | 0 |
| native_cpp | economy_rounds | 19 | pass | 4 | 0 |
| ocean_c | economy_rounds | 19 | pass | 4 | 0 |
| native_cpp | roll_buy_field | 0 | pass | 8 | 0 |
| ocean_c | roll_buy_field | 0 | pass | 8 | 0 |
| native_cpp | roll_buy_field | 1 | pass | 8 | 0 |
| ocean_c | roll_buy_field | 1 | pass | 8 | 0 |
| native_cpp | roll_buy_field | 7 | pass | 8 | 0 |
| ocean_c | roll_buy_field | 7 | pass | 8 | 0 |
| native_cpp | roll_buy_field | 19 | pass | 8 | 0 |
| ocean_c | roll_buy_field | 19 | pass | 8 | 0 |
| native_cpp | level_tempo | 0 | pass | 7 | 0 |
| ocean_c | level_tempo | 0 | pass | 7 | 0 |
| native_cpp | level_tempo | 1 | pass | 7 | 0 |
| ocean_c | level_tempo | 1 | pass | 7 | 0 |
| native_cpp | level_tempo | 7 | pass | 7 | 0 |
| ocean_c | level_tempo | 7 | pass | 7 | 0 |
| native_cpp | level_tempo | 19 | pass | 7 | 0 |
| ocean_c | level_tempo | 19 | pass | 7 | 0 |
| native_cpp | illegal_actions | 0 | pass | 4 | 0 |
| ocean_c | illegal_actions | 0 | pass | 4 | 0 |
| native_cpp | illegal_actions | 1 | pass | 4 | 0 |
| ocean_c | illegal_actions | 1 | pass | 4 | 0 |
| native_cpp | illegal_actions | 7 | pass | 4 | 0 |
| ocean_c | illegal_actions | 7 | pass | 4 | 0 |
| native_cpp | illegal_actions | 19 | pass | 4 | 0 |
| ocean_c | illegal_actions | 19 | pass | 4 | 0 |
| native_cpp | terminal_pressure | 0 | pass | 11 | 0 |
| ocean_c | terminal_pressure | 0 | pass | 11 | 0 |
| native_cpp | terminal_pressure | 1 | pass | 11 | 0 |
| ocean_c | terminal_pressure | 1 | pass | 11 | 0 |
| native_cpp | terminal_pressure | 7 | pass | 11 | 0 |
| ocean_c | terminal_pressure | 7 | pass | 11 | 0 |
| native_cpp | terminal_pressure | 19 | pass | 11 | 0 |
| ocean_c | terminal_pressure | 19 | pass | 11 | 0 |

## Limits

- native_cpp parity compares Python-compatible state signatures
- ocean_c parity allows fp32-vs-fp64 tolerance for combat float fields
- ocean_c parity compares no-reset trace rows, not Puffer's auto-reset worker loop
- parity does not prove policy quality or MuZero learning quality
