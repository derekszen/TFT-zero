# Legacy Code Map

This repo keeps several historical research lanes because tests and artifacts
still refer to them. Treat them as quarantined until their deletion gates pass.

## Active Lane

Use these for current report and overnight work:

```text
src/mini_tft/strategic/core/              canonical strategic rules
src/mini_tft/strategic/adapters/          cache, MCTS, Puffer, web/demo adapters
src/mini_tft/strategic/ocean/             PufferLib 4 Ocean C speed path
src/mini_tft/tools/run_strategic_muzero_overnight.py
src/mini_tft/tools/judge_packet.py
src/mini_tft/tools/strategic_policy_improvement_gate.py
```

## Historical But Still Referenced

| Area | Current status | Do not delete until |
| --- | --- | --- |
| `src/mini_tft/rl/puffer_env.py` and `src/mini_tft/rl/train_puffer_ppo.py` | Legacy Puffer wrapper/PPO path. PufferLib 4 does not expose all older wrapper APIs, so some tests skip when unavailable. | README/docs stop referencing the route, checkpoint loaders no longer import `MaskedActorCritic`, and `tests/test_puffer_env.py`, `tests/test_train_puffer_ppo.py`, and `tests/test_checkpoint_policy.py` are replaced or removed. |
| `src/mini_tft/search/` plus `src/mini_tft/tools/stage2_mcts.py`, `stage3_distill.py`, `stage4_dynamics.py`, `stage4_ablation_matrix.py` | Historical raw-action Stage 2-4 MuZero scaffold. | Strategic overnight/cache/search tools cover the needed behavior and `tests/test_stage2_mcts.py`, `tests/test_stage3_distill.py`, and `tests/test_stage4_dynamics.py` are retired. |
| `src/mini_tft/core/`, `src/mini_tft/rl/lobby_env.py`, and toy PPO tooling | Existing toy simulator, lobby env, UI reference, and old PPO baselines. | Web/debug UI and lobby/reference tests no longer import them, or a strategic web/demo replacement is promoted. |
| `src/mini_tft/metatft/` and `src/mini_tft/fight_model/` | Historical current-patch planner and combat-teacher experiments. | Their tests and archived docs are either moved to a separate package or explicitly dropped. |

## Safe Retirement Order

1. Mark old paths as historical in docs and avoid using them in new runbooks.
2. Replace active imports from historical modules with strategic-lane adapters.
3. Move or delete tests only after equivalent strategic tests exist.
4. Delete code only after `rg` shows no active docs, tools, or tests import it.

## Current Verification Snapshot

Run these before deleting anything from a historical lane:

```bash
env -u UV_PYTHON uv run --extra train pytest -q tests/test_puffer_env.py tests/test_train_puffer_ppo.py
env -u UV_PYTHON uv run --extra train pytest -q tests/test_stage2_mcts.py tests/test_stage3_distill.py tests/test_stage4_dynamics.py
env -u UV_PYTHON uv run --extra train pytest -q tests/test_judge_packet.py tests/test_strategic_muzero_run_loop.py tests/test_strategic_muzero_overnight_trainer.py tests/test_strategic_eval.py tests/test_strategic_policy_improvement_gate.py
```

The first command may include skips when legacy Puffer wrapper APIs are absent.
