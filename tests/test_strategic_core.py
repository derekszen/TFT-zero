from __future__ import annotations

import json

import numpy as np

from mini_tft.strategic.adapters.analytics import (
    episode_rows_frame,
    episode_rows_lazy,
    summarize_episode_rows,
)
from mini_tft.strategic.adapters.baselines import random_policy, tft_heuristic_policy
from mini_tft.strategic.adapters.muzero_cache import cache_metrics, generate_cache
from mini_tft.strategic.adapters.puffer import run_benchmark
from mini_tft.strategic.adapters.puffer.vector_env import StrategicBatchEnv
from mini_tft.strategic.adapters.web_demo import state_payload
from mini_tft.strategic.core import (
    NUM_ACTIONS,
    StrategicAction,
    StrategicConfig,
    legal_action_mask,
    observe,
    placement_proxy,
    reset,
    scenario_score,
    stage_label_for_round,
    state_signature,
    step,
)
from mini_tft.strategic.core.actions import ROLE_INDEX, action_name
from mini_tft.strategic.core.rules import run_episode, trace_episode
from mini_tft.strategic.native import plan_batch, run_native_mcts_smoke, trace_signatures
from mini_tft.tools.strategic_lane_gate import StrategicLaneGateConfig, run_gate


def test_strategic_transition_is_markov_for_cloned_state() -> None:
    config = StrategicConfig()
    left = reset(seed=7, config=config)
    right = left.clone()

    left_result = step(left, int(StrategicAction.ROLL), config)
    right_result = step(right, int(StrategicAction.ROLL), config)

    assert left_result.reward == right_result.reward
    assert left_result.info == right_result.info
    assert state_signature(left) == state_signature(right)


def test_strategic_batch_env_matches_scalar_transitions() -> None:
    config = StrategicConfig(max_round=10, max_actions_per_round=3)
    seeds = np.arange(8, dtype=np.int64)
    states = [reset(seed=int(seed), config=config) for seed in seeds]
    batch = StrategicBatchEnv(num_envs=len(seeds), seed=0, config=config)

    batch_obs = batch.reset(seed=seeds, indices=np.arange(len(seeds)))
    scalar_obs = [
        np.concatenate(
            [
                observe(state, config),
                legal_action_mask(state, config).astype(np.float32),
            ]
        )
        for state in states
    ]
    np.testing.assert_allclose(batch_obs, np.stack(scalar_obs).astype(np.float32))

    for _ in range(40):
        actions: list[int] = []
        scalar_rewards: list[float] = []
        scalar_terminals: list[bool] = []
        scalar_next_obs: list[np.ndarray] = []
        for state in states:
            if state.done:
                actions.append(int(StrategicAction.HOLD))
                scalar_rewards.append(0.0)
                scalar_terminals.append(True)
            else:
                mask = legal_action_mask(state, config)
                action = int(tft_heuristic_policy(state, mask, config))
                actions.append(action)
                result = step(state, action, config)
                scalar_rewards.append(result.reward)
                scalar_terminals.append(result.terminated)
            scalar_next_obs.append(
                np.concatenate(
                    [
                        observe(state, config),
                        legal_action_mask(state, config).astype(np.float32),
                    ]
                )
            )

        result = batch.step(np.asarray(actions, dtype=np.int64), include_infos=False)

        np.testing.assert_allclose(
            result.rewards,
            np.asarray(scalar_rewards, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(result.terminals, np.asarray(scalar_terminals))
        np.testing.assert_allclose(
            result.observations,
            np.stack(scalar_next_obs).astype(np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        if all(state.done for state in states):
            break


def test_native_trace_matches_scalar_state_signatures() -> None:
    script = [
        int(StrategicAction.BUY_HIGHEST_COST),
        int(StrategicAction.FIELD_STRONGEST),
        int(StrategicAction.HOLD),
        int(StrategicAction.BUY_HIGHEST_COST),
        int(StrategicAction.BUY_BEST_SYNERGY),
        int(StrategicAction.FIELD_STRONGEST),
        int(StrategicAction.HOLD),
        int(StrategicAction.ROLL),
        int(StrategicAction.BUY_BEST_UPGRADE),
        int(StrategicAction.FIELD_STRONGEST),
        int(StrategicAction.GREED_ECON),
        int(StrategicAction.SLAM_CARRY_ITEM),
        int(StrategicAction.SLAM_TANK_ITEM),
        int(StrategicAction.SLAM_SUPPORT_ITEM),
        int(StrategicAction.LEVEL),
        int(StrategicAction.HOLD),
    ]

    for seed in (0, 7, 19):
        assert trace_signatures(seed=seed, actions=script) == trace_episode(script, seed=seed)


def test_native_mcts_smoke_api_returns_legal_decisions() -> None:
    result = run_native_mcts_smoke(
        episodes=2,
        seed=23,
        simulations=(2,),
        max_depth=4,
        rollout_steps=2,
        prior_mode="heuristic",
    )

    decision_rows = result["decision_rows"]
    assert decision_rows
    for row in decision_rows:
        assert row["legal"] is True
        assert 0 <= row["action_id"] < NUM_ACTIONS
        assert row["action"] == action_name(row["action_id"])
        assert len(row["visit_policy"]) == NUM_ACTIONS
        assert sum(row["action_visits"].values()) == 2

    batch = plan_batch(
        seeds=[23, 24],
        simulations=2,
        max_depth=4,
        rollout_steps=2,
        prior_mode="heuristic",
    )
    assert len(batch["selected_actions"]) == 2
    assert len(batch["visit_policies"]) == 2
    assert batch["simulations_per_sec"] > 0.0


def test_role_item_slam_uses_role_slots_and_legal_mask() -> None:
    config = StrategicConfig()
    state = reset(seed=0, config=config)
    carry_index = ROLE_INDEX["carry"]

    state.owned[2] = 1
    state.fielded[0] = 2
    state.role_items[carry_index] = 1

    mask = legal_action_mask(state, config)
    assert bool(mask[StrategicAction.SLAM_CARRY_ITEM])

    result = step(state, int(StrategicAction.SLAM_CARRY_ITEM), config)

    assert result.info["legal_action"] is True
    assert state.role_items[carry_index] == 0
    assert state.role_item_slots[carry_index] == 1
    assert state.total_item_slams == 1

    state.role_items[carry_index] = 1
    state.role_item_slots[carry_index] = config.max_role_item_slots
    mask = legal_action_mask(state, config)
    assert not bool(mask[StrategicAction.SLAM_CARRY_ITEM])


def test_heuristic_baseline_can_die_under_tuned_enemy_pressure() -> None:
    config = StrategicConfig()
    final_reasons: list[str | None] = []
    final_rounds: list[int] = []
    placements: list[int] = []
    scenario_scores: list[float] = []

    for seed in range(64):
        state, _, _ = run_episode(tft_heuristic_policy, seed=seed, config=config)
        final_reasons.append(state.final_reason)
        final_rounds.append(state.round)
        placements.append(placement_proxy(state, config))
        scenario_scores.append(scenario_score(state, config))
        assert state.total_illegal_actions == 0

    death_rate = final_reasons.count("hp_zero") / len(final_reasons)
    late_game_rate = sum(round_num > 27 for round_num in final_rounds) / len(final_rounds)
    weak_bucket_rate = sum(placement >= 6 for placement in placements) / len(placements)
    assert death_rate >= 0.5
    assert final_reasons.count("max_round") == 0
    assert late_game_rate <= 0.1
    assert weak_bucket_rate >= 0.75
    assert max(scenario_scores) <= 1.0
    assert min(scenario_scores) >= 0.0

    random_placements = [
        placement_proxy(run_episode(random_policy, seed=seed, config=config)[0], config)
        for seed in range(64)
    ]
    assert max(random_placements) == 8


def test_placement_proxy_uses_stage_elimination_buckets() -> None:
    config = StrategicConfig()
    state = reset(seed=0, config=config)

    state.round = 36
    state.hp = 12
    state.final_reason = "max_round"
    assert stage_label_for_round(state.round) == "Stage 6-5"
    assert placement_proxy(state, config) == 1

    state.final_reason = "hp_zero"
    state.hp = 0
    assert placement_proxy(state, config) == 2

    state.round = 29
    assert stage_label_for_round(state.round) == "Stage 5-5"
    assert placement_proxy(state, config) == 4

    state.round = 18
    assert placement_proxy(state, config) == 6


def test_muzero_cache_rows_have_training_schema() -> None:
    rows = generate_cache(episodes=2, max_rows=24)
    metrics = cache_metrics(rows)

    assert len(rows) == 24
    assert metrics["rows"] == 24
    assert metrics["observation_dim"] == rows[0].observation.shape[0]
    assert metrics["action_dim"] == NUM_ACTIONS
    assert metrics["legal_action_rate"] == 1.0
    assert rows[0].policy_target.shape == (NUM_ACTIONS,)
    assert float(rows[0].policy_target.sum()) == 1.0
    assert bool(rows[0].legal_mask[rows[0].action])


def test_strategic_analytics_summarizes_episode_artifact_rows() -> None:
    rows = [
        {
            "episode": 0,
            "policy": "heuristic",
            "placement_proxy": 2,
            "round": 36,
            "hp": 12,
            "final_reason": "max_round",
            "scenario_score": 0.9,
            "total_illegal_actions": 0,
        },
        {
            "episode": 1,
            "policy": "heuristic",
            "placement": 6,
            "final_round": 18,
            "hp": 0,
            "final_reason": "hp_zero",
            "scenario_score": 0.2,
            "illegal_actions": 0,
        },
        {
            "episode": 2,
            "policy": "random",
            "placement": 6,
            "final_round": 20,
            "hp": 0,
            "final_reason": "hp_zero",
            "scenario_score": 0.3,
            "illegal_actions": 1,
        },
    ]

    frame = episode_rows_frame(rows)
    lazy = episode_rows_lazy(rows)
    summary = summarize_episode_rows(lazy)

    assert frame.columns == [
        "episode",
        "policy",
        "placement",
        "final_round",
        "hp",
        "final_reason",
        "scenario_score",
        "illegal_actions",
    ]
    assert frame["placement"].to_list() == [2, 6, 6]
    assert summary == {
        "episodes": 3,
        "placement_counts": {"2": 1, "6": 2},
        "survivor_rate": 1 / 3,
        "mean_final_round": 74 / 3,
        "median_final_round": 20.0,
        "min_final_round": 18,
        "max_final_round": 36,
    }


def test_puffer_benchmark_writes_quality_gate_artifacts(tmp_path) -> None:
    result = run_benchmark(episodes=2, batch_size=64, repeated_runs=1, out_dir=tmp_path)

    metrics_path = tmp_path / "metrics.json"
    decision_path = tmp_path / "decision.md"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert metrics_path.exists()
    assert decision_path.exists()
    assert result.reference_steps_per_second > 0.0
    assert result.batched_steps_per_second > 0.0
    assert result.speedup > 0.0
    assert result.decision in {"pass", "inconclusive", "smoke_only"}
    assert result.benchmark_kind in {"strategic_native_puffer_vector", "proxy_array_math_ceiling"}
    assert metrics["deliverable"] == "puffer_speed"
    assert metrics["status"] == result.decision
    assert metrics["metrics"]["semantic_parity"] == result.semantic_parity
    assert metrics["metrics"]["puffer_trainer"] == result.puffer_trainer
    assert metrics["metrics"]["speedup"] == result.speedup


def test_web_payload_exposes_playable_state_shape() -> None:
    state = reset(seed=11)
    payload = state_payload(state)

    assert payload["round"] == state.round
    assert payload["stage_label"] == "Stage 1-1"
    assert payload["hp"] == state.hp
    assert payload["role_items"] == [0, 0, 0]
    assert payload["role_item_slots"] == [0, 0, 0]
    assert 0.0 <= payload["scenario_score"] <= 1.0
    assert payload["legal_actions"]
    assert {action["name"] for action in payload["legal_actions"]} >= {"hold", "greed_econ"}


def test_strategic_lane_gate_writes_auditable_artifacts(tmp_path) -> None:
    report = run_gate(
        StrategicLaneGateConfig(
            out_dir=tmp_path,
            cache_episodes=1,
            cache_rows=8,
            benchmark_episodes=1,
            benchmark_batch_size=8,
            benchmark_repeats=1,
            calibration_seeds=4,
        )
    )

    assert report["status"] in {"pass", "smoke_only", "fail"}
    assert report["metrics"]["puffer_speed"]["decision"] in {
        "pass",
        "inconclusive",
        "smoke_only",
        "fail",
    }
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "decision.md").exists()
    assert (tmp_path / "final_report.md").exists()
    assert (tmp_path / "muzero_cache" / "rows.jsonl").exists()
    assert (tmp_path / "puffer_speed" / "metrics.json").exists()
    assert (tmp_path / "playable_demo" / "initial_payload.json").exists()
