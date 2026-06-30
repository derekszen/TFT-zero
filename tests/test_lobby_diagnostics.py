from __future__ import annotations

import json
from pathlib import Path

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import fast_level_lobby_policy, tempo_lobby_policy
from mini_tft.tools.lobby_diagnostics import (
    compare_trace_files,
    run_lobby_trace_export,
    run_roll_value_probe,
    summarize_trace_rows,
)


def test_lobby_trace_export_writes_decision_rows(tmp_path: Path) -> None:
    report = run_lobby_trace_export(
        episodes=1,
        seed=7,
        player_count=4,
        max_actions_per_player=4,
        config=EnvConfig(seed=7, starting_gold=10, max_actions_per_round=4, max_round=4),
        out_dir=tmp_path,
        hero_policy=fast_level_lobby_policy,
        opponent_policy=tempo_lobby_policy,
    )

    assert report["status"] == "pass"
    assert report["decisions"] > 0
    assert report["summary"]["chosen_buy_shop_rate"] >= 0.0
    trace_path = tmp_path / "trace.jsonl"
    assert trace_path.exists()
    first_row = trace_path.read_text(encoding="utf-8").splitlines()[0]
    assert "\"action_name\"" in first_row
    assert "\"outcome\"" in first_row
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "decision.md").exists()


def test_trace_compare_reports_action_rate_deltas(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    run_lobby_trace_export(
        episodes=1,
        seed=11,
        player_count=4,
        max_actions_per_player=4,
        config=EnvConfig(seed=11, starting_gold=10, max_actions_per_round=4, max_round=4),
        out_dir=baseline_dir,
        hero_policy=fast_level_lobby_policy,
        opponent_policy=tempo_lobby_policy,
    )
    run_lobby_trace_export(
        episodes=1,
        seed=11,
        player_count=4,
        max_actions_per_player=4,
        config=EnvConfig(seed=11, starting_gold=10, max_actions_per_round=4, max_round=4),
        out_dir=candidate_dir,
        hero_policy=tempo_lobby_policy,
        opponent_policy=tempo_lobby_policy,
    )

    report = compare_trace_files(
        baseline_trace=baseline_dir / "trace.jsonl",
        candidate_trace=candidate_dir / "trace.jsonl",
        out_dir=tmp_path / "compare",
    )

    assert report["status"] == "pass"
    assert "chosen_buy_shop_rate" in report["deltas"]["scalar"]
    assert "category_rates" in report["deltas"]
    assert "bucket_rates" in report["deltas"]
    assert "hp" in report["deltas"]["bucket_rates"]
    assert (tmp_path / "compare" / "decision.md").exists()


def test_trace_outcomes_are_episode_weighted_not_decision_weighted(
    tmp_path: Path,
) -> None:
    baseline_rows = [
        _trace_row(episode=0, decision_index=0, placement=1, final_hp=80),
        _trace_row(episode=0, decision_index=1, placement=1, final_hp=80),
        _trace_row(episode=0, decision_index=2, placement=1, final_hp=80),
        _trace_row(episode=1, decision_index=0, placement=4, final_hp=0),
    ]
    candidate_rows = [
        _trace_row(episode=0, decision_index=0, placement=1, final_hp=80),
        _trace_row(episode=1, decision_index=0, placement=4, final_hp=0),
        _trace_row(episode=1, decision_index=1, placement=4, final_hp=0),
        _trace_row(episode=1, decision_index=2, placement=4, final_hp=0),
    ]
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write_rows(baseline_path, baseline_rows)
    _write_rows(candidate_path, candidate_rows)

    baseline = summarize_trace_rows(baseline_rows)
    report = compare_trace_files(
        baseline_trace=baseline_path,
        candidate_trace=candidate_path,
    )

    assert baseline["outcome_aggregation"] == "episode"
    assert baseline["outcome_episodes"] == 2
    assert baseline["mean_placement"] == 2.5
    assert baseline["mean_final_hp"] == 40.0
    assert report["baseline"]["mean_placement"] == 2.5
    assert report["candidate"]["mean_placement"] == 2.5
    assert report["deltas"]["scalar"]["mean_placement"] == 0.0
    assert report["deltas"]["scalar"]["mean_final_hp"] == 0.0


def test_roll_value_probe_samples_legal_roll_branches(tmp_path: Path) -> None:
    report = run_roll_value_probe(
        episodes=1,
        seed=13,
        max_states=2,
        min_states=1,
        player_count=4,
        max_actions_per_player=4,
        config=EnvConfig(seed=13, starting_gold=10, max_actions_per_round=4, max_round=4),
        out_dir=tmp_path,
        hero_policy=fast_level_lobby_policy,
        opponent_policy=tempo_lobby_policy,
    )

    assert report["status"] == "pass"
    assert report["sampled_states"] >= 1
    assert "roll" in report["branch_mean_values"]
    first_result = report["results"][0]
    assert first_result["branches"]["roll"]["illegal_actions"] == 0
    assert "roll_advantage_vs_best_no_roll" in first_result
    assert (tmp_path / "metrics.json").exists()


def _trace_row(
    *,
    episode: int,
    decision_index: int,
    placement: int,
    final_hp: int,
) -> dict[str, object]:
    return {
        "episode": episode,
        "seed": 100 + episode,
        "decision_index": decision_index,
        "round": 1,
        "round_bucket": "01-10",
        "level": 3,
        "xp": 0,
        "gold": 0,
        "gold_bucket": "00-09",
        "hp": final_hp,
        "hp_bucket": "075+" if final_hp >= 75 else "000",
        "bench_size": 0,
        "board_size": 1,
        "shop_unit_ids": [],
        "shop_unit_names": [],
        "shop_costs": [],
        "shop_total_cost": 0,
        "shop_max_cost": 0,
        "legal_actions": [0],
        "legal_action_count": 1,
        "legal_roll": False,
        "legal_buy_xp": False,
        "legal_end_turn": True,
        "legal_buy_count": 0,
        "legal_sell_count": 0,
        "action": 0,
        "action_name": "end_turn",
        "action_category": "end_turn",
        "total_rolls": 0,
        "total_xp_buys": 0,
        "total_units_bought": 0,
        "total_units_sold": 0,
        "outcome": {
            "placement": placement,
            "final_hp": final_hp,
            "survived_round": 4,
            "final_board_strength": float(final_hp),
            "final_reason": "test",
        },
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
