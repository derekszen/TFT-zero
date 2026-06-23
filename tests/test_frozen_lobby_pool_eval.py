from __future__ import annotations

import pytest

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import fast_level_lobby_policy, tempo_lobby_policy
from mini_tft.tools.evaluate_frozen_lobby_pool import (
    format_markdown,
    run_lobby_policy_pool_evaluation,
)


def test_lobby_policy_pool_evaluation_reports_frozen_pool_metrics() -> None:
    report = run_lobby_policy_pool_evaluation(
        name="scripted_pool_smoke",
        hero_policy=fast_level_lobby_policy,
        hero_policy_name="fast_level",
        opponent_policies=[tempo_lobby_policy],
        opponent_policy_names=["tempo"],
        episodes=2,
        seed=7,
        player_count=4,
        max_actions_per_player=4,
        config=EnvConfig(seed=7, starting_gold=10, max_actions_per_round=4, max_round=4),
        device="cpu",
    )

    assert report["status"] == "pass"
    assert report["device"] == "cpu"
    assert report["opponent_pool"] == ["tempo"]
    assert 1.0 <= report["mean_placement"] <= 4.0
    assert 0.0 <= report["top1_rate"] <= 1.0
    assert 0.0 <= report["top4_rate"] <= 1.0
    assert report["hero_actions"] > 0
    assert sum(report["placement_histogram"].values()) == 2
    assert 0.0 <= report["hero_macro_action_rate"] <= 1.0

    markdown = format_markdown(report)
    assert "MiniTFT Frozen Lobby Pool Evaluation" in markdown
    assert "`tempo`" in markdown


def test_lobby_policy_pool_evaluation_requires_opponents() -> None:
    with pytest.raises(ValueError, match="opponent_policies"):
        run_lobby_policy_pool_evaluation(
            name="missing_pool",
            hero_policy=fast_level_lobby_policy,
            hero_policy_name="fast_level",
            opponent_policies=[],
            opponent_policy_names=[],
        )
