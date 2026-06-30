from __future__ import annotations

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import fast_level_lobby_policy, tempo_lobby_policy
from mini_tft.tools.compare_puffer_ppo import format_markdown, run_policy_comparison


def test_policy_comparison_reports_matched_seed_deltas() -> None:
    report = run_policy_comparison(
        candidate_policy=fast_level_lobby_policy,
        candidate_name="fast_level",
        baseline_policy=tempo_lobby_policy,
        baseline_name="tempo",
        opponent_policy_names=("tempo",),
        episodes=1,
        seed=9,
        player_count=4,
        max_actions_per_player=4,
        config=EnvConfig(seed=9, starting_gold=10, max_actions_per_round=4, max_round=4),
        device="cpu",
    )

    assert report["status"] == "pass"
    assert report["opponent_policy_pool"] == ["tempo"]
    assert "mean_placement" in report["deltas"]
    assert report["candidate"]["hero_policy"] == "fast_level"
    assert report["baseline"]["hero_policy"] == "tempo"

    markdown = format_markdown(report)
    assert "MiniTFT Puffer vs SB3 Comparison" in markdown
    assert "Scalar Deltas" in markdown
