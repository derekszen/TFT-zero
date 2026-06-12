from __future__ import annotations

from pathlib import Path

from mini_tft.autoresearch.evaluator import compare_metrics, run_objective_phase
from mini_tft.autoresearch.schema import MetricRule, load_objective


def test_load_lobby_objective_resolves_checkpoint_path() -> None:
    repo_root = Path.cwd()
    objective = load_objective(
        repo_root / "experiments/autoresearch/objectives/ppo_lobby_placement.toml",
        repo_root,
    )

    assert objective.name == "ppo_lobby_placement"
    assert objective.environment.kind == "lobby"
    assert objective.environment.opponent_policy == "tempo"
    assert objective.baseline.kind == "bot"
    assert objective.candidate.kind == "checkpoint"
    assert objective.seed_sets["public"].episodes == 25
    assert objective.candidate.path is not None
    assert objective.candidate.path.is_absolute()


def test_compare_metrics_applies_required_thresholds() -> None:
    rows = compare_metrics(
        {"score": 1.0, "cost": 10.0},
        {"score": 1.2, "cost": 9.0},
        (
            MetricRule("score", "higher", 0.1, True),
            MetricRule("cost", "lower", 0.5, True),
        ),
    )

    assert rows[0]["passed"] is True
    assert rows[0]["delta"] == 0.19999999999999996
    assert rows[1]["passed"] is True
    assert rows[1]["improvement"] == 1.0


def test_run_objective_phase_writes_single_player_metrics_for_bot_candidate(
    tmp_path: Path,
) -> None:
    objective_path = tmp_path / "objective.toml"
    objective_path.write_text(
        """
name = "bot_smoke"
description = "Compare FastLevelBot to itself."

[environment]
kind = "single_player"

[baseline]
kind = "bot"
name = "fast_level"

[candidate]
kind = "bot"
name = "fast_level"

[seed_sets.public]
start = 1000
episodes = 1

[[acceptance.metrics]]
name = "survival_rate"
mode = "higher"
min_delta = 0.0
required = true

[safety]
fail_on_protected_changes = false
protected_paths = []
""",
        encoding="utf-8",
    )
    repo_root = Path.cwd()
    objective = load_objective(objective_path, repo_root)

    report = run_objective_phase(objective, objective.seed_sets["public"], repo_root, tmp_path)

    assert report["status"] == "pass"
    assert (tmp_path / "metrics.json").exists()


def test_run_objective_phase_writes_lobby_metrics_for_bot_candidate(tmp_path: Path) -> None:
    objective_path = tmp_path / "objective.toml"
    objective_path.write_text(
        """
name = "lobby_bot_smoke"
description = "Compare FastLevelBot to itself in a tiny lobby."

[environment]
kind = "lobby"
opponent_policy = "tempo"
player_count = 4
max_actions_per_player = 4

[baseline]
kind = "bot"
name = "fast_level"

[candidate]
kind = "bot"
name = "fast_level"

[seed_sets.public]
start = 7
episodes = 1

[[acceptance.metrics]]
name = "mean_placement"
mode = "lower"
min_delta = 0.0
required = true

[[acceptance.metrics]]
name = "top4_rate"
mode = "higher"
min_delta = 0.0
required = true

[safety]
fail_on_protected_changes = false
protected_paths = []
""",
        encoding="utf-8",
    )
    repo_root = Path.cwd()
    objective = load_objective(objective_path, repo_root)

    report = run_objective_phase(objective, objective.seed_sets["public"], repo_root, tmp_path)

    assert report["status"] == "pass"
    assert report["environment"]["kind"] == "lobby"
    assert "mean_placement" in report["candidate"]["metrics"]
