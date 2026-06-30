"""Compare a Puffer PPO checkpoint with an SB3 PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from mini_tft.core.config import EnvConfig
from mini_tft.core.lobby_step import LobbyPolicy
from mini_tft.rl.checkpoint_policy import load_lobby_checkpoint_policy
from mini_tft.rl.lobby_env import LOBBY_POLICY_BY_NAME
from mini_tft.tools.evaluate_frozen_lobby_pool import (
    format_markdown as format_pool_markdown,
)
from mini_tft.tools.evaluate_frozen_lobby_pool import (
    run_lobby_policy_pool_evaluation,
)

SCALAR_METRICS = (
    "mean_placement",
    "median_placement",
    "top1_rate",
    "top4_rate",
    "mean_final_hp",
    "mean_survived_round",
    "mean_final_board_strength",
    "mean_lobby_rounds",
    "hero_macro_action_rate",
)


def run_puffer_vs_sb3_comparison(
    *,
    puffer_checkpoint: Path,
    sb3_checkpoint: Path,
    opponent_policy_names: Sequence[str] = ("mixed",),
    episodes: int = 20,
    seed: int = 1000,
    player_count: int = 8,
    max_actions_per_player: int | None = None,
    device: str = "cpu",
    config: EnvConfig | None = None,
) -> dict[str, Any]:
    """Evaluate Puffer and SB3 checkpoints against identical opponent seeds."""

    puffer_policy = load_lobby_checkpoint_policy(
        puffer_checkpoint,
        checkpoint_format="puffer",
        device=device,
    )
    sb3_policy = load_lobby_checkpoint_policy(
        sb3_checkpoint,
        checkpoint_format="sb3",
        device=device,
    )
    return run_policy_comparison(
        candidate_policy=puffer_policy,
        candidate_name=str(puffer_checkpoint),
        baseline_policy=sb3_policy,
        baseline_name=str(sb3_checkpoint),
        opponent_policy_names=opponent_policy_names,
        episodes=episodes,
        seed=seed,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        device=device,
        config=config,
    )


def run_policy_comparison(
    *,
    candidate_policy: LobbyPolicy,
    candidate_name: str,
    baseline_policy: LobbyPolicy,
    baseline_name: str,
    opponent_policy_names: Sequence[str] = ("mixed",),
    episodes: int = 20,
    seed: int = 1000,
    player_count: int = 8,
    max_actions_per_player: int | None = None,
    device: str = "cpu",
    config: EnvConfig | None = None,
) -> dict[str, Any]:
    """Evaluate two explicit hero policies under matched lobby settings."""

    if not opponent_policy_names:
        raise ValueError("opponent_policy_names must not be empty")
    opponent_policies = tuple(_named_lobby_policy(name) for name in opponent_policy_names)
    base_config = config or EnvConfig(seed=seed)
    candidate = run_lobby_policy_pool_evaluation(
        name="candidate",
        hero_policy=candidate_policy,
        hero_policy_name=candidate_name,
        opponent_policies=opponent_policies,
        opponent_policy_names=opponent_policy_names,
        episodes=episodes,
        seed=seed,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=base_config,
        device=device,
    )
    baseline = run_lobby_policy_pool_evaluation(
        name="baseline",
        hero_policy=baseline_policy,
        hero_policy_name=baseline_name,
        opponent_policies=opponent_policies,
        opponent_policy_names=opponent_policy_names,
        episodes=episodes,
        seed=seed,
        player_count=player_count,
        max_actions_per_player=max_actions_per_player,
        config=base_config,
        device=device,
    )
    return {
        "status": "pass" if candidate["status"] == baseline["status"] == "pass" else "fail",
        "episodes": episodes,
        "seed": seed,
        "player_count": player_count,
        "device": device,
        "opponent_policy_pool": list(opponent_policy_names),
        "candidate": candidate,
        "baseline": baseline,
        "deltas": _scalar_deltas(candidate, baseline),
        "delta_convention": "candidate - baseline; lower mean_placement is better",
    }


def format_markdown(report: dict[str, Any]) -> str:
    deltas = report["deltas"]
    rows = [
        "# MiniTFT Puffer vs SB3 Comparison",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Episodes | {report['episodes']} |",
        f"| Players | {report['player_count']} |",
        f"| Device | `{report['device']}` |",
        f"| Opponent pool | `{', '.join(report['opponent_policy_pool'])}` |",
        "",
        "## Scalar Deltas",
        "",
        "Candidate is Puffer in the checkpoint CLI. Deltas are candidate minus baseline.",
        "",
        "| Metric | Delta |",
        "| --- | ---: |",
    ]
    rows.extend(f"| {metric} | {value:.3f} |" for metric, value in deltas.items())
    rows.extend(
        [
            "",
            "## Candidate",
            "",
            format_pool_markdown(report["candidate"]),
            "",
            "## Baseline",
            "",
            format_pool_markdown(report["baseline"]),
        ]
    )
    return "\n".join(rows)


def _scalar_deltas(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for metric in SCALAR_METRICS:
        deltas[metric] = float(candidate[metric]) - float(baseline[metric])
    return deltas


def _named_lobby_policy(name: str) -> LobbyPolicy:
    try:
        return LOBBY_POLICY_BY_NAME[name]
    except KeyError as exc:
        choices = ", ".join(sorted(LOBBY_POLICY_BY_NAME))
        raise ValueError(f"unknown lobby policy {name!r}; choose one of: {choices}") from exc


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puffer-checkpoint", type=Path, required=True)
    parser.add_argument("--sb3-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent-policy",
        choices=sorted(LOBBY_POLICY_BY_NAME),
        action="append",
        default=None,
        help="Opponent policy. Repeat to build a cycling opponent pool.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--max-actions-per-player", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_puffer_vs_sb3_comparison(
        puffer_checkpoint=args.puffer_checkpoint,
        sb3_checkpoint=args.sb3_checkpoint,
        opponent_policy_names=tuple(args.opponent_policy or ("mixed",)),
        episodes=args.episodes,
        seed=args.seed,
        player_count=args.players,
        max_actions_per_player=args.max_actions_per_player,
        device=args.device,
    )
    output = (
        json.dumps(report, indent=2, sort_keys=True)
        if args.format == "json"
        else format_markdown(report)
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
