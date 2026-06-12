"""Recurring Set-1 MiniTFT simulator regression gate."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mini_tft.bots.fast_level_bot import FastLevelBot
from mini_tft.core.actions import Action
from mini_tft.core.board import generate_candidate_boards
from mini_tft.core.combat import board_strength
from mini_tft.core.config import EnvConfig
from mini_tft.core.ids import EMPTY
from mini_tft.core.rounds import is_item_drop_round, round_info
from mini_tft.core.set_data import load_set
from mini_tft.core.state import UnitInstance, new_game_state, state_signature
from mini_tft.rl.gym_env import MiniTFTGymEnv
from mini_tft.tools.combat_fixture_report import run_combat_fixture_report
from mini_tft.tools.sim_smoke import SmokeConfig, run_smoke
from mini_tft.tools.web_ui_regression_gate import WebUiGateConfig, run_web_ui_regression_gate


@dataclass(frozen=True)
class RegressionGateConfig:
    seed: int = 0
    benchmark_episodes: int = 20
    dataset_episodes: int = 20
    parallel_dataset_episodes: int = 20
    parallel_workers: int | None = 0
    eval_episodes: int = 4
    trace_steps: int = 6
    level_episodes: int = 20
    min_mean_final_level: float = 8.0
    max_mean_final_hp: float = 35.0
    run_smoke_report: bool = True


def run_simulator_regression_gate(config: RegressionGateConfig) -> dict[str, Any]:
    """Run the recurring simulator quality gate and return report data."""

    started = time.perf_counter()
    checks: list[dict[str, Any]] = []
    smoke_report = None

    if config.run_smoke_report:
        smoke_report = run_smoke(
            SmokeConfig(
                benchmark_episodes=config.benchmark_episodes,
                dataset_episodes=config.dataset_episodes,
                parallel_dataset_episodes=config.parallel_dataset_episodes,
                parallel_workers=config.parallel_workers,
                eval_episodes=config.eval_episodes,
                trace_steps=config.trace_steps,
                seed=config.seed,
                run_checks=False,
            )
        )
        checks.append(
            _check(
                "sim_smoke",
                smoke_report["status"] == "pass",
                {
                    "steps_per_sec": round(smoke_report["benchmark"]["steps_per_sec"], 3),
                    "transitions_per_sec": round(
                        smoke_report["dataset"]["transitions_per_sec"], 3
                    ),
                    "parallel_transitions_per_sec": round(
                        smoke_report["parallel_dataset"]["transitions_per_sec"], 3
                    ),
                    "failures": smoke_report["failures"],
                },
            )
        )

    checks.extend(
        [
            _determinism_check(config.seed),
            _round_schedule_check(),
            _item_flow_check(config.seed),
            _candidate_board_check(),
            _combat_fixture_check(),
            _level_pacing_check(config),
            _web_ui_check(config.seed),
        ]
    )

    failures = [check["name"] for check in checks if check["status"] != "pass"]
    return {
        "status": "pass" if not failures else "fail",
        "seed": config.seed,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "checks": checks,
        "failures": failures,
        "smoke": smoke_report,
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MiniTFT Simulator Regression Gate",
        "",
        f"**Status:** `{report['status']}`",
        f"**Seed:** `{report['seed']}`",
        f"**Elapsed:** `{report['elapsed_sec']:.3f}s`",
        "",
        "## Checks",
        "",
        "| Check | Status | Key Metrics |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            f"| `{check['name']}` | `{check['status']}` | "
            f"{_format_metrics(check['metrics'])} |"
        )
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in report["failures"])
    lines.extend(
        [
            "",
            "## Gate Meaning",
            "",
            "- `sim_smoke` checks throughput, dataset shape, bot evals, and trace output.",
            "- `combat_fixtures` checks explicit board-strength and win-probability orderings.",
            "- `candidate_boards` checks stronger legal board generation without state mutation.",
            "- `item_flow` checks scheduled PvE component drops and combine/slam behavior.",
            "- `level_pacing` checks FastLevelBot still reaches level 8 without "
            "coasting on HP.",
            "- `web_ui` checks browser payload defaults, enemy preview scaling, item "
            "actions, and moves.",
        ]
    )
    return "\n".join(lines) + "\n"


def _determinism_check(seed: int) -> dict[str, Any]:
    actions = [Action.END_TURN, Action.END_TURN, Action.END_TURN, Action.END_TURN]
    signatures = []
    for _ in range(2):
        env = MiniTFTGymEnv(config=EnvConfig(seed=seed))
        env.reset(seed=seed)
        for action in actions:
            env.step(action)
        assert env.state is not None
        signatures.append(state_signature(env.state))
    return _check(
        "determinism",
        signatures[0] == signatures[1],
        {"actions": len(actions), "seed": seed},
    )


def _round_schedule_check() -> dict[str, Any]:
    expected = {
        1: ("Stage 1-1", "pve", True),
        3: ("Stage 1-3", "pve", True),
        4: ("Stage 2-1", "pvp", False),
        10: ("Stage 2-7", "pve", True),
        11: ("Stage 3-1", "pvp", False),
    }
    rows = []
    ok = True
    for round_num, expected_row in expected.items():
        current = round_info(round_num)
        actual = (current.stage_label, current.round_type, is_item_drop_round(round_num))
        rows.append({"round": round_num, "actual": actual, "expected": expected_row})
        ok = ok and actual == expected_row
    return _check("round_schedule", ok, {"checked_rounds": len(rows), "rows": rows})


def _item_flow_check(seed: int) -> dict[str, Any]:
    env = MiniTFTGymEnv(config=EnvConfig(seed=seed))
    env.reset(seed=seed)
    assert env.state is not None

    env.step(Action.END_TURN)
    first_drop = env.state.item_bench[0] if env.state.item_bench else None
    first_drop_kind = env.data.items[first_drop].kind if first_drop is not None else None

    env.state.board[0] = UnitInstance(unit_id=2)
    env.state.item_bench = [101, 102]
    combine_legal = bool(env.action_masks()[Action.SLAM_BEST_ITEM])
    env.step(Action.SLAM_BEST_ITEM)
    combined = list(env.state.item_bench)
    slam_legal = bool(env.action_masks()[Action.SLAM_BEST_ITEM])
    env.step(Action.SLAM_BEST_ITEM)
    equipped = list(env.state.board[0].items) if env.state.board[0] is not None else []

    ok = (
        first_drop_kind == "component"
        and combine_legal
        and combined == [1]
        and slam_legal
        and equipped == [1]
    )
    return _check(
        "item_flow",
        ok,
        {
            "first_drop": first_drop,
            "first_drop_kind": first_drop_kind,
            "combined_bench": combined,
            "equipped": equipped,
        },
    )


def _candidate_board_check() -> dict[str, Any]:
    config = EnvConfig(seed=21, starting_level=2)
    data = load_set(config.dataset)
    state = new_game_state(config, seed=21, shop=[EMPTY] * config.shop_size)
    state.level = 2
    state.board[0] = UnitInstance(unit_id=1)
    state.board[1] = UnitInstance(unit_id=6)
    state.bench[0] = UnitInstance(unit_id=20)
    state.bench[1] = UnitInstance(unit_id=23)
    before = state_signature(state)
    owned = _unit_multiset([*state.board, *state.bench])
    current_strength = board_strength(state.board, data).strength

    candidates = generate_candidate_boards(state, data, config, top_k=4)
    best_strength = candidates[0].strength if candidates else 0.0
    candidate_owned = _unit_multiset([*candidates[0].board, *candidates[0].bench])
    ok = (
        bool(candidates)
        and best_strength > current_strength
        and candidate_owned == owned
        and state_signature(state) == before
    )
    return _check(
        "candidate_boards",
        ok,
        {
            "current_strength": round(current_strength, 3),
            "best_strength": round(best_strength, 3),
            "candidate_count": len(candidates),
        },
    )


def _combat_fixture_check() -> dict[str, Any]:
    report = run_combat_fixture_report()
    return _check(
        "combat_fixtures",
        report["status"] == "pass",
        {"passed": report["passed"], "total": report["total"]},
    )


def _level_pacing_check(config: RegressionGateConfig) -> dict[str, Any]:
    levels = []
    rounds = []
    hps = []
    for seed in range(1000, 1000 + config.level_episodes):
        env = MiniTFTGymEnv(config=EnvConfig(seed=seed))
        bot = FastLevelBot()
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = bot.act(env, obs, rng)
            obs, _, terminated, truncated, _ = env.step(action)
        assert env.state is not None
        levels.append(env.state.level)
        rounds.append(min(env.state.round, env.config.max_round))
        hps.append(env.state.hp)

    mean_final_level = float(np.mean(levels)) if levels else 0.0
    mean_final_hp = float(np.mean(hps)) if hps else 0.0
    survival_rate = float(np.mean([hp > 0 for hp in hps])) if hps else 0.0
    return _check(
        "level_pacing",
        mean_final_level >= config.min_mean_final_level
        and mean_final_hp <= config.max_mean_final_hp,
        {
            "episodes": config.level_episodes,
            "mean_final_level": round(mean_final_level, 3),
            "mean_final_hp": round(mean_final_hp, 3),
            "min_final_level": min(levels) if levels else 0,
            "mean_survived_round": round(float(np.mean(rounds)), 3) if rounds else 0.0,
            "survival_rate": round(survival_rate, 3),
            "min_mean_final_level": config.min_mean_final_level,
            "max_mean_final_hp": config.max_mean_final_hp,
        },
    )


def _web_ui_check(seed: int) -> dict[str, Any]:
    report = run_web_ui_regression_gate(WebUiGateConfig(seed=seed))
    return _check(
        "web_ui",
        report["status"] == "pass",
        {
            "checks": len(report["checks"]),
            "failures": report["failures"],
        },
    )


def _check(name: str, ok: bool, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "fail", "metrics": metrics}


def _unit_multiset(units: list[UnitInstance | None]) -> Counter[tuple[int, int, tuple[int, ...]]]:
    return Counter(
        (unit.unit_id, unit.stars, tuple(unit.items))
        for unit in units
        if unit is not None
    )


def _format_metrics(metrics: dict[str, Any]) -> str:
    compact = []
    for key, value in metrics.items():
        if key == "rows":
            compact.append(f"{key}={len(value)}")
        elif key == "failures":
            compact.append(f"{key}={len(value)}")
        else:
            compact.append(f"{key}={value}")
    return ", ".join(compact)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the recurring MiniTFT simulator gate.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-episodes", type=int, default=20)
    parser.add_argument("--dataset-episodes", type=int, default=20)
    parser.add_argument("--parallel-dataset-episodes", type=int, default=20)
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--trace-steps", type=int, default=6)
    parser.add_argument("--level-episodes", type=int, default=20)
    parser.add_argument("--min-mean-final-level", type=float, default=8.0)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = run_simulator_regression_gate(
        RegressionGateConfig(
            seed=args.seed,
            benchmark_episodes=args.benchmark_episodes,
            dataset_episodes=args.dataset_episodes,
            parallel_dataset_episodes=args.parallel_dataset_episodes,
            parallel_workers=args.parallel_workers,
            eval_episodes=args.eval_episodes,
            trace_steps=args.trace_steps,
            level_episodes=args.level_episodes,
            min_mean_final_level=args.min_mean_final_level,
            run_smoke_report=not args.skip_smoke,
        )
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    else:
        print(format_markdown(report), end="")
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
