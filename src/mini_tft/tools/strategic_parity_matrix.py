"""Run fixed-seed and fuzz parity checks across strategic simulator backends."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from mini_tft.strategic.adapters.baselines import tft_heuristic_policy
from mini_tft.strategic.core import (
    NUM_ACTIONS,
    StrategicAction,
    StrategicConfig,
    legal_action_mask,
    observe,
    placement_proxy,
    reset,
    scenario_score,
    state_signature,
    step,
)
from mini_tft.strategic.native import trace_signatures

REPO_ROOT = Path(__file__).resolve().parents[3]
OCEAN_SRC = REPO_ROOT / "src" / "mini_tft" / "strategic" / "ocean"
DEFAULT_OUT_DIR = Path("artifacts/strategic_lane/parity_matrix")


@dataclass(frozen=True)
class ParityScenario:
    name: str
    actions: tuple[int, ...]
    description: str
    kind: str = "fixed"
    policy: str | None = None
    seed: int | None = None


SCENARIOS: tuple[ParityScenario, ...] = (
    ParityScenario("reset_only", tuple(), "reset, observation, and legal-mask parity"),
    ParityScenario(
        "economy_rounds",
        (
            int(StrategicAction.HOLD),
            int(StrategicAction.GREED_ECON),
            int(StrategicAction.HOLD),
        ),
        "hold and greed end-round transitions",
    ),
    ParityScenario(
        "roll_buy_field",
        (
            int(StrategicAction.BUY_HIGHEST_COST),
            int(StrategicAction.FIELD_STRONGEST),
            int(StrategicAction.HOLD),
            int(StrategicAction.ROLL),
            int(StrategicAction.BUY_BEST_UPGRADE),
            int(StrategicAction.FIELD_STRONGEST),
            int(StrategicAction.HOLD),
        ),
        "shop refresh, buying, upgrade targeting, and board fielding",
    ),
    ParityScenario(
        "level_tempo",
        (
            int(StrategicAction.LEVEL),
            int(StrategicAction.LEVEL),
            int(StrategicAction.LEVEL),
            int(StrategicAction.HOLD),
            int(StrategicAction.LEVEL),
            int(StrategicAction.HOLD),
        ),
        "XP buys, illegal affordability checks, and round advancement",
    ),
    ParityScenario(
        "illegal_actions",
        (
            999,
            int(StrategicAction.SLAM_CARRY_ITEM),
            int(StrategicAction.HOLD),
        ),
        "out-of-range and currently illegal legal-mask behavior",
    ),
    ParityScenario(
        "terminal_pressure",
        (int(StrategicAction.HOLD),) * 64,
        "terminal death under the tuned hard enemy-pressure curve",
    ),
)
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}
FUZZ_POLICIES = ("random_legal", "random_mixed", "heuristic")


@dataclass(frozen=True)
class StrategicParityMatrixConfig:
    out_dir: Path = DEFAULT_OUT_DIR
    seeds: tuple[int, ...] = (0, 1, 7, 19)
    scenarios: tuple[ParityScenario, ...] = SCENARIOS
    cc: str = "cc"
    tolerance: float = 1.0e-4
    fuzz_episodes: int = 0
    fuzz_max_steps: int = 80
    fuzz_policies: tuple[str, ...] = FUZZ_POLICIES
    fuzz_seed: int = 1729


def run_strategic_parity_matrix(config: StrategicParityMatrixConfig) -> dict[str, Any]:
    if not config.seeds and config.scenarios:
        raise ValueError("at least one seed is required")
    if not config.scenarios and config.fuzz_episodes <= 0:
        raise ValueError("at least one fixed or fuzz parity scenario is required")
    if config.fuzz_episodes < 0:
        raise ValueError("fuzz_episodes must be non-negative")
    if config.fuzz_max_steps <= 0:
        raise ValueError("fuzz_max_steps must be positive")
    unknown_fuzz_policies = sorted(set(config.fuzz_policies) - set(FUZZ_POLICIES))
    if unknown_fuzz_policies:
        raise ValueError(f"unknown fuzz policies: {unknown_fuzz_policies}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    ocean_binary = _compile_ocean_trace_binary(config)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fuzz_scenarios = _generate_fuzz_scenarios(config)

    for scenario in (*config.scenarios, *fuzz_scenarios):
        scenario_seeds = (scenario.seed,) if scenario.seed is not None else config.seeds
        for seed in scenario_seeds:
            if seed is None:
                raise ValueError(f"scenario {scenario.name} has no seed")
            python_rows = _python_trace_rows(seed=seed, actions=scenario.actions)
            python_signatures = _python_trace_signatures(seed=seed, actions=scenario.actions)

            native_signatures = trace_signatures(seed=seed, actions=scenario.actions)
            native_mismatches = _compare_values(
                "signature",
                _jsonable(python_signatures),
                _jsonable(native_signatures),
                tolerance=config.tolerance,
            )
            native_result = _result_row(
                backend="native_cpp",
                scenario=scenario,
                seed=seed,
                compared_rows=len(python_rows),
                compared_fields=["state_signature"],
                mismatches=native_mismatches,
            )
            results.append(native_result)
            if native_mismatches:
                failures.append(native_result)

            ocean_rows = _run_ocean_trace_binary(
                ocean_binary,
                seed=seed,
                actions=scenario.actions,
            )
            ocean_mismatches = _compare_values(
                "trace_rows",
                python_rows,
                ocean_rows,
                tolerance=config.tolerance,
            )
            ocean_result = _result_row(
                backend="ocean_c",
                scenario=scenario,
                seed=seed,
                compared_rows=len(python_rows),
                compared_fields=[
                    "state",
                    "observation",
                    "legal_mask",
                    "reward",
                    "terminal",
                    "placement_proxy",
                    "scenario_score",
                ],
                mismatches=ocean_mismatches,
            )
            results.append(ocean_result)
            if ocean_mismatches:
                failures.append(ocean_result)

    report: dict[str, Any] = {
        "schema": "strategic-parity-matrix/v1",
        "status": "pass" if not failures else "fail",
        "backends": ["python", "native_cpp", "ocean_c"],
        "oracle_backend": "python",
        "seeds": list(config.seeds),
        "fuzz": {
            "enabled": config.fuzz_episodes > 0,
            "episodes_per_policy": config.fuzz_episodes,
            "max_steps": config.fuzz_max_steps,
            "policies": list(config.fuzz_policies),
            "seed": config.fuzz_seed,
            "cases": len(fuzz_scenarios),
        },
        "scenarios": [
            {
                "name": scenario.name,
                "kind": scenario.kind,
                "policy": scenario.policy,
                "seed": scenario.seed,
                "description": scenario.description,
                "actions": list(scenario.actions),
            }
            for scenario in (*config.scenarios, *fuzz_scenarios)
        ],
        "tolerance": config.tolerance,
        "checks": results,
        "summary": {
            "total_checks": len(results),
            "passed": sum(1 for result in results if result["status"] == "pass"),
            "failed": len(failures),
        },
        "artifacts": ["metrics.json", "matrix.jsonl", "decision.md"],
        "known_limits": [
            "native_cpp parity compares Python-compatible state signatures",
            "ocean_c parity allows fp32-vs-fp64 tolerance for combat float fields",
            "ocean_c parity compares no-reset trace rows, not Puffer's auto-reset worker loop",
            "fuzz parity is differential testing over generated traces, not exhaustive proof",
            "parity does not prove policy quality or MuZero learning quality",
        ],
    }
    _write_json(config.out_dir / "metrics.json", report)
    _write_jsonl(config.out_dir / "matrix.jsonl", results)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    return report


def _generate_fuzz_scenarios(config: StrategicParityMatrixConfig) -> tuple[ParityScenario, ...]:
    if config.fuzz_episodes <= 0:
        return tuple()
    rng = np.random.default_rng(config.fuzz_seed)
    scenarios: list[ParityScenario] = []
    for policy in config.fuzz_policies:
        for episode_index in range(config.fuzz_episodes):
            seed = int(rng.integers(0, 2_000_000_000))
            action_seed = int(rng.integers(0, 2_000_000_000))
            actions = _generate_fuzz_actions(
                seed=seed,
                policy=policy,
                max_steps=config.fuzz_max_steps,
                action_seed=action_seed,
            )
            scenarios.append(
                ParityScenario(
                    name=f"fuzz_{policy}_{episode_index:04d}",
                    actions=actions,
                    description=(
                        f"{policy} generated trace with seed={seed}, "
                        f"action_seed={action_seed}, max_steps={config.fuzz_max_steps}"
                    ),
                    kind="fuzz",
                    policy=policy,
                    seed=seed,
                )
            )
    return tuple(scenarios)


def _generate_fuzz_actions(
    *,
    seed: int,
    policy: str,
    max_steps: int,
    action_seed: int,
) -> tuple[int, ...]:
    config = StrategicConfig()
    rng = np.random.default_rng(action_seed)
    state = reset(seed=seed, config=config)
    actions: list[int] = []
    for step_index in range(max_steps):
        if state.done:
            break
        mask = legal_action_mask(state, config)
        action = _select_fuzz_action(
            state=state,
            mask=mask,
            config=config,
            policy=policy,
            rng=rng,
            step_index=step_index,
        )
        actions.append(action)
        step(state, action, config)
    return tuple(actions)


def _select_fuzz_action(
    *,
    state: Any,
    mask: np.ndarray,
    config: StrategicConfig,
    policy: str,
    rng: np.random.Generator,
    step_index: int,
) -> int:
    legal_actions = [int(action) for action in np.flatnonzero(mask)]
    if policy == "heuristic":
        return int(tft_heuristic_policy(state, mask, config))
    if policy == "random_legal":
        if not legal_actions:
            return int(StrategicAction.HOLD)
        return legal_actions[int(rng.integers(0, len(legal_actions)))]
    if policy == "random_mixed":
        if step_index % 5 == 2:
            illegal_actions = [action for action in range(NUM_ACTIONS) if not bool(mask[action])]
            illegal_actions.extend([NUM_ACTIONS, NUM_ACTIONS + 17, 999])
            return illegal_actions[int(rng.integers(0, len(illegal_actions)))]
        if not legal_actions:
            return int(StrategicAction.HOLD)
        return legal_actions[int(rng.integers(0, len(legal_actions)))]
    raise ValueError(f"unknown fuzz policy: {policy}")


def _python_trace_signatures(*, seed: int, actions: Sequence[int]) -> list[tuple[object, ...]]:
    state = reset(seed=seed)
    rows = [state_signature(state)]
    for action in actions:
        if state.done:
            break
        step(state, int(action))
        rows.append(state_signature(state))
    return rows


def _python_trace_rows(*, seed: int, actions: Sequence[int]) -> list[dict[str, Any]]:
    config = StrategicConfig()
    state = reset(seed=seed, config=config)
    rows = [_python_row(state, step_index=0, action=None, legal=True, reward=0.0, terminal=False)]
    for step_index, action in enumerate(actions, start=1):
        if state.done:
            break
        mask = legal_action_mask(state, config)
        legal = 0 <= int(action) < len(mask) and bool(mask[int(action)])
        result = step(state, int(action), config)
        rows.append(
            _python_row(
                state,
                step_index=step_index,
                action=int(action),
                legal=legal,
                reward=result.reward,
                terminal=result.terminated or result.truncated,
            )
        )
    return rows


def _python_row(
    state: Any,
    *,
    step_index: int,
    action: int | None,
    legal: bool,
    reward: float,
    terminal: bool,
) -> dict[str, Any]:
    config = StrategicConfig()
    return {
        "step": step_index,
        "action": action,
        "legal": legal,
        "reward": float(reward),
        "terminal": terminal,
        "state": {
            "seed": int(state.seed),
            "rng_key": int(state.rng_key),
            "round": int(state.round),
            "hp": int(state.hp),
            "gold": int(state.gold),
            "level": int(state.level),
            "xp": int(state.xp),
            "shop": _int_list(state.shop),
            "owned": _int_list(state.owned),
            "fielded": _int_list(state.fielded),
            "role_items": _int_list(state.role_items),
            "role_item_slots": _int_list(state.role_item_slots),
            "done": bool(state.done),
            "final_reason": state.final_reason,
            "action_count": int(state.action_count),
            "last_board_strength": float(state.last_board_strength),
            "last_enemy_strength": float(state.last_enemy_strength),
            "last_damage": int(state.last_damage),
            "last_win": bool(state.last_win),
            "total_rolls": int(state.total_rolls),
            "total_xp_buys": int(state.total_xp_buys),
            "total_units_bought": int(state.total_units_bought),
            "total_item_slams": int(state.total_item_slams),
            "total_illegal_actions": int(state.total_illegal_actions),
        },
        "observation": observe(state, config).astype(float).tolist(),
        "legal_mask": legal_action_mask(state, config).astype(bool).tolist(),
        "placement_proxy": int(placement_proxy(state, config)),
        "scenario_score": float(scenario_score(state, config)),
    }


def _compile_ocean_trace_binary(config: StrategicParityMatrixConfig) -> Path:
    trace_c = _ocean_trace_source()
    build_dir = config.out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / "strategic_ocean_trace.c"
    binary = build_dir / "strategic_ocean_trace"
    source.write_text(trace_c, encoding="utf-8")
    subprocess.run(
        [
            config.cc,
            "-O3",
            "-std=c11",
            "-D_POSIX_C_SOURCE=199309L",
            "-I",
            str(OCEAN_SRC),
            str(source),
            "-lm",
            "-o",
            str(binary),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return binary


def _run_ocean_trace_binary(
    binary: Path,
    *,
    seed: int,
    actions: Sequence[int],
) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [str(binary), str(seed), *[str(action) for action in actions]],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    return rows


def _ocean_trace_source() -> str:
    return r'''
#include <stdio.h>
#include <stdlib.h>

#include "strategic_tft.h"

static void print_int_array(const int* values, int count) {
    printf("[");
    for (int i = 0; i < count; i++) {
        if (i) printf(",");
        printf("%d", values[i]);
    }
    printf("]");
}

static void print_float_array(const float* values, int count) {
    printf("[");
    for (int i = 0; i < count; i++) {
        if (i) printf(",");
        printf("%.9g", (double)values[i]);
    }
    printf("]");
}

static void print_mask(const unsigned char* values, int count) {
    printf("[");
    for (int i = 0; i < count; i++) {
        if (i) printf(",");
        printf(values[i] ? "true" : "false");
    }
    printf("]");
}

static void print_reason(int final_reason) {
    if (final_reason == 1) printf("\"hp_zero\"");
    else if (final_reason == 2) printf("\"max_round\"");
    else printf("null");
}

static void print_row(StrategicTFT* env, int step_index, int action, int legal) {
    printf("{");
    printf("\"step\":%d,", step_index);
    if (action < 0) printf("\"action\":null,");
    else printf("\"action\":%d,", action);
    printf("\"legal\":%s,", legal ? "true" : "false");
    printf("\"reward\":%.9g,", (double)env->rewards[0]);
    printf("\"terminal\":%s,", env->terminals[0] > 0.0f ? "true" : "false");
    printf("\"state\":{");
    printf("\"seed\":%d,", env->seed);
    printf("\"rng_key\":%llu,", (unsigned long long)env->rng_key);
    printf("\"round\":%d,", env->round);
    printf("\"hp\":%d,", env->hp);
    printf("\"gold\":%d,", env->gold);
    printf("\"level\":%d,", env->level);
    printf("\"xp\":%d,", env->xp);
    printf("\"shop\":"); print_int_array(env->shop, STRATEGIC_SHOP_SIZE); printf(",");
    printf("\"owned\":"); print_int_array(env->owned, STRATEGIC_OWNED_SIZE); printf(",");
    printf("\"fielded\":"); print_int_array(env->fielded, STRATEGIC_MAX_LEVEL); printf(",");
    printf("\"role_items\":"); print_int_array(env->role_items, STRATEGIC_ROLE_COUNT); printf(",");
    printf("\"role_item_slots\":");
    print_int_array(env->role_item_slots, STRATEGIC_ROLE_COUNT);
    printf(",");
    printf("\"done\":%s,", env->done ? "true" : "false");
    printf("\"final_reason\":"); print_reason(env->final_reason); printf(",");
    printf("\"action_count\":%d,", env->action_count);
    printf("\"last_board_strength\":%.9g,", (double)env->last_board_strength);
    printf("\"last_enemy_strength\":%.9g,", (double)env->last_enemy_strength);
    printf("\"last_damage\":%d,", env->last_damage);
    printf("\"last_win\":%s,", env->last_win ? "true" : "false");
    printf("\"total_rolls\":%d,", env->total_rolls);
    printf("\"total_xp_buys\":%d,", env->total_xp_buys);
    printf("\"total_units_bought\":%d,", env->total_units_bought);
    printf("\"total_item_slams\":%d,", env->total_item_slams);
    printf("\"total_illegal_actions\":%d", env->total_illegal_actions);
    printf("},");
    printf("\"observation\":");
    print_float_array(env->observations, STRATEGIC_OBS_SIZE);
    printf(",");
    printf("\"legal_mask\":"); print_mask(env->action_mask, STRATEGIC_NUM_ACTIONS); printf(",");
    printf("\"placement_proxy\":%d,", strategic_placement_proxy(env, env->final_reason == 2));
    printf("\"scenario_score\":%.9g", (double)strategic_scenario_score(env));
    printf("}\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: strategic_ocean_trace seed [actions...]\n");
        return 2;
    }
    StrategicTFT env = {0};
    float observations[STRATEGIC_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char action_mask[STRATEGIC_NUM_ACTIONS] = {0};
    env.num_agents = 1;
    env.base_seed = atoi(argv[1]);
    env.observations = observations;
    env.actions = actions;
    env.rewards = rewards;
    env.terminals = terminals;
    env.action_mask = action_mask;
    c_reset(&env);
    print_row(&env, 0, -1, 1);
    for (int argi = 2; argi < argc && !env.done; argi++) {
        int action = atoi(argv[argi]);
        int legal = action >= 0 && action < STRATEGIC_NUM_ACTIONS && env.action_mask[action];
        env.actions[0] = (float)action;
        strategic_step_no_reset(&env);
        print_row(&env, argi - 1, action, legal);
    }
    return 0;
}
'''


def _result_row(
    *,
    backend: str,
    scenario: ParityScenario,
    seed: int,
    compared_rows: int,
    compared_fields: list[str],
    mismatches: list[str],
) -> dict[str, Any]:
    return {
        "backend": backend,
        "scenario": scenario.name,
        "scenario_kind": scenario.kind,
        "policy": scenario.policy,
        "seed": seed,
        "actions": list(scenario.actions),
        "status": "pass" if not mismatches else "fail",
        "compared_rows": compared_rows,
        "compared_fields": compared_fields,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def _compare_values(path: str, left: Any, right: Any, *, tolerance: float) -> list[str]:
    mismatches: list[str] = []
    _compare(path, left, right, tolerance=tolerance, mismatches=mismatches)
    return mismatches


def _compare(path: str, left: Any, right: Any, *, tolerance: float, mismatches: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            mismatches.append(f"{path}.{key}: missing on right")
        for key in sorted(right_keys - left_keys):
            mismatches.append(f"{path}.{key}: unexpected on right")
        for key in sorted(left_keys & right_keys):
            _compare(
                f"{path}.{key}",
                left[key],
                right[key],
                tolerance=tolerance,
                mismatches=mismatches,
            )
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            mismatches.append(f"{path}: length {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare(
                f"{path}[{index}]",
                left_item,
                right_item,
                tolerance=tolerance,
                mismatches=mismatches,
            )
        return
    if (
        isinstance(left, Real)
        and isinstance(right, Real)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
        and (isinstance(left, float) or isinstance(right, float))
    ):
        if abs(float(left) - float(right)) > tolerance:
            mismatches.append(f"{path}: {left!r} != {right!r}")
        return
    if left != right:
        mismatches.append(f"{path}: {left!r} != {right!r}")


def _int_list(values: Any) -> list[int]:
    if isinstance(values, np.ndarray):
        return [int(value) for value in values.tolist()]
    return [int(value) for value in values]


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _format_decision(report: dict[str, Any]) -> str:
    lines = [
        "# Strategic Parity Matrix",
        "",
        f"Status: `{report['status']}`",
        f"Oracle backend: `{report['oracle_backend']}`",
        f"Backends: {', '.join(report['backends'])}",
        f"Seeds: {', '.join(str(seed) for seed in report['seeds'])}",
        f"Fuzz enabled: `{report['fuzz']['enabled']}`",
        f"Fuzz cases: {report['fuzz']['cases']}",
        f"Total checks: {report['summary']['total_checks']}",
        f"Failed checks: {report['summary']['failed']}",
        "",
        "| Backend | Kind | Scenario | Seed | Status | Rows | Mismatches |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in report["checks"]:
        lines.append(
            "| "
            f"{row['backend']} | "
            f"{row['scenario_kind']} | "
            f"{row['scenario']} | "
            f"{row['seed']} | "
            f"{row['status']} | "
            f"{row['compared_rows']} | "
            f"{row['mismatch_count']} |"
        )
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {limit}" for limit in report["known_limits"])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--scenario", choices=sorted(SCENARIO_BY_NAME), action="append")
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--fuzz-episodes", type=int, default=0)
    parser.add_argument("--fuzz-max-steps", type=int, default=80)
    parser.add_argument("--fuzz-policy", choices=FUZZ_POLICIES, action="append")
    parser.add_argument("--fuzz-seed", type=int, default=1729)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenarios = (
        tuple(SCENARIO_BY_NAME[name] for name in args.scenario)
        if args.scenario
        else SCENARIOS
    )
    report = run_strategic_parity_matrix(
        StrategicParityMatrixConfig(
            out_dir=args.out_dir,
            seeds=tuple(args.seed) if args.seed else (0, 1, 7, 19),
            scenarios=scenarios,
            cc=args.cc,
            fuzz_episodes=args.fuzz_episodes,
            fuzz_max_steps=args.fuzz_max_steps,
            fuzz_policies=tuple(args.fuzz_policy) if args.fuzz_policy else FUZZ_POLICIES,
            fuzz_seed=args.fuzz_seed,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
