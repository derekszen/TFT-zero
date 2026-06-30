"""Reference-vs-batched throughput benchmark for the strategic lane."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mini_tft.strategic.adapters.baselines import tft_heuristic_policy
from mini_tft.strategic.core.actions import StrategicAction
from mini_tft.strategic.core.rules import legal_action_mask, run_episode, step
from mini_tft.strategic.core.state import DEFAULT_STRATEGIC_CONFIG, StrategicConfig


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_kind: str
    semantic_parity: bool
    puffer_trainer: bool
    reference_steps_per_second: float
    batched_steps_per_second: float
    reference_steps_per_second_std: float
    batched_steps_per_second_std: float
    speedup: float
    reference_steps: int
    batched_steps: int
    runs: int
    decision: str

    def to_dict(self) -> dict[str, bool | int | float | str]:
        return {
            "benchmark_kind": self.benchmark_kind,
            "semantic_parity": self.semantic_parity,
            "puffer_trainer": self.puffer_trainer,
            "reference_steps_per_second": self.reference_steps_per_second,
            "batched_steps_per_second": self.batched_steps_per_second,
            "reference_steps_per_second_std": self.reference_steps_per_second_std,
            "batched_steps_per_second_std": self.batched_steps_per_second_std,
            "speedup": self.speedup,
            "reference_steps": self.reference_steps,
            "batched_steps": self.batched_steps,
            "runs": self.runs,
            "decision": self.decision,
        }


def run_benchmark(
    *,
    episodes: int = 64,
    batch_size: int = 256,
    repeated_runs: int = 3,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
    out_dir: Path | None = None,
    prefer_semantic_puffer: bool = True,
) -> BenchmarkResult:
    if prefer_semantic_puffer:
        result = _try_run_native_puffer_benchmark(
            episodes=episodes,
            batch_size=batch_size,
            repeated_runs=repeated_runs,
            config=config,
        )
        if result is not None:
            if out_dir is not None:
                write_benchmark_artifacts(result, out_dir)
            return result

    reference_rates: list[float] = []
    batched_rates: list[float] = []
    reference_steps = 0
    batched_steps = 0

    for run in range(repeated_runs):
        start = time.perf_counter()
        steps = 0
        for episode in range(episodes):
            _, _, episode_steps = run_episode(
                tft_heuristic_policy,
                seed=run * 10_000 + episode,
                config=config,
            )
            steps += episode_steps
        elapsed = max(time.perf_counter() - start, 1e-9)
        reference_rates.append(steps / elapsed)
        reference_steps += steps

        start = time.perf_counter()
        batch_steps = _run_batched_greed_rollout(batch_size=batch_size, config=config)
        elapsed = max(time.perf_counter() - start, 1e-9)
        batched_rates.append(batch_steps / elapsed)
        batched_steps += batch_steps

    reference_rate = float(np.mean(reference_rates))
    batched_rate = float(np.mean(batched_rates))
    speedup = batched_rate / max(reference_rate, 1e-9)
    decision = "smoke_only" if speedup > 0.0 else "fail"
    result = BenchmarkResult(
        benchmark_kind="proxy_array_math_ceiling",
        semantic_parity=False,
        puffer_trainer=False,
        reference_steps_per_second=reference_rate,
        batched_steps_per_second=batched_rate,
        reference_steps_per_second_std=float(np.std(reference_rates)),
        batched_steps_per_second_std=float(np.std(batched_rates)),
        speedup=speedup,
        reference_steps=reference_steps,
        batched_steps=batched_steps,
        runs=repeated_runs,
        decision=decision,
    )
    if out_dir is not None:
        write_benchmark_artifacts(result, out_dir)
    return result


def write_benchmark_artifacts(result: BenchmarkResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.semantic_parity:
        known_limits = [
            "Puffer path uses native batched strategic vectorization",
            "material pass threshold is 3x repeated matched scalar throughput",
            "this is not full PPO quality evidence",
        ]
        limits = [
            "- This is throughput evidence, not full PPO quality evidence.",
            "- This is not full PPO quality evidence.",
        ]
    else:
        known_limits = [
            "batched path is a strategic-lane throughput proxy, not a full Puffer trainer",
            "batched path does not use canonical strategic rules and is not promotion evidence",
        ]
        limits = [
            "- This is proxy array math, not a matched canonical-rules benchmark.",
            "- This is not full PPO quality or Puffer trainer evidence.",
        ]
    metrics: dict[str, Any] = {
        "schema": "quality-gate/v1",
        "deliverable": "puffer_speed",
        "status": result.decision,
        "seed": 0,
        "commands": [],
        "metrics": result.to_dict(),
        "artifacts": ["metrics.json", "decision.md"],
        "known_limits": known_limits,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "decision.md").write_text(
        "\n".join(
            [
                "# Decision",
                "",
                f"Status: {result.decision}",
                "",
                "Evidence:",
                f"- Reference steps/sec: {result.reference_steps_per_second:.2f}",
                f"- Reference std: {result.reference_steps_per_second_std:.2f}",
                f"- Batched steps/sec: {result.batched_steps_per_second:.2f}",
                f"- Batched std: {result.batched_steps_per_second_std:.2f}",
                f"- Speedup: {result.speedup:.2f}x",
                f"- Semantic parity: {result.semantic_parity}",
                f"- Puffer trainer: {result.puffer_trainer}",
                "",
                "Limits:",
                *limits,
                "",
                "Next:",
                "- Run strategic PPO on the native Puffer backend and compare policy quality.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _try_run_native_puffer_benchmark(
    *,
    episodes: int,
    batch_size: int,
    repeated_runs: int,
    config: StrategicConfig,
) -> BenchmarkResult | None:
    try:
        from mini_tft.rl.puffer_env import (
            StrategicMaskInObsEnv,
            make_strategic_native_puffer_env,
        )
    except (ImportError, RuntimeError):
        return None

    reference_rates: list[float] = []
    native_rates: list[float] = []
    reference_steps = 0
    native_steps = 0
    semantic_parity = True
    script_count = max(episodes, batch_size)

    for run in range(repeated_runs):
        scripts = [
            _heuristic_action_script(seed=run * 10_000 + episode, config=config)
            for episode in range(script_count)
        ]
        replay_steps = min(24, *(len(script) for script in scripts))
        scripts = [script[:replay_steps] for script in scripts]

        reference_env = StrategicMaskInObsEnv(config=config)
        start = time.perf_counter()
        run_reference_steps = 0
        reference_traces: list[_ReplayTrace] = []
        for episode, script in enumerate(scripts):
            trace = _replay_gym_env(
                reference_env,
                seed=run * 10_000 + episode,
                actions=script,
            )
            reference_traces.append(trace)
            run_reference_steps += trace.steps
        elapsed = max(time.perf_counter() - start, 1e-9)
        reference_rates.append(run_reference_steps / elapsed)
        reference_steps += run_reference_steps
        reference_env.close()

        try:
            native_env = make_strategic_native_puffer_env(
                num_envs=script_count,
                config=config,
            )
        except RuntimeError:
            return None
        start = time.perf_counter()
        puffer_traces = _replay_native_puffer_env(
            native_env,
            seed=run * 10_000,
            actions_by_env=scripts,
        )
        run_native_steps = sum(trace.steps for trace in puffer_traces)
        elapsed = max(time.perf_counter() - start, 1e-9)
        native_rates.append(run_native_steps / elapsed)
        native_steps += run_native_steps
        native_env.close()

        semantic_parity = semantic_parity and _traces_match(reference_traces, puffer_traces)

    reference_rate = float(np.mean(reference_rates))
    native_rate = float(np.mean(native_rates))
    speedup = native_rate / max(reference_rate, 1e-9)
    if not semantic_parity:
        decision = "fail"
    elif speedup >= 3.0:
        decision = "pass"
    else:
        decision = "inconclusive"
    return BenchmarkResult(
        benchmark_kind="strategic_native_puffer_vector",
        semantic_parity=semantic_parity,
        puffer_trainer=True,
        reference_steps_per_second=reference_rate,
        batched_steps_per_second=native_rate,
        reference_steps_per_second_std=float(np.std(reference_rates)),
        batched_steps_per_second_std=float(np.std(native_rates)),
        speedup=speedup,
        reference_steps=reference_steps,
        batched_steps=native_steps,
        runs=repeated_runs,
        decision=decision,
    )


@dataclass(frozen=True)
class _ReplayTrace:
    steps: int
    rewards: tuple[float, ...]
    terminals: tuple[bool, ...]
    truncations: tuple[bool, ...]
    final_observation: tuple[float, ...]


def _heuristic_action_script(*, seed: int, config: StrategicConfig) -> list[int]:
    from mini_tft.strategic.core.rules import reset

    state = reset(seed=seed, config=config)
    actions: list[int] = []
    max_steps = config.max_round * (config.max_actions_per_round + 1)
    while not state.done and len(actions) < max_steps:
        mask = legal_action_mask(state, config)
        action = int(tft_heuristic_policy(state, mask, config))
        actions.append(action)
        step(state, action, config)
    return actions


def _replay_gym_env(
    env: Any,
    *,
    seed: int,
    actions: list[int],
) -> _ReplayTrace:
    obs, _info = env.reset(seed=seed)
    rewards: list[float] = []
    terminals: list[bool] = []
    truncations: list[bool] = []
    steps = 0
    for action in actions:
        obs, reward, terminated, truncated, _info = env.step(int(action))
        rewards.append(float(reward))
        terminals.append(bool(terminated))
        truncations.append(bool(truncated))
        steps += 1
        if bool(terminated or truncated):
            break
    return _ReplayTrace(
        steps=steps,
        rewards=tuple(rewards),
        terminals=tuple(terminals),
        truncations=tuple(truncations),
        final_observation=_flatten_observation(obs),
    )


def _replay_native_puffer_env(
    env: Any,
    *,
    seed: int,
    actions_by_env: list[list[int]],
) -> list[_ReplayTrace]:
    obs, _infos = env.reset(seed=seed)
    env_count = len(actions_by_env)
    if not actions_by_env:
        return []
    step_count = len(actions_by_env[0])
    if any(len(actions) != step_count for actions in actions_by_env):
        raise ValueError("native replay expects fixed-length action scripts")
    action_matrix = np.asarray(actions_by_env, dtype=np.int64).T
    rewards = np.zeros((step_count, env_count), dtype=np.float32)
    terminals = np.zeros((step_count, env_count), dtype=np.bool_)
    truncations = np.zeros((step_count, env_count), dtype=np.bool_)
    final_observations = np.asarray(obs, dtype=np.float32).copy()
    for step_index, actions in enumerate(action_matrix):
        next_obs, step_rewards, step_terminals, step_truncations, infos = env.step(actions)
        rewards[step_index] = np.asarray(step_rewards, dtype=np.float32)
        terminals[step_index] = np.asarray(step_terminals, dtype=np.bool_)
        truncations[step_index] = np.asarray(step_truncations, dtype=np.bool_)
        final_observations = np.asarray(next_obs, dtype=np.float32).copy()
        done = terminals[step_index] | truncations[step_index]
        if bool(done.any()):
            for env_index in np.flatnonzero(done):
                final_observations[int(env_index)] = np.asarray(
                    infos[int(env_index)].get("terminal_observation", next_obs[int(env_index)]),
                    dtype=np.float32,
                )
    return [
        _ReplayTrace(
            steps=step_count,
            rewards=tuple(float(value) for value in rewards[:, env_index].tolist()),
            terminals=tuple(bool(value) for value in terminals[:, env_index].tolist()),
            truncations=tuple(bool(value) for value in truncations[:, env_index].tolist()),
            final_observation=_flatten_observation(final_observations[env_index]),
        )
        for env_index in range(env_count)
    ]


def _flatten_observation(obs: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(obs, dtype=np.float32).reshape(-1).tolist())


def _traces_match(left: list[_ReplayTrace], right: list[_ReplayTrace]) -> bool:
    if len(left) != len(right):
        return False
    for left_trace, right_trace in zip(left, right, strict=True):
        if left_trace.steps != right_trace.steps:
            return False
        if left_trace.terminals != right_trace.terminals:
            return False
        if left_trace.truncations != right_trace.truncations:
            return False
        if not np.allclose(
            np.asarray(left_trace.rewards),
            np.asarray(right_trace.rewards),
            rtol=1e-6,
            atol=1e-6,
        ):
            return False
        if not np.allclose(
            np.asarray(left_trace.final_observation),
            np.asarray(right_trace.final_observation),
            rtol=1e-6,
            atol=1e-6,
        ):
            return False
    return True


def _run_batched_greed_rollout(
    *,
    batch_size: int,
    config: StrategicConfig,
) -> int:
    # This intentionally benchmarks fixed-shape round pressure math. The full
    # object env remains the semantic reference; this path measures the speed
    # available once strategic state is represented as arrays.
    rounds = np.ones(batch_size, dtype=np.int16)
    hp = np.full(batch_size, config.starting_hp, dtype=np.float32)
    gold = np.full(batch_size, config.starting_gold, dtype=np.float32)
    strength = np.zeros(batch_size, dtype=np.float32)
    done = np.zeros(batch_size, dtype=np.bool_)
    steps = 0
    while not bool(done.all()) and steps < config.max_round:
        enemy = (
            13.0
            + rounds.astype(np.float32) * 3.2
            + np.power(rounds.astype(np.float32), 1.18) * 1.35
        )
        damage = np.maximum(0.0, enemy - strength) / 22.0 + np.where(rounds < 8, 2.0, 4.0)
        lost = strength < enemy
        hp = np.where(done, hp, np.where(lost, np.maximum(0.0, hp - damage), hp))
        gold = np.where(
            done,
            gold,
            gold + config.base_income + np.minimum(config.max_interest, gold // 10),
        )
        strength = np.where(done, strength, strength + np.minimum(8.0, gold * 0.03))
        done = done | (hp <= 0.0) | (rounds >= config.max_round)
        rounds = np.where(done, rounds, rounds + 1)
        steps += batch_size
    return steps


def smoke_reference_step(config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG) -> None:
    state, _, _ = run_episode(tft_heuristic_policy, seed=0, config=config)
    if not state.done:
        mask = legal_action_mask(state, config)
        step(state, int(StrategicAction.HOLD if mask[StrategicAction.HOLD] else 0), config)
