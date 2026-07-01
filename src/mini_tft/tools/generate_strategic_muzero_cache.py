"""Generate MCTS-target MuZero cache rows for the strategic lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from mini_tft.strategic.adapters.mcts import StrategicMCTSConfig
from mini_tft.strategic.adapters.muzero_cache import (
    CacheRow,
    cache_metrics,
    cache_row_payload,
    generate_mcts_cache,
)
from mini_tft.strategic.core.obs import observe
from mini_tft.strategic.core.rules import legal_action_mask, reset, scenario_score, step


@dataclass(frozen=True)
class StrategicMuZeroCacheRunConfig:
    out_dir: Path
    episodes: int = 64
    max_rows: int = 1024
    seed: int = 0
    backend: str = "python"
    simulations: int = 16
    max_depth: int = 10
    rollout_steps: int = 6
    prior_mode: str = "heuristic"
    value_mode: str = "heuristic"
    checkpoint_path: Path | None = None
    checkpoint_device: str = "cpu"
    determinism_check: bool = True


def run_strategic_muzero_cache_run(
    config: StrategicMuZeroCacheRunConfig,
) -> dict[str, Any]:
    if config.episodes <= 0:
        raise ValueError("episodes must be positive")
    if config.max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if config.simulations <= 0:
        raise ValueError("simulations must be positive")
    if config.backend not in {"auto", "python", "native"}:
        raise ValueError("backend must be one of: auto, python, native")
    if config.prior_mode not in {"uniform", "heuristic", "checkpoint"}:
        raise ValueError("prior_mode must be one of: uniform, heuristic, checkpoint")
    if config.value_mode not in {"heuristic", "checkpoint"}:
        raise ValueError("value_mode must be one of: heuristic, checkpoint")
    checkpoint_guided = config.prior_mode == "checkpoint" or config.value_mode == "checkpoint"
    if checkpoint_guided:
        if config.checkpoint_path is None:
            raise ValueError("checkpoint-guided MCTS requires checkpoint_path")
        if not config.checkpoint_path.exists():
            raise ValueError(f"checkpoint does not exist: {config.checkpoint_path}")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    checkpoint_evaluator = None
    if checkpoint_guided and config.checkpoint_path is not None:
        from mini_tft.tools.train_strategic_muzero_torch import load_torch_muzero_policy_value

        checkpoint_evaluator = load_torch_muzero_policy_value(
            config.checkpoint_path,
            device=config.checkpoint_device,
        )
    mcts_config = StrategicMCTSConfig(
        simulations=config.simulations,
        max_depth=config.max_depth,
        rollout_steps=config.rollout_steps,
        prior_mode=config.prior_mode,
        value_mode=config.value_mode,
        checkpoint_path=str(config.checkpoint_path) if config.checkpoint_path else None,
        checkpoint_evaluator=checkpoint_evaluator,
    )
    rows, backend_used, fallback_reason = _generate_rows(config, mcts_config)
    elapsed_sec = perf_counter() - started
    metrics = cache_metrics(rows)
    checksum = _rows_checksum(rows)
    deterministic = None
    if config.determinism_check:
        deterministic_rows, _, _ = _generate_rows(
            StrategicMuZeroCacheRunConfig(
                out_dir=config.out_dir,
                episodes=config.episodes,
                max_rows=config.max_rows,
                seed=config.seed,
                backend=backend_used,
                simulations=config.simulations,
                max_depth=config.max_depth,
                rollout_steps=config.rollout_steps,
                prior_mode=config.prior_mode,
                value_mode=config.value_mode,
                checkpoint_path=config.checkpoint_path,
                checkpoint_device=config.checkpoint_device,
                determinism_check=False,
            ),
            mcts_config,
        )
        deterministic = checksum == _rows_checksum(deterministic_rows)

    report = {
        "schema": "quality-gate/v1",
        "deliverable": "muzero_cache",
        "status": _status(metrics, deterministic),
        "seed": config.seed,
        "commands": [
            "env -u UV_PYTHON uv run python -m "
            "mini_tft.tools.generate_strategic_muzero_cache"
        ],
        "metrics": {
            "cache": metrics,
            "search_smoke": {
                "ran": True,
                "backend": backend_used,
                "backend_requested": config.backend,
                "fallback_reason": fallback_reason,
                "total_decisions": len(rows),
                "illegal_action_count": _illegal_action_count(rows),
                "simulations": config.simulations,
                "prior_mode": config.prior_mode,
                "value_mode": config.value_mode,
                "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
                "checkpoint_device": config.checkpoint_device if checkpoint_guided else None,
                "elapsed_sec": elapsed_sec,
                "decisions_per_sec": len(rows) / elapsed_sec if elapsed_sec > 0.0 else 0.0,
            },
            "determinism": {
                "fixed_seed_reproducible": bool(deterministic),
                "checksum": checksum,
                "stable_checksum_excludes_timing": True,
            },
            "config": {
                "episodes": config.episodes,
                "max_rows": config.max_rows,
                "backend": config.backend,
                "simulations": config.simulations,
                "max_depth": config.max_depth,
                "rollout_steps": config.rollout_steps,
                "prior_mode": config.prior_mode,
                "value_mode": config.value_mode,
                "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
                "checkpoint_device": config.checkpoint_device,
                "determinism_check": config.determinism_check,
            },
        },
        "artifacts": ["rows.jsonl", "metrics.json", "decision.md"],
        "known_limits": [
            "MCTS targets are generated from the simulator-backed search, not learned dynamics",
            "this cache is readiness evidence for a tiny MuZero trainer, not policy quality",
        ],
    }
    _write_jsonl(config.out_dir / "rows.jsonl", _row_payloads(rows))
    _write_json(config.out_dir / "metrics.json", report)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    return report


def _generate_rows(
    config: StrategicMuZeroCacheRunConfig,
    mcts_config: StrategicMCTSConfig,
) -> tuple[list[CacheRow], str, str | None]:
    if config.backend in {"auto", "native"}:
        if config.prior_mode == "checkpoint" or config.value_mode == "checkpoint":
            if config.backend == "native":
                raise ValueError("native backend does not support checkpoint-guided MCTS")
            fallback_reason = "checkpoint-guided MCTS requires python backend"
        else:
            try:
                return (
                    _generate_native_mcts_cache(
                        episodes=config.episodes,
                        max_rows=config.max_rows,
                        seed=config.seed,
                        simulations=config.simulations,
                        max_depth=config.max_depth,
                        rollout_steps=config.rollout_steps,
                        prior_mode=config.prior_mode,
                    ),
                    "native",
                    None,
                )
            except (ImportError, RuntimeError, ValueError) as exc:
                if config.backend == "native":
                    raise
                fallback_reason = str(exc)
    else:
        fallback_reason = None

    return (
        generate_mcts_cache(
            episodes=config.episodes,
            max_rows=config.max_rows,
            seed=config.seed,
            mcts_config=mcts_config,
        ),
        "python",
        fallback_reason,
    )


def _generate_native_mcts_cache(
    *,
    episodes: int,
    max_rows: int,
    seed: int,
    simulations: int,
    max_depth: int,
    rollout_steps: int,
    prior_mode: str,
) -> list[CacheRow]:
    from mini_tft.strategic.native import native_available, run_native_mcts_smoke

    if not native_available():
        raise RuntimeError("native strategic MCTS extension is not available")

    native_result = run_native_mcts_smoke(
        episodes=episodes,
        seed=seed,
        simulations=(simulations,),
        max_depth=max_depth,
        rollout_steps=rollout_steps,
        prior_mode=prior_mode,
    )
    decision_rows = [
        dict(row)
        for row in native_result["decision_rows"]
        if str(dict(row).get("policy")) == f"mcts_{simulations}"
    ]
    decisions_by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in decision_rows:
        decisions_by_episode.setdefault(_as_int(row["episode"]), []).append(row)
    for episode_decisions in decisions_by_episode.values():
        episode_decisions.sort(key=lambda row: _as_int(row["step"]))

    rows: list[CacheRow] = []
    for episode in range(episodes):
        state = reset(seed=seed + episode)
        episode_rows: list[CacheRow] = []
        for decision in decisions_by_episode.get(episode, []):
            if len(rows) + len(episode_rows) >= max_rows:
                break
            obs = observe(state)
            mask = legal_action_mask(state)
            action = _as_int(decision["action_id"])
            policy_target = np.asarray(
                [float(value) for value in decision["visit_policy"]],
                dtype=np.float32,
            )
            if policy_target.shape != mask.shape:
                raise RuntimeError("native visit policy shape does not match legal action mask")
            result = step(state, action)
            next_obs = observe(state)
            episode_rows.append(
                CacheRow(
                    observation=obs,
                    legal_mask=mask,
                    action=action,
                    reward=result.reward,
                    next_observation=next_obs,
                    done=result.terminated or result.truncated,
                    policy_target=policy_target,
                    value_target=0.0,
                    metadata={
                        "episode": episode,
                        "seed": seed + episode,
                        "round": result.info["round"],
                        "hp": result.info["hp"],
                        "placement_proxy": result.info["placement_proxy"],
                        "scenario_score": scenario_score(state),
                        "legal_action": result.info["legal_action"],
                        "policy_target_source": "native_mcts",
                        "mcts_simulations": simulations,
                        "mcts_max_depth": max_depth,
                        "mcts_elapsed_ms": _as_float(decision.get("mcts_elapsed_ms", 0.0)),
                        "mcts_root_visits": int(sum(_action_visit_values(decision))),
                    },
                )
            )
        _assign_returns(episode_rows)
        rows.extend(episode_rows)
        if len(rows) >= max_rows:
            break
    if not rows:
        raise RuntimeError("native strategic MCTS produced no cache rows")
    return rows


def _assign_returns(rows: list[CacheRow], gamma: float = 0.97) -> None:
    value = 0.0
    for index in range(len(rows) - 1, -1, -1):
        row = rows[index]
        value = row.reward + gamma * value
        rows[index] = CacheRow(
            observation=row.observation,
            legal_mask=row.legal_mask,
            action=row.action,
            reward=row.reward,
            next_observation=row.next_observation,
            done=row.done,
            policy_target=row.policy_target,
            value_target=float(value),
            metadata=row.metadata,
        )


def _action_visit_values(decision: Mapping[str, Any]) -> list[int]:
    raw = decision.get("action_visits", {})
    if not isinstance(raw, Mapping):
        return []
    return [_as_int(value) for value in raw.values()]


def _status(metrics: Mapping[str, Any], deterministic: bool | None) -> str:
    if deterministic is False:
        return "fail"
    required = (
        float(metrics.get("legal_action_rate", 0.0)) == 1.0,
        float(metrics.get("mcts_target_rate", 0.0)) == 1.0,
        float(metrics.get("policy_target_valid_rate", 0.0)) == 1.0,
        float(metrics.get("value_target_finite_rate", 0.0)) == 1.0,
    )
    return "smoke_only" if all(required) else "fail"


def _illegal_action_count(rows: Sequence[CacheRow]) -> int:
    return sum(
        0 if 0 <= row.action < len(row.legal_mask) and bool(row.legal_mask[row.action]) else 1
        for row in rows
    )


def _rows_checksum(rows: Sequence[CacheRow]) -> str:
    digest = hashlib.sha256()
    for payload in _row_payloads(rows):
        stable_payload = _stable_payload(payload)
        encoded = json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode()
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    stable = dict(payload)
    metadata = dict(stable.get("metadata", {}))
    metadata.pop("mcts_elapsed_ms", None)
    stable["metadata"] = metadata
    return stable


def _row_payloads(rows: Sequence[CacheRow]) -> list[dict[str, Any]]:
    return [cache_row_payload(index, row) for index, row in enumerate(rows)]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _as_int(value: Any) -> int:
    if value is None:
        raise ValueError("expected int-compatible value, got None")
    return int(value)


def _as_float(value: Any) -> float:
    if value is None:
        raise ValueError("expected float-compatible value, got None")
    return float(value)


def _format_decision(report: Mapping[str, Any]) -> str:
    metrics = dict(report["metrics"])
    cache = dict(metrics["cache"])
    search = dict(metrics["search_smoke"])
    determinism = dict(metrics["determinism"])
    return "\n".join(
        [
            "# Strategic MuZero MCTS Cache",
            "",
            f"Status: {report['status']}",
            "",
            "Evidence:",
            f"- Rows: {cache['rows']}",
            f"- Legal action rate: {cache['legal_action_rate']:.6f}",
            f"- MCTS target rows: {cache['mcts_target_rows']}",
            f"- MCTS target rate: {cache['mcts_target_rate']:.6f}",
            f"- Policy target valid rate: {cache['policy_target_valid_rate']:.6f}",
            f"- Value target finite rate: {cache['value_target_finite_rate']:.6f}",
            f"- Decisions/sec: {search['decisions_per_sec']:.2f}",
            f"- Prior mode: {search['prior_mode']}",
            f"- Value mode: {search['value_mode']}",
            f"- Checkpoint: {search['checkpoint_path']}",
            f"- Fixed-seed reproducible: {determinism['fixed_seed_reproducible']}",
            "",
            "Limits:",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/strategic_lane/muzero_mcts_cache"),
    )
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "python", "native"], default="python")
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument(
        "--prior-mode",
        choices=["uniform", "heuristic", "checkpoint"],
        default="heuristic",
    )
    parser.add_argument("--value-mode", choices=["heuristic", "checkpoint"], default="heuristic")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-device", default="cpu")
    parser.add_argument("--skip-determinism-check", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_strategic_muzero_cache_run(
        StrategicMuZeroCacheRunConfig(
            out_dir=args.out_dir,
            episodes=args.episodes,
            max_rows=args.max_rows,
            seed=args.seed,
            backend=args.backend,
            simulations=args.simulations,
            max_depth=args.max_depth,
            rollout_steps=args.rollout_steps,
            prior_mode=args.prior_mode,
            value_mode=args.value_mode,
            checkpoint_path=args.checkpoint,
            checkpoint_device=args.checkpoint_device,
            determinism_check=not args.skip_determinism_check,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
