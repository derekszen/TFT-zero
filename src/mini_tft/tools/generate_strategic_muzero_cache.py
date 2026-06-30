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

from mini_tft.strategic.adapters.mcts import StrategicMCTSConfig
from mini_tft.strategic.adapters.muzero_cache import (
    CacheRow,
    cache_metrics,
    cache_row_payload,
    generate_mcts_cache,
)


@dataclass(frozen=True)
class StrategicMuZeroCacheRunConfig:
    out_dir: Path
    episodes: int = 64
    max_rows: int = 1024
    seed: int = 0
    simulations: int = 16
    max_depth: int = 10
    rollout_steps: int = 6
    prior_mode: str = "heuristic"
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
    config.out_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    mcts_config = StrategicMCTSConfig(
        simulations=config.simulations,
        max_depth=config.max_depth,
        rollout_steps=config.rollout_steps,
        prior_mode=config.prior_mode,
    )
    rows = generate_mcts_cache(
        episodes=config.episodes,
        max_rows=config.max_rows,
        seed=config.seed,
        mcts_config=mcts_config,
    )
    elapsed_sec = perf_counter() - started
    metrics = cache_metrics(rows)
    checksum = _rows_checksum(rows)
    deterministic = None
    if config.determinism_check:
        deterministic_rows = generate_mcts_cache(
            episodes=config.episodes,
            max_rows=config.max_rows,
            seed=config.seed,
            mcts_config=mcts_config,
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
                "backend": "python",
                "total_decisions": len(rows),
                "illegal_action_count": _illegal_action_count(rows),
                "simulations": config.simulations,
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
                "simulations": config.simulations,
                "max_depth": config.max_depth,
                "rollout_steps": config.rollout_steps,
                "prior_mode": config.prior_mode,
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
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--prior-mode", choices=["uniform", "heuristic"], default="heuristic")
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
            simulations=args.simulations,
            max_depth=args.max_depth,
            rollout_steps=args.rollout_steps,
            prior_mode=args.prior_mode,
            determinism_check=not args.skip_determinism_check,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
