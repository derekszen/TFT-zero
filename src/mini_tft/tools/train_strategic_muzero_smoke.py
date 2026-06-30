"""Train a tiny policy/value/dynamics smoke model over strategic MuZero rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StrategicMuZeroTrainSmokeConfig:
    rows_path: Path
    out_dir: Path
    epochs: int = 24
    learning_rate: float = 0.03
    seed: int = 0
    checkpoint_name: str = "train_smoke.npz"


def run_strategic_muzero_train_smoke(
    config: StrategicMuZeroTrainSmokeConfig,
) -> dict[str, Any]:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    dataset = _load_rows(config.rows_path)
    model = _init_model(dataset, seed=config.seed)
    history: list[dict[str, float]] = []
    for _ in range(config.epochs):
        losses, grads = _losses_and_grads(dataset, model)
        history.append(losses)
        for key, grad in grads.items():
            model[key] -= config.learning_rate * np.clip(grad, -5.0, 5.0)
    final_losses, _ = _losses_and_grads(dataset, model)
    history.append(final_losses)

    checkpoint_path = config.out_dir / config.checkpoint_name
    np.savez(
        checkpoint_path,
        policy_weights=model["policy_weights"],
        value_weights=model["value_weights"],
        dynamics_weights=model["dynamics_weights"],
        metadata=json.dumps(
            {
                "schema": "strategic-muzero-train-smoke-checkpoint/v1",
                "rows_path": str(config.rows_path),
                "rows": dataset.observations.shape[0],
                "observation_dim": dataset.observations.shape[1],
                "action_dim": dataset.policy_targets.shape[1],
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
            },
            sort_keys=True,
        ),
    )

    losses_finite = all(
        isfinite(value)
        for losses in (history[0], final_losses)
        for value in losses.values()
    )
    train_metrics = {
        "ran": True,
        "losses_finite": losses_finite,
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path),
        "rows": int(dataset.observations.shape[0]),
        "observation_dim": int(dataset.observations.shape[1]),
        "action_dim": int(dataset.policy_targets.shape[1]),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "elapsed_sec": perf_counter() - started,
        "policy_loss_initial": history[0]["policy_loss"],
        "value_loss_initial": history[0]["value_loss"],
        "dynamics_loss_initial": history[0]["dynamics_loss"],
        "total_loss_initial": history[0]["total_loss"],
        "policy_loss_final": final_losses["policy_loss"],
        "value_loss_final": final_losses["value_loss"],
        "dynamics_loss_final": final_losses["dynamics_loss"],
        "total_loss_final": final_losses["total_loss"],
    }
    report = {
        "schema": "quality-gate/v1",
        "deliverable": "muzero_cache",
        "status": "smoke_only" if losses_finite and checkpoint_path.exists() else "fail",
        "seed": config.seed,
        "commands": [
            "env -u UV_PYTHON uv run python -m mini_tft.tools.train_strategic_muzero_smoke"
        ],
        "metrics": {
            "train_smoke": train_metrics,
        },
        "artifacts": ["metrics.json", "decision.md", config.checkpoint_name],
        "known_limits": [
            "this is a tiny linear smoke model, not the production MuZero trainer",
            "dynamics predicts one-step observations directly from observation plus action",
        ],
    }
    _write_json(config.out_dir / "metrics.json", report)
    (config.out_dir / "decision.md").write_text(_format_decision(report), encoding="utf-8")
    return report


@dataclass(frozen=True)
class _Dataset:
    observations: NDArray[np.float64]
    actions: NDArray[np.int64]
    policy_targets: NDArray[np.float64]
    value_targets: NDArray[np.float64]
    next_observations: NDArray[np.float64]


def _load_rows(path: Path) -> _Dataset:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("rows file is empty")
    observations = np.asarray([_float_list(row["observation"]) for row in rows], dtype=np.float64)
    next_observations = np.asarray(
        [_float_list(row["next_observation"]) for row in rows],
        dtype=np.float64,
    )
    policy_targets = np.asarray(
        [_float_list(row["policy_target"]) for row in rows],
        dtype=np.float64,
    )
    actions = np.asarray([int(row["action"]) for row in rows], dtype=np.int64)
    value_targets = np.asarray([float(row["value_target"]) for row in rows], dtype=np.float64)
    _validate_dataset(observations, actions, policy_targets, value_targets, next_observations)
    return _Dataset(
        observations=observations,
        actions=actions,
        policy_targets=policy_targets,
        value_targets=value_targets,
        next_observations=next_observations,
    )


def _validate_dataset(
    observations: NDArray[np.float64],
    actions: NDArray[np.int64],
    policy_targets: NDArray[np.float64],
    value_targets: NDArray[np.float64],
    next_observations: NDArray[np.float64],
) -> None:
    if observations.ndim != 2:
        raise ValueError("observations must be a 2D array")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations must match observations shape")
    if policy_targets.ndim != 2:
        raise ValueError("policy_targets must be a 2D array")
    if len(actions) != observations.shape[0] or len(value_targets) != observations.shape[0]:
        raise ValueError("row arrays have inconsistent lengths")
    if np.any(actions < 0) or np.any(actions >= policy_targets.shape[1]):
        raise ValueError("actions are outside policy target dimension")
    if not all(
        bool(np.all(np.isfinite(array)))
        for array in (observations, next_observations, policy_targets, value_targets)
    ):
        raise ValueError("row arrays contain non-finite values")


def _init_model(dataset: _Dataset, *, seed: int) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    observation_dim = dataset.observations.shape[1]
    action_dim = dataset.policy_targets.shape[1]
    return {
        "policy_weights": rng.normal(0.0, 0.01, size=(observation_dim + 1, action_dim)),
        "value_weights": rng.normal(0.0, 0.01, size=(observation_dim + 1,)),
        "dynamics_weights": rng.normal(
            0.0,
            0.01,
            size=(observation_dim + action_dim + 1, observation_dim),
        ),
    }


def _losses_and_grads(
    dataset: _Dataset,
    model: Mapping[str, NDArray[np.float64]],
) -> tuple[dict[str, float], dict[str, NDArray[np.float64]]]:
    observations = dataset.observations
    batch_size = observations.shape[0]
    action_dim = dataset.policy_targets.shape[1]
    obs_with_bias = _with_bias(observations)

    logits = obs_with_bias @ model["policy_weights"]
    probabilities = _softmax(logits)
    policy_loss = -float(
        np.mean(np.sum(dataset.policy_targets * np.log(probabilities + 1e-12), axis=1))
    )
    policy_grad = obs_with_bias.T @ (probabilities - dataset.policy_targets) / batch_size

    value_predictions = obs_with_bias @ model["value_weights"]
    value_error = value_predictions - dataset.value_targets
    value_loss = float(np.mean(value_error**2))
    value_grad = obs_with_bias.T @ (2.0 * value_error / batch_size)

    action_one_hot = np.zeros((batch_size, action_dim), dtype=np.float64)
    action_one_hot[np.arange(batch_size), dataset.actions] = 1.0
    dynamics_input = _with_bias(np.concatenate([observations, action_one_hot], axis=1))
    next_predictions = dynamics_input @ model["dynamics_weights"]
    dynamics_error = next_predictions - dataset.next_observations
    dynamics_loss = float(np.mean(dynamics_error**2))
    dynamics_grad = (
        dynamics_input.T @ dynamics_error / float(batch_size * dataset.next_observations.shape[1])
    ) * 2.0

    total_loss = policy_loss + value_loss + dynamics_loss
    return (
        {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "dynamics_loss": dynamics_loss,
            "total_loss": total_loss,
        },
        {
            "policy_weights": policy_grad,
            "value_weights": value_grad,
            "dynamics_weights": dynamics_grad,
        },
    )


def _with_bias(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([values, np.ones((values.shape[0], 1), dtype=np.float64)], axis=1)


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("expected JSON list")
    return [float(item) for item in value]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_decision(report: Mapping[str, Any]) -> str:
    train = dict(dict(report["metrics"])["train_smoke"])
    return "\n".join(
        [
            "# Strategic MuZero Train Smoke",
            "",
            f"Status: {report['status']}",
            "",
            "Evidence:",
            f"- Rows: {train['rows']}",
            f"- Epochs: {train['epochs']}",
            f"- Losses finite: {train['losses_finite']}",
            f"- Checkpoint exists: {train['checkpoint_exists']}",
            f"- Policy loss final: {train['policy_loss_final']:.6f}",
            f"- Value loss final: {train['value_loss_final']:.6f}",
            f"- Dynamics loss final: {train['dynamics_loss_final']:.6f}",
            f"- Checkpoint: {train['checkpoint_path']}",
            "",
            "Limits:",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/strategic_lane/muzero_train_smoke"),
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-name", default="train_smoke.npz")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_strategic_muzero_train_smoke(
        StrategicMuZeroTrainSmokeConfig(
            rows_path=args.rows,
            out_dir=args.out_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            checkpoint_name=args.checkpoint_name,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
