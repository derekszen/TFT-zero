"""Train and evaluate a Torch policy/value/dynamics model over strategic cache rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from mini_tft.strategic.adapters.analytics import summarize_episode_rows
from mini_tft.strategic.adapters.baselines import random_policy, tft_heuristic_policy
from mini_tft.strategic.core import (
    NUM_ACTIONS,
    StrategicConfig,
    legal_action_mask,
    observe,
    placement_proxy,
    reset,
    scenario_score,
    step,
)
from mini_tft.strategic.core.actions import action_name


@dataclass(frozen=True)
class StrategicMuZeroTorchTrainConfig:
    rows_path: Path
    out_dir: Path
    epochs: int = 8
    learning_rate: float = 1e-3
    batch_size: int = 256
    hidden_size: int = 128
    seed: int = 0
    device: str = "auto"
    checkpoint_name: str = "strategic_muzero_torch.pt"
    value_loss_coef: float = 1.0
    dynamics_loss_coef: float = 0.25


@dataclass(frozen=True)
class StrategicTorchPolicyEvalConfig:
    checkpoint_path: Path
    out_dir: Path
    episodes: int = 32
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class _Dataset:
    observations: NDArray[np.float32]
    legal_masks: NDArray[np.bool_]
    actions: NDArray[np.int64]
    policy_targets: NDArray[np.float32]
    value_targets: NDArray[np.float32]
    next_observations: NDArray[np.float32]


class StrategicMuZeroTorchNet(nn.Module):
    """Small model with policy/value heads and one-step observation dynamics."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.representation = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, observation_dim),
        )

    def forward(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.representation(observations)
        policy_logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        action_one_hot = F.one_hot(
            actions,
            num_classes=self.action_dim,
        ).to(dtype=observations.dtype)
        next_observations = self.dynamics_head(torch.cat((hidden, action_one_hot), dim=-1))
        return policy_logits, values, next_observations

    def policy_logits(self, observations: Tensor) -> Tensor:
        hidden = self.representation(observations)
        return self.policy_head(hidden)


def run_strategic_muzero_torch_train(
    config: StrategicMuZeroTorchTrainConfig,
) -> dict[str, Any]:
    """Train the Torch smoke model and write auditable metrics/checkpoint artifacts."""

    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dataset = _load_rows(config.rows_path)
    device = _resolve_device(config.device)
    tensors = _dataset_tensors(dataset, device=device)
    model = StrategicMuZeroTorchNet(
        observation_dim=dataset.observations.shape[1],
        action_dim=dataset.policy_targets.shape[1],
        hidden_size=config.hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    initial = _losses_and_metrics(model, tensors, config)
    history: list[dict[str, float]] = [initial]
    row_count = int(dataset.observations.shape[0])
    for _ in range(config.epochs):
        order = torch.randperm(row_count, generator=generator)
        for start in range(0, row_count, config.batch_size):
            index = order[start : start + config.batch_size].to(device)
            batch = _slice_tensors(tensors, index)
            losses = _loss_tensors(model, batch, config)
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        history.append(_losses_and_metrics(model, tensors, config))

    final = history[-1]
    checkpoint_path = config.out_dir / config.checkpoint_name
    checkpoint_payload = {
        "schema": "strategic-muzero-torch-checkpoint/v1",
        "model_state_dict": model.state_dict(),
        "metadata": {
            "rows_path": str(config.rows_path),
            "rows": row_count,
            "observation_dim": int(dataset.observations.shape[1]),
            "action_dim": int(dataset.policy_targets.shape[1]),
            "hidden_size": config.hidden_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "device": str(device),
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)

    losses_finite = all(
        isfinite(float(value)) for entry in (initial, final) for value in entry.values()
    )
    checkpoint_exists = checkpoint_path.exists() and checkpoint_path.stat().st_size > 0
    train_metrics = {
        "ran": True,
        "losses_finite": losses_finite,
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_path": str(checkpoint_path),
        "rows": row_count,
        "observation_dim": int(dataset.observations.shape[1]),
        "action_dim": int(dataset.policy_targets.shape[1]),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "hidden_size": config.hidden_size,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "elapsed_sec": perf_counter() - started,
        "policy_loss_initial": initial["policy_loss"],
        "policy_loss_final": final["policy_loss"],
        "value_loss_initial": initial["value_loss"],
        "value_loss_final": final["value_loss"],
        "dynamics_loss_initial": initial["dynamics_loss"],
        "dynamics_loss_final": final["dynamics_loss"],
        "total_loss_initial": initial["total_loss"],
        "total_loss_final": final["total_loss"],
        "policy_target_top1_accuracy": final["policy_target_top1_accuracy"],
        "legal_argmax_rate": final["legal_argmax_rate"],
        "value_mae": final["value_mae"],
        "dynamics_mae": final["dynamics_mae"],
    }
    report = {
        "schema": "strategic-muzero-torch-train/v1",
        "deliverable": "muzero_torch_trainer",
        "status": "smoke_only" if losses_finite and checkpoint_exists else "fail",
        "seed": config.seed,
        "commands": [
            "env -u UV_PYTHON uv run --extra train python -m "
            "mini_tft.tools.train_strategic_muzero_torch"
        ],
        "metrics": {
            "train_torch": train_metrics,
            "train_smoke": train_metrics,
            "loss_history": history,
        },
        "artifacts": ["metrics.json", "decision.md", config.checkpoint_name],
        "known_limits": [
            "this is supervised training on simulator/MCTS cache rows, not full MuZero self-play",
            "dynamics predicts next observation directly from observation plus action",
            "train_smoke is a compatibility alias for the queue-ready verifier",
        ],
    }
    _write_json(config.out_dir / "metrics.json", report)
    (config.out_dir / "decision.md").write_text(_format_train_decision(report), encoding="utf-8")
    return report


def run_torch_checkpoint_policy_evaluation(
    config: StrategicTorchPolicyEvalConfig,
) -> dict[str, Any]:
    if config.episodes <= 0:
        raise ValueError("episodes must be positive")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    trained_policy = load_torch_muzero_policy(config.checkpoint_path, device=config.device)
    policies: tuple[tuple[str, _StrategicPolicy], ...] = (
        ("random", random_policy),
        ("heuristic", tft_heuristic_policy),
        ("torch_muzero", trained_policy),
    )
    strategic_config = StrategicConfig()
    episode_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for policy_name, policy in policies:
        policy_episode_rows: list[dict[str, Any]] = []
        policy_action_rows: list[dict[str, Any]] = []
        for episode in range(config.episodes):
            row, rows = _run_eval_episode(
                policy,
                policy_name=policy_name,
                episode=episode,
                seed=config.seed + episode,
                config=strategic_config,
            )
            policy_episode_rows.append(row)
            policy_action_rows.extend(rows)
        episode_rows.extend(policy_episode_rows)
        action_rows.extend(policy_action_rows)
        summaries[policy_name] = _summarize_policy(policy_episode_rows, policy_action_rows)

    report = {
        "schema": "strategic-muzero-torch-eval/v1",
        "status": "smoke_only"
        if sum(int(summary["illegal_action_count"]) for summary in summaries.values()) == 0
        else "fail",
        "seed": config.seed,
        "episodes_per_policy": config.episodes,
        "checkpoint_path": str(config.checkpoint_path),
        "policies": [name for name, _ in policies],
        "policy_summaries": summaries,
        "comparison": {
            "best_by_mean_scenario_score": max(
                summaries,
                key=lambda name: float(summaries[name]["mean_scenario_score"]),
            ),
            "best_by_mean_placement": min(
                summaries,
                key=lambda name: float(summaries[name]["mean_placement"]),
            ),
        },
        "artifacts": ["metrics.json", "decision.md", "episodes.jsonl", "actions.jsonl"],
        "known_limits": [
            "placement_proxy is an elimination bucket, not a real lobby placement",
            "this evaluates greedy policy-head actions, not model-backed search",
        ],
    }
    _write_json(config.out_dir / "metrics.json", report)
    _write_jsonl(config.out_dir / "episodes.jsonl", episode_rows)
    _write_jsonl(config.out_dir / "actions.jsonl", action_rows)
    (config.out_dir / "decision.md").write_text(_format_eval_decision(report), encoding="utf-8")
    return report


_StrategicPolicy = Callable[[Any, NDArray[np.bool_], StrategicConfig], int]
_StrategicPolicyValue = Callable[
    [Any, NDArray[np.bool_], StrategicConfig],
    tuple[NDArray[np.float32], float],
]
_CHECKPOINT_SCHEMA = "strategic-muzero-torch-checkpoint/v1"
_REQUIRED_CHECKPOINT_METADATA = ("observation_dim", "action_dim", "hidden_size")


def load_torch_muzero_policy_value(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
) -> _StrategicPolicyValue:
    resolved_device = _resolve_device(device)
    metadata, model_state_dict = _load_torch_checkpoint_payload(
        checkpoint_path,
        device=resolved_device,
    )
    observation_dim = _checkpoint_metadata_int(metadata, "observation_dim")
    action_dim = _checkpoint_metadata_int(metadata, "action_dim")
    hidden_size = _checkpoint_metadata_int(metadata, "hidden_size")
    model = StrategicMuZeroTorchNet(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
    ).to(resolved_device)
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint model weights do not match checkpoint metadata: "
            f"{checkpoint_path}"
        ) from exc
    model.eval()
    cache: dict[tuple[bytes, bytes], tuple[NDArray[np.float32], float]] = {}

    def policy_value(
        state: Any,
        mask: NDArray[np.bool_],
        config: StrategicConfig,
    ) -> tuple[NDArray[np.float32], float]:
        obs_array = np.asarray(observe(state, config), dtype=np.float32)
        if obs_array.ndim != 1 or obs_array.shape[0] != model.observation_dim:
            raise ValueError(
                "state observation shape does not match checkpoint metadata: "
                f"expected {(model.observation_dim,)}, got {obs_array.shape}"
            )
        mask_array = np.asarray(mask, dtype=np.bool_)
        if mask_array.ndim != 1 or mask_array.shape[0] != model.action_dim:
            raise ValueError(
                "legal mask shape does not match checkpoint metadata: "
                f"expected {(model.action_dim,)}, got {mask_array.shape}"
            )
        cache_key = (obs_array.tobytes(), mask_array.tobytes())
        cached = cache.get(cache_key)
        if cached is not None:
            priors, value = cached
            return priors.copy(), value
        obs = torch.as_tensor(
            obs_array,
            dtype=torch.float32,
            device=resolved_device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.bool, device=resolved_device)
        with torch.no_grad():
            hidden = model.representation(obs)
            logits = model.policy_head(hidden).squeeze(0)
            value = float(model.value_head(hidden).squeeze().detach().cpu().item())
            if not bool(mask_tensor.any()):
                empty_priors = np.zeros(mask_array.shape, dtype=np.float32)
                cache[cache_key] = (empty_priors.copy(), value)
                return empty_priors, value
            logits = logits.masked_fill(~mask_tensor, -1.0e9)
            priors = torch.softmax(logits, dim=-1)
            priors = priors.masked_fill(~mask_tensor, 0.0)
            total = priors.sum()
            if not bool(torch.isfinite(total)) or float(total.detach().cpu().item()) <= 0.0:
                raise RuntimeError("checkpoint policy produced no legal probability mass")
            priors = priors / total
        priors_array = priors.detach().cpu().numpy().astype(np.float32)
        cache[cache_key] = (priors_array.copy(), value)
        return priors_array, value

    return policy_value


def _load_torch_checkpoint_payload(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ValueError(f"failed to load strategic MuZero checkpoint: {checkpoint_path}") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"invalid strategic MuZero checkpoint payload: {checkpoint_path}")
    if checkpoint.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError(
            "invalid strategic MuZero checkpoint schema: "
            f"{checkpoint.get('schema')!r} in {checkpoint_path}"
        )
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"invalid strategic MuZero checkpoint metadata: {checkpoint_path}")
    missing = [key for key in _REQUIRED_CHECKPOINT_METADATA if key not in metadata]
    if missing:
        raise ValueError(
            "strategic MuZero checkpoint metadata missing required fields: "
            f"{', '.join(missing)}"
        )
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise ValueError(f"invalid strategic MuZero checkpoint model_state_dict: {checkpoint_path}")
    return dict(metadata), model_state_dict


def _checkpoint_metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid strategic MuZero checkpoint metadata field: {key}") from exc
    if value <= 0:
        raise ValueError(f"checkpoint metadata field must be positive: {key}")
    return value


def load_torch_muzero_policy(checkpoint_path: Path, *, device: str = "cpu") -> _StrategicPolicy:
    policy_value = load_torch_muzero_policy_value(checkpoint_path, device=device)

    def policy(state: Any, mask: NDArray[np.bool_], config: StrategicConfig) -> int:
        priors, _ = policy_value(state, mask, config)
        action = int(np.argmax(priors))
        return action

    return policy


def _load_rows(path: Path) -> _Dataset:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("rows file is empty")
    observations = np.asarray([_float_list(row["observation"]) for row in rows], dtype=np.float32)
    next_observations = np.asarray(
        [_float_list(row["next_observation"]) for row in rows],
        dtype=np.float32,
    )
    legal_masks = np.asarray([_bool_list(row["legal_mask"]) for row in rows], dtype=np.bool_)
    policy_targets = np.asarray(
        [_float_list(row["policy_target"]) for row in rows],
        dtype=np.float32,
    )
    actions = np.asarray([int(row["action"]) for row in rows], dtype=np.int64)
    value_targets = np.asarray([float(row["value_target"]) for row in rows], dtype=np.float32)
    _validate_dataset(
        observations,
        legal_masks,
        actions,
        policy_targets,
        value_targets,
        next_observations,
    )
    return _Dataset(
        observations=observations,
        legal_masks=legal_masks,
        actions=actions,
        policy_targets=policy_targets,
        value_targets=value_targets,
        next_observations=next_observations,
    )


def _validate_dataset(
    observations: NDArray[np.float32],
    legal_masks: NDArray[np.bool_],
    actions: NDArray[np.int64],
    policy_targets: NDArray[np.float32],
    value_targets: NDArray[np.float32],
    next_observations: NDArray[np.float32],
) -> None:
    if observations.ndim != 2:
        raise ValueError("observations must be a 2D array")
    if next_observations.shape != observations.shape:
        raise ValueError("next_observations must match observations shape")
    if legal_masks.ndim != 2:
        raise ValueError("legal_masks must be a 2D array")
    if policy_targets.shape != legal_masks.shape:
        raise ValueError("policy_targets must match legal_masks shape")
    row_count = observations.shape[0]
    if legal_masks.shape[0] != row_count or len(actions) != row_count:
        raise ValueError("row arrays have inconsistent lengths")
    if len(value_targets) != row_count:
        raise ValueError("value_targets length does not match observations")
    if np.any(actions < 0) or np.any(actions >= legal_masks.shape[1]):
        raise ValueError("actions are outside action dimension")
    if not bool(np.all(legal_masks[np.arange(row_count), actions])):
        raise ValueError("selected actions must be legal")
    if not all(
        bool(np.all(np.isfinite(array)))
        for array in (observations, next_observations, policy_targets, value_targets)
    ):
        raise ValueError("row arrays contain non-finite values")
    if np.any(policy_targets < 0.0):
        raise ValueError("policy_targets must be non-negative")
    policy_sums = policy_targets.sum(axis=1)
    if not bool(np.allclose(policy_sums, 1.0, atol=1e-5)):
        raise ValueError("policy_targets must sum to 1.0")
    illegal_policy_mass = np.where(legal_masks, 0.0, policy_targets).sum(axis=1)
    if not bool(np.all(illegal_policy_mass <= 1e-5)):
        raise ValueError("policy_targets put mass on illegal actions")


def _dataset_tensors(dataset: _Dataset, *, device: torch.device) -> dict[str, Tensor]:
    return {
        "observations": torch.as_tensor(dataset.observations, dtype=torch.float32, device=device),
        "legal_masks": torch.as_tensor(dataset.legal_masks, dtype=torch.bool, device=device),
        "actions": torch.as_tensor(dataset.actions, dtype=torch.long, device=device),
        "policy_targets": torch.as_tensor(
            dataset.policy_targets,
            dtype=torch.float32,
            device=device,
        ),
        "value_targets": torch.as_tensor(dataset.value_targets, dtype=torch.float32, device=device),
        "next_observations": torch.as_tensor(
            dataset.next_observations,
            dtype=torch.float32,
            device=device,
        ),
    }


def _slice_tensors(tensors: Mapping[str, Tensor], index: Tensor) -> dict[str, Tensor]:
    return {key: value.index_select(0, index) for key, value in tensors.items()}


def _loss_tensors(
    model: StrategicMuZeroTorchNet,
    tensors: Mapping[str, Tensor],
    config: StrategicMuZeroTorchTrainConfig,
) -> dict[str, Tensor]:
    logits, values, next_predictions = model(tensors["observations"], tensors["actions"])
    masked_logits = logits.masked_fill(~tensors["legal_masks"], -1.0e9)
    policy_log_probs = F.log_softmax(masked_logits, dim=-1)
    policy_loss = -(tensors["policy_targets"] * policy_log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, tensors["value_targets"])
    dynamics_loss = F.mse_loss(next_predictions, tensors["next_observations"])
    total_loss = (
        policy_loss
        + config.value_loss_coef * value_loss
        + config.dynamics_loss_coef * dynamics_loss
    )
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "dynamics_loss": dynamics_loss,
        "total_loss": total_loss,
    }


def _losses_and_metrics(
    model: StrategicMuZeroTorchNet,
    tensors: Mapping[str, Tensor],
    config: StrategicMuZeroTorchTrainConfig,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        losses = _loss_tensors(model, tensors, config)
        logits, values, next_predictions = model(tensors["observations"], tensors["actions"])
        masked_logits = logits.masked_fill(~tensors["legal_masks"], -1.0e9)
        predicted_actions = torch.argmax(masked_logits, dim=-1)
        target_actions = torch.argmax(tensors["policy_targets"], dim=-1)
        legal_argmax = tensors["legal_masks"].gather(1, predicted_actions.unsqueeze(1)).squeeze(1)
        output = {
            key: float(value.detach().cpu().item()) for key, value in losses.items()
        }
        output["policy_target_top1_accuracy"] = float(
            (predicted_actions == target_actions).to(dtype=torch.float32).mean().cpu().item()
        )
        output["legal_argmax_rate"] = float(
            legal_argmax.to(dtype=torch.float32).mean().cpu().item()
        )
        output["value_mae"] = float(
            torch.mean(torch.abs(values - tensors["value_targets"])).cpu().item()
        )
        output["dynamics_mae"] = float(
            torch.mean(torch.abs(next_predictions - tensors["next_observations"])).cpu().item()
        )
    model.train()
    return output


def _run_eval_episode(
    policy: _StrategicPolicy,
    *,
    policy_name: str,
    episode: int,
    seed: int,
    config: StrategicConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = reset(seed=seed, config=config)
    total_reward = 0.0
    steps = 0
    action_rows: list[dict[str, Any]] = []
    max_steps = config.max_round * (config.max_actions_per_round + 1)
    while not state.done and steps < max_steps:
        observation = observe(state, config)
        mask = legal_action_mask(state, config)
        action = int(policy(state, mask, config))
        legal = 0 <= action < len(mask) and bool(mask[action])
        result = step(state, action, config)
        total_reward += result.reward
        action_rows.append(
            {
                "policy": policy_name,
                "episode": episode,
                "seed": seed,
                "step": steps,
                "round": _as_int(result.info["round"]),
                "action_id": action,
                "action": action_name(action),
                "legal": legal,
                "legal_action_count": int(mask.sum()),
                "observation_dim": int(observation.shape[0]),
                "reward": float(result.reward),
                "ended_round": _as_bool(result.info["ended_round"]),
                "hp": _as_int(result.info["hp"]),
                "gold": _as_int(result.info["gold"]),
                "level": _as_int(result.info["level"]),
                "placement_proxy": _as_int(result.info["placement_proxy"]),
                "scenario_score": _as_float(result.info["scenario_score"]),
            }
        )
        steps += 1

    episode_row = {
        "policy": policy_name,
        "episode": episode,
        "seed": seed,
        "placement": placement_proxy(state, config),
        "final_round": state.round,
        "hp": state.hp,
        "final_reason": state.final_reason,
        "scenario_score": scenario_score(state, config),
        "illegal_actions": state.total_illegal_actions,
        "total_reward": total_reward,
        "steps": steps,
    }
    return episode_row, action_rows


def _summarize_policy(
    episode_rows: Sequence[dict[str, Any]],
    action_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    action_counts = Counter(str(row["action"]) for row in action_rows)
    total_actions = sum(action_counts.values())
    final_reasons = Counter(str(row["final_reason"]) for row in episode_rows)
    placement_values = [int(row["placement"]) for row in episode_rows]
    score_values = [float(row["scenario_score"]) for row in episode_rows]
    return {
        **summarize_episode_rows(episode_rows),
        "mean_placement": _mean(placement_values),
        "mean_scenario_score": _mean(score_values),
        "illegal_action_count": sum(int(row["illegal_actions"]) for row in episode_rows),
        "total_actions": total_actions,
        "action_counts": {
            action_name(action_id): int(action_counts[action_name(action_id)])
            for action_id in range(NUM_ACTIONS)
        },
        "action_rates": {
            action_name(action_id): _rate(int(action_counts[action_name(action_id)]), total_actions)
            for action_id in range(NUM_ACTIONS)
        },
        "final_reason_counts": {
            reason: int(count) for reason, count in sorted(final_reasons.items())
        },
        "mean_reward": _mean(float(row["total_reward"]) for row in episode_rows),
        "mean_steps": _mean(int(row["steps"]) for row in episode_rows),
    }


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(device)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return [float(item) for item in value]


def _bool_list(value: Any) -> list[bool]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return [bool(item) for item in value]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _as_int(value: Any) -> int:
    if value is None:
        raise ValueError("expected int-compatible value, got None")
    return int(value)


def _as_float(value: Any) -> float:
    if value is None:
        raise ValueError("expected float-compatible value, got None")
    return float(value)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _format_train_decision(report: Mapping[str, Any]) -> str:
    metrics = dict(report["metrics"])
    train = dict(metrics["train_torch"])
    return "\n".join(
        [
            "# Strategic MuZero Torch Train",
            "",
            f"Status: `{report['status']}`",
            "",
            "## Evidence",
            "",
            f"- Rows: {train['rows']}",
            f"- Device: {train['device']}",
            f"- Checkpoint: `{train['checkpoint_path']}`",
            f"- Initial total loss: {train['total_loss_initial']:.6f}",
            f"- Final total loss: {train['total_loss_final']:.6f}",
            f"- Policy target top-1 accuracy: {train['policy_target_top1_accuracy']:.3f}",
            f"- Legal argmax rate: {train['legal_argmax_rate']:.3f}",
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )


def _format_eval_decision(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strategic MuZero Torch Eval",
        "",
        f"Status: `{report['status']}`",
        "",
        (
            "| Policy | Mean placement | Survivor rate | Mean final round | "
            "Mean scenario score | Illegal actions |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    summaries = dict(report["policy_summaries"])
    for policy in report["policies"]:
        summary = dict(summaries[str(policy)])
        lines.append(
            "| "
            f"{policy} | "
            f"{float(summary['mean_placement']):.3f} | "
            f"{float(summary['survivor_rate']):.3f} | "
            f"{float(summary['mean_final_round']):.3f} | "
            f"{float(summary['mean_scenario_score']):.3f} | "
            f"{int(summary['illegal_action_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            *[f"- {limit}" for limit in report["known_limits"]],
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-name", default="strategic_muzero_torch.pt")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_strategic_muzero_torch_train(
        StrategicMuZeroTorchTrainConfig(
            rows_path=args.rows_path,
            out_dir=args.out_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            seed=args.seed,
            device=args.device,
            checkpoint_name=args.checkpoint_name,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
