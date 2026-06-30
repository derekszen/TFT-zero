"""MuZero-lite dynamics/search helpers for Stage 4."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import NUM_ACTIONS, Action, action_name
from mini_tft.core.config import EnvConfig
from mini_tft.core.featurize import featurize_state, observation_dim
from mini_tft.core.lobby import Set1LobbyState, clone_lobby_state, lobby_action_mask
from mini_tft.core.lobby_step import (
    LobbyPolicy,
    PlayerOrderMode,
    apply_lobby_action,
)
from mini_tft.core.set_data import GameData, load_set
from mini_tft.rl.lobby_env import MiniTFTLobbyEnv
from mini_tft.search.distill import (
    outcome_value,
    parse_hidden_sizes,
    policy_target_from_trace,
    write_json,
)
from mini_tft.search.mcts import (
    RAW_PRIMITIVE_LEGACY_ACTION_SURFACE,
    MCTSConfig,
    RootMCTSPlanner,
    _state_value,
)


@dataclass(frozen=True)
class TransitionDataset:
    """Stage 4 one-step simulator transition dataset."""

    obs: NDArray[np.float32]
    masks: NDArray[np.bool_]
    actions: NDArray[np.int64]
    next_obs: NDArray[np.float32]
    next_masks: NDArray[np.bool_]
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]
    policy_targets: NDArray[np.float32]
    value_targets: NDArray[np.float32]
    placements: NDArray[np.float32]
    final_hp: NDArray[np.float32]
    final_board_strength: NDArray[np.float32]
    rounds: NDArray[np.int64]
    levels: NDArray[np.int64]
    gold: NDArray[np.int64]
    hp: NDArray[np.int64]
    seeds: NDArray[np.int64]
    decision_indices: NDArray[np.int64]
    suite_ids: NDArray[np.int64]
    suite_names: tuple[str, ...]

    @property
    def size(self) -> int:
        return int(self.obs.shape[0])


@dataclass(frozen=True)
class ModelSearchConfig:
    """Root model-search configuration for Stage 4."""

    action_surface: str = RAW_PRIMITIVE_LEGACY_ACTION_SURFACE
    simulations: int = 8
    max_root_actions: int = 8
    exploration: float = 1.0
    discount: float = 0.997
    prior_improvement_margin: float = 1.0
    simulator_value_guard: bool = False
    simulator_value_threshold: float = 0.05
    simulator_value_max_actions: int = 12
    simulator_value_illegal_penalty: float = 0.25


@dataclass
class ModelActionStats:
    """Visit/value accumulator for one model-backed root action."""

    action: int
    visits: int = 0
    value_sum: float = 0.0
    reward_sum: float = 0.0
    prior_logit: float = 0.0

    @property
    def mean_value(self) -> float | None:
        if self.visits <= 0:
            return None
        return self.value_sum / self.visits

    @property
    def mean_reward(self) -> float | None:
        if self.visits <= 0:
            return None
        return self.reward_sum / self.visits


@dataclass(frozen=True)
class ModelSearchDecision:
    """Model-backed root search decision plus diagnostics."""

    selected_action: int
    legal_actions: tuple[int, ...]
    considered_actions: tuple[int, ...]
    action_stats: tuple[ModelActionStats, ...]
    simulations_run: int
    elapsed_ms: float
    trace: dict[str, Any]


def generate_transition_dataset(
    *,
    out_path: Path,
    suite_name: str,
    episodes: int,
    seed: int,
    player_count: int,
    max_actions_per_player: int | None,
    env_config: EnvConfig,
    mcts_config: MCTSConfig,
    opponent_policy: LobbyPolicy,
    prior_policy: LobbyPolicy,
    rollout_policy: LobbyPolicy | None = None,
    player_order_mode: PlayerOrderMode = "fixed",
) -> dict[str, Any]:
    """Run PPO-guided MCTS and save one-step dynamics rows."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rollout = rollout_policy or prior_policy
    rows: list[dict[str, Any]] = []
    started = perf_counter()

    for episode in range(episodes):
        episode_seed = seed + episode
        planner = RootMCTSPlanner(
            hero_player_id=0,
            config=mcts_config,
            opponent_policy=opponent_policy,
            rollout_policy=rollout,
            prior_policy=prior_policy,
        )
        planner.set_episode_context(episode=episode, seed=episode_seed)
        episode_rows: list[dict[str, Any]] = []

        def policy(
            player_id: int,
            state: Set1LobbyState,
            mask: NDArray[np.bool_],
            game_data: GameData,
            config: EnvConfig,
            rng: np.random.Generator,
            _planner: RootMCTSPlanner = planner,
            _episode_rows: list[dict[str, Any]] = episode_rows,
            _episode_seed: int = episode_seed,
        ) -> int:
            if player_id != 0:
                return int(opponent_policy(player_id, state, mask, game_data, config, rng))

            obs = featurize_state(state.players[0], game_data, config)
            decision = _planner.plan(
                state=state,
                mask=mask,
                data=game_data,
                env_config=config,
                rng=rng,
            )
            branch_state = clone_lobby_state(state)
            branch_rng = _clone_rng(rng)
            applied = apply_lobby_action(
                branch_state,
                0,
                decision.selected_action,
                game_data,
                config,
                branch_rng,
            )
            reward = _transition_reward(
                before=state,
                after=branch_state,
                data=game_data,
                config=config,
                applied=applied,
            )
            _episode_rows.append(
                {
                    "obs": obs,
                    "mask": mask.copy(),
                    "action": int(decision.selected_action),
                    "next_obs": featurize_state(branch_state.players[0], game_data, config),
                    "next_mask": lobby_action_mask(branch_state, 0, game_data, config),
                    "reward": reward,
                    "done": bool(branch_state.players[0].done or branch_state.done),
                    "policy_target": policy_target_from_trace(decision.trace),
                    "round": int(state.round),
                    "level": int(state.players[0].level),
                    "gold": int(state.players[0].gold),
                    "hp": int(state.players[0].hp),
                    "seed": int(_episode_seed),
                    "decision_index": int(decision.trace["decision_index"]),
                    "suite_id": 0,
                }
            )
            return decision.selected_action

        env = MiniTFTLobbyEnv(
            seed=episode_seed,
            config=env_config,
            player_count=player_count,
            policy=policy,
            max_actions_per_player=max_actions_per_player,
            player_order_mode=player_order_mode,
        )
        _obs, _info = env.reset(seed=episode_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            _obs, _reward, terminated, truncated, _info = env.step(None)
        summary = env.episode_summary(0)
        placement = _summary_float(summary, "placement")
        final_hp = _summary_float(summary, "final_hp")
        final_board_strength = _summary_float(summary, "final_board_strength")
        value_target = outcome_value(
            placement=placement,
            final_hp=final_hp,
            final_board_strength=final_board_strength,
            player_count=player_count,
            env_config=env_config,
        )
        for row in episode_rows:
            row["placement"] = placement
            row["final_hp"] = final_hp
            row["final_board_strength"] = final_board_strength
            row["value_target"] = value_target
        rows.extend(episode_rows)
        env.close()

    dataset = rows_to_transition_dataset(rows, suite_names=(suite_name,))
    save_transition_dataset(out_path, dataset)
    metrics = transition_dataset_metrics(dataset)
    metrics.update(
        {
            "status": "pass" if dataset.size > 0 else "fail",
            "kind": "stage4_transition_dataset",
            "suite": suite_name,
            "episodes": episodes,
            "seed": seed,
            "elapsed_sec": perf_counter() - started,
            "dataset": str(out_path),
        }
    )
    write_json(out_path.with_suffix(".metrics.json"), metrics)
    return metrics


def rows_to_transition_dataset(
    rows: Sequence[dict[str, Any]],
    *,
    suite_names: tuple[str, ...],
) -> TransitionDataset:
    if not rows:
        return empty_transition_dataset(suite_names=suite_names)
    return TransitionDataset(
        obs=np.stack([row["obs"] for row in rows]).astype(np.float32),
        masks=np.stack([row["mask"] for row in rows]).astype(np.bool_),
        actions=np.asarray([row["action"] for row in rows], dtype=np.int64),
        next_obs=np.stack([row["next_obs"] for row in rows]).astype(np.float32),
        next_masks=np.stack([row["next_mask"] for row in rows]).astype(np.bool_),
        rewards=np.asarray([row["reward"] for row in rows], dtype=np.float32),
        dones=np.asarray([row["done"] for row in rows], dtype=np.bool_),
        policy_targets=np.stack([row["policy_target"] for row in rows]).astype(np.float32),
        value_targets=np.asarray([row["value_target"] for row in rows], dtype=np.float32),
        placements=np.asarray([row["placement"] for row in rows], dtype=np.float32),
        final_hp=np.asarray([row["final_hp"] for row in rows], dtype=np.float32),
        final_board_strength=np.asarray(
            [row["final_board_strength"] for row in rows],
            dtype=np.float32,
        ),
        rounds=np.asarray([row["round"] for row in rows], dtype=np.int64),
        levels=np.asarray([row["level"] for row in rows], dtype=np.int64),
        gold=np.asarray([row["gold"] for row in rows], dtype=np.int64),
        hp=np.asarray([row["hp"] for row in rows], dtype=np.int64),
        seeds=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        decision_indices=np.asarray([row["decision_index"] for row in rows], dtype=np.int64),
        suite_ids=np.asarray([row["suite_id"] for row in rows], dtype=np.int64),
        suite_names=suite_names,
    )


def empty_transition_dataset(*, suite_names: tuple[str, ...]) -> TransitionDataset:
    return TransitionDataset(
        obs=np.zeros((0, 0), dtype=np.float32),
        masks=np.zeros((0, NUM_ACTIONS), dtype=np.bool_),
        actions=np.zeros((0,), dtype=np.int64),
        next_obs=np.zeros((0, 0), dtype=np.float32),
        next_masks=np.zeros((0, NUM_ACTIONS), dtype=np.bool_),
        rewards=np.zeros((0,), dtype=np.float32),
        dones=np.zeros((0,), dtype=np.bool_),
        policy_targets=np.zeros((0, NUM_ACTIONS), dtype=np.float32),
        value_targets=np.zeros((0,), dtype=np.float32),
        placements=np.zeros((0,), dtype=np.float32),
        final_hp=np.zeros((0,), dtype=np.float32),
        final_board_strength=np.zeros((0,), dtype=np.float32),
        rounds=np.zeros((0,), dtype=np.int64),
        levels=np.zeros((0,), dtype=np.int64),
        gold=np.zeros((0,), dtype=np.int64),
        hp=np.zeros((0,), dtype=np.int64),
        seeds=np.zeros((0,), dtype=np.int64),
        decision_indices=np.zeros((0,), dtype=np.int64),
        suite_ids=np.zeros((0,), dtype=np.int64),
        suite_names=suite_names,
    )


def save_transition_dataset(path: Path, dataset: TransitionDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=dataset.obs,
        masks=dataset.masks,
        actions=dataset.actions,
        next_obs=dataset.next_obs,
        next_masks=dataset.next_masks,
        rewards=dataset.rewards,
        dones=dataset.dones,
        policy_targets=dataset.policy_targets,
        value_targets=dataset.value_targets,
        placements=dataset.placements,
        final_hp=dataset.final_hp,
        final_board_strength=dataset.final_board_strength,
        rounds=dataset.rounds,
        levels=dataset.levels,
        gold=dataset.gold,
        hp=dataset.hp,
        seeds=dataset.seeds,
        decision_indices=dataset.decision_indices,
        suite_ids=dataset.suite_ids,
        suite_names=np.asarray(dataset.suite_names),
    )


def load_transition_dataset(path: Path) -> TransitionDataset:
    with np.load(path, allow_pickle=False) as payload:
        return TransitionDataset(
            obs=payload["obs"].astype(np.float32),
            masks=payload["masks"].astype(np.bool_),
            actions=payload["actions"].astype(np.int64),
            next_obs=payload["next_obs"].astype(np.float32),
            next_masks=payload["next_masks"].astype(np.bool_),
            rewards=payload["rewards"].astype(np.float32),
            dones=payload["dones"].astype(np.bool_),
            policy_targets=payload["policy_targets"].astype(np.float32),
            value_targets=payload["value_targets"].astype(np.float32),
            placements=payload["placements"].astype(np.float32),
            final_hp=payload["final_hp"].astype(np.float32),
            final_board_strength=payload["final_board_strength"].astype(np.float32),
            rounds=payload["rounds"].astype(np.int64),
            levels=payload["levels"].astype(np.int64),
            gold=payload["gold"].astype(np.int64),
            hp=payload["hp"].astype(np.int64),
            seeds=payload["seeds"].astype(np.int64),
            decision_indices=payload["decision_indices"].astype(np.int64),
            suite_ids=payload["suite_ids"].astype(np.int64),
            suite_names=tuple(str(item) for item in payload["suite_names"].tolist()),
        )


def merge_transition_datasets(paths: Sequence[Path]) -> TransitionDataset:
    datasets = [load_transition_dataset(path) for path in paths]
    if not datasets:
        raise ValueError("at least one dataset path is required")
    suite_names: list[str] = []
    for dataset in datasets:
        for name in dataset.suite_names:
            if name not in suite_names:
                suite_names.append(name)
    return TransitionDataset(
        obs=np.concatenate([dataset.obs for dataset in datasets], axis=0),
        masks=np.concatenate([dataset.masks for dataset in datasets], axis=0),
        actions=np.concatenate([dataset.actions for dataset in datasets], axis=0),
        next_obs=np.concatenate([dataset.next_obs for dataset in datasets], axis=0),
        next_masks=np.concatenate([dataset.next_masks for dataset in datasets], axis=0),
        rewards=np.concatenate([dataset.rewards for dataset in datasets], axis=0),
        dones=np.concatenate([dataset.dones for dataset in datasets], axis=0),
        policy_targets=np.concatenate(
            [dataset.policy_targets for dataset in datasets],
            axis=0,
        ),
        value_targets=np.concatenate([dataset.value_targets for dataset in datasets], axis=0),
        placements=np.concatenate([dataset.placements for dataset in datasets], axis=0),
        final_hp=np.concatenate([dataset.final_hp for dataset in datasets], axis=0),
        final_board_strength=np.concatenate(
            [dataset.final_board_strength for dataset in datasets],
            axis=0,
        ),
        rounds=np.concatenate([dataset.rounds for dataset in datasets], axis=0),
        levels=np.concatenate([dataset.levels for dataset in datasets], axis=0),
        gold=np.concatenate([dataset.gold for dataset in datasets], axis=0),
        hp=np.concatenate([dataset.hp for dataset in datasets], axis=0),
        seeds=np.concatenate([dataset.seeds for dataset in datasets], axis=0),
        decision_indices=np.concatenate(
            [dataset.decision_indices for dataset in datasets],
            axis=0,
        ),
        suite_ids=np.concatenate([dataset.suite_ids for dataset in datasets], axis=0),
        suite_names=tuple(suite_names),
    )


def transition_dataset_metrics(dataset: TransitionDataset) -> dict[str, Any]:
    if dataset.size == 0:
        return {"rows": 0}
    target_sums = dataset.policy_targets.sum(axis=1)
    legal_mass = (dataset.policy_targets * dataset.masks.astype(np.float32)).sum(axis=1)
    persistence_mse = np.mean(np.square(dataset.next_obs - dataset.obs))
    return {
        "rows": dataset.size,
        "obs_dim": int(dataset.obs.shape[1]),
        "action_dim": int(dataset.policy_targets.shape[1]),
        "mean_policy_target_sum": float(np.mean(target_sums)),
        "min_policy_target_sum": float(np.min(target_sums)),
        "mean_legal_policy_mass": float(np.mean(legal_mass)),
        "mean_reward": float(np.mean(dataset.rewards)),
        "mean_value_target": float(np.mean(dataset.value_targets)),
        "mean_placement": float(np.mean(dataset.placements)),
        "mean_final_hp": float(np.mean(dataset.final_hp)),
        "persistence_next_obs_mse": float(persistence_mse),
    }


def validate_transition_dataset_schema(
    dataset: TransitionDataset,
    *,
    env_config: EnvConfig | None = None,
) -> None:
    data = load_set((env_config or EnvConfig()).dataset)
    expected_obs_dim = observation_dim(data, env_config or EnvConfig())
    if dataset.size <= 0:
        raise ValueError("dataset must contain at least one row")
    if dataset.obs.shape != (dataset.size, expected_obs_dim):
        raise ValueError(f"unexpected obs shape: {dataset.obs.shape}")
    if dataset.next_obs.shape != dataset.obs.shape:
        raise ValueError(f"unexpected next obs shape: {dataset.next_obs.shape}")
    if dataset.masks.shape != (dataset.size, NUM_ACTIONS):
        raise ValueError(f"unexpected mask shape: {dataset.masks.shape}")
    if dataset.next_masks.shape != (dataset.size, NUM_ACTIONS):
        raise ValueError(f"unexpected next mask shape: {dataset.next_masks.shape}")
    if dataset.policy_targets.shape != (dataset.size, NUM_ACTIONS):
        raise ValueError(f"unexpected policy target shape: {dataset.policy_targets.shape}")
    if dataset.actions.shape != (dataset.size,):
        raise ValueError(f"unexpected action shape: {dataset.actions.shape}")
    target_sums = dataset.policy_targets.sum(axis=1)
    if not np.allclose(target_sums, 1.0, atol=1e-4):
        raise ValueError("policy targets must sum to 1")
    legal_mass = (dataset.policy_targets * dataset.masks.astype(np.float32)).sum(axis=1)
    if not np.allclose(legal_mass, 1.0, atol=1e-4):
        raise ValueError("policy targets must put all mass on legal actions")
    if not np.all(dataset.masks[np.arange(dataset.size), dataset.actions]):
        raise ValueError("actions must be legal")


def train_dynamics_model(
    *,
    dataset_paths: Sequence[Path],
    out_path: Path,
    epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    hidden_sizes: Sequence[int] = (128, 128),
    latent_dim: int = 64,
    dynamics_coef: float = 1.0,
    reward_coef: float = 1.0,
    value_coef: float = 0.5,
    policy_coef: float = 1.0,
    seed: int = 0,
    device: str = "cpu",
    resume: Path | None = None,
) -> dict[str, Any]:
    """Train or resume the MuZero-lite representation/dynamics/prediction model."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    dataset = merge_transition_datasets(dataset_paths)
    if dataset.size == 0:
        raise ValueError("cannot train on an empty dataset")

    torch.manual_seed(seed)
    actual_device = torch.device(device)
    model = MuZeroLiteNet(
        input_dim=int(dataset.obs.shape[1]),
        action_dim=int(dataset.policy_targets.shape[1]),
        hidden_sizes=tuple(hidden_sizes),
        latent_dim=latent_dim,
    ).to(actual_device)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=actual_device)
        model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset.size, dtype=np.int64)
    losses: list[dict[str, float]] = []
    started = perf_counter()

    obs_all = torch.as_tensor(dataset.obs, dtype=torch.float32, device=actual_device)
    masks_all = torch.as_tensor(dataset.masks, dtype=torch.bool, device=actual_device)
    actions_all = torch.as_tensor(dataset.actions, dtype=torch.long, device=actual_device)
    next_obs_all = torch.as_tensor(dataset.next_obs, dtype=torch.float32, device=actual_device)
    policy_all = torch.as_tensor(
        dataset.policy_targets,
        dtype=torch.float32,
        device=actual_device,
    )
    value_all = torch.as_tensor(dataset.value_targets, dtype=torch.float32, device=actual_device)
    reward_all = torch.as_tensor(dataset.rewards, dtype=torch.float32, device=actual_device)
    persistence_next_obs_mse = float(np.mean(np.square(dataset.next_obs - dataset.obs)))

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "reward_loss": 0.0,
            "next_obs_loss": 0.0,
        }
        seen = 0
        correct = 0
        for batch_indices_np in _batches(indices, batch_size):
            batch_indices = torch.as_tensor(
                batch_indices_np,
                dtype=torch.long,
                device=actual_device,
            )
            obs = obs_all[batch_indices]
            masks = masks_all[batch_indices]
            actions = actions_all[batch_indices]
            next_obs = next_obs_all[batch_indices]
            policy_targets = policy_all[batch_indices]
            value_targets = value_all[batch_indices]
            reward_targets = reward_all[batch_indices]

            latent, logits, values = model.initial(obs)
            next_latent, rewards = model.recurrent(latent, actions)
            decoded_next_obs = obs + model.decode(next_latent)
            masked_logits = logits.masked_fill(~masks, -1.0e9)
            log_probs = torch.log_softmax(masked_logits, dim=1)
            policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(values.squeeze(-1), value_targets)
            reward_loss = nn.functional.mse_loss(rewards.squeeze(-1), reward_targets)
            next_obs_loss = nn.functional.mse_loss(decoded_next_obs, next_obs)
            loss = (
                policy_coef * policy_loss
                + value_coef * value_loss
                + reward_coef * reward_loss
                + dynamics_coef * next_obs_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size_actual = int(obs.shape[0])
            totals["loss"] += float(loss.item()) * batch_size_actual
            totals["policy_loss"] += float(policy_loss.item()) * batch_size_actual
            totals["value_loss"] += float(value_loss.item()) * batch_size_actual
            totals["reward_loss"] += float(reward_loss.item()) * batch_size_actual
            totals["next_obs_loss"] += float(next_obs_loss.item()) * batch_size_actual
            predicted = masked_logits.argmax(dim=1)
            correct += int((predicted == actions).sum().item())
            seen += batch_size_actual

        losses.append(
            {
                "epoch": float(epoch),
                "loss": totals["loss"] / max(1, seen),
                "policy_loss": totals["policy_loss"] / max(1, seen),
                "value_loss": totals["value_loss"] / max(1, seen),
                "reward_loss": totals["reward_loss"] / max(1, seen),
                "next_obs_loss": totals["next_obs_loss"] / max(1, seen),
                "selected_action_acc": correct / max(1, seen),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "input_dim": int(dataset.obs.shape[1]),
        "action_dim": int(dataset.policy_targets.shape[1]),
        "hidden_sizes": tuple(int(size) for size in hidden_sizes),
        "latent_dim": int(latent_dim),
        "dataset_paths": [str(path) for path in dataset_paths],
        "dataset_metrics": transition_dataset_metrics(dataset),
        "losses": losses,
        "seed": seed,
    }
    torch.save(payload, out_path)
    first = losses[0] if losses else {}
    final = losses[-1] if losses else {}
    report = {
        "status": "pass",
        "kind": "stage4_dynamics_train",
        "checkpoint": str(out_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_sizes": [int(size) for size in hidden_sizes],
        "latent_dim": latent_dim,
        "dynamics_coef": dynamics_coef,
        "reward_coef": reward_coef,
        "value_coef": value_coef,
        "policy_coef": policy_coef,
        "device": str(actual_device),
        "elapsed_sec": perf_counter() - started,
        "dataset_metrics": payload["dataset_metrics"],
        "persistence_next_obs_mse": persistence_next_obs_mse,
        "losses": losses,
        "initial_loss": first.get("loss"),
        "final_loss": final.get("loss"),
        "initial_reward_loss": first.get("reward_loss"),
        "final_reward_loss": final.get("reward_loss"),
        "initial_value_loss": first.get("value_loss"),
        "final_value_loss": final.get("value_loss"),
        "final_next_obs_loss": final.get("next_obs_loss"),
        "final_selected_action_acc": final.get("selected_action_acc"),
        "beats_persistence_next_obs": (
            final.get("next_obs_loss", float("inf")) < persistence_next_obs_mse
        ),
    }
    write_json(out_path.with_suffix(".metrics.json"), report)
    return report


def load_dynamics_model(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load a Stage 4 dynamics checkpoint."""

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    actual_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=actual_device)
    model = MuZeroLiteNet(
        input_dim=int(checkpoint["input_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_sizes=tuple(int(size) for size in checkpoint["hidden_sizes"]),
        latent_dim=int(checkpoint["latent_dim"]),
    ).to(actual_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_model_search_policy(
    checkpoint_path: Path,
    *,
    search_config: ModelSearchConfig | None = None,
    device: str = "cpu",
    prior_policy: LobbyPolicy | None = None,
) -> LobbyPolicy:
    """Load a Stage 4 checkpoint as a model-backed root-search policy."""

    planner = ModelBackedRootPlanner(
        checkpoint_path=checkpoint_path,
        config=search_config or ModelSearchConfig(),
        device=device,
        prior_policy=prior_policy,
    )
    return planner.policy


class ModelBackedRootPlanner:
    """Root search using the learned dynamics model instead of cloned simulator branches."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        config: ModelSearchConfig | None = None,
        device: str = "cpu",
        prior_policy: LobbyPolicy | None = None,
    ) -> None:
        self.config = config or ModelSearchConfig()
        if self.config.action_surface != RAW_PRIMITIVE_LEGACY_ACTION_SURFACE:
            raise ValueError(
                "ModelBackedRootPlanner currently supports only raw primitive legacy "
                "actions; candidate-choice search requires candidate-action policy "
                "targets and recurrent-action encoding."
            )
        if self.config.simulations <= 0:
            raise ValueError("model-search simulations must be positive")
        if self.config.max_root_actions <= 0:
            raise ValueError("model-search max_root_actions must be positive")
        self.device = device
        self.model, self.checkpoint = load_dynamics_model(checkpoint_path, device=device)
        self.prior_policy = prior_policy
        self.traces: list[dict[str, Any]] = []
        self._episode: int | None = None
        self._episode_seed: int | None = None
        self._decision_index = 0

    def set_episode_context(self, *, episode: int, seed: int) -> None:
        self._episode = episode
        self._episode_seed = seed
        self._decision_index = 0

    def policy(
        self,
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> int:
        if player_id != 0:
            raise ValueError("ModelBackedRootPlanner only owns player 0")
        decision = self.plan(
            state=state,
            mask=mask,
            data=data,
            env_config=env_config,
            rng=rng,
        )
        self.traces.append(decision.trace)
        self._decision_index += 1
        return decision.selected_action

    def plan(
        self,
        *,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> ModelSearchDecision:
        start = perf_counter()
        legal_actions = tuple(int(action) for action in np.flatnonzero(mask))
        if not legal_actions:
            return self._empty_decision(start)
        prior_action = self._prior_action(state, mask, data, env_config, rng)
        obs = featurize_state(state.players[0], data, env_config)
        latent, logits, root_value = self._initial(obs)
        logits_np = logits.reshape(-1)
        candidates = self._candidate_actions(
            legal_actions=legal_actions,
            logits=logits_np,
            prior_action=prior_action,
        )
        stats = {
            action: ModelActionStats(action=action, prior_logit=float(logits_np[action]))
            for action in candidates
        }
        simulations_run = 0

        for simulation_index in range(self.config.simulations):
            action = self._select_action(stats, simulations_run)
            value, reward = self._evaluate_action(latent, action)
            item = stats[action]
            item.visits += 1
            item.value_sum += value
            item.reward_sum += reward
            simulations_run = simulation_index + 1

        selected_action = self._select_final_action(stats, prior_action)
        guard_trace: dict[str, Any] | None = None
        if self.config.simulator_value_guard and prior_action is not None:
            selected_action, guard_trace = self._simulator_value_guard(
                state=state,
                mask=mask,
                data=data,
                env_config=env_config,
                rng=rng,
                prior_action=prior_action,
                selected_action=selected_action,
                considered_actions=candidates,
            )
        elapsed_ms = (perf_counter() - start) * 1000.0
        action_stats_tuple = tuple(stats[action] for action in candidates)
        trace = self._trace_row(
            state=state,
            legal_actions=legal_actions,
            considered_actions=candidates,
            selected_action=selected_action,
            action_stats=action_stats_tuple,
            prior_action=prior_action,
            root_value=float(root_value),
            guard_trace=guard_trace,
            simulations_run=simulations_run,
            elapsed_ms=elapsed_ms,
        )
        return ModelSearchDecision(
            selected_action=selected_action,
            legal_actions=legal_actions,
            considered_actions=candidates,
            action_stats=action_stats_tuple,
            simulations_run=simulations_run,
            elapsed_ms=elapsed_ms,
            trace=trace,
        )

    def _empty_decision(self, start: float) -> ModelSearchDecision:
        elapsed_ms = (perf_counter() - start) * 1000.0
        trace = {
            "kind": "model_search_decision",
            "action_surface": self.config.action_surface,
            "episode": self._episode,
            "seed": self._episode_seed,
            "decision_index": self._decision_index,
            "selected_action": int(Action.END_TURN),
            "selected_action_name": action_name(int(Action.END_TURN)),
            "legal_actions": [],
            "considered_actions": [],
            "action_stats": [],
            "simulations_run": 0,
            "elapsed_ms": elapsed_ms,
        }
        return ModelSearchDecision(
            selected_action=int(Action.END_TURN),
            legal_actions=(),
            considered_actions=(),
            action_stats=(),
            simulations_run=0,
            elapsed_ms=elapsed_ms,
            trace=trace,
        )

    def _initial(self, obs: NDArray[np.float32]) -> tuple[Any, NDArray[np.float32], float]:
        import torch

        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=torch.device(self.device),
            ).unsqueeze(0)
            latent, logits, value = self.model.initial(obs_tensor)
            return (
                latent,
                logits.detach().cpu().numpy().astype(np.float32)[0],
                float(value.squeeze(-1).item()),
            )

    def _evaluate_action(self, latent: Any, action: int) -> tuple[float, float]:
        import torch

        with torch.no_grad():
            action_tensor = torch.as_tensor([action], dtype=torch.long, device=latent.device)
            next_latent, reward = self.model.recurrent(latent, action_tensor)
            _logits, value = self.model.predict(next_latent)
            reward_value = float(reward.squeeze(-1).item())
            value_value = float(value.squeeze(-1).item())
            return reward_value + self.config.discount * value_value, reward_value

    def _prior_action(
        self,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> int | None:
        if self.prior_policy is None:
            return None
        action = int(self.prior_policy(0, state, mask, data, env_config, _clone_rng(rng)))
        if 0 <= action < len(mask) and bool(mask[action]):
            return action
        return None

    def _candidate_actions(
        self,
        *,
        legal_actions: tuple[int, ...],
        logits: NDArray[np.float32],
        prior_action: int | None,
    ) -> tuple[int, ...]:
        ranked = sorted(
            [action for action in legal_actions if action != prior_action],
            key=lambda action: float(logits[action]),
            reverse=True,
        )
        if prior_action is not None:
            return (prior_action, *ranked[: max(0, self.config.max_root_actions - 1)])
        return tuple(ranked[: min(self.config.max_root_actions, len(ranked))])

    def _select_action(
        self,
        stats: dict[int, ModelActionStats],
        total_visits: int,
    ) -> int:
        for action, item in stats.items():
            if item.visits == 0:
                return action
        log_total = np.log(max(2, total_visits + 1))
        return max(
            stats.values(),
            key=lambda item: (
                (item.mean_value or 0.0)
                + self.config.exploration * np.sqrt(log_total / item.visits),
                item.prior_logit,
                -item.action,
            ),
        ).action

    def _select_final_action(
        self,
        stats: dict[int, ModelActionStats],
        prior_action: int | None,
    ) -> int:
        best = max(
            stats.values(),
            key=lambda item: (
                item.mean_value if item.mean_value is not None else float("-inf"),
                item.visits,
                item.prior_logit,
                -item.action,
            ),
        )
        if prior_action is None or prior_action not in stats:
            return best.action
        prior = stats[prior_action]
        if prior.mean_value is None or best.mean_value is None:
            return prior_action
        if best.action == prior_action:
            return prior_action
        if best.mean_value < prior.mean_value + self.config.prior_improvement_margin:
            return prior_action
        return best.action

    def _simulator_value_guard(
        self,
        *,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
        prior_action: int,
        selected_action: int,
        considered_actions: tuple[int, ...],
    ) -> tuple[int, dict[str, Any]]:
        actions = self._guard_candidate_actions(
            mask=mask,
            prior_action=prior_action,
            selected_action=selected_action,
            considered_actions=considered_actions,
        )
        values = {
            action: self._simulator_branch_value(
                state=state,
                action=action,
                data=data,
                env_config=env_config,
                rng=rng,
            )
            for action in actions
        }
        prior_value = values[prior_action]
        best_action = max(actions, key=lambda action: (values[action], -action))
        best_value = values[best_action]
        threshold = self.config.simulator_value_threshold
        guarded_action = best_action if best_value >= prior_value + threshold else prior_action
        return guarded_action, {
            "enabled": True,
            "threshold": threshold,
            "max_actions": self.config.simulator_value_max_actions,
            "prior_action": prior_action,
            "prior_value": prior_value,
            "model_selected_action": selected_action,
            "best_action": best_action,
            "best_value": best_value,
            "selected_action": guarded_action,
            "selected_from_prior": guarded_action == prior_action,
            "action_values": [
                {
                    "action": action,
                    "action_name": action_name(action),
                    "value": values[action],
                }
                for action in actions
            ],
        }

    def _guard_candidate_actions(
        self,
        *,
        mask: NDArray[np.bool_],
        prior_action: int,
        selected_action: int,
        considered_actions: tuple[int, ...],
    ) -> tuple[int, ...]:
        priority = (
            prior_action,
            selected_action,
            *considered_actions,
            int(Action.FIELD_BEST_BOARD),
            int(Action.SLAM_BEST_ITEM),
            int(Action.BUY_XP),
            int(Action.ROLL),
            int(Action.END_TURN),
            int(Action.BUY_SHOP_0),
            int(Action.BUY_SHOP_1),
            int(Action.BUY_SHOP_2),
            int(Action.BUY_SHOP_3),
            int(Action.BUY_SHOP_4),
        )
        actions: list[int] = []
        for action in priority:
            if (
                0 <= int(action) < len(mask)
                and bool(mask[int(action)])
                and int(action) not in actions
            ):
                actions.append(int(action))
            if len(actions) >= self.config.simulator_value_max_actions:
                break
        return tuple(actions)

    def _simulator_branch_value(
        self,
        *,
        state: Set1LobbyState,
        action: int,
        data: GameData,
        env_config: EnvConfig,
        rng: np.random.Generator,
    ) -> float:
        branch_state = clone_lobby_state(state)
        branch_rng = _clone_rng(rng)
        applied = apply_lobby_action(branch_state, 0, action, data, env_config, branch_rng)
        value = _state_value(branch_state, 0, data, env_config)
        if not applied:
            value -= self.config.simulator_value_illegal_penalty
        return float(value)

    def _trace_row(
        self,
        *,
        state: Set1LobbyState,
        legal_actions: tuple[int, ...],
        considered_actions: tuple[int, ...],
        selected_action: int,
        action_stats: tuple[ModelActionStats, ...],
        prior_action: int | None,
        root_value: float,
        guard_trace: dict[str, Any] | None,
        simulations_run: int,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        player = state.players[0]
        return {
            "kind": "model_search_decision",
            "action_surface": self.config.action_surface,
            "episode": self._episode,
            "seed": self._episode_seed,
            "decision_index": self._decision_index,
            "round": state.round,
            "level": player.level,
            "gold": player.gold,
            "hp": player.hp,
            "selected_action": int(selected_action),
            "selected_action_name": action_name(selected_action),
            "prior_action": prior_action,
            "prior_action_name": action_name(prior_action) if prior_action is not None else None,
            "selected_from_prior": selected_action == prior_action,
            "simulator_value_guard": guard_trace,
            "root_value": root_value,
            "legal_actions": list(legal_actions),
            "considered_actions": list(considered_actions),
            "action_stats": [
                {
                    "action": item.action,
                    "action_name": action_name(item.action),
                    "visits": item.visits,
                    "mean_value": item.mean_value,
                    "mean_reward": item.mean_reward,
                    "prior_logit": item.prior_logit,
                }
                for item in action_stats
            ],
            "simulations_requested": self.config.simulations,
            "simulations_run": simulations_run,
            "elapsed_ms": elapsed_ms,
        }


def make_muzero_lite_net(
    *,
    input_dim: int,
    action_dim: int = NUM_ACTIONS,
    hidden_sizes: Sequence[int] = (128, 128),
    latent_dim: int = 64,
) -> Any:
    return MuZeroLiteNet(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        latent_dim=latent_dim,
    )


class MuZeroLiteNet:
    """Torch-backed MuZero-lite network created lazily."""

    def __new__(
        cls,
        *,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        latent_dim: int,
    ) -> Any:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

        class _MuZeroLiteNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.action_dim = action_dim
                self.representation = _mlp(input_dim, hidden_sizes, latent_dim)
                self.dynamics_body = _mlp(latent_dim + action_dim, hidden_sizes, latent_dim)
                self.reward_head = nn.Linear(latent_dim, 1)
                self.policy_head = nn.Linear(latent_dim, action_dim)
                self.value_head = nn.Linear(latent_dim, 1)
                self.decoder = _mlp(latent_dim, hidden_sizes, input_dim)
                decoder_last = self.decoder[-1]
                if isinstance(decoder_last, nn.Linear):
                    nn.init.zeros_(decoder_last.weight)
                    nn.init.zeros_(decoder_last.bias)

            def initial(self, obs: Any) -> tuple[Any, Any, Any]:
                latent = torch.tanh(self.representation(obs))
                logits, value = self.predict(latent)
                return latent, logits, value

            def recurrent(self, latent: Any, actions: Any) -> tuple[Any, Any]:
                one_hot = nn.functional.one_hot(actions, num_classes=self.action_dim).to(
                    dtype=latent.dtype
                )
                next_latent = torch.tanh(self.dynamics_body(torch.cat([latent, one_hot], dim=1)))
                reward = self.reward_head(next_latent)
                return next_latent, reward

            def predict(self, latent: Any) -> tuple[Any, Any]:
                return self.policy_head(latent), self.value_head(latent)

            def decode(self, latent: Any) -> Any:
                return self.decoder(latent)

        return _MuZeroLiteNet()


def parse_stage4_hidden_sizes(value: str) -> tuple[int, ...]:
    return parse_hidden_sizes(value)


def _mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> Any:
    from torch import nn

    layers: list[Any] = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(size)))
        layers.append(nn.ReLU())
        last_dim = int(size)
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def _transition_reward(
    *,
    before: Set1LobbyState,
    after: Set1LobbyState,
    data: GameData,
    config: EnvConfig,
    applied: bool,
) -> float:
    value_delta = _state_value(after, 0, data, config) - _state_value(before, 0, data, config)
    if applied:
        return float(value_delta)
    return float(value_delta - 1.0)


def _summary_float(summary: dict[str, int | float | str | None], key: str) -> float:
    value = summary[key]
    if not isinstance(value, int | float):
        raise TypeError(f"expected numeric summary value for {key}")
    return float(value)


def _clone_rng(rng: np.random.Generator) -> np.random.Generator:
    cloned = np.random.default_rng()
    cloned.bit_generator.state = deepcopy(rng.bit_generator.state)
    return cloned


def _batches(indices: NDArray[np.int64], batch_size: int) -> Sequence[NDArray[np.int64]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]
