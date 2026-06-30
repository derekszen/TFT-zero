"""Search-trace distillation helpers for Stage 3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import NUM_ACTIONS
from mini_tft.core.config import EnvConfig
from mini_tft.core.featurize import featurize_state, observation_dim
from mini_tft.core.lobby import Set1LobbyState
from mini_tft.core.lobby_step import LobbyPolicy, PlayerOrderMode
from mini_tft.core.set_data import GameData, load_set
from mini_tft.rl.lobby_env import MiniTFTLobbyEnv
from mini_tft.search.mcts import MCTSConfig, RootMCTSPlanner


@dataclass(frozen=True)
class SearchDataset:
    """In-memory Stage 3 supervised search-target dataset."""

    obs: NDArray[np.float32]
    masks: NDArray[np.bool_]
    policy_targets: NDArray[np.float32]
    value_targets: NDArray[np.float32]
    selected_actions: NDArray[np.int64]
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


def generate_teacher_dataset(
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
    """Run PPO-guided MCTS and save supervised policy/value rows."""

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
            decision = _planner.plan(
                state=state,
                mask=mask,
                data=game_data,
                env_config=config,
                rng=rng,
            )
            _episode_rows.append(
                {
                    "obs": featurize_state(state.players[0], game_data, config),
                    "mask": mask.copy(),
                    "policy_target": policy_target_from_trace(decision.trace),
                    "selected_action": decision.selected_action,
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

    dataset = rows_to_dataset(rows, suite_names=(suite_name,))
    save_search_dataset(out_path, dataset)
    metrics = dataset_metrics(dataset)
    metrics.update(
        {
            "status": "pass" if dataset.size > 0 else "fail",
            "kind": "stage3_teacher_dataset",
            "suite": suite_name,
            "episodes": episodes,
            "seed": seed,
            "elapsed_sec": perf_counter() - started,
            "dataset": str(out_path),
        }
    )
    write_json(out_path.with_suffix(".metrics.json"), metrics)
    return metrics


def policy_target_from_trace(trace: dict[str, Any]) -> NDArray[np.float32]:
    """Convert MCTS action visit counts into a dense policy target."""

    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    total_visits = 0
    for item in trace.get("action_stats", []):
        action = int(item["action"])
        visits = int(item["visits"])
        if 0 <= action < NUM_ACTIONS and visits > 0:
            target[action] += float(visits)
            total_visits += visits
    if total_visits <= 0:
        selected = int(trace["selected_action"])
        target[selected] = 1.0
        return target
    target /= float(total_visits)
    return target


def rows_to_dataset(
    rows: Sequence[dict[str, Any]],
    *,
    suite_names: tuple[str, ...],
) -> SearchDataset:
    if not rows:
        return empty_dataset(suite_names=suite_names)
    return SearchDataset(
        obs=np.stack([row["obs"] for row in rows]).astype(np.float32),
        masks=np.stack([row["mask"] for row in rows]).astype(np.bool_),
        policy_targets=np.stack([row["policy_target"] for row in rows]).astype(np.float32),
        value_targets=np.asarray([row["value_target"] for row in rows], dtype=np.float32),
        selected_actions=np.asarray([row["selected_action"] for row in rows], dtype=np.int64),
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


def empty_dataset(*, suite_names: tuple[str, ...]) -> SearchDataset:
    return SearchDataset(
        obs=np.zeros((0, 0), dtype=np.float32),
        masks=np.zeros((0, NUM_ACTIONS), dtype=np.bool_),
        policy_targets=np.zeros((0, NUM_ACTIONS), dtype=np.float32),
        value_targets=np.zeros((0,), dtype=np.float32),
        selected_actions=np.zeros((0,), dtype=np.int64),
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


def save_search_dataset(path: Path, dataset: SearchDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=dataset.obs,
        masks=dataset.masks,
        policy_targets=dataset.policy_targets,
        value_targets=dataset.value_targets,
        selected_actions=dataset.selected_actions,
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


def load_search_dataset(path: Path) -> SearchDataset:
    with np.load(path, allow_pickle=False) as payload:
        return SearchDataset(
            obs=payload["obs"].astype(np.float32),
            masks=payload["masks"].astype(np.bool_),
            policy_targets=payload["policy_targets"].astype(np.float32),
            value_targets=payload["value_targets"].astype(np.float32),
            selected_actions=payload["selected_actions"].astype(np.int64),
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


def merge_search_datasets(paths: Sequence[Path]) -> SearchDataset:
    datasets = [load_search_dataset(path) for path in paths]
    if not datasets:
        raise ValueError("at least one dataset path is required")
    suite_names: list[str] = []
    for dataset in datasets:
        for name in dataset.suite_names:
            if name not in suite_names:
                suite_names.append(name)
    return SearchDataset(
        obs=np.concatenate([dataset.obs for dataset in datasets], axis=0),
        masks=np.concatenate([dataset.masks for dataset in datasets], axis=0),
        policy_targets=np.concatenate(
            [dataset.policy_targets for dataset in datasets],
            axis=0,
        ),
        value_targets=np.concatenate([dataset.value_targets for dataset in datasets], axis=0),
        selected_actions=np.concatenate(
            [dataset.selected_actions for dataset in datasets],
            axis=0,
        ),
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


def dataset_metrics(dataset: SearchDataset) -> dict[str, Any]:
    if dataset.size == 0:
        return {"rows": 0}
    target_sums = dataset.policy_targets.sum(axis=1)
    legal_mass = (dataset.policy_targets * dataset.masks.astype(np.float32)).sum(axis=1)
    return {
        "rows": dataset.size,
        "obs_dim": int(dataset.obs.shape[1]),
        "action_dim": int(dataset.policy_targets.shape[1]),
        "mean_policy_target_sum": float(np.mean(target_sums)),
        "min_policy_target_sum": float(np.min(target_sums)),
        "mean_legal_policy_mass": float(np.mean(legal_mass)),
        "mean_value_target": float(np.mean(dataset.value_targets)),
        "mean_placement": float(np.mean(dataset.placements)),
        "mean_final_hp": float(np.mean(dataset.final_hp)),
    }


def train_student(
    *,
    dataset_paths: Sequence[Path],
    out_path: Path,
    epochs: int = 5,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    value_coef: float = 0.5,
    selected_action_coef: float = 1.0,
    hidden_sizes: Sequence[int] = (128, 128),
    seed: int = 0,
    device: str = "cpu",
    resume: Path | None = None,
) -> dict[str, Any]:
    """Train or resume a masked policy/value student."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    dataset = merge_search_datasets(dataset_paths)
    if dataset.size == 0:
        raise ValueError("cannot train on an empty dataset")

    torch.manual_seed(seed)
    actual_device = torch.device(device)
    model = PolicyValueNet(
        input_dim=int(dataset.obs.shape[1]),
        action_dim=int(dataset.policy_targets.shape[1]),
        hidden_sizes=tuple(hidden_sizes),
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
    policy_all = torch.as_tensor(
        dataset.policy_targets,
        dtype=torch.float32,
        device=actual_device,
    )
    value_all = torch.as_tensor(dataset.value_targets, dtype=torch.float32, device=actual_device)
    selected_all = torch.as_tensor(
        dataset.selected_actions,
        dtype=torch.long,
        device=actual_device,
    )

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
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
            policy_targets = policy_all[batch_indices]
            value_targets = value_all[batch_indices]
            selected = selected_all[batch_indices]

            logits, values = model(obs)
            masked_logits = logits.masked_fill(~masks, -1.0e9)
            log_probs = torch.log_softmax(masked_logits, dim=1)
            policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
            selected_loss = nn.functional.cross_entropy(masked_logits, selected)
            value_loss = nn.functional.mse_loss(values.squeeze(-1), value_targets)
            loss = policy_loss + selected_action_coef * selected_loss + value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size_actual = int(obs.shape[0])
            epoch_loss += float(loss.item()) * batch_size_actual
            epoch_policy_loss += float(policy_loss.item()) * batch_size_actual
            epoch_value_loss += float(value_loss.item()) * batch_size_actual
            predicted = masked_logits.argmax(dim=1)
            correct += int((predicted == selected).sum().item())
            seen += batch_size_actual

        losses.append(
            {
                "epoch": float(epoch),
                "loss": epoch_loss / max(1, seen),
                "policy_loss": epoch_policy_loss / max(1, seen),
                "value_loss": epoch_value_loss / max(1, seen),
                "selected_action_acc": correct / max(1, seen),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "input_dim": int(dataset.obs.shape[1]),
        "action_dim": int(dataset.policy_targets.shape[1]),
        "hidden_sizes": tuple(int(size) for size in hidden_sizes),
        "dataset_paths": [str(path) for path in dataset_paths],
        "dataset_metrics": dataset_metrics(dataset),
        "losses": losses,
        "seed": seed,
    }
    torch.save(payload, out_path)
    report = {
        "status": "pass",
        "kind": "stage3_student_train",
        "checkpoint": str(out_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "value_coef": value_coef,
        "selected_action_coef": selected_action_coef,
        "device": str(actual_device),
        "elapsed_sec": perf_counter() - started,
        "dataset_metrics": payload["dataset_metrics"],
        "losses": losses,
        "final_loss": losses[-1]["loss"] if losses else None,
        "final_selected_action_acc": (
            losses[-1]["selected_action_acc"] if losses else None
        ),
    }
    write_json(out_path.with_suffix(".metrics.json"), report)
    return report


class PolicyValueNetProtocol:
    pass


def load_student_policy(checkpoint_path: Path, *, device: str = "cpu") -> LobbyPolicy:
    """Load a Stage 3 checkpoint as a deterministic legal lobby policy."""

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    actual_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=actual_device)
    model = PolicyValueNet(
        input_dim=int(checkpoint["input_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_sizes=tuple(int(size) for size in checkpoint["hidden_sizes"]),
    ).to(actual_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    def policy(
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        config: EnvConfig,
        _rng: np.random.Generator,
    ) -> int:
        obs = featurize_state(state.players[player_id], data, config)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=actual_device,
            ).unsqueeze(0)
            logits, _value = model(obs_tensor)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=actual_device).unsqueeze(0)
            masked_logits = logits.masked_fill(~mask_tensor, -1.0e9)
            action = int(masked_logits.argmax(dim=1).item())
        if 0 <= action < len(mask) and bool(mask[action]):
            return action
        legal = np.flatnonzero(mask)
        return int(legal[0]) if len(legal) else 0

    return policy


def make_policy_value_net(
    *,
    input_dim: int,
    action_dim: int = NUM_ACTIONS,
    hidden_sizes: Sequence[int] = (128, 128),
) -> Any:
    return PolicyValueNet(input_dim=input_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)


class PolicyValueNet:  # intentionally torch-light at import time
    """Torch-backed MLP created lazily when training dependencies are installed."""

    def __new__(
        cls,
        *,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
    ) -> Any:
        try:
            from torch import nn
        except ImportError as exc:
            raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

        class _PolicyValueNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[nn.Module] = []
                last_dim = input_dim
                for size in hidden_sizes:
                    layers.append(nn.Linear(last_dim, int(size)))
                    layers.append(nn.ReLU())
                    last_dim = int(size)
                self.body = nn.Sequential(*layers)
                self.policy_head = nn.Linear(last_dim, action_dim)
                self.value_head = nn.Linear(last_dim, 1)

            def forward(self, obs: Any) -> tuple[Any, Any]:
                hidden = self.body(obs)
                return self.policy_head(hidden), self.value_head(hidden)

        return _PolicyValueNet()


def outcome_value(
    *,
    placement: float,
    final_hp: float,
    final_board_strength: float,
    player_count: int,
    env_config: EnvConfig,
) -> float:
    placement_score = (player_count + 1.0 - placement) / max(1.0, float(player_count))
    hp_score = max(0.0, final_hp) / max(1.0, float(env_config.starting_hp))
    board_score = min(1.0, max(0.0, final_board_strength) / 400.0)
    return float(placement_score + 0.30 * hp_score + 0.20 * board_score)


def validate_dataset_schema(dataset: SearchDataset, *, env_config: EnvConfig | None = None) -> None:
    data = load_set((env_config or EnvConfig()).dataset)
    expected_obs_dim = observation_dim(data, env_config or EnvConfig())
    if dataset.size <= 0:
        raise ValueError("dataset must contain at least one row")
    if dataset.obs.shape != (dataset.size, expected_obs_dim):
        raise ValueError(f"unexpected obs shape: {dataset.obs.shape}")
    if dataset.masks.shape != (dataset.size, NUM_ACTIONS):
        raise ValueError(f"unexpected mask shape: {dataset.masks.shape}")
    if dataset.policy_targets.shape != (dataset.size, NUM_ACTIONS):
        raise ValueError(f"unexpected policy target shape: {dataset.policy_targets.shape}")
    target_sums = dataset.policy_targets.sum(axis=1)
    if not np.allclose(target_sums, 1.0, atol=1e-4):
        raise ValueError("policy targets must sum to 1")
    legal_mass = (dataset.policy_targets * dataset.masks.astype(np.float32)).sum(axis=1)
    if not np.allclose(legal_mass, 1.0, atol=1e-4):
        raise ValueError("policy targets must put all mass on legal actions")
    if not np.all(dataset.masks[np.arange(dataset.size), dataset.selected_actions]):
        raise ValueError("selected actions must be legal")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_float(summary: dict[str, int | float | str | None], key: str) -> float:
    value = summary[key]
    if not isinstance(value, int | float):
        raise RuntimeError(f"summary field {key!r} is not numeric: {value!r}")
    return float(value)


def _batches(indices: NDArray[np.int64], batch_size: int) -> list[NDArray[np.int64]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("hidden sizes must be positive integers")
    return sizes
