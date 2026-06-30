"""Lobby policy adapters for trained checkpoint formats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from mini_tft.core.config import EnvConfig
from mini_tft.core.featurize import featurize_state
from mini_tft.core.lobby import Set1LobbyState
from mini_tft.core.lobby_step import LobbyPolicy
from mini_tft.core.set_data import GameData
from mini_tft.rl.train_ppo import parse_hidden_sizes
from mini_tft.rl.train_puffer_ppo import MaskedActorCritic

CheckpointFormat = Literal["auto", "sb3", "puffer"]
ResolvedCheckpointFormat = Literal["sb3", "puffer"]


@dataclass(frozen=True)
class PufferCheckpointPolicy:
    """Deterministic lobby policy backed by a local Puffer PPO checkpoint."""

    checkpoint: Path
    model: MaskedActorCritic
    observation_dim: int
    action_dim: int
    device: torch.device

    def predict(
        self,
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        config: EnvConfig,
    ) -> int:
        legal_mask = _validated_mask(mask, action_dim=self.action_dim)
        obs = np.concatenate(
            [
                featurize_state(state.players[player_id], data, config).astype(np.float32),
                legal_mask.astype(np.float32),
            ]
        )
        if obs.shape != (self.observation_dim,):
            raise ValueError(
                f"Puffer checkpoint expects observation_dim={self.observation_dim}, "
                f"got {obs.shape[0]}"
            )

        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(legal_mask[None, :], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, _value = self.model(obs_tensor)
            masked_logits = logits.masked_fill(~mask_tensor, -1.0e9)
            action = int(torch.argmax(masked_logits, dim=-1).item())
        if bool(legal_mask[action]):
            return action
        return _first_legal_action(legal_mask)


def load_lobby_checkpoint_policy(
    checkpoint: Path,
    *,
    checkpoint_format: CheckpointFormat = "auto",
    device: str = "cpu",
) -> LobbyPolicy:
    """Load an SB3 or Puffer checkpoint as a MiniTFT lobby policy."""

    resolved_format = resolve_checkpoint_format(checkpoint, checkpoint_format)
    if resolved_format == "puffer":
        puffer_policy = load_puffer_checkpoint_policy(checkpoint, device=device)

        def policy(
            player_id: int,
            state: Set1LobbyState,
            mask: NDArray[np.bool_],
            data: GameData,
            config: EnvConfig,
            _rng: np.random.Generator,
        ) -> int:
            return puffer_policy.predict(player_id, state, mask, data, config)

        return policy

    return _sb3_checkpoint_policy(checkpoint, device=device)


def resolve_checkpoint_format(
    checkpoint: Path,
    checkpoint_format: CheckpointFormat,
) -> ResolvedCheckpointFormat:
    if checkpoint_format != "auto":
        return checkpoint_format
    if checkpoint.suffix.lower() in {".pt", ".pth"}:
        return "puffer"
    return "sb3"


def load_puffer_checkpoint_policy(
    checkpoint: Path,
    *,
    device: str = "cpu",
) -> PufferCheckpointPolicy:
    """Load a Puffer PPO checkpoint produced by ``train_puffer_ppo``."""

    resolved_device = torch.device(device)
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint} is not a Puffer PPO checkpoint dictionary")
    if payload.get("kind") != "puffer_ppo":
        raise ValueError(f"{checkpoint} has kind={payload.get('kind')!r}, expected 'puffer_ppo'")

    resolved = payload.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError(f"{checkpoint} is missing resolved checkpoint metadata")
    observation_dim = int(resolved["observation_dim"])
    action_dim = int(resolved["action_dim"])
    hidden_sizes = _hidden_sizes(payload)
    model = MaskedActorCritic(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint} is missing model_state_dict")
    model.load_state_dict(cast(dict[str, Any], state_dict))
    model.to(resolved_device)
    model.eval()
    return PufferCheckpointPolicy(
        checkpoint=checkpoint,
        model=model,
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=resolved_device,
    )


def _sb3_checkpoint_policy(checkpoint: Path, *, device: str) -> LobbyPolicy:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit("Install training dependencies with `uv sync --extra train`.") from exc

    model = MaskablePPO.load(checkpoint, device=device)

    def policy(
        player_id: int,
        state: Set1LobbyState,
        mask: NDArray[np.bool_],
        data: GameData,
        config: EnvConfig,
        _rng: np.random.Generator,
    ) -> int:
        obs = featurize_state(state.players[player_id], data, config)
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=mask,
        )
        return int(action)

    return policy


def _hidden_sizes(payload: dict[Any, Any]) -> list[int]:
    resolved = payload.get("resolved")
    if isinstance(resolved, dict):
        hidden_sizes = resolved.get("hidden_sizes")
        if isinstance(hidden_sizes, list | tuple):
            return [int(size) for size in hidden_sizes]

    args = payload.get("args")
    if isinstance(args, dict):
        hidden_sizes_arg = args.get("hidden_sizes")
        if isinstance(hidden_sizes_arg, str):
            return parse_hidden_sizes(hidden_sizes_arg)

    raise ValueError("Puffer checkpoint is missing hidden_sizes metadata")


def _validated_mask(mask: NDArray[np.bool_], *, action_dim: int) -> NDArray[np.bool_]:
    legal_mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if legal_mask.shape != (action_dim,):
        raise ValueError(f"action mask has shape {legal_mask.shape}, expected ({action_dim},)")
    if np.any(legal_mask):
        return legal_mask
    fallback = legal_mask.copy()
    fallback[0] = True
    return fallback


def _first_legal_action(mask: NDArray[np.bool_]) -> int:
    legal_actions = np.flatnonzero(mask)
    if legal_actions.size == 0:
        return 0
    return int(legal_actions[0])
