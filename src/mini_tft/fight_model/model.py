"""PyTorch FightValueNet and dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import Dataset

from mini_tft.fight_model.schema import MAX_POSITION, MAX_UNITS
from mini_tft.fight_model.storage import load_arrays, read_manifest

BOARD_KEYS = (
    "unit_ids",
    "stars",
    "position_ids",
    "item_ids",
    "chosen_traits",
)


@dataclass(frozen=True)
class FightModelConfig:
    max_unit_id: int
    max_item_id: int
    max_chosen_trait_id: int = 64
    embed_dim: int = 64
    hidden_dim: int = 256


class FightValueNet(nn.Module):
    """Small board-matchup value model."""

    def __init__(self, config: FightModelConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim
        self.unit_embed = nn.Embedding(config.max_unit_id + 1, embed_dim, padding_idx=0)
        self.star_embed = nn.Embedding(4, embed_dim, padding_idx=0)
        self.position_embed = nn.Embedding(MAX_POSITION + 1, embed_dim)
        self.item_embed = nn.Embedding(config.max_item_id + 1, embed_dim, padding_idx=0)
        self.chosen_embed = nn.Embedding(config.max_chosen_trait_id + 1, embed_dim, padding_idx=0)
        self.board_mlp = nn.Sequential(
            nn.Linear(embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.matchup_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 4 + 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.win_head = nn.Linear(config.hidden_dim, 1)
        self.damage_head = nn.Linear(config.hidden_dim, 2)
        self.survivor_head = nn.Linear(config.hidden_dim, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        board_a = self._encode_board(batch, "board_a")
        board_b = self._encode_board(batch, "board_b")
        scalars = torch.stack(
            [
                batch["stage"].float() / 8.0,
                batch["level_a"].float() / MAX_UNITS,
                batch["level_b"].float() / MAX_UNITS,
            ],
            dim=-1,
        )
        matchup = torch.cat(
            [board_a, board_b, board_a - board_b, torch.abs(board_a - board_b), scalars],
            dim=-1,
        )
        hidden = self.matchup_mlp(matchup)
        return {
            "win_logit": self.win_head(hidden).squeeze(-1),
            "damage": torch.nn.functional.softplus(self.damage_head(hidden)),
            "survivor_value": torch.nn.functional.softplus(self.survivor_head(hidden)),
        }

    def _encode_board(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        unit_ids = batch[f"{prefix}_unit_ids"].long().clamp(0, self.config.max_unit_id)
        stars = batch[f"{prefix}_stars"].long().clamp(0, 3)
        positions = batch[f"{prefix}_position_ids"].long().clamp(0, MAX_POSITION)
        item_ids = batch[f"{prefix}_item_ids"].long().clamp(0, self.config.max_item_id)
        chosen = batch[f"{prefix}_chosen_traits"].long().clamp(0, self.config.max_chosen_trait_id)
        token = (
            self.unit_embed(unit_ids)
            + self.star_embed(stars)
            + self.position_embed(positions)
            + self.item_embed(item_ids).sum(dim=2)
            + self.chosen_embed(chosen)
        )
        mask = (unit_ids > 0).float().unsqueeze(-1)
        pooled = (token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.board_mlp(pooled)


class FightArrayDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: dict[str, NDArray[Any]], indices: NDArray[np.int64]) -> None:
        self.arrays = arrays
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        return {name: torch.as_tensor(array[index]) for name, array in self.arrays.items()}


def load_fight_dataset(
    root: Path,
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, NDArray[Any]], NDArray[np.int64], NDArray[np.int64], FightModelConfig]:
    arrays = load_arrays(root)
    count = len(arrays["a_win"])
    rng = np.random.default_rng(seed)
    indices = np.arange(count, dtype=np.int64)
    rng.shuffle(indices)
    val_count = max(1, int(count * val_fraction)) if count > 1 else 1
    val_indices = indices[:val_count]
    train_indices = indices[val_count:] if count > 1 else indices
    max_unit_id = int(
        max(
            arrays["board_a_unit_ids"].max(initial=1),
            arrays["board_b_unit_ids"].max(initial=1),
        )
    )
    max_item_id = int(
        max(
            arrays["board_a_item_ids"].max(initial=1),
            arrays["board_b_item_ids"].max(initial=1),
        )
    )
    return arrays, train_indices, val_indices, FightModelConfig(max_unit_id, max_item_id)


def batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def checkpoint_payload(
    model: FightValueNet,
    manifest_root: Path,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "config": model.config.__dict__,
        "dataset_manifest": read_manifest(manifest_root),
        "metrics": metrics,
    }


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> FightValueNet:
    payload = load_checkpoint_payload(path, map_location=map_location)
    config = FightModelConfig(**payload["config"])
    model = FightValueNet(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def load_checkpoint_payload(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
