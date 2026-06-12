"""Training and evaluation loops for FightValueNet."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from mini_tft.fight_model.model import (
    FightValueNet,
    checkpoint_payload,
    load_fight_dataset,
)


@dataclass(frozen=True)
class TrainReport:
    train_loss: float
    val_loss: float
    val_win_accuracy: float
    val_damage_mae: float
    elapsed_sec: float
    path: str


def train_fight_value_model(
    dataset_root: Path,
    output: Path,
    device_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    seed: int,
) -> TrainReport:
    started = time.perf_counter()
    arrays, train_indices, val_indices, config = load_fight_dataset(
        dataset_root,
        val_fraction,
        seed,
    )
    device = torch.device(device_name)
    tensor_arrays = _tensorize_arrays(arrays, device)
    train_indices_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    val_indices_tensor = torch.as_tensor(val_indices, dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model = FightValueNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()
    train_loss = 0.0
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in _iter_batches(
            tensor_arrays,
            train_indices_tensor,
            batch_size,
            shuffle=True,
            generator=generator,
        ):
            out = model(batch)
            loss = _loss(out, batch, bce, huber)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = int(batch["a_win"].shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count
        train_loss = total_loss / max(1, total_count)

    metrics = evaluate_model(model, tensor_arrays, val_indices_tensor, batch_size, generator)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, dataset_root, metrics), output)
    elapsed = time.perf_counter() - started
    return TrainReport(
        train_loss=train_loss,
        val_loss=metrics["loss"],
        val_win_accuracy=metrics["win_accuracy"],
        val_damage_mae=metrics["damage_mae"],
        elapsed_sec=elapsed,
        path=str(output),
    )


@torch.no_grad()
def evaluate_model(
    model: FightValueNet,
    tensor_arrays: dict[str, torch.Tensor],
    indices: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    huber = nn.SmoothL1Loss(reduction="sum")
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct = 0
    damage_abs = 0.0
    for batch in _iter_batches(
        tensor_arrays,
        indices,
        batch_size,
        shuffle=False,
        generator=generator,
    ):
        out = model(batch)
        loss = _loss(out, batch, bce, huber)
        a_win = batch["a_win"].float()
        pred_win = torch.sigmoid(out["win_logit"]) >= 0.5
        damage_target = torch.stack([batch["damage_to_a"], batch["damage_to_b"]], dim=-1).float()
        correct += int((pred_win == a_win.bool()).sum().detach().cpu())
        damage_abs += float(torch.abs(out["damage"] - damage_target).sum().detach().cpu())
        batch_count = int(a_win.shape[0])
        total_loss += float(loss.detach().cpu())
        total_count += batch_count
    return {
        "loss": total_loss / max(1, total_count),
        "win_accuracy": correct / max(1, total_count),
        "damage_mae": damage_abs / max(1, total_count * 2),
    }


def _tensorize_arrays(
    arrays: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(array, device=device) for name, array in arrays.items()}


def _iter_batches(
    tensor_arrays: dict[str, torch.Tensor],
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None,
) -> Iterator[dict[str, torch.Tensor]]:
    if shuffle:
        order = indices[
            torch.randperm(
                int(indices.shape[0]),
                device=indices.device,
                generator=generator,
            )
        ]
    else:
        order = indices
    for start in range(0, int(order.shape[0]), batch_size):
        batch_indices = order[start : start + batch_size]
        yield {
            name: tensor.index_select(0, batch_indices)
            for name, tensor in tensor_arrays.items()
        }


def _loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    bce: nn.Module,
    huber: nn.Module,
) -> torch.Tensor:
    a_win = batch["a_win"].float()
    damage_target = torch.stack([batch["damage_to_a"], batch["damage_to_b"]], dim=-1).float()
    survivor_target = torch.stack(
        [batch["survivor_value_a"], batch["survivor_value_b"]],
        dim=-1,
    ).float()
    return (
        bce(out["win_logit"], a_win)
        + 0.25 * huber(out["damage"], damage_target)
        + 0.10 * huber(out["survivor_value"], survivor_target)
    )
