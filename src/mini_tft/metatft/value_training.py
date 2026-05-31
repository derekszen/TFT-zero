"""Current-patch MetaTFT board-value training data and model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from mini_tft.metatft.catalog import MetaTFTCatalog
from mini_tft.metatft.encoders import (
    CurrentBoardEncoder,
    derive_stage_line_states,
    final_board_state,
)


@dataclass(frozen=True)
class BoardValueBatch:
    scalars: NDArray[np.float32]
    board_unit_ids: NDArray[np.int64]
    board_item_ids: NDArray[np.int64]
    active_trait_ids: NDArray[np.int64]
    augment_ids: NDArray[np.int64]
    target_comp_ids: NDArray[np.int64]
    targets: NDArray[np.float32]
    weights: NDArray[np.float32]
    comp_ids: tuple[str, ...]
    lines: tuple[str, ...]


@dataclass(frozen=True)
class CurrentPatchRankingReport:
    examples: int
    pairwise_accuracy: float
    spearman: float
    top_k_overlap: float


@dataclass(frozen=True)
class CurrentPatchValueConfig:
    max_unit_id: int
    max_item_id: int
    max_trait_id: int
    max_augment_id: int
    max_comp_id: int
    scalar_dim: int
    embed_dim: int = 32
    hidden_dim: int = 128


@dataclass(frozen=True)
class CurrentPatchValueTrainReport:
    examples: int
    loss: float
    pairwise_accuracy: float
    elapsed_sec: float
    path: str
    train_examples: int = 0
    heldout_examples: int = 0
    heldout_comp_count: int = 0
    heldout_pairwise_accuracy: float | None = None
    heldout_spearman: float | None = None
    heldout_top_k_overlap: float | None = None
    target_metadata_blinded: bool = False


class CurrentPatchValueNet(nn.Module):
    """Small value model over encoded current-patch symbolic board states."""

    def __init__(self, config: CurrentPatchValueConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.unit_embed = nn.Embedding(config.max_unit_id + 1, dim, padding_idx=0)
        self.item_embed = nn.Embedding(config.max_item_id + 1, dim, padding_idx=0)
        self.trait_embed = nn.Embedding(config.max_trait_id + 1, dim, padding_idx=0)
        self.augment_embed = nn.Embedding(config.max_augment_id + 1, dim, padding_idx=0)
        self.comp_embed = nn.Embedding(config.max_comp_id + 1, dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(config.scalar_dim + dim * 5, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        board_units = _masked_mean(self.unit_embed(batch["board_unit_ids"]))
        board_items = _masked_mean(self.item_embed(batch["board_item_ids"]).sum(dim=2))
        traits = _masked_mean(self.trait_embed(batch["active_trait_ids"]))
        augments = _masked_mean(self.augment_embed(batch["augment_ids"]))
        comps = self.comp_embed(batch["target_comp_ids"])
        features = torch.cat(
            [batch["scalars"].float(), board_units, board_items, traits, augments, comps],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)


def build_value_training_batch(
    catalog: MetaTFTCatalog,
    *,
    include_stage_lines: bool = True,
    encoder: CurrentBoardEncoder | None = None,
) -> BoardValueBatch:
    encoder = encoder or CurrentBoardEncoder(catalog)
    rows: list[dict[str, Any]] = []
    for comp in catalog.comps:
        states = [final_board_state(catalog, comp.comp_id)]
        if include_stage_lines:
            states.extend(
                state
                for state in derive_stage_line_states(catalog, comp.comp_id)
                if state.metadata.get("line") != "final"
            )
        for state in states:
            encoding = encoder.encode(state)
            avg = float(state.metadata.get("avg_placement", comp.avg_placement))
            count = int(state.metadata.get("count", comp.count))
            rows.append(
                {
                    "encoding": encoding,
                    "target": -avg,
                    "weight": np.log1p(max(1, count)),
                    "comp_id": comp.comp_id,
                    "line": str(state.metadata.get("line", "final")),
                }
            )
    if not rows:
        raise ValueError("catalog produced no value-training rows")
    return BoardValueBatch(
        scalars=np.stack([row["encoding"].scalars for row in rows]).astype(np.float32),
        board_unit_ids=np.stack([row["encoding"].board_unit_ids for row in rows]).astype(np.int64),
        board_item_ids=np.stack([row["encoding"].board_item_ids for row in rows]).astype(np.int64),
        active_trait_ids=np.stack([row["encoding"].active_trait_ids for row in rows]).astype(
            np.int64
        ),
        augment_ids=np.stack([row["encoding"].augment_ids for row in rows]).astype(np.int64),
        target_comp_ids=np.asarray(
            [row["encoding"].target_comp_id for row in rows],
            dtype=np.int64,
        ),
        targets=np.asarray([row["target"] for row in rows], dtype=np.float32),
        weights=np.asarray([row["weight"] for row in rows], dtype=np.float32),
        comp_ids=tuple(str(row["comp_id"]) for row in rows),
        lines=tuple(str(row["line"]) for row in rows),
    )


def train_current_patch_value_model(
    catalog: MetaTFTCatalog,
    *,
    output: Path,
    device_name: str,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    seed: int = 0,
    include_stage_lines: bool = True,
    embed_dim: int = 32,
    hidden_dim: int = 128,
    validation_fraction: float = 0.0,
    blind_target_metadata: bool = False,
    min_avg_gap: float = 0.05,
    top_k: int = 4,
) -> CurrentPatchValueTrainReport:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    started = time.perf_counter()
    torch.manual_seed(seed)
    encoder = CurrentBoardEncoder(
        catalog,
        include_target_stats=not blind_target_metadata,
        include_target_comp_id=not blind_target_metadata,
    )
    batch = build_value_training_batch(
        catalog,
        include_stage_lines=include_stage_lines,
        encoder=encoder,
    )
    config = CurrentPatchValueConfig(
        max_unit_id=catalog.unit_count,
        max_item_id=catalog.item_count,
        max_trait_id=catalog.tag_count,
        max_augment_id=catalog.augment_count,
        max_comp_id=catalog.comp_count,
        scalar_dim=encoder.scalar_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    device = torch.device(device_name)
    model = CurrentPatchValueNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_mask, heldout_mask, heldout_comp_ids = _split_batch_masks(
        batch,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train_batch = _slice_batch(batch, train_mask)
    tensors = _batch_to_tensors(train_batch, device)
    targets = _standardize(tensors["targets"])
    weights = tensors["weights"] / tensors["weights"].mean().clamp_min(1e-6)
    last_loss = 0.0
    for _ in range(epochs):
        predictions = model(tensors)
        reg_loss = ((predictions - targets) ** 2 * weights).mean()
        pair_loss = _pairwise_loss(predictions, tensors["targets"])
        loss = reg_loss + 0.2 * pair_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    with torch.no_grad():
        predictions = model(tensors).detach().cpu().numpy()
    train_ranking = ranking_report(
        train_batch.targets,
        predictions,
        min_target_gap=min_avg_gap,
        top_k=top_k,
    )
    heldout_ranking: CurrentPatchRankingReport | None = None
    heldout_eval_batch: BoardValueBatch | None = None
    if heldout_mask.any():
        line_labels = np.asarray(batch.lines, dtype=object)
        final_mask = heldout_mask & (line_labels == "final")
        heldout_eval_batch = _slice_batch(batch, final_mask if final_mask.any() else heldout_mask)
        with torch.no_grad():
            heldout_predictions = model(
                _batch_to_tensors(heldout_eval_batch, device),
            ).detach().cpu().numpy()
        heldout_ranking = ranking_report(
            heldout_eval_batch.targets,
            heldout_predictions,
            min_target_gap=min_avg_gap,
            top_k=top_k,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "catalog_metadata": catalog.metadata.__dict__,
            "unit_namespace": catalog.unit_namespace,
            "training_source": "metatft_current_patch_board_encoder",
            "encoder": {
                "include_target_stats": not blind_target_metadata,
                "include_target_comp_id": not blind_target_metadata,
            },
            "metrics": {
                "examples": len(batch.targets),
                "train_examples": len(train_batch.targets),
                "loss": last_loss,
                "pairwise_accuracy": train_ranking.pairwise_accuracy,
                "heldout_examples": heldout_ranking.examples if heldout_ranking else 0,
                "heldout_comp_count": len(heldout_comp_ids),
                "heldout_pairwise_accuracy": (
                    heldout_ranking.pairwise_accuracy if heldout_ranking else None
                ),
                "heldout_spearman": heldout_ranking.spearman if heldout_ranking else None,
                "heldout_top_k_overlap": (
                    heldout_ranking.top_k_overlap if heldout_ranking else None
                ),
                "target_metadata_blinded": blind_target_metadata,
            },
        },
        output,
    )
    return CurrentPatchValueTrainReport(
        examples=len(batch.targets),
        loss=last_loss,
        pairwise_accuracy=train_ranking.pairwise_accuracy,
        elapsed_sec=time.perf_counter() - started,
        path=str(output),
        train_examples=len(train_batch.targets),
        heldout_examples=heldout_ranking.examples if heldout_ranking else 0,
        heldout_comp_count=len(heldout_comp_ids),
        heldout_pairwise_accuracy=heldout_ranking.pairwise_accuracy if heldout_ranking else None,
        heldout_spearman=heldout_ranking.spearman if heldout_ranking else None,
        heldout_top_k_overlap=heldout_ranking.top_k_overlap if heldout_ranking else None,
        target_metadata_blinded=blind_target_metadata,
    )


def ranking_report(
    targets: NDArray[np.float32],
    predictions: NDArray[np.float32],
    *,
    min_target_gap: float = 0.05,
    top_k: int = 4,
) -> CurrentPatchRankingReport:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    return CurrentPatchRankingReport(
        examples=len(targets),
        pairwise_accuracy=_pairwise_accuracy(targets, predictions, min_target_gap),
        spearman=_spearman(targets, predictions),
        top_k_overlap=_top_k_overlap(targets, predictions, top_k),
    )


def _batch_to_tensors(batch: BoardValueBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "scalars": torch.as_tensor(batch.scalars, device=device),
        "board_unit_ids": torch.as_tensor(batch.board_unit_ids, device=device),
        "board_item_ids": torch.as_tensor(batch.board_item_ids, device=device),
        "active_trait_ids": torch.as_tensor(batch.active_trait_ids, device=device),
        "augment_ids": torch.as_tensor(batch.augment_ids, device=device),
        "target_comp_ids": torch.as_tensor(batch.target_comp_ids, device=device),
        "targets": torch.as_tensor(batch.targets, device=device),
        "weights": torch.as_tensor(batch.weights, device=device),
    }


def _slice_batch(batch: BoardValueBatch, mask: NDArray[np.bool_]) -> BoardValueBatch:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError("cannot slice an empty BoardValueBatch")
    return BoardValueBatch(
        scalars=batch.scalars[indices],
        board_unit_ids=batch.board_unit_ids[indices],
        board_item_ids=batch.board_item_ids[indices],
        active_trait_ids=batch.active_trait_ids[indices],
        augment_ids=batch.augment_ids[indices],
        target_comp_ids=batch.target_comp_ids[indices],
        targets=batch.targets[indices],
        weights=batch.weights[indices],
        comp_ids=tuple(batch.comp_ids[index] for index in indices),
        lines=tuple(batch.lines[index] for index in indices),
    )


def _split_batch_masks(
    batch: BoardValueBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], frozenset[str]]:
    comp_ids = sorted(set(batch.comp_ids))
    if validation_fraction <= 0.0 or len(comp_ids) < 2:
        train_mask = np.ones((len(batch.targets),), dtype=bool)
        return train_mask, np.zeros_like(train_mask), frozenset()
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(np.asarray(comp_ids, dtype=object)))
    heldout_count = max(1, int(round(len(shuffled) * validation_fraction)))
    heldout_count = min(heldout_count, len(shuffled) - 1)
    heldout_comp_ids = frozenset(str(comp_id) for comp_id in shuffled[:heldout_count])
    heldout_mask = np.asarray(
        [comp_id in heldout_comp_ids for comp_id in batch.comp_ids],
        dtype=bool,
    )
    return ~heldout_mask, heldout_mask, heldout_comp_ids


def _masked_mean(values: torch.Tensor) -> torch.Tensor:
    mask = values.abs().sum(dim=-1, keepdim=True).gt(0).float()
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std().clamp_min(1e-6)


def _pairwise_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    better = targets[:, None] > targets[None, :]
    if not bool(better.any()):
        return torch.zeros((), device=predictions.device)
    margins = predictions[:, None] - predictions[None, :]
    return torch.nn.functional.softplus(-margins[better]).mean()


def _pairwise_accuracy(
    targets: NDArray[np.float32],
    predictions: NDArray[np.float32],
    min_target_gap: float = 0.0,
) -> float:
    correct = 0
    total = 0
    for left_index, left_target in enumerate(targets):
        for right_index in range(left_index + 1, len(targets)):
            right_target = targets[right_index]
            if abs(float(left_target - right_target)) < max(1e-6, min_target_gap):
                continue
            total += 1
            left_better = left_target > right_target
            left_pred_better = predictions[left_index] > predictions[right_index]
            if left_better == left_pred_better:
                correct += 1
    return correct / total if total else 0.0


def _spearman(targets: NDArray[np.float32], predictions: NDArray[np.float32]) -> float:
    count = len(targets)
    if count < 2:
        return 0.0
    target_order = np.argsort(-targets)
    prediction_order = np.argsort(-predictions)
    target_ranks = np.empty((count,), dtype=np.int64)
    prediction_ranks = np.empty((count,), dtype=np.int64)
    target_ranks[target_order] = np.arange(1, count + 1)
    prediction_ranks[prediction_order] = np.arange(1, count + 1)
    diff_sq = np.square(target_ranks - prediction_ranks).sum()
    return float(1.0 - (6.0 * diff_sq) / (count * (count * count - 1)))


def _top_k_overlap(
    targets: NDArray[np.float32],
    predictions: NDArray[np.float32],
    top_k: int,
) -> float:
    if len(targets) == 0:
        return 0.0
    k = min(max(1, top_k), len(targets))
    target_top = set(np.argsort(-targets)[:k])
    predicted_top = set(np.argsort(-predictions)[:k])
    return len(target_top & predicted_top) / k
