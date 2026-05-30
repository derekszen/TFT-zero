"""Train FightValueNet as a current-patch MetaTFT comp ranking model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mini_tft.fight_model.calibration import (
    CalibrationReport,
    CompStrengthRecord,
    calibration_report,
    metatft_unit_id_lookup,
    score_records_with_fight_value,
)
from mini_tft.fight_model.model import FightModelConfig, FightValueNet
from mini_tft.fight_model.schema import MAX_ITEMS, MAX_POSITION, MAX_UNITS, FightBoard, FightUnit
from mini_tft.fight_model.simulator_adapter import FightValueCombatModel, baseline_fight_board


@dataclass(frozen=True)
class MetaTFTTrainReport:
    train_pairwise_accuracy: float
    eval_pairwise_accuracy: float
    eval_spearman: float
    eval_top_k_overlap: float
    eval_passed: bool
    train_loss: float
    records_train: int
    records_eval: int
    elapsed_sec: float
    path: str


def train_metatft_ranking_model(
    *,
    train_records: list[CompStrengthRecord],
    eval_records: list[CompStrengthRecord],
    output: Path,
    device_name: str,
    epochs: int,
    learning_rate: float,
    seed: int,
    hidden_dim: int = 128,
    embed_dim: int = 64,
    min_avg_gap: float = 0.03,
) -> MetaTFTTrainReport:
    if len(train_records) < 2:
        raise ValueError("at least two train records are required")
    started = time.perf_counter()
    torch.manual_seed(seed)
    device = torch.device(device_name)
    all_records = [*train_records, *eval_records]
    unit_lookup = metatft_unit_id_lookup(all_records)
    config = FightModelConfig(
        max_unit_id=max(unit_lookup.values()),
        max_item_id=1,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    model = FightValueNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_batch = _records_batch(train_records, unit_lookup, device)
    train_targets = _target_scores(train_records, device)
    train_pairs = _pair_indices(train_records, min_avg_gap, device)
    mse = nn.MSELoss()
    train_loss = 0.0

    for _ in range(epochs):
        model.train()
        scores = _score_batch(model, train_batch)
        pair_loss = _pairwise_loss(scores, train_pairs)
        reg_loss = mse(_standardize(scores), train_targets)
        loss = pair_loss + 0.15 * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "metrics": {
                "train_loss": train_loss,
                "train_records": len(train_records),
                "eval_records": len(eval_records),
            },
            "metatft_unit_id_lookup": unit_lookup,
            "training_source": "metatft_current_patch_comp_rankings",
            "unit_namespace": "metatft_current_patch",
        },
        output,
    )

    evaluator = FightValueCombatModel(output, device_name=device_name)
    train_scores = score_records_with_fight_value(train_records, evaluator)
    eval_scores = score_records_with_fight_value(eval_records, evaluator)
    train_report = calibration_report(
        train_records,
        train_scores,
        min_avg_gap=min_avg_gap,
        min_pairwise_accuracy=0.0,
        min_spearman=-1.0,
        min_top_k_overlap=0.0,
    )
    eval_report = calibration_report(eval_records, eval_scores, min_avg_gap=min_avg_gap)
    elapsed = time.perf_counter() - started
    return _train_report(
        train_report,
        eval_report,
        train_loss,
        len(train_records),
        len(eval_records),
        elapsed,
        output,
    )


def split_records_for_eval_exclusion(
    records: list[CompStrengthRecord],
    eval_records: list[CompStrengthRecord],
) -> list[CompStrengthRecord]:
    eval_ids = {record.cluster_id for record in eval_records}
    return [record for record in records if record.cluster_id not in eval_ids]


def _records_batch(
    records: list[CompStrengthRecord],
    unit_lookup: dict[str, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    boards = [_record_board(record, unit_lookup) for record in records]
    baselines = [baseline_fight_board(board.level) for board in boards]
    return _boards_batch(boards, baselines, device)


def _record_board(record: CompStrengthRecord, unit_lookup: dict[str, int]) -> FightBoard:
    units = []
    for position, unit_name in enumerate(record.units[:MAX_UNITS]):
        units.append(
            FightUnit(
                unit_id=unit_lookup[unit_name],
                stars=1,
                position=position,
            )
        )
    return FightBoard(units=tuple(units), level=max(1, min(MAX_UNITS, len(units))))


def _boards_batch(
    boards: list[FightBoard],
    baselines: list[FightBoard],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count = len(boards)
    row: dict[str, Any] = {
        "stage": [5 for _ in range(count)],
        "level_a": [board.level for board in boards],
        "level_b": [board.level for board in baselines],
    }
    for prefix, source_boards in (("board_a", boards), ("board_b", baselines)):
        row[f"{prefix}_unit_ids"] = [[0 for _ in range(MAX_UNITS)] for _ in range(count)]
        row[f"{prefix}_stars"] = [[0 for _ in range(MAX_UNITS)] for _ in range(count)]
        row[f"{prefix}_position_ids"] = [[0 for _ in range(MAX_UNITS)] for _ in range(count)]
        row[f"{prefix}_item_ids"] = [
            [[0 for _ in range(MAX_ITEMS)] for _ in range(MAX_UNITS)]
            for _ in range(count)
        ]
        row[f"{prefix}_chosen_traits"] = [[0 for _ in range(MAX_UNITS)] for _ in range(count)]
        for row_index, board in enumerate(source_boards):
            for unit_index, unit in enumerate(board.units[:MAX_UNITS]):
                row[f"{prefix}_unit_ids"][row_index][unit_index] = unit.unit_id
                row[f"{prefix}_stars"][row_index][unit_index] = unit.stars
                row[f"{prefix}_position_ids"][row_index][unit_index] = min(
                    unit.position,
                    MAX_POSITION,
                )
    return {name: torch.as_tensor(value, device=device) for name, value in row.items()}


def _score_batch(model: FightValueNet, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(batch)
    return (
        out["win_logit"]
        - out["damage"][:, 0] * 0.05
        + out["damage"][:, 1] * 0.05
        + out["survivor_value"][:, 0] * 0.02
    )


def _target_scores(records: list[CompStrengthRecord], device: torch.device) -> torch.Tensor:
    raw = torch.as_tensor(
        [-record.avg_placement for record in records],
        dtype=torch.float32,
        device=device,
    )
    return _standardize(raw)


def _standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std().clamp_min(1e-6)


def _pair_indices(
    records: list[CompStrengthRecord],
    min_avg_gap: float,
    device: torch.device,
) -> torch.Tensor:
    pairs = []
    for left_index, left in enumerate(records):
        for right_index, right in enumerate(records):
            if left_index == right_index:
                continue
            if left.avg_placement + min_avg_gap < right.avg_placement:
                pairs.append((left_index, right_index))
    if not pairs:
        raise ValueError("no train pairs found")
    return torch.as_tensor(pairs, dtype=torch.long, device=device)


def _pairwise_loss(scores: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    better = scores.index_select(0, pairs[:, 0])
    worse = scores.index_select(0, pairs[:, 1])
    return torch.nn.functional.softplus(-(better - worse)).mean()


def _train_report(
    train_report: CalibrationReport,
    eval_report: CalibrationReport,
    train_loss: float,
    records_train: int,
    records_eval: int,
    elapsed_sec: float,
    output: Path,
) -> MetaTFTTrainReport:
    return MetaTFTTrainReport(
        train_pairwise_accuracy=train_report.pairwise_accuracy,
        eval_pairwise_accuracy=eval_report.pairwise_accuracy,
        eval_spearman=eval_report.spearman,
        eval_top_k_overlap=eval_report.top_k_overlap,
        eval_passed=eval_report.passed,
        train_loss=train_loss,
        records_train=records_train,
        records_eval=records_eval,
        elapsed_sec=elapsed_sec,
        path=str(output),
    )
