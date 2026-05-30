"""Calibration helpers for ranking learned fight values against real comp data."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mini_tft.fight_model.simulator_adapter import (
    FightValueCombatModel,
    baseline_fight_board,
    metatft_units_to_fight_board,
)


@dataclass(frozen=True)
class CompStrengthRecord:
    cluster_id: str
    name: str
    units: tuple[str, ...]
    avg_placement: float
    top4_rate: float
    win_rate: float
    count: int


@dataclass(frozen=True)
class CalibrationReport:
    records: int
    pairwise_accuracy: float
    spearman: float
    top_k_overlap: float
    min_pairwise_accuracy: float
    min_spearman: float
    min_top_k_overlap: float
    passed: bool


def load_comp_strength_records(path: Path) -> tuple[dict[str, object], list[CompStrengthRecord]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [
        CompStrengthRecord(
            cluster_id=str(row["cluster_id"]),
            name=str(row["name"]),
            units=tuple(str(unit) for unit in row["units"]),
            avg_placement=float(row["avg_placement"]),
            top4_rate=float(row["top4_rate"]),
            win_rate=float(row["win_rate"]),
            count=int(row["count"]),
        )
        for row in payload["records"]
    ]
    return payload["source"], records


def metatft_unit_id_lookup(records: list[CompStrengthRecord]) -> dict[str, int]:
    unit_names = sorted({unit for record in records for unit in record.units})
    return {unit_name: index + 1 for index, unit_name in enumerate(unit_names)}


def score_records_with_fight_value(
    records: list[CompStrengthRecord],
    evaluator: FightValueCombatModel,
) -> dict[str, float]:
    unit_lookup = evaluator.metatft_unit_id_lookup or metatft_unit_id_lookup(records)
    scores = {}
    for record in records:
        if any(unit not in unit_lookup for unit in record.units):
            continue
        board = metatft_units_to_fight_board(record.units, unit_lookup)
        scores[record.cluster_id] = evaluator.score_fight_board(
            board,
            stage=5,
            baseline_board=baseline_fight_board(board.level),
            enemy_strength=100.0,
        )
    return scores


def score_records_by_metric(
    records: list[CompStrengthRecord],
    metric: Callable[[CompStrengthRecord], float],
) -> dict[str, float]:
    return {record.cluster_id: metric(record) for record in records}


def calibration_report(
    records: list[CompStrengthRecord],
    predicted_scores: dict[str, float],
    *,
    min_avg_gap: float = 0.05,
    top_k: int = 4,
    min_pairwise_accuracy: float = 0.65,
    min_spearman: float = 0.30,
    min_top_k_overlap: float = 0.50,
) -> CalibrationReport:
    comparable = [record for record in records if record.cluster_id in predicted_scores]
    pairwise_accuracy = _pairwise_accuracy(comparable, predicted_scores, min_avg_gap)
    spearman = _spearman(comparable, predicted_scores)
    top_overlap = _top_k_overlap(comparable, predicted_scores, top_k)
    passed = (
        pairwise_accuracy >= min_pairwise_accuracy
        and spearman >= min_spearman
        and top_overlap >= min_top_k_overlap
    )
    return CalibrationReport(
        records=len(comparable),
        pairwise_accuracy=pairwise_accuracy,
        spearman=spearman,
        top_k_overlap=top_overlap,
        min_pairwise_accuracy=min_pairwise_accuracy,
        min_spearman=min_spearman,
        min_top_k_overlap=min_top_k_overlap,
        passed=passed,
    )


def _pairwise_accuracy(
    records: list[CompStrengthRecord],
    scores: dict[str, float],
    min_avg_gap: float,
) -> float:
    correct = 0
    total = 0
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            placement_gap = abs(left.avg_placement - right.avg_placement)
            if placement_gap < min_avg_gap:
                continue
            total += 1
            better = left if left.avg_placement < right.avg_placement else right
            worse = right if better is left else left
            if scores[better.cluster_id] > scores[worse.cluster_id]:
                correct += 1
    return correct / total if total else 0.0


def _spearman(records: list[CompStrengthRecord], scores: dict[str, float]) -> float:
    count = len(records)
    if count < 2:
        return 0.0
    target_order = sorted(records, key=lambda record: record.avg_placement)
    predicted_order = sorted(records, key=lambda record: scores[record.cluster_id], reverse=True)
    target_rank = {record.cluster_id: rank for rank, record in enumerate(target_order, start=1)}
    predicted_rank = {
        record.cluster_id: rank for rank, record in enumerate(predicted_order, start=1)
    }
    diff_sq = sum(
        (target_rank[record.cluster_id] - predicted_rank[record.cluster_id]) ** 2
        for record in records
    )
    return 1.0 - (6.0 * diff_sq) / (count * (count * count - 1))


def _top_k_overlap(
    records: list[CompStrengthRecord],
    scores: dict[str, float],
    top_k: int,
) -> float:
    if not records:
        return 0.0
    k = min(top_k, len(records))
    target = {
        record.cluster_id
        for record in sorted(records, key=lambda record: record.avg_placement)[:k]
    }
    predicted = {
        record.cluster_id
        for record in sorted(records, key=lambda record: scores[record.cluster_id], reverse=True)[
            :k
        ]
    }
    return len(target & predicted) / k
