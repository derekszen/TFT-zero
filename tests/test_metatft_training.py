from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from mini_tft.fight_model.calibration import CompStrengthRecord, score_records_with_fight_value
from mini_tft.fight_model.metatft_training import (
    split_records_for_eval_exclusion,
    train_metatft_ranking_model,
)
from mini_tft.fight_model.simulator_adapter import FightValueCombatModel


def test_split_records_for_eval_exclusion_removes_matching_clusters() -> None:
    train_records = [
        _record("a", "top", ("A", "B"), 3.8),
        _record("b", "middle", ("C", "D"), 4.5),
        _record("c", "bottom", ("E", "F"), 5.2),
    ]
    eval_records = [_record("b", "middle", ("C", "D"), 4.5)]

    split = split_records_for_eval_exclusion(train_records, eval_records)

    assert [record.cluster_id for record in split] == ["a", "c"]


def test_metatft_ranking_training_writes_current_patch_lookup_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "metatft_current.pt"
    train_records = [
        _record("top-a", "top a", ("A", "B", "C", "D"), 3.6),
        _record("top-b", "top b", ("A", "B", "C", "E"), 3.8),
        _record("low-a", "low a", ("W", "X", "Y", "Z"), 5.2),
        _record("low-b", "low b", ("V", "W", "Y", "Z"), 5.4),
    ]

    report = train_metatft_ranking_model(
        train_records=train_records,
        eval_records=train_records,
        output=checkpoint,
        device_name="cpu",
        epochs=80,
        learning_rate=3e-3,
        seed=0,
        hidden_dim=32,
        embed_dim=16,
    )
    evaluator = FightValueCombatModel(checkpoint, device_name="cpu")
    scores = score_records_with_fight_value(train_records, evaluator)

    assert checkpoint.exists()
    assert evaluator.metatft_unit_id_lookup is not None
    assert report.records_train == 4
    assert report.records_eval == 4
    assert report.train_pairwise_accuracy >= 0.8
    assert scores["top-a"] > scores["low-a"]


def _record(
    cluster_id: str,
    name: str,
    units: tuple[str, ...],
    avg_placement: float,
) -> CompStrengthRecord:
    return CompStrengthRecord(
        cluster_id=cluster_id,
        name=name,
        units=units,
        avg_placement=avg_placement,
        top4_rate=max(0.0, min(1.0, (8.0 - avg_placement) / 8.0)),
        win_rate=max(0.0, min(1.0, (5.0 - avg_placement) / 10.0)),
        count=1000,
    )
