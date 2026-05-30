from __future__ import annotations

from pathlib import Path

from mini_tft.fight_model.calibration import (
    calibration_report,
    load_comp_strength_records,
    metatft_unit_id_lookup,
    score_records_by_metric,
)

FIXTURE = Path("tests/fixtures/metatft_set17_comp_strength_2026-05-31.json")


def test_metatft_comp_strength_fixture_loads_real_snapshot() -> None:
    source, records = load_comp_strength_records(FIXTURE)

    assert source["name"] == "MetaTFT current-patch comp stats"
    assert source["tft_set"] == "TFTSet17"
    assert source["retrieved_at"] == "2026-05-31"
    assert len(records) == 16
    assert min(record.count for record in records) >= 12_000
    assert records[0].avg_placement < records[-1].avg_placement
    assert len(metatft_unit_id_lookup(records)) >= 30


def test_calibration_ranking_gate_accepts_correct_real_data_ordering() -> None:
    _, records = load_comp_strength_records(FIXTURE)
    scores = score_records_by_metric(records, lambda record: -record.avg_placement)

    report = calibration_report(records, scores)

    assert report.passed
    assert report.pairwise_accuracy == 1.0
    assert report.spearman == 1.0
    assert report.top_k_overlap == 1.0


def test_calibration_ranking_gate_rejects_inverted_real_data_ordering() -> None:
    _, records = load_comp_strength_records(FIXTURE)
    scores = score_records_by_metric(records, lambda record: record.avg_placement)

    report = calibration_report(records, scores)

    assert not report.passed
    assert report.pairwise_accuracy == 0.0
    assert report.spearman < 0.0
    assert report.top_k_overlap == 0.0
