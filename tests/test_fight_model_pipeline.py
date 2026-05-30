from __future__ import annotations

import json

import pytest

pytest.importorskip("zarr")
pytest.importorskip("torch")

from mini_tft.fight_model.benchmark import benchmark_model
from mini_tft.fight_model.generation import generate_fight_labels
from mini_tft.fight_model.storage import load_arrays, validate_dataset
from mini_tft.fight_model.training import train_fight_value_model


def test_heuristic_fight_label_generation_writes_resumable_zarr(tmp_path) -> None:
    out = tmp_path / "fight_labels"

    report = generate_fight_labels(
        out=out,
        target_fights=24,
        workers=1,
        shard_size=8,
        seed=123,
        teacher_name="heuristic",
        teacher_root=None,
    )
    repeat = generate_fight_labels(
        out=out,
        target_fights=24,
        workers=1,
        shard_size=8,
        seed=123,
        teacher_name="heuristic",
        teacher_root=None,
    )

    assert report.written_fights == 24
    assert repeat.written_fights == 0
    summary = validate_dataset(out)
    assert summary == {"total_fights": 24, "shards": 3}

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["teacher"] == "heuristic"
    assert manifest["sampling"] == "staged_v1"

    arrays = load_arrays(out)
    assert arrays["board_a_unit_ids"].shape == (24, 9)
    assert arrays["board_a_item_ids"].shape == (24, 9, 3)
    assert arrays["a_win"].dtype == bool


def test_fight_value_model_training_and_benchmark_smoke(tmp_path) -> None:
    dataset = tmp_path / "fight_labels"
    checkpoint = tmp_path / "fight_value.pt"
    generate_fight_labels(
        out=dataset,
        target_fights=48,
        workers=1,
        shard_size=16,
        seed=7,
        teacher_name="heuristic",
        teacher_root=None,
    )

    train_report = train_fight_value_model(
        dataset_root=dataset,
        output=checkpoint,
        device_name="cpu",
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        val_fraction=0.25,
        seed=0,
    )
    bench = benchmark_model(checkpoint, device_name="cpu", batch_size=16, batches=2)

    assert checkpoint.exists()
    assert train_report.val_loss >= 0.0
    assert 0.0 <= train_report.val_win_accuracy <= 1.0
    assert bench["fights"] == 32
    assert bench["fights_per_sec"] > 0.0
