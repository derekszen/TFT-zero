"""Benchmark FightValueNet inference throughput."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from mini_tft.fight_model.model import load_checkpoint
from mini_tft.fight_model.schema import MAX_ITEMS, MAX_UNITS


def benchmark_model(
    checkpoint: Path,
    device_name: str,
    batch_size: int,
    batches: int,
) -> dict[str, float | int | str]:
    device = torch.device(device_name)
    model = load_checkpoint(checkpoint, map_location=device).to(device)
    config = model.config
    batch = {
        "board_a_unit_ids": torch.randint(
            0,
            config.max_unit_id + 1,
            (batch_size, MAX_UNITS),
            device=device,
        ),
        "board_a_stars": torch.randint(0, 4, (batch_size, MAX_UNITS), device=device),
        "board_a_position_ids": torch.randint(0, 28, (batch_size, MAX_UNITS), device=device),
        "board_a_item_ids": torch.randint(
            0,
            config.max_item_id + 1,
            (batch_size, MAX_UNITS, MAX_ITEMS),
            device=device,
        ),
        "board_a_chosen_traits": torch.zeros(
            (batch_size, MAX_UNITS),
            dtype=torch.long,
            device=device,
        ),
        "board_b_unit_ids": torch.randint(
            0,
            config.max_unit_id + 1,
            (batch_size, MAX_UNITS),
            device=device,
        ),
        "board_b_stars": torch.randint(0, 4, (batch_size, MAX_UNITS), device=device),
        "board_b_position_ids": torch.randint(0, 28, (batch_size, MAX_UNITS), device=device),
        "board_b_item_ids": torch.randint(
            0,
            config.max_item_id + 1,
            (batch_size, MAX_UNITS, MAX_ITEMS),
            device=device,
        ),
        "board_b_chosen_traits": torch.zeros(
            (batch_size, MAX_UNITS),
            dtype=torch.long,
            device=device,
        ),
        "stage": torch.full((batch_size,), 4, device=device),
        "level_a": torch.full((batch_size,), 7, device=device),
        "level_b": torch.full((batch_size,), 7, device=device),
    }
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(batches):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    fights = batch_size * batches
    return {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "batch_size": batch_size,
        "batches": batches,
        "fights": fights,
        "elapsed_sec": elapsed,
        "fights_per_sec": fights / elapsed if elapsed else 0.0,
    }
