"""Fight specification sampling for teacher-label generation."""

from __future__ import annotations

import numpy as np

from mini_tft.fight_model.schema import FightBoard, FightSpec, FightUnit, SamplingBucket

DEFAULT_MAX_UNIT_ID = 58
DEFAULT_MAX_ITEM_ID = 45


def sample_staged_v1(
    rng: np.random.Generator,
    seed: int,
    max_unit_id: int = DEFAULT_MAX_UNIT_ID,
    max_item_id: int = DEFAULT_MAX_ITEM_ID,
) -> FightSpec:
    bucket = SamplingBucket(
        rng.choice(
            [
                SamplingBucket.EARLY,
                SamplingBucket.MID,
                SamplingBucket.LATE,
                SamplingBucket.CLOSE,
                SamplingBucket.STRESS,
            ],
            p=[0.25, 0.25, 0.20, 0.20, 0.10],
        )
    )

    if bucket == SamplingBucket.EARLY:
        return _random_match(seed, bucket, rng, 2, 4, 1, 2, max_unit_id, max_item_id)
    if bucket == SamplingBucket.MID:
        return _random_match(seed, bucket, rng, 5, 7, 1, 3, max_unit_id, max_item_id)
    if bucket == SamplingBucket.LATE:
        return _random_match(seed, bucket, rng, 7, 9, 2, 5, max_unit_id, max_item_id)
    if bucket == SamplingBucket.CLOSE:
        base = _random_board(rng, rng.integers(5, 9), 2, 5, max_unit_id, max_item_id)
        board_b = _mutate_board(base, rng, max_unit_id, max_item_id)
        return FightSpec(
            base,
            board_b,
            stage=int(rng.integers(3, 6)),
            seed=seed,
            sampling_bucket=bucket,
        )
    return _stress_match(seed, rng, max_unit_id, max_item_id)


def _random_match(
    seed: int,
    bucket: SamplingBucket,
    rng: np.random.Generator,
    min_level: int,
    max_level: int,
    min_cost_band: int,
    max_cost_band: int,
    max_unit_id: int,
    max_item_id: int,
) -> FightSpec:
    level_a = int(rng.integers(min_level, max_level + 1))
    level_b = int(rng.integers(min_level, max_level + 1))
    board_a = _random_board(rng, level_a, min_cost_band, max_cost_band, max_unit_id, max_item_id)
    board_b = _random_board(rng, level_b, min_cost_band, max_cost_band, max_unit_id, max_item_id)
    stage = max(1, min(6, (level_a + level_b) // 3))
    return FightSpec(board_a, board_b, stage=stage, seed=seed, sampling_bucket=bucket)


def _random_board(
    rng: np.random.Generator,
    level: int,
    min_cost_band: int,
    max_cost_band: int,
    max_unit_id: int,
    max_item_id: int,
) -> FightBoard:
    unit_count = int(min(level, rng.integers(max(1, level - 1), level + 1)))
    positions = rng.choice(np.arange(28), size=unit_count, replace=False)
    units = []
    for unit_index in range(unit_count):
        cost_band = int(rng.integers(min_cost_band, max_cost_band + 1))
        unit_id = _unit_id_from_cost_band(rng, cost_band, max_unit_id)
        stars = _sample_stars(rng, cost_band)
        item_count = _sample_item_count(rng, level, unit_index)
        items = tuple(int(x) for x in rng.integers(1, max_item_id + 1, size=item_count))
        units.append(
            FightUnit(
                unit_id=unit_id,
                stars=stars,
                position=int(positions[unit_index]),
                items=items,
                chosen_trait=0,
            )
        )
    return FightBoard(units=tuple(units), level=level)


def _mutate_board(
    board: FightBoard,
    rng: np.random.Generator,
    max_unit_id: int,
    max_item_id: int,
) -> FightBoard:
    units = list(board.units)
    if units:
        index = int(rng.integers(0, len(units)))
        unit = units[index]
        if rng.random() < 0.5:
            mutated_unit_id = _bounded_unit_id(
                unit.unit_id + int(rng.choice([-2, -1, 1, 2])),
                max_unit_id,
            )
            units[index] = FightUnit(
                unit_id=mutated_unit_id,
                stars=unit.stars,
                position=unit.position,
                items=unit.items,
                chosen_trait=unit.chosen_trait,
            )
        else:
            units[index] = FightUnit(
                unit_id=unit.unit_id,
                stars=min(3, unit.stars + 1),
                position=unit.position,
                items=unit.items,
                chosen_trait=unit.chosen_trait,
            )
    if rng.random() < 0.35 and len(units) < board.level:
        positions = {unit.position for unit in units}
        available = [position for position in range(28) if position not in positions]
        units.append(
            FightUnit(
                unit_id=int(rng.integers(1, max_unit_id + 1)),
                stars=1,
                position=int(rng.choice(available)),
                items=tuple(int(x) for x in rng.integers(1, max_item_id + 1, size=1)),
            )
        )
    return FightBoard(units=tuple(units), level=board.level)


def _stress_match(
    seed: int,
    rng: np.random.Generator,
    max_unit_id: int,
    max_item_id: int,
) -> FightSpec:
    level = int(rng.integers(7, 10))
    carry_items = tuple(int(x) for x in rng.integers(1, max_item_id + 1, size=3))
    tank_items = tuple(int(x) for x in rng.integers(1, max_item_id + 1, size=3))
    board_a = FightBoard(
        units=(
            FightUnit(_bounded_unit_id(max_unit_id - 4, max_unit_id), 2, 3, tank_items),
            FightUnit(_bounded_unit_id(max_unit_id - 3, max_unit_id), 2, 10),
            FightUnit(_bounded_unit_id(max_unit_id - 2, max_unit_id), 2, 24, carry_items),
            FightUnit(_bounded_unit_id(max_unit_id - 1, max_unit_id), 1, 26),
        ),
        level=level,
    )
    board_b = _random_board(rng, level, 2, 5, max_unit_id, max_item_id)
    return FightSpec(board_a, board_b, stage=5, seed=seed, sampling_bucket=SamplingBucket.STRESS)


def _unit_id_from_cost_band(
    rng: np.random.Generator,
    cost_band: int,
    max_unit_id: int,
) -> int:
    band_size = max(1, max_unit_id // 5)
    low = min(max_unit_id, (cost_band - 1) * band_size + 1)
    high = max(low, min(max_unit_id, cost_band * band_size))
    return int(rng.integers(low, high + 1))


def _sample_stars(rng: np.random.Generator, cost_band: int) -> int:
    if cost_band <= 1:
        return int(rng.choice([1, 2, 3], p=[0.70, 0.25, 0.05]))
    if cost_band <= 3:
        return int(rng.choice([1, 2, 3], p=[0.78, 0.20, 0.02]))
    return int(rng.choice([1, 2], p=[0.90, 0.10]))


def _sample_item_count(rng: np.random.Generator, level: int, unit_index: int) -> int:
    if level < 5:
        return int(rng.choice([0, 1], p=[0.82, 0.18]))
    if unit_index < 2:
        return int(rng.choice([0, 1, 2, 3], p=[0.25, 0.30, 0.25, 0.20]))
    return int(rng.choice([0, 1, 2], p=[0.74, 0.21, 0.05]))


def _bounded_unit_id(unit_id: int, max_unit_id: int) -> int:
    return max(1, min(max_unit_id, unit_id))
