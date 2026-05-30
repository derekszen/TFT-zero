"""Serializable schemas for fight-label generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

MAX_UNITS = 9
MAX_ITEMS = 3
MAX_POSITION = 27
SCHEMA_VERSION = 1


class SamplingBucket(IntEnum):
    EARLY = 1
    MID = 2
    LATE = 3
    CLOSE = 4
    STRESS = 5
    POLICY = 6


@dataclass(frozen=True)
class FightUnit:
    unit_id: int
    stars: int
    position: int
    items: tuple[int, ...] = ()
    chosen_trait: int = 0

    def __post_init__(self) -> None:
        if self.unit_id <= 0:
            raise ValueError("unit_id must be positive")
        if self.stars not in {1, 2, 3}:
            raise ValueError("stars must be 1, 2, or 3")
        if not 0 <= self.position <= MAX_POSITION:
            raise ValueError(f"position must be in [0, {MAX_POSITION}]")
        if len(self.items) > MAX_ITEMS:
            raise ValueError(f"at most {MAX_ITEMS} items are supported")
        if any(item_id < 0 for item_id in self.items):
            raise ValueError("item ids must be non-negative")


@dataclass(frozen=True)
class FightBoard:
    units: tuple[FightUnit, ...]
    level: int

    def __post_init__(self) -> None:
        if not 1 <= self.level <= MAX_UNITS:
            raise ValueError(f"level must be in [1, {MAX_UNITS}]")
        if len(self.units) > MAX_UNITS:
            raise ValueError(f"at most {MAX_UNITS} units are supported")


@dataclass(frozen=True)
class FightSpec:
    board_a: FightBoard
    board_b: FightBoard
    stage: int
    seed: int
    sampling_bucket: SamplingBucket

    def __post_init__(self) -> None:
        if self.stage <= 0:
            raise ValueError("stage must be positive")


@dataclass(frozen=True)
class FightLabel:
    a_win: bool
    damage_to_a: float
    damage_to_b: float
    survivor_value_a: float
    survivor_value_b: float
    teacher_ms: float
    error: str = ""


FIGHT_ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "board_a_unit_ids": np.dtype("int16"),
    "board_a_stars": np.dtype("int8"),
    "board_a_position_ids": np.dtype("int8"),
    "board_a_item_ids": np.dtype("int16"),
    "board_a_chosen_traits": np.dtype("int16"),
    "board_b_unit_ids": np.dtype("int16"),
    "board_b_stars": np.dtype("int8"),
    "board_b_position_ids": np.dtype("int8"),
    "board_b_item_ids": np.dtype("int16"),
    "board_b_chosen_traits": np.dtype("int16"),
    "stage": np.dtype("int8"),
    "level_a": np.dtype("int8"),
    "level_b": np.dtype("int8"),
    "seed": np.dtype("int64"),
    "sampling_bucket": np.dtype("int8"),
    "a_win": np.dtype("bool"),
    "damage_to_a": np.dtype("float32"),
    "damage_to_b": np.dtype("float32"),
    "survivor_value_a": np.dtype("float32"),
    "survivor_value_b": np.dtype("float32"),
    "teacher_ms": np.dtype("float32"),
}


def empty_arrays(count: int) -> dict[str, NDArray[Any]]:
    arrays: dict[str, NDArray[Any]] = {}
    for name, dtype in FIGHT_ARRAY_DTYPES.items():
        shape = _array_shape(name, count)
        arrays[name] = np.zeros(shape, dtype=dtype)
    return arrays


def encode_example(
    arrays: dict[str, NDArray[Any]],
    index: int,
    spec: FightSpec,
    label: FightLabel,
) -> None:
    _encode_board(arrays, "board_a", index, spec.board_a)
    _encode_board(arrays, "board_b", index, spec.board_b)
    arrays["stage"][index] = spec.stage
    arrays["level_a"][index] = spec.board_a.level
    arrays["level_b"][index] = spec.board_b.level
    arrays["seed"][index] = spec.seed
    arrays["sampling_bucket"][index] = int(spec.sampling_bucket)
    arrays["a_win"][index] = label.a_win
    arrays["damage_to_a"][index] = label.damage_to_a
    arrays["damage_to_b"][index] = label.damage_to_b
    arrays["survivor_value_a"][index] = label.survivor_value_a
    arrays["survivor_value_b"][index] = label.survivor_value_b
    arrays["teacher_ms"][index] = label.teacher_ms


def _encode_board(
    arrays: dict[str, NDArray[Any]],
    prefix: str,
    index: int,
    board: FightBoard,
) -> None:
    for unit_index, unit in enumerate(board.units[:MAX_UNITS]):
        arrays[f"{prefix}_unit_ids"][index, unit_index] = unit.unit_id
        arrays[f"{prefix}_stars"][index, unit_index] = unit.stars
        arrays[f"{prefix}_position_ids"][index, unit_index] = unit.position
        arrays[f"{prefix}_chosen_traits"][index, unit_index] = unit.chosen_trait
        for item_index, item_id in enumerate(unit.items[:MAX_ITEMS]):
            arrays[f"{prefix}_item_ids"][index, unit_index, item_index] = item_id


def _array_shape(name: str, count: int) -> tuple[int, ...]:
    if name.endswith("_item_ids"):
        return (count, MAX_UNITS, MAX_ITEMS)
    if name.startswith("board_"):
        return (count, MAX_UNITS)
    return (count,)
