"""Current-patch board state schema for MetaTFT value/planning models."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_BOARD_TOKENS = 12
MAX_BENCH_TOKENS = 12
MAX_ITEMS_PER_UNIT = 3
MAX_LEVEL = 10


@dataclass(frozen=True)
class CurrentBoardUnit:
    unit_key: str
    stars: int = 1
    item_keys: tuple[str, ...] = ()
    position: int = 0
    chosen_trait_key: str | None = None

    def __post_init__(self) -> None:
        if not self.unit_key:
            raise ValueError("unit_key is required")
        if self.stars not in {1, 2, 3, 4}:
            raise ValueError("stars must be 1, 2, 3, or 4")
        if len(self.item_keys) > MAX_ITEMS_PER_UNIT:
            raise ValueError(f"at most {MAX_ITEMS_PER_UNIT} item keys are supported")
        if self.position < 0:
            raise ValueError("position must be non-negative")


@dataclass(frozen=True)
class CurrentBoardState:
    stage: int
    stage_round: int
    level: int
    board: tuple[CurrentBoardUnit, ...]
    gold: int = 0
    hp: int = 100
    bench: tuple[CurrentBoardUnit, ...] = ()
    active_trait_keys: tuple[str, ...] = ()
    augment_keys: tuple[str, ...] = ()
    target_comp_id: str | None = None
    source: str = "manual"
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage < 1:
            raise ValueError("stage must be positive")
        if self.stage_round < 1:
            raise ValueError("stage_round must be positive")
        if not 1 <= self.level <= MAX_LEVEL:
            raise ValueError(f"level must be in [1, {MAX_LEVEL}]")
        if not 0 <= self.gold <= 300:
            raise ValueError("gold must be in [0, 300]")
        if not 0 <= self.hp <= 100:
            raise ValueError("hp must be in [0, 100]")
        if len(self.board) > MAX_BOARD_TOKENS:
            raise ValueError(f"at most {MAX_BOARD_TOKENS} board tokens are supported")
        if len(self.bench) > MAX_BENCH_TOKENS:
            raise ValueError(f"at most {MAX_BENCH_TOKENS} bench tokens are supported")

    @property
    def stage_label(self) -> str:
        return f"{self.stage}-{self.stage_round}"

    @property
    def board_unit_keys(self) -> tuple[str, ...]:
        return tuple(unit.unit_key for unit in self.board)
