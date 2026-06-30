"""Markov state for the clean strategic MiniTFT lane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mini_tft.strategic.core.actions import ROLE_NAMES

MAX_LEVEL = 9
SHOP_SIZE = 5
UNIT_COUNT = 9
ROLE_COUNT = len(ROLE_NAMES)


@dataclass(frozen=True)
class StrategicConfig:
    max_round: int = 36
    max_actions_per_round: int = 3
    starting_hp: int = 100
    starting_gold: int = 3
    starting_level: int = 3
    starting_xp: int = 0
    max_level: int = MAX_LEVEL
    shop_size: int = SHOP_SIZE
    roll_cost: int = 2
    xp_buy_cost: int = 4
    xp_per_buy: int = 4
    base_income: int = 5
    max_interest: int = 5
    win_gold: int = 1
    max_role_item_slots: int = 3
    item_drop_interval: int = 4
    combat_sigmoid_scale: float = 14.0
    enemy_strength_multiplier: float = 2.9
    combat_noise: float = 2.0
    max_owned_copies: int = 27


DEFAULT_STRATEGIC_CONFIG = StrategicConfig()


@dataclass(frozen=True)
class StrategicUnit:
    unit_id: int
    name: str
    cost: int
    role: str
    trait: str
    power: float


@dataclass
class StrategicState:
    seed: int
    rng_key: int
    round: int
    hp: int
    gold: int
    level: int
    xp: int
    shop: NDArray[np.int16]
    owned: NDArray[np.int16]
    fielded: NDArray[np.int16]
    role_items: NDArray[np.int16]
    role_item_slots: NDArray[np.int16]
    done: bool = False
    final_reason: str | None = None
    action_count: int = 0
    last_board_strength: float = 0.0
    last_enemy_strength: float = 0.0
    last_damage: int = 0
    last_win: bool = False
    total_rolls: int = 0
    total_xp_buys: int = 0
    total_units_bought: int = 0
    total_item_slams: int = 0
    total_illegal_actions: int = 0

    def clone(self) -> StrategicState:
        return StrategicState(
            seed=self.seed,
            rng_key=self.rng_key,
            round=self.round,
            hp=self.hp,
            gold=self.gold,
            level=self.level,
            xp=self.xp,
            shop=self.shop.copy(),
            owned=self.owned.copy(),
            fielded=self.fielded.copy(),
            role_items=self.role_items.copy(),
            role_item_slots=self.role_item_slots.copy(),
            done=self.done,
            final_reason=self.final_reason,
            action_count=self.action_count,
            last_board_strength=self.last_board_strength,
            last_enemy_strength=self.last_enemy_strength,
            last_damage=self.last_damage,
            last_win=self.last_win,
            total_rolls=self.total_rolls,
            total_xp_buys=self.total_xp_buys,
            total_units_bought=self.total_units_bought,
            total_item_slams=self.total_item_slams,
            total_illegal_actions=self.total_illegal_actions,
        )


@dataclass(frozen=True)
class StrategicStepResult:
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, int | float | bool | str | None]


UNITS: tuple[StrategicUnit, ...] = (
    StrategicUnit(1, "Vanguard", 1, "tank", "noble", 10.0),
    StrategicUnit(2, "Duelist", 1, "carry", "ranger", 11.0),
    StrategicUnit(3, "Mystic", 1, "support", "wild", 8.5),
    StrategicUnit(4, "Knight", 2, "tank", "noble", 16.0),
    StrategicUnit(5, "Sharpshooter", 2, "carry", "ranger", 17.5),
    StrategicUnit(6, "Invoker", 2, "support", "sorcerer", 14.0),
    StrategicUnit(7, "Guardian", 3, "tank", "glacial", 24.0),
    StrategicUnit(8, "Blademaster", 3, "carry", "imperial", 27.0),
    StrategicUnit(9, "Enchanter", 3, "support", "sorcerer", 21.0),
)
UNIT_BY_ID = {unit.unit_id: unit for unit in UNITS}


def empty_shop(config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG) -> NDArray[np.int16]:
    return np.zeros(config.shop_size, dtype=np.int16)


def empty_field(config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG) -> NDArray[np.int16]:
    return np.zeros(config.max_level, dtype=np.int16)


def empty_owned() -> NDArray[np.int16]:
    return np.zeros(UNIT_COUNT + 1, dtype=np.int16)


def empty_role_counts() -> NDArray[np.int16]:
    return np.zeros(ROLE_COUNT, dtype=np.int16)


def state_signature(state: StrategicState) -> tuple[object, ...]:
    return (
        state.seed,
        state.rng_key,
        state.round,
        state.hp,
        state.gold,
        state.level,
        state.xp,
        tuple(int(value) for value in state.shop.tolist()),
        tuple(int(value) for value in state.owned.tolist()),
        tuple(int(value) for value in state.fielded.tolist()),
        tuple(int(value) for value in state.role_items.tolist()),
        tuple(int(value) for value in state.role_item_slots.tolist()),
        state.done,
        state.final_reason,
        state.action_count,
        round(state.last_board_strength, 6),
        round(state.last_enemy_strength, 6),
        state.last_damage,
        state.last_win,
        state.total_rolls,
        state.total_xp_buys,
        state.total_units_bought,
        state.total_item_slams,
        state.total_illegal_actions,
    )
