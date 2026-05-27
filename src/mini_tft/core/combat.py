"""Abstract board strength and combat resolution."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData
from mini_tft.core.state import UnitInstance
from mini_tft.core.traits import active_trait_effects

STAR_MULTIPLIER = {
    1: 1.0,
    2: 1.8,
    3: 3.4,
}

FRONT_SLOTS = {0, 1, 2}
BACK_SLOTS = {6, 7, 8}
STAR_TEMPO_BONUS = {
    1: 0.0,
    2: 4.0,
    3: 9.0,
}


@dataclass(frozen=True)
class CombatStats:
    strength: float
    enemy_power_penalty: float
    active_traits: dict[str, dict[str, float]]


@dataclass(frozen=True)
class CombatResult:
    won: bool
    damage: int
    my_strength: float
    enemy_strength: float
    p_win: float


def unit_standalone_power(unit: UnitInstance, data: GameData) -> float:
    unit_def = data.units[unit.unit_id]
    return unit_def.base_power * STAR_MULTIPLIER[unit.stars]


def board_strength(board: list[UnitInstance | None], data: GameData) -> CombatStats:
    role_power = {"carry": 0.0, "tank": 0.0, "support": 0.0}
    item_flat_power = 0.0
    item_enemy_penalty = 0.0
    star_tempo_power = 0.0
    formation_power = 0.0

    for slot, unit in enumerate(board):
        if unit is None:
            continue
        unit_def = data.units[unit.unit_id]
        item_multiplier = 1.0
        for item_id in unit.items:
            item = data.items[item_id]
            item_flat_power += item.effects.get("flat_power", 0.0)
            item_enemy_penalty += item.effects.get("enemy_power_penalty", 0.0)
            item_multiplier *= item.effects.get(f"{unit_def.role}_multiplier", 1.0)
        role_power[unit_def.role] += unit_standalone_power(unit, data) * item_multiplier
        star_tempo_power += STAR_TEMPO_BONUS.get(unit.stars, 0.0)
        formation_power += _formation_bonus(slot, unit_def.role)

    carry_multiplier = 1.0
    tank_multiplier = 1.0
    support_multiplier = 1.0
    flat_power = item_flat_power
    enemy_penalty = item_enemy_penalty
    active_traits = active_trait_effects(board, data)

    for effects in active_traits.values():
        flat_power += effects.get("flat_power", 0.0)
        carry_multiplier *= effects.get("carry_multiplier", 1.0)
        tank_multiplier *= effects.get("tank_multiplier", 1.0)
        support_multiplier *= effects.get("support_multiplier", 1.0)
        enemy_penalty += effects.get("enemy_power_penalty", 0.0)

    strength = (
        role_power["carry"] * carry_multiplier
        + role_power["tank"] * tank_multiplier
        + role_power["support"] * support_multiplier
        + flat_power
        + star_tempo_power
        + formation_power
        + _board_balance_bonus(role_power)
    )
    return CombatStats(
        strength=max(0.0, strength),
        enemy_power_penalty=enemy_penalty,
        active_traits=active_traits,
    )


def resolve_combat(
    board: list[UnitInstance | None],
    round_num: int,
    data: GameData,
    config: EnvConfig,
    rng: np.random.Generator,
) -> CombatResult:
    stats = board_strength(board, data)
    enemy_index = min(round_num - 1, len(data.enemy_curve) - 1)
    enemy_strength = data.enemy_curve[enemy_index] + float(rng.normal(0.0, config.combat_noise_std))
    enemy_strength = max(0.0, enemy_strength - stats.enemy_power_penalty)
    diff = stats.strength - enemy_strength
    p_win = 1.0 / (1.0 + exp(-(diff / config.combat_sigmoid_scale)))
    won = bool(rng.random() < p_win)
    damage = 0
    if not won:
        margin_damage = int(max(0.0, enemy_strength - stats.strength) / 20)
        damage = base_damage_by_round(round_num) + margin_damage
    return CombatResult(
        won=won,
        damage=damage,
        my_strength=stats.strength,
        enemy_strength=enemy_strength,
        p_win=p_win,
    )


def base_damage_by_round(round_num: int) -> int:
    if round_num < 8:
        return 2
    if round_num < 16:
        return 4
    if round_num < 24:
        return 6
    if round_num < 32:
        return 8
    return 10


def _board_balance_bonus(role_power: dict[str, float]) -> float:
    has_tank = role_power["tank"] > 0
    has_carry = role_power["carry"] > 0
    tank_count_bonus = 8.0 if role_power["tank"] >= 24 and role_power["carry"] >= 24 else 0.0
    if has_tank and has_carry:
        return 5.0 + tank_count_bonus
    if not has_tank and not has_carry:
        return -16.0
    if not has_tank:
        return -8.0
    if not has_carry:
        return -8.0
    return 0.0


def _formation_bonus(slot: int, role: str) -> float:
    if role == "tank":
        if slot in FRONT_SLOTS:
            return 1.5
        if slot in BACK_SLOTS:
            return -1.5
        return 0.0
    if role in {"carry", "support"}:
        if slot in BACK_SLOTS:
            return 1.0
        if slot in FRONT_SLOTS:
            return -0.8
    return 0.0
