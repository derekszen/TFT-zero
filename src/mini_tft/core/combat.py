"""Abstract board strength and combat resolution."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData
from mini_tft.core.state import UnitInstance

STAR_MULTIPLIER = {
    1: 1.0,
    2: 1.8,
    3: 3.4,
}
STAR_RELIABILITY_BONUS = {
    1: 0.0,
    2: 1.55,
    3: 5.20,
}
MATCHED_ITEM_ROLE_BONUS = {
    "carry": 7.0,
    "tank": 5.0,
    "support": 4.0,
}
MISMATCHED_ITEM_ROLE_PENALTY = {
    "carry": 9.0,
    "tank": 5.0,
    "support": 2.5,
}
TRAIT_BREAKPOINT_FLAT_BONUS = {
    ("ranger", 2): 5.0,
    ("noble", 6): 35.0,
}
ASSASSIN_PRESSURE_POWER_RATIO = 0.14
ASSASSIN_PAIR_ACCESS_BONUS = 3.75
ASSASSIN_PRESSURE_CAP = 8.0


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
    role_counts = {"carry": 0, "tank": 0, "support": 0}
    item_flat_power = 0.0
    item_enemy_penalty = 0.0
    item_fit_power = 0.0
    upgrade_bonus = 0.0
    assassin_count = 0
    assassin_power = 0.0
    unit_ids: set[int] = set()

    for slot_index, unit in enumerate(board):
        if unit is None:
            continue
        unit_def = data.units[unit.unit_id]
        unit_ids.add(unit.unit_id)
        item_multiplier = 1.0
        for item_id in unit.items:
            item = data.items[item_id]
            item_flat_power += item.effects.get("flat_power", 0.0)
            item_enemy_penalty += item.effects.get("enemy_power_penalty", 0.0)
            item_multiplier *= item.effects.get(f"{unit_def.role}_multiplier", 1.0)
            item_fit_power += _item_role_fit_power(unit_def.role, item.tags)

        positioned_power = (
            unit_standalone_power(unit, data)
            * item_multiplier
            * _role_position_multiplier(slot_index, unit_def.role)
        )
        role_power[unit_def.role] += positioned_power
        role_counts[unit_def.role] += 1
        upgrade_bonus += _upgrade_reliability_bonus(unit, data)
        if "assassin" in unit_def.traits:
            assassin_count += 1
            assassin_power += positioned_power

    carry_multiplier = 1.0
    tank_multiplier = 1.0
    support_multiplier = 1.0
    flat_power = item_flat_power
    enemy_penalty = item_enemy_penalty
    counts = _trait_counts_from_unit_ids(unit_ids, data)
    active_traits = _active_trait_effects_from_counts(counts, data)

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
        + item_fit_power
        + _board_balance_bonus(role_power)
        + _skirmish_structure_bonus(role_power, role_counts)
        + _assassin_pressure_bonus(assassin_count, assassin_power, role_power)
        + _trait_breakpoint_bonus(counts, data)
        + upgrade_bonus
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
    has_board_units = any(unit is not None for unit in board)
    if not has_board_units and enemy_strength > 0.0:
        damage = base_damage_by_round(round_num) + int(enemy_strength / 20)
        return CombatResult(
            won=False,
            damage=damage,
            my_strength=stats.strength,
            enemy_strength=enemy_strength,
            p_win=0.0,
        )
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


def _role_position_multiplier(slot_index: int, role: str) -> float:
    """Approximate combat uptime from TFT-like front/mid/back rows.

    Slots 0-2 are treated as frontline, 3-5 as midline, and 6-8 as backline.
    This keeps combat scalar and fast while letting manual placement matter.
    """

    row = slot_index // 3
    if row == 0:
        return {"tank": 1.18, "carry": 0.82, "support": 0.90}.get(role, 1.0)
    if row == 1:
        return {"tank": 1.02, "carry": 1.00, "support": 1.05}.get(role, 1.0)
    return {"tank": 0.72, "carry": 1.14, "support": 1.10}.get(role, 1.0)


def _skirmish_structure_bonus(role_power: dict[str, float], role_counts: dict[str, int]) -> float:
    """Reward boards that look like stable skirmishes, not just raw piles."""

    tank_power = role_power["tank"]
    carry_power = role_power["carry"]
    support_power = role_power["support"]

    bonus = 0.0
    if tank_power > 0.0 and carry_power > 0.0:
        bonus += min(tank_power, carry_power) * 0.18
    if support_power > 0.0 and tank_power + carry_power > 0.0:
        bonus += min(support_power * 0.14, (tank_power + carry_power) * 0.06)
    if role_counts["tank"] == 0 and role_counts["carry"] >= 2:
        bonus -= 4.0
    return bonus


def _assassin_pressure_bonus(
    assassin_count: int,
    assassin_power: float,
    role_power: dict[str, float],
) -> float:
    """Approximate assassin jump pressure without simulating targeting."""

    if assassin_count <= 0:
        return 0.0

    bonus = min(assassin_power * ASSASSIN_PRESSURE_POWER_RATIO, ASSASSIN_PRESSURE_CAP)
    if assassin_count >= 2:
        bonus += ASSASSIN_PAIR_ACCESS_BONUS
    if assassin_count >= 3:
        bonus += min((assassin_count - 2) * 2.0, 6.0)

    if role_power["tank"] <= 0.0:
        bonus *= 0.55
    if role_power["carry"] <= 0.0:
        bonus *= 0.70
    return bonus


def _upgrade_reliability_bonus(unit: UnitInstance, data: GameData) -> float:
    if unit.stars <= 1:
        return 0.0
    unit_def = data.units[unit.unit_id]
    return unit_def.base_power * STAR_RELIABILITY_BONUS.get(unit.stars, 0.0)


def _item_role_fit_power(role: str, tags: tuple[str, ...]) -> float:
    if not tags:
        return 0.0
    if role in tags:
        return MATCHED_ITEM_ROLE_BONUS.get(role, 0.0)
    return -sum(MISMATCHED_ITEM_ROLE_PENALTY.get(tag, 0.0) for tag in tags)


def _trait_breakpoint_bonus(counts: dict[str, int], data: GameData) -> float:
    bonus = 0.0
    for trait_id, count in counts.items():
        trait = data.traits[trait_id]
        reached = [breakpoint for breakpoint in trait.breakpoints if breakpoint <= count]
        if not reached:
            continue
        breakpoint = max(reached)
        bonus += TRAIT_BREAKPOINT_FLAT_BONUS.get((trait_id, breakpoint), 0.0)
    return bonus


def _trait_counts_from_unit_ids(unit_ids: set[int], data: GameData) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit_id in unit_ids:
        for trait in data.units[unit_id].traits:
            counts[trait] = counts.get(trait, 0) + 1
    return counts


def _active_trait_effects_from_counts(
    counts: dict[str, int],
    data: GameData,
) -> dict[str, dict[str, float]]:
    active: dict[str, dict[str, float]] = {}
    for trait_id, count in counts.items():
        trait = data.traits[trait_id]
        reached = [breakpoint for breakpoint in trait.breakpoints if breakpoint <= count]
        if reached:
            active[trait_id] = trait.breakpoints[max(reached)]
    return active
