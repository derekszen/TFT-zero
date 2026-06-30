"""Strategic MiniTFT rules.

The transition function is intentionally small and Markov: transition and reward
depend only on canonical state, action, config, and the RNG key stored in state.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from math import exp

import numpy as np
from numpy.typing import NDArray

from mini_tft.strategic.core.actions import (
    NUM_ACTIONS,
    ROLE_BY_SLAM_ACTION,
    ROLE_INDEX,
    ROLE_NAMES,
    StrategicAction,
)
from mini_tft.strategic.core.state import (
    DEFAULT_STRATEGIC_CONFIG,
    UNIT_BY_ID,
    UNITS,
    StrategicConfig,
    StrategicState,
    StrategicStepResult,
    empty_field,
    empty_owned,
    empty_role_counts,
    empty_shop,
)

_UINT64_MASK = (1 << 64) - 1
_LCG_MULTIPLIER = 6364136223846793005
_LCG_INCREMENT = 1442695040888963407
_STAR_MULTIPLIER = {0: 0.0, 1: 1.0, 2: 1.9, 3: 3.5}
_ROLE_ITEM_POWER = np.asarray([9.0, 8.0, 6.5], dtype=np.float32)


def reset(seed: int = 0, config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG) -> StrategicState:
    state = StrategicState(
        seed=int(seed),
        rng_key=(int(seed) ^ 0x9E3779B97F4A7C15) & _UINT64_MASK,
        round=1,
        hp=config.starting_hp,
        gold=config.starting_gold,
        level=config.starting_level,
        xp=config.starting_xp,
        shop=empty_shop(config),
        owned=empty_owned(),
        fielded=empty_field(config),
        role_items=empty_role_counts(),
        role_item_slots=empty_role_counts(),
    )
    refresh_shop(state, config)
    return state


def legal_action_mask(
    state: StrategicState,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> NDArray[np.bool_]:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    if state.done:
        return mask

    mask[StrategicAction.HOLD] = True
    mask[StrategicAction.GREED_ECON] = True
    mask[StrategicAction.LEVEL] = (
        state.gold >= config.xp_buy_cost and state.level < config.max_level
    )
    mask[StrategicAction.ROLL] = state.gold >= config.roll_cost
    mask[StrategicAction.BUY_BEST_UPGRADE] = _best_buy(state, config, "upgrade") is not None
    mask[StrategicAction.BUY_BEST_SYNERGY] = _best_buy(state, config, "synergy") is not None
    mask[StrategicAction.BUY_HIGHEST_COST] = _best_buy(state, config, "highest_cost") is not None
    mask[StrategicAction.FIELD_STRONGEST] = _strongest_field_signature(state, config) != tuple(
        int(unit_id) for unit_id in state.fielded.tolist()
    )

    present_roles = fielded_roles(state)
    for role, action in (
        ("carry", StrategicAction.SLAM_CARRY_ITEM),
        ("tank", StrategicAction.SLAM_TANK_ITEM),
        ("support", StrategicAction.SLAM_SUPPORT_ITEM),
    ):
        role_index = ROLE_INDEX[role]
        mask[action] = (
            state.role_items[role_index] > 0
            and state.role_item_slots[role_index] < config.max_role_item_slots
            and role in present_roles
        )
    return mask


def step(
    state: StrategicState,
    action: int,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> StrategicStepResult:
    if state.done:
        raise RuntimeError("Strategic episode is done. Call reset() before stepping.")

    action = int(action)
    mask = legal_action_mask(state, config)
    legal = 0 <= action < len(mask) and bool(mask[action])
    reward = 0.0
    ended_round = False

    if not legal:
        state.total_illegal_actions += 1
        reward -= 1.0
    else:
        strategic_action = StrategicAction(action)
        if strategic_action == StrategicAction.HOLD:
            reward += _end_round(state, config, greed=False)
            ended_round = True
        elif strategic_action == StrategicAction.GREED_ECON:
            reward += _end_round(state, config, greed=True)
            ended_round = True
        else:
            reward += _apply_non_terminal_action(state, strategic_action, config)
            state.action_count += 1
            if state.action_count >= config.max_actions_per_round and not state.done:
                reward -= 0.05
                reward += _end_round(state, config, greed=False)
                ended_round = True

    terminated = state.done and state.final_reason in {"hp_zero", "max_round"}
    info = {
        "legal_action": legal,
        "ended_round": ended_round,
        "round": state.round,
        "hp": state.hp,
        "gold": state.gold,
        "level": state.level,
        "board_strength": state.last_board_strength,
        "enemy_strength": state.last_enemy_strength,
        "placement_proxy": placement_proxy(state, config),
        "scenario_score": scenario_score(state, config),
        "final_reason": state.final_reason,
    }
    return StrategicStepResult(
        reward=float(reward),
        terminated=terminated,
        truncated=False,
        info=info,
    )


def refresh_shop(
    state: StrategicState,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> None:
    for index in range(config.shop_size):
        state.shop[index] = _sample_unit_id(state, config)


def field_strongest(
    state: StrategicState,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> bool:
    next_units = _strongest_field_signature(state, config)
    current_units = tuple(int(unit_id) for unit_id in state.fielded.tolist())
    if next_units == current_units:
        return False
    state.fielded[:] = np.asarray(next_units, dtype=np.int16)
    return True


def board_strength(
    state: StrategicState,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> float:
    del config
    strength = 0.0
    trait_counts: Counter[str] = Counter()
    role_counts = np.zeros(len(ROLE_NAMES), dtype=np.int16)
    for unit_id in state.fielded:
        unit_id = int(unit_id)
        if unit_id == 0:
            continue
        unit = UNIT_BY_ID[unit_id]
        stars = star_level(int(state.owned[unit_id]))
        strength += unit.power * _STAR_MULTIPLIER[stars]
        trait_counts[unit.trait] += 1
        role_counts[ROLE_INDEX[unit.role]] += 1

    for trait_count in trait_counts.values():
        if trait_count >= 3:
            strength += 8.0
        elif trait_count >= 2:
            strength += 3.0

    for role_index, slots in enumerate(state.role_item_slots):
        if role_counts[role_index] > 0:
            strength += float(slots) * float(_ROLE_ITEM_POWER[role_index])

    if role_counts[ROLE_INDEX["tank"]] > 0 and role_counts[ROLE_INDEX["carry"]] > 0:
        strength += 6.0
    if role_counts.sum() == 0:
        strength -= 12.0
    return max(0.0, strength)


def enemy_strength_for_round(
    round_num: int,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> float:
    round_float = float(max(1, round_num))
    base = 13.0 + round_float * 3.2 + (round_float ** 1.18) * 1.35
    return base * config.enemy_strength_multiplier


def stage_round_for_round(round_num: int) -> tuple[int, int]:
    if round_num < 1:
        raise ValueError(f"round_num must be positive: {round_num}")
    if round_num <= 3:
        return 1, round_num
    offset = round_num - 4
    return 2 + offset // 7, 1 + offset % 7


def stage_label_for_round(round_num: int) -> str:
    stage, stage_round = stage_round_for_round(round_num)
    return f"Stage {stage}-{stage_round}"


def placement_proxy(
    state: StrategicState,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> int:
    if state.final_reason == "max_round" or (state.round >= config.max_round and state.hp > 0):
        return 1

    if state.round >= 36:
        return 2
    if state.round >= 32:
        return 3
    if state.round >= 29:
        return 4
    if state.round >= 25:
        return 5
    if state.round >= 18:
        return 6
    if state.round >= 11:
        return 7
    return 8


def scenario_score(
    state: StrategicState,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> float:
    round_frac = min(1.0, max(0.0, state.round / config.max_round))
    hp_frac = min(1.0, max(0.0, state.hp / config.starting_hp))
    enemy = max(1.0, enemy_strength_for_round(max(1, state.round), config))
    strength_ratio = min(1.4, state.last_board_strength / enemy) / 1.4
    return float(np.clip(0.45 * round_frac + 0.25 * hp_frac + 0.30 * strength_ratio, 0.0, 1.0))


def fielded_roles(state: StrategicState) -> set[str]:
    roles: set[str] = set()
    for unit_id in state.fielded:
        unit_id = int(unit_id)
        if unit_id:
            roles.add(UNIT_BY_ID[unit_id].role)
    return roles


def star_level(copies: int) -> int:
    if copies >= 9:
        return 3
    if copies >= 3:
        return 2
    if copies >= 1:
        return 1
    return 0


def xp_to_next_level(level: int) -> int:
    return 2 + max(0, level - 1) * 2


def run_episode(
    policy,
    *,
    seed: int,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> tuple[StrategicState, float, int]:
    state = reset(seed=seed, config=config)
    total_reward = 0.0
    steps = 0
    while not state.done and steps < config.max_round * (config.max_actions_per_round + 1):
        mask = legal_action_mask(state, config)
        action = int(policy(state, mask, config))
        result = step(state, action, config)
        total_reward += result.reward
        steps += 1
    return state, total_reward, steps


def _apply_non_terminal_action(
    state: StrategicState,
    action: StrategicAction,
    config: StrategicConfig,
) -> float:
    if action == StrategicAction.LEVEL:
        state.gold -= config.xp_buy_cost
        state.xp += config.xp_per_buy
        leveled = False
        while state.level < config.max_level and state.xp >= xp_to_next_level(state.level):
            state.xp -= xp_to_next_level(state.level)
            state.level += 1
            leveled = True
        state.total_xp_buys += 1
        return 0.08 if leveled else 0.01
    if action == StrategicAction.ROLL:
        state.gold -= config.roll_cost
        state.total_rolls += 1
        refresh_shop(state, config)
        return 0.0
    if action in {
        StrategicAction.BUY_BEST_UPGRADE,
        StrategicAction.BUY_BEST_SYNERGY,
        StrategicAction.BUY_HIGHEST_COST,
    }:
        mode = {
            StrategicAction.BUY_BEST_UPGRADE: "upgrade",
            StrategicAction.BUY_BEST_SYNERGY: "synergy",
            StrategicAction.BUY_HIGHEST_COST: "highest_cost",
        }[action]
        return _buy_best(state, config, mode)
    if action == StrategicAction.FIELD_STRONGEST:
        before = board_strength(state, config)
        changed = field_strongest(state, config)
        after = board_strength(state, config)
        return max(0.0, (after - before) * 0.01) if changed else 0.0
    if action in ROLE_BY_SLAM_ACTION:
        return _slam_role_item(state, ROLE_BY_SLAM_ACTION[action], config)
    return 0.0


def _end_round(state: StrategicState, config: StrategicConfig, *, greed: bool) -> float:
    previous_strength = state.last_board_strength
    current_strength = board_strength(state, config)
    enemy_strength = enemy_strength_for_round(state.round, config) + _normal_noise(state, config)
    p_win = 1.0 / (1.0 + exp(-((current_strength - enemy_strength) / config.combat_sigmoid_scale)))
    won = _random_float(state) < p_win
    damage = 0 if won else _damage_from_margin(state.round, enemy_strength, current_strength)
    previous_hp = state.hp
    state.hp = max(0, state.hp - damage)

    state.last_board_strength = current_strength
    state.last_enemy_strength = enemy_strength
    state.last_damage = damage
    state.last_win = won

    state.gold += config.base_income + min(config.max_interest, state.gold // 10)
    if won:
        state.gold += config.win_gold
    if greed and state.gold >= 10:
        state.gold += 1

    _maybe_drop_role_item(state, config)
    refresh_shop(state, config)
    state.action_count = 0

    if state.hp <= 0:
        state.done = True
        state.final_reason = "hp_zero"
    elif state.round >= config.max_round:
        state.done = True
        state.final_reason = "max_round"
    else:
        state.round += 1

    reward = (state.hp - previous_hp) * 0.04
    reward += (current_strength - previous_strength) * 0.015
    reward += 0.25 if won else -0.10
    reward += scenario_score(state, config) * 0.10
    if greed:
        reward += 0.04 if state.hp >= config.starting_hp * 0.45 else -0.08
    if state.done and state.final_reason == "max_round":
        reward += 1.0
    if state.done and state.final_reason == "hp_zero":
        reward -= 0.8
    return reward


def _buy_best(state: StrategicState, config: StrategicConfig, mode: str) -> float:
    candidate = _best_buy(state, config, mode)
    if candidate is None:
        return 0.0
    shop_index, unit_id = candidate
    unit = UNIT_BY_ID[unit_id]
    state.gold -= unit.cost
    state.owned[unit_id] += 1
    state.shop[shop_index] = 0
    state.total_units_bought += 1
    return 0.10 if state.owned[unit_id] in {3, 9} else 0.04


def _best_buy(
    state: StrategicState,
    config: StrategicConfig,
    mode: str,
) -> tuple[int, int] | None:
    candidates: list[tuple[float, int, int]] = []
    total_owned = int(state.owned.sum())
    if total_owned >= config.max_owned_copies:
        return None

    owned_traits = {
        UNIT_BY_ID[int(unit_id)].trait
        for unit_id, count in enumerate(state.owned)
        if unit_id in UNIT_BY_ID and count > 0
    }
    owned_roles = fielded_roles(state)
    for shop_index, unit_id_raw in enumerate(state.shop):
        unit_id = int(unit_id_raw)
        if unit_id == 0:
            continue
        unit = UNIT_BY_ID[unit_id]
        if state.gold < unit.cost:
            continue
        copies = int(state.owned[unit_id])
        if mode == "upgrade" and copies <= 0:
            continue
        if mode == "upgrade":
            immediate = 1.0 if copies + 1 in {3, 9} else 0.0
            distance = min((3 - ((copies + 1) % 3)) % 3, (9 - (copies + 1)) % 9)
            score = immediate * 1000.0 + (20.0 - distance) + unit.power
        elif mode == "synergy":
            trait_match = 1.0 if unit.trait in owned_traits else 0.0
            role_need = 1.0 if unit.role not in owned_roles else 0.0
            score = trait_match * 100.0 + role_need * 20.0 + unit.power + unit.cost
        elif mode == "highest_cost":
            score = unit.cost * 100.0 + unit.power
        else:
            raise ValueError(f"unknown buy mode: {mode}")
        candidates.append((score, shop_index, unit_id))
    if not candidates:
        return None
    _, shop_index, unit_id = max(candidates)
    return shop_index, unit_id


def _slam_role_item(state: StrategicState, role: str, config: StrategicConfig) -> float:
    role_index = ROLE_INDEX[role]
    if state.role_items[role_index] <= 0:
        return 0.0
    if state.role_item_slots[role_index] >= config.max_role_item_slots:
        return 0.0
    if role not in fielded_roles(state):
        return 0.0
    state.role_items[role_index] -= 1
    state.role_item_slots[role_index] += 1
    state.total_item_slams += 1
    return float(_ROLE_ITEM_POWER[role_index]) * 0.02


def _strongest_field_signature(
    state: StrategicState,
    config: StrategicConfig,
) -> tuple[int, ...]:
    candidates = [
        unit.unit_id
        for unit in UNITS
        if state.owned[unit.unit_id] > 0
    ]
    candidates.sort(
        key=lambda unit_id: (
            _unit_field_power(unit_id, int(state.owned[unit_id])),
            UNIT_BY_ID[unit_id].cost,
            -unit_id,
        ),
        reverse=True,
    )
    fielded = candidates[: min(config.max_level, state.level)]
    fielded.extend([0] * (config.max_level - len(fielded)))
    return tuple(fielded)


def _unit_field_power(unit_id: int, copies: int) -> float:
    unit = UNIT_BY_ID[unit_id]
    return unit.power * _STAR_MULTIPLIER[star_level(copies)]


def _sample_unit_id(state: StrategicState, config: StrategicConfig) -> int:
    roll = _random_float(state)
    if state.level <= 3:
        tiers = (0.72, 0.26, 0.02)
    elif state.level <= 5:
        tiers = (0.48, 0.42, 0.10)
    elif state.level <= 7:
        tiers = (0.25, 0.50, 0.25)
    else:
        tiers = (0.12, 0.43, 0.45)
    cost = 1 if roll < tiers[0] else 2 if roll < tiers[0] + tiers[1] else 3
    units = [unit.unit_id for unit in UNITS if unit.cost == cost]
    return units[_random_int(state, len(units))]


def _maybe_drop_role_item(state: StrategicState, config: StrategicConfig) -> None:
    if state.round % config.item_drop_interval != 0:
        return
    role_index = _random_int(state, len(ROLE_NAMES))
    state.role_items[role_index] += 1


def _damage_from_margin(round_num: int, enemy_strength: float, board: float) -> int:
    base = 2 if round_num < 8 else 4 if round_num < 16 else 6 if round_num < 24 else 8
    return int(base + max(0.0, enemy_strength - board) / 22.0)


def _normal_noise(state: StrategicState, config: StrategicConfig) -> float:
    if config.combat_noise <= 0:
        return 0.0
    # Irwin-Hall style bounded noise. Cheap, deterministic, and Markov via rng_key.
    return ((_random_float(state) + _random_float(state) + _random_float(state)) - 1.5) * (
        config.combat_noise * 1.6
    )


def _random_int(state: StrategicState, limit: int) -> int:
    if limit <= 0:
        raise ValueError("limit must be positive")
    return int(_next_u64(state) % limit)


def _random_float(state: StrategicState) -> float:
    return (_next_u64(state) >> 11) * (1.0 / (1 << 53))


def _next_u64(state: StrategicState) -> int:
    state.rng_key = (state.rng_key * _LCG_MULTIPLIER + _LCG_INCREMENT) & _UINT64_MASK
    return state.rng_key


def trace_episode(
    actions: Iterable[int],
    *,
    seed: int = 0,
    config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
) -> list[tuple[object, ...]]:
    from mini_tft.strategic.core.state import state_signature

    state = reset(seed=seed, config=config)
    rows = [state_signature(state)]
    for action in actions:
        if state.done:
            break
        step(state, int(action), config)
        rows.append(state_signature(state))
    return rows
