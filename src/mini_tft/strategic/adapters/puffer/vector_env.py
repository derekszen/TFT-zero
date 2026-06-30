"""Native batched strategic-lane transition core.

This module mirrors :mod:`mini_tft.strategic.core.rules` with fixed-shape NumPy
arrays. It is intentionally scoped to the clean strategic lane so Puffer can
step many independent episodes without one Gym object per slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.strategic.core.actions import NUM_ACTIONS, ROLE_INDEX, StrategicAction
from mini_tft.strategic.core.obs import observation_dim
from mini_tft.strategic.core.rules import enemy_strength_for_round
from mini_tft.strategic.core.state import (
    DEFAULT_STRATEGIC_CONFIG,
    ROLE_COUNT,
    UNIT_BY_ID,
    UNIT_COUNT,
    UNITS,
    StrategicConfig,
)

_UINT64_MASK = np.uint64((1 << 64) - 1)
_LCG_MULTIPLIER = np.uint64(6364136223846793005)
_LCG_INCREMENT = np.uint64(1442695040888963407)
_SEED_XOR = np.uint64(0x9E3779B97F4A7C15)
_STAR_MULTIPLIERS = np.asarray([0.0, 1.0, 1.9, 3.5], dtype=np.float32)
_ROLE_ITEM_POWER = np.asarray([9.0, 8.0, 6.5], dtype=np.float32)
_FINAL_REASON_NONE = np.int8(0)
_FINAL_REASON_HP_ZERO = np.int8(1)
_FINAL_REASON_MAX_ROUND = np.int8(2)
_UNIT_ID_MAX = max(UNIT_BY_ID)

_UNIT_COST = np.zeros(UNIT_COUNT + 1, dtype=np.int16)
_UNIT_POWER = np.zeros(UNIT_COUNT + 1, dtype=np.float32)
_UNIT_ROLE = np.full(UNIT_COUNT + 1, -1, dtype=np.int16)
_UNIT_TRAIT = np.full(UNIT_COUNT + 1, -1, dtype=np.int16)
_TRAIT_BY_NAME = {
    trait: index for index, trait in enumerate(sorted({unit.trait for unit in UNITS}))
}
for _unit in UNITS:
    _UNIT_COST[_unit.unit_id] = _unit.cost
    _UNIT_POWER[_unit.unit_id] = _unit.power
    _UNIT_ROLE[_unit.unit_id] = ROLE_INDEX[_unit.role]
    _UNIT_TRAIT[_unit.unit_id] = _TRAIT_BY_NAME[_unit.trait]

_UNITS_BY_COST = {
    1: np.asarray([unit.unit_id for unit in UNITS if unit.cost == 1], dtype=np.int16),
    2: np.asarray([unit.unit_id for unit in UNITS if unit.cost == 2], dtype=np.int16),
    3: np.asarray([unit.unit_id for unit in UNITS if unit.cost == 3], dtype=np.int16),
}


@dataclass(frozen=True)
class BatchStepResult:
    observations: NDArray[np.float32]
    rewards: NDArray[np.float32]
    terminals: NDArray[np.bool_]
    truncations: NDArray[np.bool_]
    infos: list[dict[str, Any]]


class StrategicBatchEnv:
    """Fixed-shape vector environment for independent strategic episodes."""

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int = 0,
        config: StrategicConfig = DEFAULT_STRATEGIC_CONFIG,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.num_envs = int(num_envs)
        self.config = config
        self.base_observation_dim = observation_dim(config)
        self.observation_dim = self.base_observation_dim + NUM_ACTIONS
        self.seed = np.zeros(self.num_envs, dtype=np.int64)
        self.rng_key = np.zeros(self.num_envs, dtype=np.uint64)
        self.round = np.ones(self.num_envs, dtype=np.int16)
        self.hp = np.full(self.num_envs, config.starting_hp, dtype=np.int16)
        self.gold = np.full(self.num_envs, config.starting_gold, dtype=np.int16)
        self.level = np.full(self.num_envs, config.starting_level, dtype=np.int16)
        self.xp = np.full(self.num_envs, config.starting_xp, dtype=np.int16)
        self.shop = np.zeros((self.num_envs, config.shop_size), dtype=np.int16)
        self.owned = np.zeros((self.num_envs, UNIT_COUNT + 1), dtype=np.int16)
        self.fielded = np.zeros((self.num_envs, config.max_level), dtype=np.int16)
        self.role_items = np.zeros((self.num_envs, ROLE_COUNT), dtype=np.int16)
        self.role_item_slots = np.zeros((self.num_envs, ROLE_COUNT), dtype=np.int16)
        self.done = np.zeros(self.num_envs, dtype=np.bool_)
        self.final_reason_code = np.zeros(self.num_envs, dtype=np.int8)
        self.action_count = np.zeros(self.num_envs, dtype=np.int16)
        self.last_board_strength = np.zeros(self.num_envs, dtype=np.float32)
        self.last_enemy_strength = np.zeros(self.num_envs, dtype=np.float32)
        self.last_damage = np.zeros(self.num_envs, dtype=np.int16)
        self.last_win = np.zeros(self.num_envs, dtype=np.bool_)
        self.total_rolls = np.zeros(self.num_envs, dtype=np.int32)
        self.total_xp_buys = np.zeros(self.num_envs, dtype=np.int32)
        self.total_units_bought = np.zeros(self.num_envs, dtype=np.int32)
        self.total_item_slams = np.zeros(self.num_envs, dtype=np.int32)
        self.total_illegal_actions = np.zeros(self.num_envs, dtype=np.int32)
        self._cached_action_mask = np.zeros((self.num_envs, NUM_ACTIONS), dtype=np.bool_)
        self.reset(seed=seed)

    def reset(
        self,
        seed: int | NDArray[np.integer[Any]] | list[int] | tuple[int, ...] = 0,
        indices: NDArray[np.integer[Any]] | list[int] | tuple[int, ...] | None = None,
    ) -> NDArray[np.float32]:
        index_array = self._indices(indices)
        seed_array = np.asarray(seed, dtype=np.int64)
        if seed_array.ndim == 0:
            seeds = int(seed_array) + index_array.astype(np.int64)
        else:
            seeds = seed_array.reshape(-1).astype(np.int64)
            if seeds.shape[0] != index_array.shape[0]:
                raise ValueError("seed array length must match indices length")

        self.seed[index_array] = seeds
        self.rng_key[index_array] = seeds.astype(np.uint64) ^ _SEED_XOR
        self.round[index_array] = 1
        self.hp[index_array] = self.config.starting_hp
        self.gold[index_array] = self.config.starting_gold
        self.level[index_array] = self.config.starting_level
        self.xp[index_array] = self.config.starting_xp
        self.shop[index_array, :] = 0
        self.owned[index_array, :] = 0
        self.fielded[index_array, :] = 0
        self.role_items[index_array, :] = 0
        self.role_item_slots[index_array, :] = 0
        self.done[index_array] = False
        self.final_reason_code[index_array] = _FINAL_REASON_NONE
        self.action_count[index_array] = 0
        self.last_board_strength[index_array] = 0.0
        self.last_enemy_strength[index_array] = 0.0
        self.last_damage[index_array] = 0
        self.last_win[index_array] = False
        self.total_rolls[index_array] = 0
        self.total_xp_buys[index_array] = 0
        self.total_units_bought[index_array] = 0
        self.total_item_slams[index_array] = 0
        self.total_illegal_actions[index_array] = 0
        self._refresh_shop(index_array)
        self._cached_action_mask = self.legal_action_mask()
        return self.observe(action_mask=self._cached_action_mask)

    def step(
        self,
        actions: NDArray[np.integer[Any]] | list[int] | tuple[int, ...],
        *,
        include_infos: bool = True,
    ) -> BatchStepResult:
        action_array = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        before_mask = self._cached_action_mask
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminals = np.zeros(self.num_envs, dtype=np.bool_)
        truncations = np.zeros(self.num_envs, dtype=np.bool_)
        ended_round = np.zeros(self.num_envs, dtype=np.bool_)
        legal = (
            (~self.done)
            & (action_array >= 0)
            & (action_array < NUM_ACTIONS)
            & before_mask[np.arange(self.num_envs), np.clip(action_array, 0, NUM_ACTIONS - 1)]
        )

        illegal = (~self.done) & (~legal)
        if bool(illegal.any()):
            self.total_illegal_actions[illegal] += 1
            rewards[illegal] -= 1.0

        for action in StrategicAction:
            selected = legal & (action_array == int(action))
            if not bool(selected.any()):
                continue
            indices = np.flatnonzero(selected).astype(np.int64)
            if action == StrategicAction.HOLD:
                rewards[indices] += self._end_round(indices, greed=False)
                ended_round[indices] = True
            elif action == StrategicAction.GREED_ECON:
                rewards[indices] += self._end_round(indices, greed=True)
                ended_round[indices] = True
            else:
                rewards[indices] += self._apply_non_terminal_action(indices, action)
                self.action_count[indices] += 1
                capped = indices[
                    (self.action_count[indices] >= self.config.max_actions_per_round)
                    & (~self.done[indices])
                ]
                if capped.size:
                    rewards[capped] -= 0.05
                    rewards[capped] += self._end_round(capped, greed=False)
                    ended_round[capped] = True

        terminals[:] = self.done & (
            (self.final_reason_code == _FINAL_REASON_HP_ZERO)
            | (self.final_reason_code == _FINAL_REASON_MAX_ROUND)
        )
        after_mask = self.legal_action_mask()
        self._cached_action_mask = after_mask
        observations = self.observe(action_mask=after_mask)
        if include_infos:
            infos = [
                self._info(
                    env_index,
                    bool(legal[env_index]),
                    bool(ended_round[env_index]),
                    after_mask,
                )
                for env_index in range(self.num_envs)
            ]
        else:
            infos = [{} for _ in range(self.num_envs)]
        return BatchStepResult(
            observations=observations,
            rewards=rewards,
            terminals=terminals,
            truncations=truncations,
            infos=infos,
        )

    def observe(self, action_mask: NDArray[np.bool_] | None = None) -> NDArray[np.float32]:
        obs = np.zeros((self.num_envs, self.observation_dim), dtype=np.float32)
        config = self.config
        cursor = 0
        obs[:, cursor] = self.round.astype(np.float32) / config.max_round
        cursor += 1
        obs[:, cursor] = self.hp.astype(np.float32) / config.starting_hp
        cursor += 1
        obs[:, cursor] = self.gold.astype(np.float32) / 100.0
        cursor += 1
        obs[:, cursor] = self.level.astype(np.float32) / config.max_level
        cursor += 1
        obs[:, cursor] = self.xp.astype(np.float32) / max(1.0, float(config.max_level * 2))
        cursor += 1
        obs[:, cursor] = self.action_count.astype(np.float32) / max(
            1,
            config.max_actions_per_round,
        )
        cursor += 1
        obs[:, cursor] = self.last_board_strength / 200.0
        cursor += 1
        obs[:, cursor] = self.last_enemy_strength / 200.0
        cursor += 1
        obs[:, cursor : cursor + config.shop_size] = self.shop.astype(np.float32) / _UNIT_ID_MAX
        cursor += config.shop_size
        obs[:, cursor : cursor + UNIT_COUNT + 1] = self.owned.astype(np.float32) / 9.0
        cursor += UNIT_COUNT + 1
        obs[:, cursor : cursor + config.max_level] = self.fielded.astype(np.float32) / _UNIT_ID_MAX
        cursor += config.max_level
        obs[:, cursor : cursor + ROLE_COUNT] = self.role_items.astype(np.float32) / 5.0
        cursor += ROLE_COUNT
        obs[:, cursor : cursor + ROLE_COUNT] = self.role_item_slots.astype(np.float32) / (
            config.max_role_item_slots
        )
        cursor += ROLE_COUNT
        mask = self.legal_action_mask() if action_mask is None else action_mask
        obs[:, cursor : cursor + NUM_ACTIONS] = mask.astype(np.float32)
        return obs

    def legal_action_mask(self) -> NDArray[np.bool_]:
        config = self.config
        mask = np.zeros((self.num_envs, NUM_ACTIONS), dtype=np.bool_)
        alive = ~self.done
        mask[:, StrategicAction.HOLD] = alive
        mask[:, StrategicAction.GREED_ECON] = alive
        mask[:, StrategicAction.LEVEL] = alive & (self.gold >= config.xp_buy_cost) & (
            self.level < config.max_level
        )
        mask[:, StrategicAction.ROLL] = alive & (self.gold >= config.roll_cost)
        for action, mode in (
            (StrategicAction.BUY_BEST_UPGRADE, "upgrade"),
            (StrategicAction.BUY_BEST_SYNERGY, "synergy"),
            (StrategicAction.BUY_HIGHEST_COST, "highest_cost"),
        ):
            shop_index, _unit_id = self._best_buy(mode)
            mask[:, action] = alive & (shop_index >= 0)
        mask[:, StrategicAction.FIELD_STRONGEST] = alive & np.any(
            self._strongest_field() != self.fielded,
            axis=1,
        )
        present_roles = self._fielded_role_presence()
        for role_name, action in (
            ("carry", StrategicAction.SLAM_CARRY_ITEM),
            ("tank", StrategicAction.SLAM_TANK_ITEM),
            ("support", StrategicAction.SLAM_SUPPORT_ITEM),
        ):
            role_index = ROLE_INDEX[role_name]
            mask[:, action] = (
                alive
                & (self.role_items[:, role_index] > 0)
                & (self.role_item_slots[:, role_index] < config.max_role_item_slots)
                & present_roles[:, role_index]
            )
        return mask

    def final_reason(self, env_index: int) -> str | None:
        code = int(self.final_reason_code[env_index])
        if code == int(_FINAL_REASON_HP_ZERO):
            return "hp_zero"
        if code == int(_FINAL_REASON_MAX_ROUND):
            return "max_round"
        return None

    @property
    def cached_action_mask(self) -> NDArray[np.bool_]:
        return self._cached_action_mask

    def placement_proxy(self) -> NDArray[np.int16]:
        placements = np.full(self.num_envs, 8, dtype=np.int16)
        placements[(self.round >= 11)] = 7
        placements[(self.round >= 18)] = 6
        placements[(self.round >= 25)] = 5
        placements[(self.round >= 29)] = 4
        placements[(self.round >= 32)] = 3
        placements[(self.round >= 36)] = 2
        max_round_mask = (self.final_reason_code == _FINAL_REASON_MAX_ROUND) | (
            (self.round >= self.config.max_round) & (self.hp > 0)
        )
        placements[max_round_mask] = 1
        return placements

    def scenario_score(self) -> NDArray[np.float32]:
        round_frac = np.clip(
            self.round.astype(np.float32) / float(self.config.max_round),
            0.0,
            1.0,
        )
        hp_frac = np.clip(
            self.hp.astype(np.float32) / float(self.config.starting_hp),
            0.0,
            1.0,
        )
        enemy = np.maximum(
            1.0,
            self._enemy_strength(self.round.astype(np.int64)),
        )
        strength_ratio = np.minimum(1.4, self.last_board_strength / enemy) / 1.4
        return np.clip(
            0.45 * round_frac + 0.25 * hp_frac + 0.30 * strength_ratio,
            0.0,
            1.0,
        ).astype(np.float32)

    def _apply_non_terminal_action(
        self,
        indices: NDArray[np.int64],
        action: StrategicAction,
    ) -> NDArray[np.float32]:
        rewards = np.zeros(indices.shape[0], dtype=np.float32)
        if action == StrategicAction.LEVEL:
            self.gold[indices] -= self.config.xp_buy_cost
            self.xp[indices] += self.config.xp_per_buy
            leveled = np.zeros(indices.shape[0], dtype=np.bool_)
            while True:
                xp_needed = self._xp_to_next_level(self.level[indices])
                can_level = (self.level[indices] < self.config.max_level) & (
                    self.xp[indices] >= xp_needed
                )
                if not bool(can_level.any()):
                    break
                active = indices[can_level]
                self.xp[active] -= xp_needed[can_level]
                self.level[active] += 1
                leveled[can_level] = True
            self.total_xp_buys[indices] += 1
            rewards[:] = np.where(leveled, 0.08, 0.01).astype(np.float32)
            return rewards
        if action == StrategicAction.ROLL:
            self.gold[indices] -= self.config.roll_cost
            self.total_rolls[indices] += 1
            self._refresh_shop(indices)
            return rewards
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
            return self._buy_best(indices, mode)
        if action == StrategicAction.FIELD_STRONGEST:
            before = self._board_strength(indices)
            self.fielded[indices] = self._strongest_field(indices)
            after = self._board_strength(indices)
            return np.maximum(0.0, (after - before) * 0.01).astype(np.float32)
        role_by_action = {
            StrategicAction.SLAM_CARRY_ITEM: ROLE_INDEX["carry"],
            StrategicAction.SLAM_TANK_ITEM: ROLE_INDEX["tank"],
            StrategicAction.SLAM_SUPPORT_ITEM: ROLE_INDEX["support"],
        }
        if action in role_by_action:
            role_index = role_by_action[action]
            self.role_items[indices, role_index] -= 1
            self.role_item_slots[indices, role_index] += 1
            self.total_item_slams[indices] += 1
            rewards[:] = float(_ROLE_ITEM_POWER[role_index]) * 0.02
        return rewards

    def _end_round(self, indices: NDArray[np.int64], *, greed: bool) -> NDArray[np.float32]:
        previous_strength = self.last_board_strength[indices].copy()
        current_strength = self._board_strength(indices)
        enemy_strength = self._enemy_strength(self.round[indices].astype(np.int64))
        enemy_strength = enemy_strength + self._normal_noise(indices)
        p_win = 1.0 / (
            1.0 + np.exp(-((current_strength - enemy_strength) / self.config.combat_sigmoid_scale))
        )
        won = self._random_float(indices) < p_win
        damage = np.where(
            won,
            0,
            self._damage_from_margin(
                self.round[indices].astype(np.int64),
                enemy_strength,
                current_strength,
            ),
        ).astype(np.int16)
        previous_hp = self.hp[indices].copy()
        self.hp[indices] = np.maximum(0, self.hp[indices] - damage).astype(np.int16)
        self.last_board_strength[indices] = current_strength
        self.last_enemy_strength[indices] = enemy_strength
        self.last_damage[indices] = damage
        self.last_win[indices] = won

        self.gold[indices] += self.config.base_income + np.minimum(
            self.config.max_interest,
            self.gold[indices] // 10,
        ).astype(np.int16)
        self.gold[indices[won]] += self.config.win_gold
        if greed:
            greed_bonus = indices[self.gold[indices] >= 10]
            self.gold[greed_bonus] += 1

        self._maybe_drop_role_item(indices)
        self._refresh_shop(indices)
        self.action_count[indices] = 0

        hp_zero = self.hp[indices] <= 0
        max_round = self.round[indices] >= self.config.max_round
        self.done[indices[hp_zero]] = True
        self.final_reason_code[indices[hp_zero]] = _FINAL_REASON_HP_ZERO
        max_round_indices = indices[(~hp_zero) & max_round]
        self.done[max_round_indices] = True
        self.final_reason_code[max_round_indices] = _FINAL_REASON_MAX_ROUND
        advance = indices[(~hp_zero) & (~max_round)]
        self.round[advance] += 1

        rewards = (self.hp[indices].astype(np.float32) - previous_hp.astype(np.float32)) * 0.04
        rewards += (current_strength - previous_strength) * 0.015
        rewards += np.where(won, 0.25, -0.10).astype(np.float32)
        rewards += self.scenario_score()[indices] * 0.10
        if greed:
            rewards += np.where(
                self.hp[indices] >= self.config.starting_hp * 0.45,
                0.04,
                -0.08,
            ).astype(np.float32)
        rewards += np.where(
            self.done[indices] & (self.final_reason_code[indices] == _FINAL_REASON_MAX_ROUND),
            1.0,
            0.0,
        ).astype(np.float32)
        rewards += np.where(
            self.done[indices] & (self.final_reason_code[indices] == _FINAL_REASON_HP_ZERO),
            -0.8,
            0.0,
        ).astype(np.float32)
        return rewards.astype(np.float32)

    def _buy_best(self, indices: NDArray[np.int64], mode: str) -> NDArray[np.float32]:
        shop_indices, unit_ids = self._best_buy(mode, indices)
        rewards = np.zeros(indices.shape[0], dtype=np.float32)
        valid = shop_indices >= 0
        if not bool(valid.any()):
            return rewards
        active = indices[valid]
        active_shop_indices = shop_indices[valid]
        active_unit_ids = unit_ids[valid]
        self.gold[active] -= _UNIT_COST[active_unit_ids]
        self.owned[active, active_unit_ids] += 1
        self.shop[active, active_shop_indices] = 0
        self.total_units_bought[active] += 1
        copies = self.owned[active, active_unit_ids]
        rewards[valid] = np.where((copies == 3) | (copies == 9), 0.10, 0.04).astype(np.float32)
        return rewards

    def _best_buy(
        self,
        mode: str,
        indices: NDArray[np.int64] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int16]]:
        index_array = self._indices(indices)
        score = np.full((index_array.shape[0], self.config.shop_size), -np.inf, dtype=np.float32)
        shop_units = self.shop[index_array]
        total_owned = self.owned[index_array].sum(axis=1)
        can_hold_more = total_owned < self.config.max_owned_copies
        owned_trait_presence = self._owned_trait_presence(index_array)
        fielded_role_presence = self._fielded_role_presence(index_array)

        for shop_index in range(self.config.shop_size):
            unit_ids = shop_units[:, shop_index]
            nonempty = unit_ids > 0
            affordable = self.gold[index_array] >= _UNIT_COST[unit_ids]
            valid = can_hold_more & nonempty & affordable
            copies = self.owned[index_array, unit_ids]
            if mode == "upgrade":
                valid &= copies > 0
                immediate = np.isin(copies + 1, np.asarray([3, 9], dtype=np.int16))
                distance = np.minimum((3 - ((copies + 1) % 3)) % 3, (9 - (copies + 1)) % 9)
                raw_score = (
                    immediate.astype(np.float32) * 1000.0
                    + (20.0 - distance)
                    + _UNIT_POWER[unit_ids]
                )
            elif mode == "synergy":
                rows = np.arange(index_array.shape[0])
                trait_match = owned_trait_presence[rows, _UNIT_TRAIT[unit_ids]]
                role_need = ~fielded_role_presence[rows, _UNIT_ROLE[unit_ids]]
                raw_score = (
                    trait_match.astype(np.float32) * 100.0
                    + role_need.astype(np.float32) * 20.0
                    + _UNIT_POWER[unit_ids]
                    + _UNIT_COST[unit_ids].astype(np.float32)
                )
            elif mode == "highest_cost":
                raw_score = _UNIT_COST[unit_ids].astype(np.float32) * 100.0 + _UNIT_POWER[unit_ids]
            else:
                raise ValueError(f"unknown buy mode: {mode}")
            tie_break = shop_index * 1.0e-3 + unit_ids.astype(np.float32) * 1.0e-6
            score[:, shop_index] = np.where(valid, raw_score + tie_break, -np.inf)

        best_shop = np.argmax(score, axis=1).astype(np.int64)
        has_candidate = np.isfinite(score[np.arange(index_array.shape[0]), best_shop])
        best_shop = np.where(has_candidate, best_shop, -1).astype(np.int64)
        unit_ids = np.where(
            has_candidate,
            shop_units[np.arange(index_array.shape[0]), np.maximum(best_shop, 0)],
            0,
        ).astype(np.int16)
        return best_shop, unit_ids

    def _strongest_field(
        self,
        indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int16]:
        index_array = self._indices(indices)
        score = np.full((index_array.shape[0], UNIT_COUNT), -np.inf, dtype=np.float32)
        unit_ids = np.arange(1, UNIT_COUNT + 1, dtype=np.int16)
        copies = self.owned[index_array][:, 1:]
        stars = self._star_levels(copies)
        score = _UNIT_POWER[1:][None, :] * _STAR_MULTIPLIERS[stars]
        score += _UNIT_COST[1:].astype(np.float32)[None, :] * 1.0e-3
        score -= unit_ids.astype(np.float32)[None, :] * 1.0e-6
        score = np.where(copies > 0, score, -np.inf)
        fielded = np.zeros((index_array.shape[0], self.config.max_level), dtype=np.int16)
        order = np.argsort(-score, axis=1)
        ranks = np.arange(self.config.max_level)
        allowed = ranks[None, :] < self.level[index_array, None]
        chosen = unit_ids[order[:, : self.config.max_level]]
        chosen_score = np.take_along_axis(score, order[:, : self.config.max_level], axis=1)
        fielded[:, :] = np.where(allowed & np.isfinite(chosen_score), chosen, 0).astype(np.int16)
        return fielded

    def _board_strength(self, indices: NDArray[np.int64]) -> NDArray[np.float32]:
        fielded = self.fielded[indices]
        owned = self.owned[indices]
        strength = np.zeros(indices.shape[0], dtype=np.float32)
        trait_counts = np.zeros((indices.shape[0], len(_TRAIT_BY_NAME)), dtype=np.int16)
        role_counts = np.zeros((indices.shape[0], ROLE_COUNT), dtype=np.int16)
        for slot in range(self.config.max_level):
            unit_ids = fielded[:, slot]
            present = unit_ids > 0
            if not bool(present.any()):
                continue
            copies = owned[np.arange(indices.shape[0]), unit_ids]
            stars = self._star_levels(copies)
            strength += np.where(present, _UNIT_POWER[unit_ids] * _STAR_MULTIPLIERS[stars], 0.0)
            rows = np.flatnonzero(present)
            np.add.at(trait_counts, (rows, _UNIT_TRAIT[unit_ids[present]]), 1)
            np.add.at(role_counts, (rows, _UNIT_ROLE[unit_ids[present]]), 1)

        strength += np.sum(
            np.where(trait_counts >= 3, 8.0, np.where(trait_counts >= 2, 3.0, 0.0)),
            axis=1,
        )
        strength += np.sum(
            np.where(role_counts > 0, self.role_item_slots[indices], 0).astype(np.float32)
            * _ROLE_ITEM_POWER[None, :],
            axis=1,
        )
        strength += np.where(
            (role_counts[:, ROLE_INDEX["tank"]] > 0) & (role_counts[:, ROLE_INDEX["carry"]] > 0),
            6.0,
            0.0,
        ).astype(np.float32)
        strength += np.where(role_counts.sum(axis=1) == 0, -12.0, 0.0).astype(np.float32)
        return np.maximum(0.0, strength).astype(np.float32)

    def _refresh_shop(self, indices: NDArray[np.int64]) -> None:
        for shop_index in range(self.config.shop_size):
            self.shop[indices, shop_index] = self._sample_unit_id(indices)

    def _sample_unit_id(self, indices: NDArray[np.int64]) -> NDArray[np.int16]:
        rolls = self._random_float(indices)
        levels = self.level[indices]
        low_level = levels <= 3
        mid_level = (levels > 3) & (levels <= 5)
        high_level = (levels > 5) & (levels <= 7)
        tier_one = np.where(
            low_level,
            0.72,
            np.where(mid_level, 0.48, np.where(high_level, 0.25, 0.12)),
        )
        tier_two = np.where(
            low_level,
            0.26,
            np.where(mid_level, 0.42, np.where(high_level, 0.50, 0.43)),
        )
        cost = np.where(rolls < tier_one, 1, np.where(rolls < tier_one + tier_two, 2, 3))
        draws = self._random_int(indices, 3)
        sampled = np.empty(indices.shape[0], dtype=np.int16)
        for cost_value, unit_ids in _UNITS_BY_COST.items():
            selected = cost == cost_value
            sampled[selected] = unit_ids[draws[selected]]
        return sampled

    def _maybe_drop_role_item(self, indices: NDArray[np.int64]) -> None:
        selected = indices[self.round[indices] % self.config.item_drop_interval == 0]
        if not selected.size:
            return
        role_indices = self._random_int(selected, ROLE_COUNT)
        self.role_items[selected, role_indices] += 1

    def _normal_noise(self, indices: NDArray[np.int64]) -> NDArray[np.float32]:
        if self.config.combat_noise <= 0:
            return np.zeros(indices.shape[0], dtype=np.float32)
        return (
            (
                self._random_float(indices)
                + self._random_float(indices)
                + self._random_float(indices)
            )
            - 1.5
        ).astype(np.float32) * (self.config.combat_noise * 1.6)

    def _random_int(self, indices: NDArray[np.int64], limit: int) -> NDArray[np.int64]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return (self._next_u64(indices) % np.uint64(limit)).astype(np.int64)

    def _random_float(self, indices: NDArray[np.int64]) -> NDArray[np.float64]:
        return ((self._next_u64(indices) >> np.uint64(11)).astype(np.float64)) * (
            1.0 / float(1 << 53)
        )

    def _next_u64(self, indices: NDArray[np.int64]) -> NDArray[np.uint64]:
        self.rng_key[indices] = self.rng_key[indices] * _LCG_MULTIPLIER + _LCG_INCREMENT
        return self.rng_key[indices]

    def _owned_trait_presence(
        self,
        indices: NDArray[np.int64],
    ) -> NDArray[np.bool_]:
        presence = np.zeros((indices.shape[0], len(_TRAIT_BY_NAME)), dtype=np.bool_)
        owned = self.owned[indices]
        for unit_id in range(1, UNIT_COUNT + 1):
            presence[:, _UNIT_TRAIT[unit_id]] |= owned[:, unit_id] > 0
        return presence

    def _fielded_role_presence(
        self,
        indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.bool_]:
        index_array = self._indices(indices)
        presence = np.zeros((index_array.shape[0], ROLE_COUNT), dtype=np.bool_)
        fielded = self.fielded[index_array]
        for slot in range(self.config.max_level):
            unit_ids = fielded[:, slot]
            present = unit_ids > 0
            if bool(present.any()):
                presence[present, _UNIT_ROLE[unit_ids[present]]] = True
        return presence

    def _enemy_strength(self, rounds: NDArray[np.int64]) -> NDArray[np.float32]:
        round_float = np.maximum(1, rounds).astype(np.float32)
        base = 13.0 + round_float * 3.2 + np.power(round_float, 1.18) * 1.35
        return (base * self.config.enemy_strength_multiplier).astype(np.float32)

    def _damage_from_margin(
        self,
        rounds: NDArray[np.int64],
        enemy_strength: NDArray[np.float32],
        board: NDArray[np.float32],
    ) -> NDArray[np.int16]:
        base = np.where(rounds < 8, 2, np.where(rounds < 16, 4, np.where(rounds < 24, 6, 8)))
        return (base + np.maximum(0.0, enemy_strength - board) / 22.0).astype(np.int16)

    def _info(
        self,
        env_index: int,
        legal: bool,
        ended_round: bool,
        action_mask: NDArray[np.bool_],
    ) -> dict[str, Any]:
        return {
            "action_mask": action_mask[env_index],
            "legal_action": legal,
            "ended_round": ended_round,
            "round": int(self.round[env_index]),
            "hp": int(self.hp[env_index]),
            "gold": int(self.gold[env_index]),
            "level": int(self.level[env_index]),
            "board_strength": float(self.last_board_strength[env_index]),
            "enemy_strength": float(self.last_enemy_strength[env_index]),
            "placement_proxy": int(self.placement_proxy()[env_index]),
            "scenario_score": float(self.scenario_score()[env_index]),
            "final_reason": self.final_reason(env_index),
        }

    def _indices(
        self,
        indices: NDArray[np.integer[Any]] | list[int] | tuple[int, ...] | None,
    ) -> NDArray[np.int64]:
        if indices is None:
            return np.arange(self.num_envs, dtype=np.int64)
        return np.asarray(indices, dtype=np.int64).reshape(-1)

    @staticmethod
    def _star_levels(copies: NDArray[np.integer[Any]]) -> NDArray[np.int64]:
        return np.where(
            copies >= 9,
            3,
            np.where(copies >= 3, 2, np.where(copies >= 1, 1, 0)),
        ).astype(
            np.int64,
        )

    @staticmethod
    def _xp_to_next_level(levels: NDArray[np.integer[Any]]) -> NDArray[np.int16]:
        return (2 + np.maximum(0, levels.astype(np.int16) - 1) * 2).astype(np.int16)


def scalar_enemy_strength(round_num: int, config: StrategicConfig) -> float:
    """Expose the scalar enemy formula for parity-focused diagnostics."""

    return enemy_strength_for_round(round_num, config)
