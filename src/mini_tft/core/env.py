"""Gymnasium-compatible V0 simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from mini_tft.core.actions import (
    BUY_SHOP_OFFSET,
    NUM_ACTIONS,
    SELL_BENCH_OFFSET,
    Action,
    decode_move_bench_to_board_action,
    decode_move_board_to_bench_action,
    is_move_bench_to_board_action,
    is_move_board_to_bench_action,
)
from mini_tft.core.board import field_best_board
from mini_tft.core.combat import CombatResult, board_strength, resolve_combat
from mini_tft.core.config import EnvConfig
from mini_tft.core.economy import apply_xp, income_after_combat, sell_value
from mini_tft.core.featurize import OBS_CLIP_HIGH, OBS_CLIP_LOW, featurize_state, observation_dim
from mini_tft.core.ids import EMPTY
from mini_tft.core.items import maybe_drop_item, slam_best_item
from mini_tft.core.masks import legal_action_mask
from mini_tft.core.rewards import action_reward, end_turn_reward, illegal_action_reward
from mini_tft.core.rounds import round_info
from mini_tft.core.set_data import GameData, load_set
from mini_tft.core.shop import sample_shop
from mini_tft.core.state import GameState, UnitInstance, new_game_state
from mini_tft.core.upgrades import auto_combine

if TYPE_CHECKING:
    from mini_tft.fight_model.simulator_adapter import FightValueCombatModel


class MiniTFTEnv(gym.Env[NDArray[np.float32], int]):
    """Single-player Set-1-like abstract TFT simulator."""

    metadata = {"render_modes": ["text", "ansi"]}

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.data: GameData = load_set(self.config.dataset)
        self.rng = np.random.default_rng(self.config.seed)
        self.state: GameState | None = None
        self.fight_value_model: FightValueCombatModel | None = self._load_fight_value_model()
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=OBS_CLIP_LOW,
            high=OBS_CLIP_HIGH,
            shape=(observation_dim(self.data, self.config),),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        actual_seed = self.config.seed if seed is None else seed
        if actual_seed is None:
            actual_seed = int(np.random.SeedSequence().entropy)
        self.rng = np.random.default_rng(actual_seed)
        shop = sample_shop(self.data, self.config.starting_level, self.config.shop_size, self.rng)
        self.state = new_game_state(self.config, actual_seed, shop)
        obs = self._observe()
        return obs, self._info()

    def step(
        self,
        action: int,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        if state.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        action = int(action)
        mask = self.action_masks()
        legal = 0 <= action < len(mask) and bool(mask[action])
        auto_end_turn = False
        reward = action_reward(action == Action.END_TURN)

        if legal:
            reward += self._apply_action(action)
        else:
            state.total_illegal_actions += 1
            reward += illegal_action_reward()

        if action != Action.END_TURN and not state.done:
            state.round_action_count += 1
            if state.round_action_count >= self.config.max_actions_per_round:
                auto_end_turn = True
                reward -= 0.05
                reward += self._end_turn()

        state.step_count += 1
        if state.step_count >= self.config.max_steps_per_episode and not state.done:
            state.done = True
            state.final_reason = "max_steps"

        terminated = state.done and state.final_reason in {"hp_zero", "max_round"}
        truncated = state.done and state.final_reason == "max_steps"
        return (
            self._observe(),
            float(reward),
            terminated,
            truncated,
            self._info(legal_action=legal, auto_end_turn=auto_end_turn),
        )

    def action_masks(self) -> NDArray[np.bool_]:
        state = self._require_state()
        return legal_action_mask(state, self.data, self.config)

    def render(self) -> str:
        from mini_tft.tools.render_text import render_state

        return render_state(self)

    def episode_summary(self) -> dict[str, int | float | bool | str | None]:
        state = self._require_state()
        return {
            "final_hp": state.hp,
            "survived_round": min(state.round, self.config.max_round),
            "survival_rate": state.round > self.config.max_round,
            "final_board_strength": self._board_value(state),
            "total_rolls": state.total_rolls,
            "total_xp_buys": state.total_xp_buys,
            "total_units_bought": state.total_units_bought,
            "total_units_sold": state.total_units_sold,
            "total_item_slams": state.total_item_slams,
            "total_illegal_actions": state.total_illegal_actions,
            "final_reason": state.final_reason,
        }

    def _apply_action(self, action: int) -> float:
        if action == Action.END_TURN:
            return self._end_turn()
        if action == Action.ROLL:
            return self._roll()
        if action == Action.BUY_XP:
            return self._buy_xp()
        if Action.BUY_SHOP_0 <= action <= Action.BUY_SHOP_4:
            return self._buy_shop(action - BUY_SHOP_OFFSET)
        if Action.SELL_BENCH_0 <= action <= Action.SELL_BENCH_8:
            return self._sell_bench(action - SELL_BENCH_OFFSET)
        if action == Action.FIELD_BEST_BOARD:
            return 0.08 if field_best_board(self._require_state(), self.data, self.config) else 0.0
        if action == Action.SLAM_BEST_ITEM:
            return 0.02 if slam_best_item(self._require_state(), self.data, self.config) else 0.0
        if is_move_bench_to_board_action(action):
            bench_index, board_index = decode_move_bench_to_board_action(action)
            return 0.05 if self._move_bench_to_board(bench_index, board_index) else 0.0
        if is_move_board_to_bench_action(action):
            board_index, bench_index = decode_move_board_to_bench_action(action)
            return 0.01 if self._move_board_to_bench(board_index, bench_index) else 0.0
        return 0.0

    def _roll(self) -> float:
        state = self._require_state()
        state.gold -= self.config.roll_cost
        state.shop = sample_shop(self.data, state.level, self.config.shop_size, self.rng)
        state.total_rolls += 1
        return 0.0

    def _buy_xp(self) -> float:
        state = self._require_state()
        previous_level = state.level
        state.gold -= self.config.xp_buy_cost
        state.level, state.xp = apply_xp(
            state.level,
            state.xp,
            self.config.xp_per_buy,
            self.config.max_level,
        )
        state.total_xp_buys += 1
        return 0.01 if state.level > previous_level else 0.0

    def _buy_shop(self, shop_index: int) -> float:
        state = self._require_state()
        unit_id = state.shop[shop_index]
        if unit_id == EMPTY:
            return 0.0
        bench_index = next(index for index, unit in enumerate(state.bench) if unit is None)
        state.gold -= self.data.units[unit_id].cost
        state.bench[bench_index] = UnitInstance(unit_id=unit_id)
        state.shop[shop_index] = EMPTY
        state.total_units_bought += 1
        auto_combine(state)
        return 0.06

    def _sell_bench(self, bench_index: int) -> float:
        state = self._require_state()
        unit = state.bench[bench_index]
        if unit is None:
            return 0.0
        state.gold += sell_value(unit, self.data)
        state.bench[bench_index] = None
        state.total_units_sold += 1
        return -0.01

    def _move_bench_to_board(self, bench_index: int, board_index: int) -> bool:
        state = self._require_state()
        unit = state.bench[bench_index]
        if unit is None:
            return False
        target = state.board[board_index]
        if target is None and sum(slot is not None for slot in state.board) >= state.level:
            return False
        state.bench[bench_index], state.board[board_index] = target, unit
        return True

    def _move_board_to_bench(self, board_index: int, bench_index: int) -> bool:
        state = self._require_state()
        unit = state.board[board_index]
        if unit is None:
            return False
        state.board[board_index], state.bench[bench_index] = state.bench[bench_index], unit
        return True

    def _end_turn(self) -> float:
        state = self._require_state()
        combat_round = state.round
        previous_strength = state.last_board_strength
        result = self._resolve_combat(state)
        state.last_board_strength = result.my_strength
        state.last_enemy_strength = result.enemy_strength
        state.last_win = result.won
        state.hp = max(0, state.hp - result.damage)
        state.gold += income_after_combat(state.gold, result.won, self.config)
        maybe_drop_item(state, self.data, self.config, self.rng)
        state.shop = sample_shop(self.data, state.level, self.config.shop_size, self.rng)

        survived_max_round = state.round >= self.config.max_round and state.hp > 0
        if state.hp <= 0:
            state.done = True
            state.final_reason = "hp_zero"
        elif survived_max_round:
            state.done = True
            state.final_reason = "max_round"
        else:
            state.round += 1

        board_units = sum(unit is not None for unit in state.board)
        reward = end_turn_reward(
            won=result.won,
            damage=result.damage,
            board_strength_delta=result.my_strength - previous_strength,
            terminated=state.done,
            survived_max_round=survived_max_round,
            hp=state.hp,
        )
        if combat_round > 1 and board_units == 0:
            reward -= 1.0
        elif board_units > 0:
            reward += min(result.my_strength * 0.002, 0.4)
        state.round_action_count = 0
        return reward

    def _observe(self) -> NDArray[np.float32]:
        return featurize_state(self._require_state(), self.data, self.config)

    def _info(
        self,
        legal_action: bool | None = None,
        auto_end_turn: bool = False,
    ) -> dict[str, Any]:
        state = self._require_state()
        current_round = round_info(state.round)
        info: dict[str, Any] = {
            "action_mask": self.action_masks(),
            "round": state.round,
            "stage": current_round.stage,
            "stage_round": current_round.stage_round,
            "stage_label": current_round.stage_label,
            "round_type": current_round.round_type,
            "is_pve_round": current_round.is_pve,
            "round_action_count": state.round_action_count,
            "hp": state.hp,
            "gold": state.gold,
            "level": state.level,
            "board_strength": self._board_value(state),
            "combat_model": self.config.combat_model,
            "final_reason": state.final_reason,
            "auto_end_turn": auto_end_turn,
        }
        if legal_action is not None:
            info["legal_action"] = legal_action
        return info

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Call reset() before using the environment.")
        return self.state

    def _resolve_combat(self, state: GameState) -> CombatResult:
        if self.config.combat_model == "abstract":
            return resolve_combat(state.board, state.round, self.data, self.config, self.rng)
        if self.config.combat_model == "fight_value":
            if self.fight_value_model is None:
                raise RuntimeError("fight_value combat selected without a loaded evaluator")
            from mini_tft.fight_model.simulator_adapter import resolve_combat_with_fight_value

            return resolve_combat_with_fight_value(
                state.board,
                state.round,
                self.data,
                self.config,
                self.rng,
                self.fight_value_model,
            )
        raise ValueError(f"unsupported combat_model: {self.config.combat_model}")

    def _board_value(self, state: GameState) -> float:
        if self.config.combat_model == "fight_value" and self.fight_value_model is not None:
            return self.fight_value_model.predict_mini_board(
                state.board,
                state.round,
                self.data,
                self.config,
            ).learned_strength
        return board_strength(state.board, self.data).strength

    def _load_fight_value_model(self) -> FightValueCombatModel | None:
        if self.config.combat_model == "abstract":
            return None
        if self.config.combat_model != "fight_value":
            raise ValueError(f"unsupported combat_model: {self.config.combat_model}")
        if not self.config.fight_value_checkpoint:
            raise ValueError("fight_value combat_model requires fight_value_checkpoint")
        from mini_tft.fight_model.simulator_adapter import FightValueCombatModel

        evaluator = FightValueCombatModel(
            self.config.fight_value_checkpoint,
            device_name=self.config.fight_value_device,
        )
        if evaluator.metatft_unit_id_lookup is not None:
            raise ValueError(
                "MetaTFT current-patch fight value checkpoints cannot be used with "
                "MiniTFTEnv until the simulator state uses the same unit namespace"
            )
        return evaluator
