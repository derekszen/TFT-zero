"""All-player action stepping for the Set-1 lobby layer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import (
    BUY_SHOP_OFFSET,
    NUM_ACTIONS,
    SELL_BENCH_OFFSET,
    Action,
    action_name,
    decode_move_bench_to_board_action,
    decode_move_board_to_bench_action,
    is_move_bench_to_board_action,
    is_move_board_to_bench_action,
)
from mini_tft.core.board import field_best_board
from mini_tft.core.config import EnvConfig
from mini_tft.core.economy import apply_xp
from mini_tft.core.items import slam_best_item
from mini_tft.core.lobby import (
    LobbyCombatResult,
    Set1LobbyState,
    buy_shop_unit,
    lobby_action_mask,
    refresh_player_shop,
    resolve_lobby_round,
    sell_bench_unit,
)
from mini_tft.core.set_data import GameData
from mini_tft.core.state import GameState

PlayerOrderMode = Literal["fixed", "random", "rotating"]
LobbyPolicy = Callable[
    [
        int,
        Set1LobbyState,
        NDArray[np.bool_],
        GameData,
        EnvConfig,
        np.random.Generator,
    ],
    int,
]


@dataclass(frozen=True)
class LobbyActionRecord:
    """One attempted player action during a synchronized lobby round."""

    player_id: int
    action: int
    name: str
    legal: bool
    applied: bool
    forced: bool = False


@dataclass(frozen=True)
class LobbyRoundStepResult:
    """Summary from one all-player action phase plus combat resolution."""

    round: int
    player_order: tuple[int, ...]
    actions: tuple[LobbyActionRecord, ...]
    combat_results: tuple[LobbyCombatResult, ...]
    forced_players: tuple[int, ...]
    illegal_actions: int

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def resolved_fights(self) -> int:
        return len(self.combat_results)


def lobby_legal_action_mask(
    state: Set1LobbyState,
    player_id: int,
    data: GameData,
    config: EnvConfig,
) -> NDArray[np.bool_]:
    """Return a MiniTFT action mask with shared-pool availability applied."""

    return lobby_action_mask(state, player_id, data, config)


def apply_lobby_action(
    state: Set1LobbyState,
    player_id: int,
    action: int,
    data: GameData,
    config: EnvConfig,
    rng: np.random.Generator,
) -> bool:
    """Apply one legal MiniTFT action to one lobby player."""

    action = int(action)
    if not _is_action_legal(state, player_id, action, data, config):
        return False

    player = state.players[player_id]
    if action == Action.END_TURN:
        return True
    if action == Action.ROLL:
        player.gold -= config.roll_cost
        refresh_player_shop(state, player_id, data, config, rng)
        player.total_rolls += 1
        return True
    if action == Action.BUY_XP:
        previous_level = player.level
        player.gold -= config.xp_buy_cost
        player.level, player.xp = apply_xp(
            player.level,
            player.xp,
            config.xp_per_buy,
            config.max_level,
        )
        player.total_xp_buys += 1
        return player.level >= previous_level
    if Action.BUY_SHOP_0 <= action <= Action.BUY_SHOP_4:
        return buy_shop_unit(state, player_id, action - BUY_SHOP_OFFSET, data)
    if Action.SELL_BENCH_0 <= action <= Action.SELL_BENCH_8:
        return sell_bench_unit(state, player_id, action - SELL_BENCH_OFFSET, data)
    if action == Action.FIELD_BEST_BOARD:
        return field_best_board(player, data, config)
    if action == Action.SLAM_BEST_ITEM:
        return slam_best_item(player, data, config)
    if is_move_bench_to_board_action(action):
        bench_index, board_index = decode_move_bench_to_board_action(action)
        return _move_bench_to_board(state, player_id, bench_index, board_index)
    if is_move_board_to_bench_action(action):
        board_index, bench_index = decode_move_board_to_bench_action(action)
        return _move_board_to_bench(state, player_id, board_index, bench_index)
    return False


def step_lobby_round_with_policies(
    state: Set1LobbyState,
    data: GameData,
    config: EnvConfig,
    rng: np.random.Generator,
    policy: LobbyPolicy,
    *,
    max_actions_per_player: int | None = None,
    player_order: Sequence[int] | None = None,
    player_order_mode: PlayerOrderMode = "fixed",
    rotation_offset: int = 0,
    matchups: list[tuple[int, int]] | None = None,
) -> LobbyRoundStepResult:
    """Let every active player act, then resolve one lobby combat round."""

    round_number = state.round
    if state.done:
        return LobbyRoundStepResult(
            round=round_number,
            player_order=(),
            actions=(),
            combat_results=(),
            forced_players=(),
            illegal_actions=0,
        )

    budget = (
        config.max_actions_per_round
        if max_actions_per_player is None
        else max_actions_per_player
    )
    budget = max(0, int(budget))
    order = (
        tuple(player_order)
        if player_order is not None
        else ordered_active_players(state, rng, player_order_mode, rotation_offset)
    )
    actions: list[LobbyActionRecord] = []
    forced_players: list[int] = []
    illegal_actions = 0

    for player_id in order:
        if player_id < 0 or player_id >= state.player_count:
            continue
        player = state.players[player_id]
        if player.done:
            continue

        actions_taken = 0
        while actions_taken < budget:
            mask = lobby_legal_action_mask(state, player_id, data, config)
            chosen_action = int(policy(player_id, state, mask, data, config, rng))
            legal = 0 <= chosen_action < NUM_ACTIONS and bool(mask[chosen_action])

            if not legal:
                player.total_illegal_actions += 1
                illegal_actions += 1
                actions.append(
                    LobbyActionRecord(
                        player_id=player_id,
                        action=chosen_action,
                        name=action_name(chosen_action),
                        legal=False,
                        applied=False,
                    )
                )
                break

            if chosen_action == Action.END_TURN:
                actions.append(
                    LobbyActionRecord(
                        player_id=player_id,
                        action=chosen_action,
                        name=action_name(chosen_action),
                        legal=True,
                        applied=True,
                    )
                )
                break

            applied = apply_lobby_action(
                state,
                player_id,
                chosen_action,
                data,
                config,
                rng,
            )
            if applied:
                player.round_action_count += 1
                actions_taken += 1
            else:
                player.total_illegal_actions += 1
                illegal_actions += 1

            actions.append(
                LobbyActionRecord(
                    player_id=player_id,
                    action=chosen_action,
                    name=action_name(chosen_action),
                    legal=True,
                    applied=applied,
                )
            )

            if not applied:
                break
            if player.round_action_count >= budget:
                forced_players.append(player_id)
                actions.append(
                    LobbyActionRecord(
                        player_id=player_id,
                        action=int(Action.END_TURN),
                        name=action_name(int(Action.END_TURN)),
                        legal=True,
                        applied=True,
                        forced=True,
                    )
                )
                break

    combat_results = resolve_lobby_round(state, data, config, rng, matchups=matchups)
    return LobbyRoundStepResult(
        round=round_number,
        player_order=order,
        actions=tuple(actions),
        combat_results=tuple(combat_results),
        forced_players=tuple(forced_players),
        illegal_actions=illegal_actions,
    )


def ordered_active_players(
    state: Set1LobbyState,
    rng: np.random.Generator,
    mode: PlayerOrderMode = "fixed",
    rotation_offset: int = 0,
) -> tuple[int, ...]:
    """Return active player order for a synchronized action phase."""

    active = tuple(state.active_player_ids())
    if len(active) <= 1 or mode == "fixed":
        return active
    if mode == "random":
        return tuple(int(player_id) for player_id in rng.permutation(np.asarray(active)))
    if mode == "rotating":
        offset = rotation_offset % len(active)
        return active[offset:] + active[:offset]
    raise ValueError(f"unknown player order mode: {mode}")


def first_legal_action_policy(
    _player_id: int,
    _state: Set1LobbyState,
    mask: NDArray[np.bool_],
    _data: GameData,
    _config: EnvConfig,
    _rng: np.random.Generator,
) -> int:
    """Tiny deterministic policy for smoke tests."""

    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return int(Action.END_TURN)
    return int(legal[0])


def random_lobby_policy(
    _player_id: int,
    _state: Set1LobbyState,
    mask: NDArray[np.bool_],
    _data: GameData,
    _config: EnvConfig,
    rng: np.random.Generator,
) -> int:
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return int(Action.END_TURN)
    return int(rng.choice(legal))


def fast_level_lobby_policy(
    player_id: int,
    state: Set1LobbyState,
    mask: NDArray[np.bool_],
    data: GameData,
    _config: EnvConfig,
    _rng: np.random.Generator,
) -> int:
    """Lobby policy mirroring `FastLevelBot` on one synchronized player state."""

    player = state.players[player_id]
    if mask[Action.SLAM_BEST_ITEM]:
        return int(Action.SLAM_BEST_ITEM)
    if mask[Action.FIELD_BEST_BOARD]:
        return int(Action.FIELD_BEST_BOARD)
    if mask[Action.BUY_XP] and (player.level < 6 or player.gold >= 50):
        return int(Action.BUY_XP)
    buy = _highest_cost_buy(player, mask, data)
    if buy is not None:
        return buy
    return int(Action.END_TURN)


def tempo_lobby_policy(
    player_id: int,
    state: Set1LobbyState,
    mask: NDArray[np.bool_],
    data: GameData,
    _config: EnvConfig,
    _rng: np.random.Generator,
) -> int:
    """Simple policy that buys strong units, fields board, then ends."""

    player = state.players[player_id]
    buy = _highest_cost_buy(player, mask, data)
    if buy is not None:
        return buy
    if mask[Action.FIELD_BEST_BOARD]:
        return int(Action.FIELD_BEST_BOARD)
    if mask[Action.SLAM_BEST_ITEM]:
        return int(Action.SLAM_BEST_ITEM)
    if mask[Action.BUY_XP] and player.gold >= 12:
        return int(Action.BUY_XP)
    return int(Action.END_TURN)


def mixed_lobby_policy(
    player_id: int,
    state: Set1LobbyState,
    mask: NDArray[np.bool_],
    data: GameData,
    config: EnvConfig,
    rng: np.random.Generator,
) -> int:
    """Deterministic mix of baseline lobby policies for broader opponent coverage."""

    policy_index = (player_id + state.round) % 3
    if policy_index == 0:
        return tempo_lobby_policy(player_id, state, mask, data, config, rng)
    if policy_index == 1:
        return fast_level_lobby_policy(player_id, state, mask, data, config, rng)
    return random_lobby_policy(player_id, state, mask, data, config, rng)


def _highest_cost_buy(
    player_state: GameState,
    mask: NDArray[np.bool_],
    data: GameData,
) -> int | None:
    candidates: list[tuple[float, int]] = []
    for shop_index, unit_id in enumerate(player_state.shop):
        action = BUY_SHOP_OFFSET + shop_index
        if mask[action] and unit_id in data.units:
            unit = data.units[unit_id]
            candidates.append((unit.cost * 100.0 + unit.base_power, action))
    if not candidates:
        return None
    return max(candidates)[1]


def _is_action_legal(
    state: Set1LobbyState,
    player_id: int,
    action: int,
    data: GameData,
    config: EnvConfig,
) -> bool:
    mask = lobby_legal_action_mask(state, player_id, data, config)
    return 0 <= action < len(mask) and bool(mask[action])


def _move_bench_to_board(
    state: Set1LobbyState,
    player_id: int,
    bench_index: int,
    board_index: int,
) -> bool:
    player = state.players[player_id]
    unit = player.bench[bench_index]
    if unit is None:
        return False
    target = player.board[board_index]
    board_count = sum(slot is not None for slot in player.board)
    if target is None and board_count >= player.level:
        return False
    player.bench[bench_index], player.board[board_index] = target, unit
    return True


def _move_board_to_bench(
    state: Set1LobbyState,
    player_id: int,
    board_index: int,
    bench_index: int,
) -> bool:
    player = state.players[player_id]
    unit = player.board[board_index]
    if unit is None:
        return False
    player.board[board_index], player.bench[bench_index] = player.bench[bench_index], unit
    return True
