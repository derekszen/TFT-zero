"""Set-1-like multi-player lobby layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mini_tft.core.actions import BUY_SHOP_OFFSET
from mini_tft.core.combat import base_damage_by_round, board_strength
from mini_tft.core.config import EnvConfig
from mini_tft.core.economy import income_after_combat, sell_value
from mini_tft.core.ids import EMPTY
from mini_tft.core.items import maybe_drop_item
from mini_tft.core.masks import legal_action_mask
from mini_tft.core.pool import SharedUnitPool, sample_shop_from_pool
from mini_tft.core.set_data import GameData
from mini_tft.core.state import GameState, UnitInstance, new_game_state, state_signature
from mini_tft.core.upgrades import auto_combine

DEFAULT_LOBBY_PLAYERS = 8


@dataclass(frozen=True)
class LobbyCombatResult:
    round: int
    player_a: int
    player_b: int
    strength_a: float
    strength_b: float
    p_a_win: float
    winner: int
    loser: int
    damage: int


@dataclass
class LobbyPlayerSummary:
    player_id: int
    placement: int
    final_hp: int
    survived_round: int
    final_board_strength: float
    final_reason: str | None
    total_rolls: int
    total_xp_buys: int
    total_units_bought: int
    total_units_sold: int
    total_item_slams: int
    total_illegal_actions: int


@dataclass
class Set1LobbyState:
    """Synchronized state for a simplified 8-player TFT lobby."""

    seed: int
    players: list[GameState]
    pool: SharedUnitPool
    round: int = 1
    placements: dict[int, int] = field(default_factory=dict)
    eliminated_order: list[int] = field(default_factory=list)
    last_matchups: list[tuple[int, int]] = field(default_factory=list)
    combat_log: list[LobbyCombatResult] = field(default_factory=list)
    done: bool = False
    final_reason: str | None = None

    @property
    def player_count(self) -> int:
        return len(self.players)

    def active_player_ids(self) -> list[int]:
        return [index for index, player in enumerate(self.players) if not player.done]

    def clone(self) -> Set1LobbyState:
        return clone_lobby_state(self)


def new_lobby_state(
    config: EnvConfig,
    data: GameData,
    *,
    seed: int,
    player_count: int = DEFAULT_LOBBY_PLAYERS,
    pool: SharedUnitPool | None = None,
) -> Set1LobbyState:
    rng = np.random.default_rng(seed)
    shared_pool = pool.clone() if pool is not None else SharedUnitPool.from_data(data)
    players = [
        new_game_state(
            config,
            seed + player_id,
            sample_shop_from_pool(data, shared_pool, config.starting_level, config.shop_size, rng),
        )
        for player_id in range(player_count)
    ]
    return Set1LobbyState(seed=seed, players=players, pool=shared_pool)


def clone_lobby_state(state: Set1LobbyState) -> Set1LobbyState:
    return Set1LobbyState(
        seed=state.seed,
        players=[player.clone() for player in state.players],
        pool=state.pool.clone(),
        round=state.round,
        placements=dict(state.placements),
        eliminated_order=list(state.eliminated_order),
        last_matchups=list(state.last_matchups),
        combat_log=list(state.combat_log),
        done=state.done,
        final_reason=state.final_reason,
    )


def restore_lobby_state(target: Set1LobbyState, source: Set1LobbyState) -> None:
    """Overwrite `target` with a deep copy of `source`."""

    clone = clone_lobby_state(source)
    target.seed = clone.seed
    target.players = clone.players
    target.pool = clone.pool
    target.round = clone.round
    target.placements = clone.placements
    target.eliminated_order = clone.eliminated_order
    target.last_matchups = clone.last_matchups
    target.combat_log = clone.combat_log
    target.done = clone.done
    target.final_reason = clone.final_reason


def lobby_signature(state: Set1LobbyState) -> tuple[Any, ...]:
    return (
        state.seed,
        state.round,
        tuple(state_signature(player) for player in state.players),
        state.pool.signature(),
        tuple(sorted(state.placements.items())),
        tuple(state.eliminated_order),
        tuple(state.last_matchups),
        tuple(state.combat_log),
        state.done,
        state.final_reason,
    )


def lobby_action_mask(
    state: Set1LobbyState,
    player_id: int,
    data: GameData,
    config: EnvConfig,
) -> NDArray[np.bool_]:
    """Return a single player's legal action mask with shared-pool buy checks."""

    player = state.players[player_id]
    mask = legal_action_mask(player, data, config)
    if player.done:
        return mask

    for shop_index, unit_id in enumerate(player.shop):
        action_index = BUY_SHOP_OFFSET + shop_index
        if unit_id != EMPTY and not state.pool.can_take(unit_id):
            mask[action_index] = False
    return mask


def buy_shop_unit(
    state: Set1LobbyState,
    player_id: int,
    shop_index: int,
    data: GameData,
) -> bool:
    player = state.players[player_id]
    if player.done or not 0 <= shop_index < len(player.shop):
        return False

    unit_id = player.shop[shop_index]
    if unit_id == EMPTY:
        return False
    unit_cost = data.units[unit_id].cost
    if player.gold < unit_cost or not any(unit is None for unit in player.bench):
        return False
    if not state.pool.take(unit_id):
        return False

    bench_index = next(index for index, unit in enumerate(player.bench) if unit is None)
    player.gold -= unit_cost
    player.bench[bench_index] = UnitInstance(unit_id=unit_id)
    player.shop[shop_index] = EMPTY
    player.total_units_bought += 1
    auto_combine(player)
    return True


def sell_bench_unit(
    state: Set1LobbyState,
    player_id: int,
    bench_index: int,
    data: GameData,
) -> bool:
    player = state.players[player_id]
    if player.done or not 0 <= bench_index < len(player.bench):
        return False

    unit = player.bench[bench_index]
    if unit is None:
        return False
    player.gold += sell_value(unit, data)
    state.pool.return_unit(unit)
    player.bench[bench_index] = None
    player.total_units_sold += 1
    return True


def refresh_player_shop(
    state: Set1LobbyState,
    player_id: int,
    data: GameData,
    config: EnvConfig,
    rng: np.random.Generator,
) -> None:
    player = state.players[player_id]
    if not player.done:
        player.shop = sample_shop_from_pool(data, state.pool, player.level, config.shop_size, rng)


def pair_active_players(
    state: Set1LobbyState,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    active = state.active_player_ids()
    if len(active) <= 1:
        return []
    shuffled = list(rng.permutation(np.asarray(active, dtype=np.int64)))
    return [
        (int(shuffled[index]), int(shuffled[index + 1]))
        for index in range(0, len(shuffled) - 1, 2)
    ]


def resolve_lobby_round(
    state: Set1LobbyState,
    data: GameData,
    config: EnvConfig,
    rng: np.random.Generator,
    *,
    matchups: list[tuple[int, int]] | None = None,
) -> list[LobbyCombatResult]:
    """Resolve one synchronized lobby combat round with scalar board-vs-board fights."""

    if state.done:
        return []

    pairs = matchups if matchups is not None else pair_active_players(state, rng)
    state.last_matchups = list(pairs)
    results: list[LobbyCombatResult] = []

    for player_a, player_b in pairs:
        if state.players[player_a].done or state.players[player_b].done:
            continue
        result = _resolve_pair(state, player_a, player_b, data, config, rng)
        state.combat_log.append(result)
        results.append(result)

    for player_id in state.active_player_ids():
        refresh_player_shop(state, player_id, data, config, rng)
        state.players[player_id].round_action_count = 0

    _sync_round_and_done(state, data, config)
    return results


def eliminate_player(state: Set1LobbyState, player_id: int) -> None:
    player = state.players[player_id]
    if player.done:
        return
    _return_owned_units(state, player)
    player.done = True
    player.final_reason = "eliminated"
    player.hp = max(0, player.hp)
    placement = state.player_count - len(state.eliminated_order)
    state.placements[player_id] = placement
    state.eliminated_order.append(player_id)


def lobby_player_summary(
    state: Set1LobbyState,
    player_id: int,
    data: GameData,
) -> LobbyPlayerSummary:
    player = state.players[player_id]
    placement = state.placements.get(player_id, state.player_count)
    visible_strength = board_strength(player.board, data).strength
    return LobbyPlayerSummary(
        player_id=player_id,
        placement=placement,
        final_hp=player.hp,
        survived_round=min(player.round, state.round),
        final_board_strength=max(visible_strength, player.last_board_strength),
        final_reason=player.final_reason,
        total_rolls=player.total_rolls,
        total_xp_buys=player.total_xp_buys,
        total_units_bought=player.total_units_bought,
        total_units_sold=player.total_units_sold,
        total_item_slams=player.total_item_slams,
        total_illegal_actions=player.total_illegal_actions,
    )


def _resolve_pair(
    state: Set1LobbyState,
    player_a: int,
    player_b: int,
    data: GameData,
    config: EnvConfig,
    rng: np.random.Generator,
) -> LobbyCombatResult:
    left = state.players[player_a]
    right = state.players[player_b]
    left_stats = board_strength(left.board, data)
    right_stats = board_strength(right.board, data)
    strength_a = max(0.0, left_stats.strength - right_stats.enemy_power_penalty)
    strength_b = max(0.0, right_stats.strength - left_stats.enemy_power_penalty)
    diff = strength_a - strength_b
    p_a_win = 1.0 / (1.0 + exp(-(diff / config.combat_sigmoid_scale)))
    a_won = bool(rng.random() < p_a_win)
    winner = player_a if a_won else player_b
    loser = player_b if a_won else player_a
    damage = base_damage_by_round(state.round) + int(abs(diff) / 20.0)

    loser_state = state.players[loser]
    loser_state.hp = max(0, loser_state.hp - damage)
    left.last_board_strength = strength_a
    left.last_enemy_strength = strength_b
    left.last_win = a_won
    right.last_board_strength = strength_b
    right.last_enemy_strength = strength_a
    right.last_win = not a_won
    left.gold += income_after_combat(left.gold, a_won, config)
    right.gold += income_after_combat(right.gold, not a_won, config)
    maybe_drop_item(left, data, config, rng)
    maybe_drop_item(right, data, config, rng)

    if loser_state.hp <= 0:
        eliminate_player(state, loser)

    return LobbyCombatResult(
        round=state.round,
        player_a=player_a,
        player_b=player_b,
        strength_a=strength_a,
        strength_b=strength_b,
        p_a_win=p_a_win,
        winner=winner,
        loser=loser,
        damage=damage,
    )


def _sync_round_and_done(state: Set1LobbyState, data: GameData, config: EnvConfig) -> None:
    active = state.active_player_ids()
    if len(active) <= 1:
        state.done = True
        state.final_reason = "one_player_remaining"
        if active:
            winner = active[0]
            state.players[winner].done = True
            state.players[winner].final_reason = "winner"
            state.placements[winner] = 1
        return

    if state.round >= config.max_round:
        state.done = True
        state.final_reason = "max_round"
        _assign_remaining_placements_by_standings(state, data)
        return

    state.round += 1
    for player in state.players:
        if not player.done:
            player.round = state.round


def _assign_remaining_placements_by_standings(state: Set1LobbyState, data: GameData) -> None:
    remaining = sorted(
        state.active_player_ids(),
        key=lambda player_id: (
            state.players[player_id].hp,
            board_strength(state.players[player_id].board, data).strength,
            -player_id,
        ),
        reverse=True,
    )
    for placement, player_id in enumerate(remaining, start=1):
        state.placements[player_id] = placement
        state.players[player_id].done = True
        state.players[player_id].final_reason = "max_round"


def _return_owned_units(state: Set1LobbyState, player: GameState) -> None:
    for slots in (player.board, player.bench):
        for index, unit in enumerate(slots):
            if unit is not None:
                state.pool.return_unit(unit)
                slots[index] = None
