"""Serializable V0 game state."""

from __future__ import annotations

from dataclasses import dataclass, field

from mini_tft.core.config import EnvConfig


@dataclass
class UnitInstance:
    unit_id: int
    stars: int = 1
    items: list[int] = field(default_factory=list)

    def clone(self) -> UnitInstance:
        return UnitInstance(unit_id=self.unit_id, stars=self.stars, items=list(self.items))


@dataclass
class GameState:
    seed: int
    round: int
    hp: int
    gold: int
    level: int
    xp: int
    shop: list[int]
    bench: list[UnitInstance | None]
    board: list[UnitInstance | None]
    item_bench: list[int]
    done: bool = False
    step_count: int = 0
    round_action_count: int = 0
    final_reason: str | None = None
    last_board_strength: float = 0.0
    last_enemy_strength: float = 0.0
    last_win: bool = False
    total_rolls: int = 0
    total_xp_buys: int = 0
    total_units_bought: int = 0
    total_units_sold: int = 0
    total_item_slams: int = 0
    total_illegal_actions: int = 0

    def clone(self) -> GameState:
        """Return an independent copy suitable for simulator branch/restore."""

        return GameState(
            seed=self.seed,
            round=self.round,
            hp=self.hp,
            gold=self.gold,
            level=self.level,
            xp=self.xp,
            shop=list(self.shop),
            bench=[unit.clone() if unit is not None else None for unit in self.bench],
            board=[unit.clone() if unit is not None else None for unit in self.board],
            item_bench=list(self.item_bench),
            done=self.done,
            step_count=self.step_count,
            round_action_count=self.round_action_count,
            final_reason=self.final_reason,
            last_board_strength=self.last_board_strength,
            last_enemy_strength=self.last_enemy_strength,
            last_win=self.last_win,
            total_rolls=self.total_rolls,
            total_xp_buys=self.total_xp_buys,
            total_units_bought=self.total_units_bought,
            total_units_sold=self.total_units_sold,
            total_item_slams=self.total_item_slams,
            total_illegal_actions=self.total_illegal_actions,
        )


def new_game_state(config: EnvConfig, seed: int, shop: list[int]) -> GameState:
    """Create a fresh episode state."""

    return GameState(
        seed=seed,
        round=1,
        hp=config.starting_hp,
        gold=config.starting_gold,
        level=config.starting_level,
        xp=config.starting_xp,
        shop=shop,
        bench=[None for _ in range(config.bench_size)],
        board=[None for _ in range(config.max_level)],
        item_bench=[],
    )


def state_signature(state: GameState) -> tuple[object, ...]:
    """Return a deterministic immutable signature for tests and debugging."""

    return (
        state.round,
        state.hp,
        state.gold,
        state.level,
        state.xp,
        tuple(state.shop),
        tuple(_unit_signature(unit) for unit in state.bench),
        tuple(_unit_signature(unit) for unit in state.board),
        tuple(state.item_bench),
        state.done,
        state.step_count,
        state.round_action_count,
        state.final_reason,
        state.last_board_strength,
        state.last_enemy_strength,
        state.last_win,
        state.total_rolls,
        state.total_xp_buys,
        state.total_units_bought,
        state.total_units_sold,
        state.total_item_slams,
        state.total_illegal_actions,
    )


def _unit_signature(unit: UnitInstance | None) -> tuple[int, int, tuple[int, ...]] | None:
    if unit is None:
        return None
    return (unit.unit_id, unit.stars, tuple(unit.items))
