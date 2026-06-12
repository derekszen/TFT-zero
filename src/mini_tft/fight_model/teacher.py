"""Teacher simulators for fight labels."""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import numpy as np

from mini_tft.fight_model.schema import FightBoard, FightLabel, FightSpec


class FightTeacher(Protocol):
    name: str
    max_unit_id: int
    max_item_id: int

    def simulate(self, spec: FightSpec) -> FightLabel:
        """Return a deterministic label for a fight spec."""
        ...


@dataclass
class HeuristicFightTeacher:
    """Fast deterministic fallback teacher for tests and smoke runs.

    This is not meant to be the real training teacher. It exercises the pipeline
    without importing the third-party Set 4 simulator.
    """

    name: str = "heuristic"
    max_unit_id: int = 58
    max_item_id: int = 45

    def simulate(self, spec: FightSpec) -> FightLabel:
        started = time.perf_counter()
        score_a = _heuristic_board_score(spec.board_a)
        score_b = _heuristic_board_score(spec.board_b)
        rng = np.random.default_rng(spec.seed)
        score_a += float(rng.normal(0.0, 0.75))
        score_b += float(rng.normal(0.0, 0.75))
        a_win = score_a >= score_b
        margin = abs(score_a - score_b)
        damage = 1.0 + margin / 14.0
        survivor_value = margin / 8.0
        return FightLabel(
            a_win=a_win,
            damage_to_a=0.0 if a_win else damage,
            damage_to_b=damage if a_win else 0.0,
            survivor_value_a=survivor_value if a_win else 0.0,
            survivor_value_b=0.0 if a_win else survivor_value,
            teacher_ms=(time.perf_counter() - started) * 1000.0,
        )


class TFTMuZeroSet4Teacher:
    """Adapter for the vendored TFTMuZeroAgent Set 4 detailed combat simulator."""

    name = "tft_muzero_set4"

    def __init__(self, root: str | Path = "third_party/TFTMuZeroAgent") -> None:
        self.root = Path(root).resolve()
        if not (self.root / "Simulator").is_dir():
            raise FileNotFoundError(
                f"TFTMuZeroAgent teacher not found at {self.root}. "
                "Expected a checkout/snapshot with a Simulator/ directory."
            )
        self._load_modules()

    def _load_modules(self) -> None:
        root_str = str(self.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        self._pool_mod = importlib.import_module("Simulator.pool")
        self._player_mod = importlib.import_module("Simulator.player")
        self._champion_mod = importlib.import_module("Simulator.champion")
        self._game_round_mod = importlib.import_module("Simulator.game_round")
        self._sim_config = importlib.import_module("Simulator.config")
        self._player_manager_mod = importlib.import_module("Simulator.player_manager")
        self._obs_mod = importlib.import_module("Simulator.observation.vector.observation")
        self._action_mod = importlib.import_module("Simulator.observation.token.action")
        self._utils_mod = importlib.import_module("Simulator.utils")
        self._agent_stats = importlib.import_module("Simulator.default_agent_stats")
        self._item_stats = importlib.import_module("Simulator.item_stats")
        self._global_config = importlib.import_module("config")
        global_config = cast(Any, self._global_config)
        sim_config = cast(Any, self._sim_config)
        global_config.AUTO_BATTLER_PERCENTAGE = 0
        global_config.DEBUG = False
        sim_config.LOGMESSAGES = False
        sim_config.PRINTMESSAGES = False

        unit_lists = [
            self._agent_stats.ONE_COST_UNITS,
            self._agent_stats.TWO_COST_UNITS,
            self._agent_stats.THREE_COST_UNITS,
            self._agent_stats.FOUR_COST_UNITS,
            self._agent_stats.FIVE_COST_UNITS,
        ]
        excluded_units = {"azir", "kayn"}
        self.unit_names = tuple(
            name
            for unit_list in unit_lists
            for name in unit_list
            if name not in excluded_units
        )
        self.item_names = tuple(
            name
            for name in self._item_stats.items
            if name not in set(getattr(self._item_stats, "uncraftable_items", []))
        )
        self.max_unit_id = len(self.unit_names)
        self.max_item_id = len(self.item_names)

        self._pool = self._pool_mod.pool()
        tft_config = SimpleNamespace(
            observation_class=self._obs_mod.ObservationVector,
            action_class=self._action_mod.ActionToken,
        )
        self._manager = self._player_manager_mod.PlayerManager(2, self._pool, tft_config)
        self._game_round = self._game_round_mod.Game_Round(None, self._pool, self._manager)

    def simulate(self, spec: FightSpec) -> FightLabel:
        started = time.perf_counter()
        player_a = self._build_player(0, spec.board_a)
        player_b = self._build_player(1, spec.board_b)
        player_a.opponent = player_b
        player_b.opponent = player_a
        result, damage = self._game_round.single_combat_phase([player_a, player_b])
        teacher_ms = (time.perf_counter() - started) * 1000.0
        a_win = result == 1
        b_win = result == 2
        return FightLabel(
            a_win=a_win,
            damage_to_a=float(damage if b_win or result == 0 else 0.0),
            damage_to_b=float(damage if a_win or result == 0 else 0.0),
            survivor_value_a=_survivor_value(player_a) if a_win else 0.0,
            survivor_value_b=_survivor_value(player_b) if b_win else 0.0,
            teacher_ms=teacher_ms,
        )

    def _build_player(self, player_num: int, board: FightBoard):
        player = self._player_mod.Player(self._pool, player_num)
        player.level = board.level
        player.max_units = board.level
        for unit in board.units:
            champion = self._champion_mod.champion(
                self._unit_name(unit.unit_id),
                stars=unit.stars,
                itemlist=[self._item_name(item_id) for item_id in unit.items if item_id > 0],
            )
            player.add_to_bench(champion)
            x, y = self._utils_mod.coord_to_x_y(unit.position)
            if not player.move_bench_to_board(0, x, y):
                raise RuntimeError(f"failed to place unit {unit} at {(x, y)}")
        return player

    def _unit_name(self, unit_id: int) -> str:
        return self.unit_names[(unit_id - 1) % len(self.unit_names)]

    def _item_name(self, item_id: int) -> str:
        return self.item_names[(item_id - 1) % len(self.item_names)]


def make_teacher(name: str, root: str | Path | None = None) -> FightTeacher:
    normalized = name.replace("-", "_")
    if normalized == "heuristic":
        return HeuristicFightTeacher()
    if normalized in {"tft_muzero", "tft_muzero_set4"}:
        return TFTMuZeroSet4Teacher(root or "third_party/TFTMuZeroAgent")
    raise ValueError(f"unknown fight teacher {name!r}")


def _heuristic_board_score(board: FightBoard) -> float:
    score = 0.0
    positions = set()
    for unit in board.units:
        positions.add(unit.position)
        row = unit.position // 7
        position_mult = 1.0
        if row == 0:
            position_mult = 1.08
        elif row == 3:
            position_mult = 1.05
        score += (4.0 + unit.unit_id * 0.55) * (1.75 ** (unit.stars - 1)) * position_mult
        score += len(unit.items) * (2.5 + unit.unit_id % 5)
        score += sum((item_id % 9) * 0.18 for item_id in unit.items)
    score += min(len(board.units), board.level) * 1.7
    score -= max(0, len(board.units) - len(positions)) * 3.0
    return score


def _survivor_value(player: object) -> float:
    value = 0.0
    for column in getattr(player, "board", []):
        for unit in column:
            if unit is not None:
                value += float(getattr(unit, "cost", 1)) * float(getattr(unit, "stars", 1))
    return value
