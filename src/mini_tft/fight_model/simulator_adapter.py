"""Adapter from MiniTFT boards to FightValueNet combat predictions."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_tft.core.combat import CombatResult, base_damage_by_round
from mini_tft.core.config import EnvConfig
from mini_tft.core.set_data import GameData
from mini_tft.core.state import UnitInstance
from mini_tft.fight_model.model import FightModelConfig, FightValueNet, load_checkpoint_payload
from mini_tft.fight_model.schema import MAX_ITEMS, MAX_POSITION, MAX_UNITS, FightBoard, FightUnit


@dataclass(frozen=True)
class FightValuePrediction:
    win_probability: float
    damage_to_a: float
    damage_to_b: float
    survivor_value_a: float
    survivor_value_b: float
    win_logit: float
    learned_strength: float
    enemy_strength: float


class FightValueCombatModel:
    """Reusable FightValueNet scorer for simulator combat and ranking checks."""

    def __init__(self, checkpoint: str | Path, device_name: str = "cpu") -> None:
        self.device = torch.device(device_name)
        self.payload = load_checkpoint_payload(Path(checkpoint), map_location=self.device)
        config = FightModelConfig(**self.payload["config"])
        self.model = FightValueNet(config).to(self.device)
        self.model.load_state_dict(self.payload["model_state"])
        self.model.eval()
        self.unit_namespace = str(self.payload.get("unit_namespace", "raw_unit_ids"))
        raw_lookup = self.payload.get("metatft_unit_id_lookup")
        self.metatft_unit_id_lookup = (
            {str(unit): int(unit_id) for unit, unit_id in raw_lookup.items()}
            if isinstance(raw_lookup, dict)
            else None
        )

    @torch.no_grad()
    def predict_matchup(
        self,
        board_a: FightBoard,
        board_b: FightBoard,
        *,
        stage: int,
        enemy_strength: float,
        sigmoid_scale: float,
    ) -> FightValuePrediction:
        batch = _batch_from_boards(board_a, board_b, stage, self.device)
        out = self.model(batch)
        win_logit = float(out["win_logit"][0].detach().cpu())
        win_probability = 1.0 / (1.0 + np.exp(-win_logit))
        damage = out["damage"][0].detach().cpu().tolist()
        survivor = out["survivor_value"][0].detach().cpu().tolist()
        learned_strength = max(0.0, enemy_strength + win_logit * sigmoid_scale)
        return FightValuePrediction(
            win_probability=float(win_probability),
            damage_to_a=float(damage[0]),
            damage_to_b=float(damage[1]),
            survivor_value_a=float(survivor[0]),
            survivor_value_b=float(survivor[1]),
            win_logit=win_logit,
            learned_strength=learned_strength,
            enemy_strength=enemy_strength,
        )

    def predict_mini_board(
        self,
        board: list[UnitInstance | None],
        round_num: int,
        data: GameData,
        config: EnvConfig,
    ) -> FightValuePrediction:
        if self.metatft_unit_id_lookup is not None:
            raise ValueError(
                "MetaTFT current-patch checkpoints use MetaTFT unit-name IDs and cannot "
                "score MiniTFT integer unit IDs directly"
            )
        board_a = mini_board_to_fight_board(board)
        enemy_board, enemy_strength = enemy_proxy_board(round_num, data)
        return self.predict_matchup(
            board_a,
            enemy_board,
            stage=round_to_stage(round_num),
            enemy_strength=enemy_strength,
            sigmoid_scale=config.combat_sigmoid_scale,
        )

    def score_fight_board(
        self,
        board: FightBoard,
        *,
        stage: int = 5,
        baseline_board: FightBoard | None = None,
        enemy_strength: float = 100.0,
        sigmoid_scale: float = 12.0,
    ) -> float:
        baseline = baseline_board or baseline_fight_board(board.level)
        prediction = self.predict_matchup(
            board,
            baseline,
            stage=stage,
            enemy_strength=enemy_strength,
            sigmoid_scale=sigmoid_scale,
        )
        return (
            prediction.win_logit
            - prediction.damage_to_a * 0.05
            + prediction.damage_to_b * 0.05
            + prediction.survivor_value_a * 0.02
        )


def resolve_combat_with_fight_value(
    board: list[UnitInstance | None],
    round_num: int,
    data: GameData,
    config: EnvConfig,
    rng: np.random.Generator,
    evaluator: FightValueCombatModel,
) -> CombatResult:
    prediction = evaluator.predict_mini_board(board, round_num, data, config)
    won = bool(rng.random() < prediction.win_probability)
    damage = 0
    if not won:
        damage = max(base_damage_by_round(round_num), int(round(prediction.damage_to_a)))
    return CombatResult(
        won=won,
        damage=damage,
        my_strength=prediction.learned_strength,
        enemy_strength=prediction.enemy_strength,
        p_win=prediction.win_probability,
    )


def mini_board_to_fight_board(board: list[UnitInstance | None]) -> FightBoard:
    units = []
    for position, unit in enumerate(board[:MAX_UNITS]):
        if unit is None:
            continue
        units.append(
            FightUnit(
                unit_id=max(1, int(unit.unit_id)),
                stars=max(1, min(3, int(unit.stars))),
                position=min(position, MAX_POSITION),
                items=tuple(int(item_id) for item_id in unit.items[:MAX_ITEMS]),
            )
        )
    return FightBoard(units=tuple(units), level=max(1, min(MAX_UNITS, len(units) or 1)))


def enemy_proxy_board(round_num: int, data: GameData) -> tuple[FightBoard, float]:
    enemy_index = min(max(0, round_num - 1), len(data.enemy_curve) - 1)
    enemy_strength = float(data.enemy_curve[enemy_index])
    level = max(1, min(MAX_UNITS, 3 + round_num // 4))
    unit_ids = sorted(
        data.units,
        key=lambda unit_id: (data.units[unit_id].cost, data.units[unit_id].base_power),
        reverse=True,
    )
    star_level = 1
    if enemy_strength >= 120:
        star_level = 2
    if enemy_strength >= 260:
        star_level = 3
    units = []
    for position, unit_id in enumerate(unit_ids[:level]):
        units.append(
            FightUnit(
                unit_id=int(unit_id),
                stars=star_level,
                position=min(position, MAX_POSITION),
            )
        )
    return FightBoard(units=tuple(units), level=level), enemy_strength


def baseline_fight_board(level: int = 8) -> FightBoard:
    clamped_level = max(1, min(MAX_UNITS, level))
    units = tuple(
        FightUnit(unit_id=unit_id, stars=2 if unit_id <= clamped_level // 2 else 1, position=index)
        for index, unit_id in enumerate(range(1, clamped_level + 1))
    )
    return FightBoard(units=units, level=clamped_level)


def metatft_units_to_fight_board(
    units: tuple[str, ...],
    unit_id_lookup: dict[str, int],
) -> FightBoard:
    encoded_units = []
    for position, unit_name in enumerate(units[:MAX_UNITS]):
        encoded_units.append(
            FightUnit(
                unit_id=unit_id_lookup[unit_name],
                stars=1,
                position=position,
            )
        )
    return FightBoard(units=tuple(encoded_units), level=max(1, min(MAX_UNITS, len(encoded_units))))


def round_to_stage(round_num: int) -> int:
    return max(1, min(8, 2 + (round_num - 1) // 6))


def _batch_from_boards(
    board_a: FightBoard,
    board_b: FightBoard,
    stage: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    row: dict[str, Any] = {
        "stage": [stage],
        "level_a": [board_a.level],
        "level_b": [board_b.level],
    }
    for prefix, board in (("board_a", board_a), ("board_b", board_b)):
        row[f"{prefix}_unit_ids"] = [[0 for _ in range(MAX_UNITS)]]
        row[f"{prefix}_stars"] = [[0 for _ in range(MAX_UNITS)]]
        row[f"{prefix}_position_ids"] = [[0 for _ in range(MAX_UNITS)]]
        row[f"{prefix}_item_ids"] = [[[0 for _ in range(MAX_ITEMS)] for _ in range(MAX_UNITS)]]
        row[f"{prefix}_chosen_traits"] = [[0 for _ in range(MAX_UNITS)]]
        for unit_index, unit in enumerate(board.units[:MAX_UNITS]):
            row[f"{prefix}_unit_ids"][0][unit_index] = unit.unit_id
            row[f"{prefix}_stars"][0][unit_index] = unit.stars
            row[f"{prefix}_position_ids"][0][unit_index] = unit.position
            row[f"{prefix}_chosen_traits"][0][unit_index] = unit.chosen_trait
            for item_index, item_id in enumerate(unit.items[:MAX_ITEMS]):
                row[f"{prefix}_item_ids"][0][unit_index][item_index] = item_id
    return {name: torch.as_tensor(value, device=device) for name, value in row.items()}


def logit(probability: float) -> float:
    clamped = min(1.0 - 1e-6, max(1e-6, probability))
    return log(clamped / (1.0 - clamped))
