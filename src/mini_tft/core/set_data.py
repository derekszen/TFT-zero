"""Data loading and validation for packaged sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UnitDef:
    id: int
    name: str
    cost: int
    traits: tuple[str, ...]
    role: str
    base_power: float


@dataclass(frozen=True)
class TraitDef:
    id: str
    breakpoints: dict[int, dict[str, float]]


@dataclass(frozen=True)
class ItemDef:
    id: int
    name: str
    tags: tuple[str, ...]
    effects: dict[str, float]
    kind: str = "completed"
    components: tuple[int, int] = ()


@dataclass(frozen=True)
class GameData:
    units: dict[int, UnitDef]
    traits: dict[str, TraitDef]
    items: dict[int, ItemDef]
    shop_odds: dict[int, NDArray[np.float64]]
    enemy_curve: tuple[float, ...]
    units_by_cost: dict[int, tuple[int, ...]]
    trait_ids: tuple[str, ...]
    max_unit_id: int
    max_item_id: int
    component_item_ids: tuple[int, ...]
    completed_item_ids: tuple[int, ...]
    item_recipes: dict[tuple[int, int], int]


def load_set(name: str = "set1") -> GameData:
    """Load a packaged data set."""

    units_raw = _load_json(f"{name}_units.json")
    traits_raw = _load_json(f"{name}_traits.json")
    items_raw = _load_json(f"{name}_items.json")
    shop_raw = _load_json("shop_odds.json")
    curves_raw = _load_json("enemy_curves.json")

    traits = {
        trait_id: TraitDef(
            id=trait_id,
            breakpoints={
                int(count): {effect: float(value) for effect, value in effects.items()}
                for count, effects in spec["breakpoints"].items()
            },
        )
        for trait_id, spec in traits_raw.items()
    }
    trait_ids = tuple(sorted(traits))

    units = {
        int(unit["id"]): UnitDef(
            id=int(unit["id"]),
            name=str(unit["name"]),
            cost=int(unit["cost"]),
            traits=tuple(str(trait) for trait in unit["traits"]),
            role=str(unit["role"]),
            base_power=float(unit["base_power"]),
        )
        for unit in units_raw
    }

    items = {
        int(item["id"]): ItemDef(
            id=int(item["id"]),
            name=str(item["name"]),
            tags=tuple(str(tag) for tag in item["tags"]),
            effects={str(effect): float(value) for effect, value in item["effects"].items()},
            kind=str(item.get("kind", "completed")),
            components=tuple(sorted(int(component) for component in item.get("components", ()))),
        )
        for item in items_raw
    }
    component_item_ids = tuple(
        sorted(item.id for item in items.values() if item.kind == "component")
    )
    completed_item_ids = tuple(
        sorted(item.id for item in items.values() if item.kind == "completed")
    )
    item_recipes = {
        item.components: item.id
        for item in items.values()
        if item.kind == "completed" and item.components
    }

    shop_odds = {
        int(level): np.asarray(weights, dtype=np.float64)
        for level, weights in shop_raw.items()
    }
    enemy_curve = tuple(float(value) for value in curves_raw["default"])

    units_by_cost: dict[int, list[int]] = {}
    for unit in units.values():
        units_by_cost.setdefault(unit.cost, []).append(unit.id)

    data = GameData(
        units=units,
        traits=traits,
        items=items,
        shop_odds=shop_odds,
        enemy_curve=enemy_curve,
        units_by_cost={cost: tuple(ids) for cost, ids in units_by_cost.items()},
        trait_ids=trait_ids,
        max_unit_id=max(units),
        max_item_id=max(items),
        component_item_ids=component_item_ids,
        completed_item_ids=completed_item_ids,
        item_recipes=item_recipes,
    )
    validate_data(data)
    return data


def validate_data(data: GameData) -> None:
    """Raise if the set is internally inconsistent."""

    if len(data.enemy_curve) < 36:
        raise ValueError("enemy curve must cover at least 36 rounds")

    for unit in data.units.values():
        if unit.cost not in {1, 2, 3, 4, 5}:
            raise ValueError(f"{unit.name} has invalid cost {unit.cost}")
        missing_traits = [trait for trait in unit.traits if trait not in data.traits]
        if missing_traits:
            raise ValueError(f"{unit.name} references missing traits {missing_traits}")

    for level, weights in data.shop_odds.items():
        if weights.shape != (5,):
            raise ValueError(f"level {level} shop odds must have 5 costs")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"level {level} shop odds must sum to 1.0")

    for cost in range(1, 6):
        if cost not in data.units_by_cost:
            raise ValueError(f"no units for cost {cost}")

    for item in data.items.values():
        if item.kind not in {"component", "completed"}:
            raise ValueError(f"{item.name} has invalid item kind {item.kind}")
        if item.kind == "component" and item.components:
            raise ValueError(f"{item.name} component item cannot have a recipe")
        if item.components:
            for component_id in item.components:
                component = data.items.get(component_id)
                if component is None:
                    raise ValueError(f"{item.name} references missing component {component_id}")
                if component.kind != "component":
                    raise ValueError(f"{item.name} recipe uses non-component {component.name}")

    for recipe, completed_id in data.item_recipes.items():
        if len(recipe) != 2:
            raise ValueError(f"item recipe for {completed_id} must have exactly two components")

    if not data.completed_item_ids:
        raise ValueError("set must define at least one completed item")


def _load_json(filename: str) -> Any:
    data_path = files("mini_tft.data").joinpath(filename)
    return json.loads(data_path.read_text(encoding="utf-8"))
