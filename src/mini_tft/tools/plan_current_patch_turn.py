"""Run one current-patch MetaTFT shop/econ planning smoke turn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_tft.metatft import (
    CurrentBoardState,
    CurrentBoardUnit,
    CurrentPatchShopEconPolicy,
    ShopEconPolicyConfig,
    load_catalog_from_comp_strength,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comp-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-actions", type=int, default=8)
    args = parser.parse_args()

    catalog = load_catalog_from_comp_strength(args.catalog)
    comp = catalog.comp(args.comp_id) if args.comp_id else catalog.comps[0]
    state, shops = _demo_state_and_shops(comp_id=comp.comp_id, unit_keys=comp.unit_keys)
    policy = CurrentPatchShopEconPolicy.from_checkpoint(
        catalog,
        args.checkpoint,
        device_name=args.device,
        config=ShopEconPolicyConfig(max_actions_per_turn=args.max_actions),
    )
    plan = policy.plan_turn(state, shops=shops)
    payload = {
        "catalog": str(args.catalog),
        "checkpoint": str(args.checkpoint),
        "comp_id": comp.comp_id,
        "comp_name": comp.name,
        "initial_board": [unit.unit_key for unit in state.board],
        "initial_bench": [unit.unit_key for unit in state.bench],
        "initial_shop": list(shops[0]) if shops else [],
        "decisions": [
            {
                "rank": decision.rank,
                "action": decision.action,
                "type": decision.transition.metadata.get("type"),
                "after_value": decision.after_value,
                "delta": decision.delta,
                "gold_after": decision.transition.after.gold,
                "level_after": decision.transition.after.level,
            }
            for decision in plan.decisions
        ],
        "final_board": [unit.unit_key for unit in plan.final_state.board],
        "final_bench": [unit.unit_key for unit in plan.final_state.bench],
        "final_gold": plan.final_state.gold,
        "final_level": plan.final_state.level,
        "final_shop": list(plan.final_shop),
        "stopped": plan.stopped,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _demo_state_and_shops(
    *,
    comp_id: str,
    unit_keys: tuple[str, ...],
) -> tuple[CurrentBoardState, tuple[tuple[str, ...], ...]]:
    board_units = unit_keys[:3]
    bench_units = unit_keys[3:5]
    first_shop = unit_keys[5:10]
    second_shop = unit_keys[2:7]
    state = CurrentBoardState(
        stage=3,
        stage_round=2,
        level=5,
        gold=30,
        board=tuple(
            CurrentBoardUnit(unit_key=unit_key, position=index)
            for index, unit_key in enumerate(board_units)
        ),
        bench=tuple(CurrentBoardUnit(unit_key=unit_key) for unit_key in bench_units),
        target_comp_id=comp_id,
        source="current_patch_policy_smoke",
        metadata={"line": "policy_smoke"},
    )
    return state, (tuple(first_shop), tuple(second_shop))


if __name__ == "__main__":
    main()
