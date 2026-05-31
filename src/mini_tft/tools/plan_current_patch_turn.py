"""Run one current-patch MetaTFT shop/econ planning smoke turn."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mini_tft.metatft import (
    CurrentPatchShopEconPolicy,
    ShopEconPolicyConfig,
    demo_state_and_shops,
    load_catalog_from_comp_strength,
    top_comp_match_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comp-id", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--demo-level", type=int, default=5)
    parser.add_argument("--match-levels", default="8,9")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.75)
    args = parser.parse_args()

    catalog = load_catalog_from_comp_strength(args.catalog)
    comp = catalog.comp(args.comp_id) if args.comp_id else catalog.comps[0]
    state, shops = demo_state_and_shops(
        comp_id=comp.comp_id,
        unit_keys=comp.unit_keys,
        level=args.demo_level,
    )
    policy = CurrentPatchShopEconPolicy.from_checkpoint(
        catalog,
        args.checkpoint,
        device_name=args.device,
        config=ShopEconPolicyConfig(max_actions_per_turn=args.max_actions),
    )
    plan = policy.plan_turn(state, shops=shops)
    match_levels = _parse_levels(args.match_levels)
    top_comp_matches = top_comp_match_report(
        catalog,
        plan.final_state.board_unit_keys,
        board_level=plan.final_state.level,
        levels=match_levels,
        top_k=args.top_k,
        min_recall=args.min_recall,
    )
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
        "top_comp_match": [asdict(match) for match in top_comp_matches],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

def _parse_levels(value: str) -> tuple[int, ...]:
    levels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not levels:
        raise ValueError("--match-levels must include at least one level")
    if any(level < 1 for level in levels):
        raise ValueError("--match-levels must contain positive integers")
    return levels


if __name__ == "__main__":
    main()
