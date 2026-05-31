"""Batch-evaluate current-patch planner traces against top MetaTFT comps."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mini_tft.metatft import (
    CurrentPatchShopEconPolicy,
    ShopEconPolicyConfig,
    evaluate_planner_trace_batch,
    load_catalog_from_comp_strength,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--comp-ids", default=None)
    parser.add_argument("--comp-limit", type=int, default=16)
    parser.add_argument("--demo-levels", default="8,9")
    parser.add_argument("--match-levels", default="8,9")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    catalog = load_catalog_from_comp_strength(args.catalog)
    policy = CurrentPatchShopEconPolicy.from_checkpoint(
        catalog,
        args.checkpoint,
        device_name=args.device,
        config=ShopEconPolicyConfig(max_actions_per_turn=args.max_actions),
    )
    report = evaluate_planner_trace_batch(
        catalog,
        policy,
        comp_ids=_parse_optional_strings(args.comp_ids),
        comp_limit=args.comp_limit,
        demo_levels=_parse_ints(args.demo_levels, "--demo-levels"),
        match_levels=_parse_ints(args.match_levels, "--match-levels"),
        top_k=args.top_k,
        min_recall=args.min_recall,
    )
    payload = {
        "catalog": str(args.catalog),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        **asdict(report),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


def _parse_optional_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    return parsed or None


def _parse_ints(value: str, flag: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"{flag} must include at least one integer")
    if any(item < 1 for item in parsed):
        raise ValueError(f"{flag} must contain positive integers")
    return parsed


if __name__ == "__main__":
    main()
