"""Batch-evaluate current-patch planner traces against top MetaTFT comps."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from mini_tft.metatft import (
    CurrentPatchShopEconPolicy,
    PlannerGateMetric,
    PlannerMetricRequirement,
    ShopEconPolicyConfig,
    evaluate_planner_batch_gate,
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
    parser.add_argument(
        "--trace-mode",
        choices=("completion", "shop-planning"),
        default="completion",
        help="Regression trace generator to use.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument(
        "--require-exact-match-rate",
        action="append",
        default=None,
        metavar="LEVEL:MIN",
        help="Fail if exact_match_rate for LEVEL is below MIN.",
    )
    parser.add_argument(
        "--require-good-enough-rate",
        action="append",
        default=None,
        metavar="LEVEL:MIN",
        help="Fail if good_enough_rate for LEVEL is below MIN.",
    )
    parser.add_argument(
        "--require-eligible-good-enough-rate",
        action="append",
        default=None,
        metavar="LEVEL:MIN",
        help="Fail if eligible_good_enough_rate for LEVEL is below MIN.",
    )
    parser.add_argument(
        "--require-mean-recall",
        action="append",
        default=None,
        metavar="LEVEL:MIN",
        help="Fail if mean_recall for LEVEL is below MIN.",
    )
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
        trace_mode=args.trace_mode,
    )
    gate = evaluate_planner_batch_gate(
        report,
        _parse_gate_requirements(args),
    )
    payload = {
        "catalog": str(args.catalog),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "gate": asdict(gate),
        **asdict(report),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not gate.passed:
        print(
            "planner regression gate failed; see gate.failures in the JSON report",
            file=sys.stderr,
        )
        raise SystemExit(1)


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


def _parse_gate_requirements(args: argparse.Namespace) -> tuple[PlannerMetricRequirement, ...]:
    requirements: list[PlannerMetricRequirement] = []
    requirements.extend(
        _parse_metric_requirements(
            args.require_exact_match_rate,
            metric="exact_match_rate",
            flag="--require-exact-match-rate",
        )
    )
    requirements.extend(
        _parse_metric_requirements(
            args.require_good_enough_rate,
            metric="good_enough_rate",
            flag="--require-good-enough-rate",
        )
    )
    requirements.extend(
        _parse_metric_requirements(
            args.require_eligible_good_enough_rate,
            metric="eligible_good_enough_rate",
            flag="--require-eligible-good-enough-rate",
        )
    )
    requirements.extend(
        _parse_metric_requirements(
            args.require_mean_recall,
            metric="mean_recall",
            flag="--require-mean-recall",
        )
    )
    return tuple(requirements)


def _parse_metric_requirements(
    values: Sequence[str] | None,
    *,
    metric: PlannerGateMetric,
    flag: str,
) -> tuple[PlannerMetricRequirement, ...]:
    if values is None:
        return ()
    parsed = []
    for value in values:
        parts = value.split(":", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{flag} must use LEVEL:MIN")
        level_text, minimum_text = parts
        level = int(level_text)
        minimum = float(minimum_text)
        if level < 1:
            raise ValueError(f"{flag} level must be positive")
        if not 0.0 <= minimum <= 1.0:
            raise ValueError(f"{flag} minimum must be in [0, 1]")
        parsed.append(
            PlannerMetricRequirement(
                level=level,
                metric=metric,
                minimum=minimum,
            )
        )
    return tuple(parsed)


if __name__ == "__main__":
    main()
