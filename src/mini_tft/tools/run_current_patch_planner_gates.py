"""Run recurring current-patch planner regression gate suites."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from mini_tft.metatft import (
    PLANNER_TRACE_MODES,
    CurrentPatchPlannerScorer,
    CurrentPatchShopEconPolicy,
    PlannerGateMetric,
    PlannerMetricRequirement,
    PlannerTraceMode,
    ShopEconPolicyConfig,
    evaluate_planner_batch_gate,
    evaluate_planner_trace_batch,
    load_catalog_from_comp_strength,
)

MINIMUM_GATE_MODES: tuple[PlannerTraceMode, ...] = ("shop-planning",)
STRICT_GATE_MODES: tuple[PlannerTraceMode, ...] = (
    "shop-planning",
    "distractor-heavy",
    "multi-roll",
)
DEFAULT_MODE_MAX_ACTIONS: dict[PlannerTraceMode, int] = {
    "completion": 8,
    "shop-planning": 8,
    "distractor-heavy": 8,
    "multi-roll": 10,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--suite", choices=("minimum", "strict"), default="minimum")
    parser.add_argument(
        "--trace-modes",
        default=None,
        help="Comma-separated trace modes. Overrides --suite when set.",
    )
    parser.add_argument("--comp-ids", default=None)
    parser.add_argument("--comp-limit", type=int, default=8)
    parser.add_argument("--demo-levels", default="8,9")
    parser.add_argument("--match-levels", default="8,9")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Override per-mode max action defaults.",
    )
    parser.add_argument(
        "--print-full-report",
        action="store_true",
        help="Print full trace payload to stdout. By default stdout is compact.",
    )
    parser.add_argument(
        "--require-exact-match-rate",
        action="append",
        default=None,
        metavar="LEVEL:MIN",
        help="Fail if exact_match_rate for LEVEL is below MIN. Defaults to 8:1.0 and 9:1.0.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    modes = trace_modes_for_suite(args.suite, trace_modes=args.trace_modes)
    requirements = _parse_gate_requirements(args.require_exact_match_rate)
    catalog = load_catalog_from_comp_strength(args.catalog)
    scorer = CurrentPatchPlannerScorer.from_checkpoint(
        catalog,
        args.checkpoint,
        device_name=args.device,
    )
    unit_costs = {
        unit.key: unit.cost
        for unit in catalog.units
        if unit.cost is not None and unit.cost > 0
    }

    mode_reports = []
    failures = []
    for mode in modes:
        max_actions = mode_max_actions(mode, override=args.max_actions)
        policy = CurrentPatchShopEconPolicy(
            scorer,
            config=ShopEconPolicyConfig(max_actions_per_turn=max_actions),
            unit_costs=unit_costs,
            catalog=catalog,
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
            trace_mode=mode,
        )
        gate = evaluate_planner_batch_gate(report, requirements)
        if not gate.passed:
            failures.extend(
                {
                    "trace_mode": mode,
                    **asdict(failure),
                }
                for failure in gate.failures
            )
        mode_reports.append(
            {
                "trace_mode": mode,
                "max_actions": max_actions,
                "gate": asdict(gate),
                "action_mix": _action_mix(report),
                **asdict(report),
            }
        )

    payload = {
        "catalog": str(args.catalog),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "suite": args.suite,
        "trace_modes": modes,
        "passed": not failures,
        "failures": failures,
        "requirements": [asdict(requirement) for requirement in requirements],
        "reports": mode_reports,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    stdout_payload = payload if args.print_full_report else compact_gate_suite_payload(payload)
    print(json.dumps(stdout_payload, indent=2, sort_keys=True))
    if failures:
        print(
            "planner gate suite failed; see failures in the JSON report",
            file=sys.stderr,
        )
        raise SystemExit(1)


def trace_modes_for_suite(
    suite: str,
    *,
    trace_modes: str | None = None,
) -> tuple[PlannerTraceMode, ...]:
    if trace_modes is not None:
        modes = tuple(part.strip() for part in trace_modes.split(",") if part.strip())
        if not modes:
            raise ValueError("--trace-modes must include at least one mode")
        _validate_trace_modes(modes)
        return modes
    if suite == "minimum":
        return MINIMUM_GATE_MODES
    if suite == "strict":
        return STRICT_GATE_MODES
    raise ValueError("--suite must be 'minimum' or 'strict'")


def mode_max_actions(mode: PlannerTraceMode, *, override: int | None = None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("--max-actions must be positive")
        return override
    return DEFAULT_MODE_MAX_ACTIONS[mode]


def compact_gate_suite_payload(payload: dict[str, object]) -> dict[str, object]:
    reports = payload.get("reports", ())
    if not isinstance(reports, list):
        reports = []
    compact_reports = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        compact_reports.append(
            {
                "trace_mode": report.get("trace_mode"),
                "max_actions": report.get("max_actions"),
                "gate": report.get("gate"),
                "summaries": report.get("summaries"),
                "action_mix": report.get("action_mix"),
            }
        )
    return {
        "catalog": payload.get("catalog"),
        "checkpoint": payload.get("checkpoint"),
        "device": payload.get("device"),
        "suite": payload.get("suite"),
        "trace_modes": payload.get("trace_modes"),
        "passed": payload.get("passed"),
        "failures": payload.get("failures"),
        "requirements": payload.get("requirements"),
        "reports": compact_reports,
    }


def _action_mix(report) -> dict[str, int]:
    counts = Counter(
        action_type
        for trace in report.traces
        for action_type in trace.decision_action_types
    )
    return dict(sorted(counts.items()))


def _validate_trace_modes(modes: Sequence[str]) -> None:
    invalid = [mode for mode in modes if mode not in PLANNER_TRACE_MODES]
    if invalid:
        supported = ", ".join(PLANNER_TRACE_MODES)
        raise ValueError(f"unsupported trace mode(s): {', '.join(invalid)}; supported: {supported}")


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


def _parse_gate_requirements(
    values: Sequence[str] | None,
) -> tuple[PlannerMetricRequirement, ...]:
    if values is None:
        values = ("8:1.0", "9:1.0")
    return _parse_metric_requirements(
        values,
        metric="exact_match_rate",
        flag="--require-exact-match-rate",
    )


def _parse_metric_requirements(
    values: Sequence[str],
    *,
    metric: PlannerGateMetric,
    flag: str,
) -> tuple[PlannerMetricRequirement, ...]:
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
