"""Batch evaluation for current-patch planner traces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from mini_tft.metatft.catalog import MetaTFTCatalog
from mini_tft.metatft.metrics import (
    TopCompMatch,
    target_comp_units_for_level,
    top_comp_match_report,
)
from mini_tft.metatft.policy import PolicyTurnPlan
from mini_tft.metatft.schema import CurrentBoardState, CurrentBoardUnit


class TurnPlanner(Protocol):
    def plan_turn(
        self,
        state: CurrentBoardState,
        *,
        shops: Sequence[Sequence[str]],
        unit_costs: Mapping[str, int] | None = None,
        rank_by: Literal["after_value", "delta"] = "after_value",
    ) -> PolicyTurnPlan:
        """Return a deterministic turn plan for a current-patch state."""
        ...


PlannerTraceMode = Literal[
    "completion",
    "shop-planning",
    "distractor-heavy",
    "multi-roll",
]
PLANNER_TRACE_MODES: tuple[PlannerTraceMode, ...] = (
    "completion",
    "shop-planning",
    "distractor-heavy",
    "multi-roll",
)


@dataclass(frozen=True)
class PlannerTraceEvaluation:
    comp_id: str
    comp_rank: int
    comp_name: str
    demo_level: int
    final_level: int
    stopped: bool
    decisions_count: int
    decision_actions: tuple[str, ...]
    decision_action_types: tuple[str, ...]
    initial_board: tuple[str, ...]
    final_board: tuple[str, ...]
    matches: tuple[TopCompMatch, ...]


@dataclass(frozen=True)
class PlannerUnitFrequency:
    unit_key: str
    count: int


@dataclass(frozen=True)
class PlannerExactFailureExample:
    comp_id: str
    comp_rank: int
    comp_name: str
    demo_level: int
    match_level: int
    final_level: int
    board_unit_count: int
    target_unit_count: int
    matched_comp_id: str
    recall: float
    jaccard: float
    missing_units: tuple[str, ...]
    extra_units: tuple[str, ...]


@dataclass(frozen=True)
class PlannerExactFailureSummary:
    level: int
    failed_count: int
    underleveled_count: int
    underfilled_count: int
    unit_mismatch_count: int
    top_missing_units: tuple[PlannerUnitFrequency, ...]
    top_extra_units: tuple[PlannerUnitFrequency, ...]
    examples: tuple[PlannerExactFailureExample, ...]


@dataclass(frozen=True)
class PlannerLevelMatchSummary:
    level: int
    trace_count: int
    eligible_count: int
    exact_match_count: int
    partial_match_count: int
    good_enough_count: int
    exact_match_rate: float
    partial_match_rate: float
    good_enough_rate: float
    eligible_good_enough_rate: float
    mean_recall: float
    mean_jaccard: float


PlannerGateMetric = Literal[
    "exact_match_rate",
    "partial_match_rate",
    "good_enough_rate",
    "eligible_good_enough_rate",
    "mean_recall",
    "mean_jaccard",
]


@dataclass(frozen=True)
class PlannerMetricRequirement:
    level: int
    metric: PlannerGateMetric
    minimum: float


@dataclass(frozen=True)
class PlannerGateFailure:
    level: int
    metric: PlannerGateMetric
    actual: float
    minimum: float


@dataclass(frozen=True)
class PlannerGateResult:
    passed: bool
    requirements: tuple[PlannerMetricRequirement, ...]
    failures: tuple[PlannerGateFailure, ...]


@dataclass(frozen=True)
class PlannerBatchEvaluation:
    comp_ids: tuple[str, ...]
    trace_mode: PlannerTraceMode
    demo_levels: tuple[int, ...]
    match_levels: tuple[int, ...]
    top_k: int
    min_recall: float
    traces: tuple[PlannerTraceEvaluation, ...]
    summaries: tuple[PlannerLevelMatchSummary, ...]
    exact_failure_summaries: tuple[PlannerExactFailureSummary, ...]


def evaluate_planner_trace_batch(
    catalog: MetaTFTCatalog,
    planner: TurnPlanner,
    *,
    comp_ids: Sequence[str] | None = None,
    comp_limit: int | None = 16,
    demo_levels: Sequence[int] = (8, 9),
    match_levels: Sequence[int] = (8, 9),
    top_k: int = 10,
    min_recall: float = 0.75,
    trace_mode: PlannerTraceMode = "completion",
) -> PlannerBatchEvaluation:
    """Run deterministic planner traces and summarize top-comp match rates."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be in [0, 1]")
    selected_comp_ids = _select_comp_ids(catalog, comp_ids, comp_limit)
    _validate_trace_mode(trace_mode)
    demo_level_values = tuple(demo_levels)
    match_level_values = tuple(match_levels)
    if not demo_level_values:
        raise ValueError("demo_levels must include at least one level")
    if not match_level_values:
        raise ValueError("match_levels must include at least one level")

    traces: list[PlannerTraceEvaluation] = []
    for comp_id in selected_comp_ids:
        comp = catalog.comp(comp_id)
        for demo_level in demo_level_values:
            state, shops = planner_trace_state_and_shops(
                catalog=catalog,
                comp_id=comp.comp_id,
                unit_keys=target_comp_units_for_level(comp, demo_level),
                level=demo_level,
                trace_mode=trace_mode,
            )
            plan = planner.plan_turn(state, shops=shops)
            matches = top_comp_match_report(
                catalog,
                plan.final_state.board_unit_keys,
                board_level=plan.final_state.level,
                levels=match_level_values,
                top_k=top_k,
                min_recall=min_recall,
            )
            traces.append(
                PlannerTraceEvaluation(
                    comp_id=comp.comp_id,
                    comp_rank=comp.rank,
                    comp_name=comp.name,
                    demo_level=demo_level,
                    final_level=plan.final_state.level,
                    stopped=plan.stopped,
                    decisions_count=len(plan.decisions),
                    decision_actions=tuple(decision.action for decision in plan.decisions),
                    decision_action_types=tuple(
                        str(decision.transition.metadata.get("type", "unknown"))
                        for decision in plan.decisions
                    ),
                    initial_board=state.board_unit_keys,
                    final_board=plan.final_state.board_unit_keys,
                    matches=matches,
                )
            )

    return PlannerBatchEvaluation(
        comp_ids=selected_comp_ids,
        trace_mode=trace_mode,
        demo_levels=demo_level_values,
        match_levels=match_level_values,
        top_k=top_k,
        min_recall=min_recall,
        traces=tuple(traces),
        summaries=_summarize_matches(traces, match_level_values),
        exact_failure_summaries=_summarize_exact_failures(traces, match_level_values),
    )


def evaluate_planner_batch_gate(
    report: PlannerBatchEvaluation,
    requirements: Sequence[PlannerMetricRequirement],
) -> PlannerGateResult:
    """Check level-specific planner metrics for use as a regression gate."""

    summary_by_level = {summary.level: summary for summary in report.summaries}
    failures = []
    for requirement in requirements:
        summary = summary_by_level.get(requirement.level)
        actual = 0.0 if summary is None else float(getattr(summary, requirement.metric))
        if actual < requirement.minimum:
            failures.append(
                PlannerGateFailure(
                    level=requirement.level,
                    metric=requirement.metric,
                    actual=actual,
                    minimum=requirement.minimum,
                )
            )
    requirement_values = tuple(requirements)
    return PlannerGateResult(
        passed=not failures,
        requirements=requirement_values,
        failures=tuple(failures),
    )


def demo_state_and_shops(
    *,
    comp_id: str,
    unit_keys: tuple[str, ...],
    level: int,
) -> tuple[CurrentBoardState, tuple[tuple[str, ...], ...]]:
    """Build a deterministic partial-board trace seed for planner evaluation."""

    if level < 1:
        raise ValueError("level must be positive")
    if not unit_keys:
        raise ValueError("unit_keys must include at least one unit")
    board_count = max(1, min(len(unit_keys), max(1, level - 2)))
    board_units = unit_keys[:board_count]
    bench_units = unit_keys[board_count : board_count + 2]
    shop_pool = unit_keys[board_count + 2 :] + unit_keys[: board_count + 2]
    first_shop = shop_pool[:5]
    second_shop = shop_pool[2:7]
    state = CurrentBoardState(
        stage=4 if level >= 8 else 3,
        stage_round=2,
        level=level,
        gold=30,
        board=tuple(
            CurrentBoardUnit(unit_key=unit_key, position=index)
            for index, unit_key in enumerate(board_units)
        ),
        bench=tuple(CurrentBoardUnit(unit_key=unit_key) for unit_key in bench_units),
        target_comp_id=comp_id,
        source="current_patch_policy_eval",
        metadata={"line": "policy_eval"},
    )
    return state, (tuple(first_shop), tuple(second_shop))


def planner_trace_state_and_shops(
    *,
    catalog: MetaTFTCatalog | None = None,
    comp_id: str,
    unit_keys: tuple[str, ...],
    level: int,
    trace_mode: PlannerTraceMode,
) -> tuple[CurrentBoardState, tuple[tuple[str, ...], ...]]:
    """Build a deterministic trace seed for a planner regression mode."""

    _validate_trace_mode(trace_mode)
    if trace_mode == "completion":
        return demo_state_and_shops(
            comp_id=comp_id,
            unit_keys=unit_keys,
            level=level,
        )
    if trace_mode == "shop-planning":
        return hard_shop_state_and_shops(
            comp_id=comp_id,
            unit_keys=unit_keys,
            level=level,
        )
    if catalog is None:
        raise ValueError(f"{trace_mode} trace mode requires a catalog")
    if trace_mode == "distractor-heavy":
        return distractor_heavy_state_and_shops(
            catalog=catalog,
            comp_id=comp_id,
            unit_keys=unit_keys,
            level=level,
        )
    return multi_roll_state_and_shops(
        catalog=catalog,
        comp_id=comp_id,
        unit_keys=unit_keys,
        level=level,
    )


def hard_shop_state_and_shops(
    *,
    comp_id: str,
    unit_keys: tuple[str, ...],
    level: int,
) -> tuple[CurrentBoardState, tuple[tuple[str, ...], ...]]:
    """Build a trace that requires buying missing target units across shops."""

    board_units, missing_units = _hard_trace_parts(unit_keys=unit_keys, level=level)
    first_missing_count = max(1, min(2, len(missing_units)))
    first_shop = _pad_shop(missing_units[:first_missing_count], board_units)
    second_shop = _pad_shop(
        missing_units[first_missing_count:],
        board_units[::-1],
    )
    state = _hard_trace_state(
        comp_id=comp_id,
        level=level,
        board_units=board_units,
        gold=40,
        source="current_patch_policy_hard_eval",
        line="policy_hard_eval",
    )
    return state, (first_shop, second_shop)


def distractor_heavy_state_and_shops(
    *,
    catalog: MetaTFTCatalog,
    comp_id: str,
    unit_keys: tuple[str, ...],
    level: int,
) -> tuple[CurrentBoardState, tuple[tuple[str, ...], ...]]:
    """Build a trace with off-target shop distractors around visible targets."""

    board_units, missing_units = _hard_trace_parts(unit_keys=unit_keys, level=level)
    distractors = _distractor_units(catalog, unit_keys)
    first_target_count = min(1, len(missing_units))
    first_shop = _pad_shop(
        missing_units[:first_target_count],
        distractors,
    )
    second_shop = _pad_shop(
        missing_units[first_target_count:],
        _rotate(distractors, 5),
    )
    state = _hard_trace_state(
        comp_id=comp_id,
        level=level,
        board_units=board_units,
        gold=44,
        source="current_patch_policy_distractor_eval",
        line="policy_distractor_eval",
    )
    return state, (first_shop, second_shop)


def multi_roll_state_and_shops(
    *,
    catalog: MetaTFTCatalog,
    comp_id: str,
    unit_keys: tuple[str, ...],
    level: int,
) -> tuple[CurrentBoardState, tuple[tuple[str, ...], ...]]:
    """Build a trace that requires skipping a bad shop and rolling twice."""

    board_units, missing_units = _hard_trace_parts(unit_keys=unit_keys, level=level)
    distractors = _distractor_units(catalog, unit_keys)
    split = max(1, len(missing_units) // 2)
    first_shop = _pad_shop((), distractors)
    second_shop = _pad_shop(missing_units[:split], _rotate(distractors, 5))
    third_shop = _pad_shop(missing_units[split:], _rotate(distractors, 10))
    state = _hard_trace_state(
        comp_id=comp_id,
        level=level,
        board_units=board_units,
        gold=50,
        source="current_patch_policy_multi_roll_eval",
        line="policy_multi_roll_eval",
    )
    return state, (first_shop, second_shop, third_shop)


def _select_comp_ids(
    catalog: MetaTFTCatalog,
    comp_ids: Sequence[str] | None,
    comp_limit: int | None,
) -> tuple[str, ...]:
    if comp_ids:
        selected = tuple(comp_ids)
    else:
        selected = tuple(comp.comp_id for comp in catalog.comps)
    if comp_limit is not None:
        if comp_limit < 1:
            raise ValueError("comp_limit must be at least 1")
        selected = selected[:comp_limit]
    if not selected:
        raise ValueError("no comp ids selected")
    for comp_id in selected:
        catalog.comp(comp_id)
    return selected


def _validate_trace_mode(trace_mode: str) -> None:
    if trace_mode not in PLANNER_TRACE_MODES:
        supported = ", ".join(PLANNER_TRACE_MODES)
        raise ValueError(f"trace_mode must be one of: {supported}")


def _hard_trace_parts(
    *,
    unit_keys: tuple[str, ...],
    level: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if level < 1:
        raise ValueError("level must be positive")
    if not unit_keys:
        raise ValueError("unit_keys must include at least one unit")
    board_count = max(1, min(len(unit_keys), max(1, level - 4)))
    board_units = unit_keys[:board_count]
    missing_units = unit_keys[board_count:level]
    return board_units, missing_units


def _hard_trace_state(
    *,
    comp_id: str,
    level: int,
    board_units: Sequence[str],
    gold: int,
    source: str,
    line: str,
) -> CurrentBoardState:
    return CurrentBoardState(
        stage=4 if level >= 8 else 3,
        stage_round=2,
        level=level,
        gold=gold,
        board=tuple(
            CurrentBoardUnit(unit_key=unit_key, position=index)
            for index, unit_key in enumerate(board_units)
        ),
        bench=(),
        target_comp_id=comp_id,
        source=source,
        metadata={"line": line},
    )


def _distractor_units(
    catalog: MetaTFTCatalog,
    target_units: Sequence[str],
) -> tuple[str, ...]:
    target_unit_set = set(target_units)
    distractors = tuple(
        unit.key for unit in catalog.units if unit.key not in target_unit_set
    )
    return distractors or tuple(unit for unit in target_units if unit)


def _rotate(units: Sequence[str], offset: int) -> tuple[str, ...]:
    values = tuple(units)
    if not values:
        return ()
    index = offset % len(values)
    return values[index:] + values[:index]


def _pad_shop(
    priority_units: Sequence[str],
    filler_units: Sequence[str],
    *,
    size: int = 5,
) -> tuple[str, ...]:
    shop = [unit_key for unit_key in priority_units if unit_key][:size]
    filler = tuple(unit_key for unit_key in filler_units if unit_key) or tuple(shop)
    cursor = 0
    while len(shop) < size and filler:
        shop.append(filler[cursor % len(filler)])
        cursor += 1
    return tuple(shop)


def _summarize_matches(
    traces: Sequence[PlannerTraceEvaluation],
    levels: Sequence[int],
) -> tuple[PlannerLevelMatchSummary, ...]:
    summaries = []
    for level in levels:
        matches = [
            match
            for trace in traces
            for match in trace.matches
            if trace.demo_level == level and match.level == level
        ]
        trace_count = len(matches)
        eligible_count = sum(1 for match in matches if match.eligible)
        exact_count = sum(1 for match in matches if match.exact_match)
        partial_count = sum(1 for match in matches if match.partial_match)
        good_count = sum(1 for match in matches if match.good_enough)
        summaries.append(
            PlannerLevelMatchSummary(
                level=level,
                trace_count=trace_count,
                eligible_count=eligible_count,
                exact_match_count=exact_count,
                partial_match_count=partial_count,
                good_enough_count=good_count,
                exact_match_rate=_rate(exact_count, trace_count),
                partial_match_rate=_rate(partial_count, trace_count),
                good_enough_rate=_rate(good_count, trace_count),
                eligible_good_enough_rate=_rate(good_count, eligible_count),
                mean_recall=_mean(match.recall for match in matches),
                mean_jaccard=_mean(match.jaccard for match in matches),
            )
        )
    return tuple(summaries)


def _summarize_exact_failures(
    traces: Sequence[PlannerTraceEvaluation],
    levels: Sequence[int],
    *,
    max_units: int = 8,
    max_examples: int = 5,
) -> tuple[PlannerExactFailureSummary, ...]:
    summaries = []
    for level in levels:
        failures = [
            (trace, match)
            for trace in traces
            for match in trace.matches
            if trace.demo_level == level and match.level == level and not match.exact_match
        ]
        missing = Counter(
            unit for _, match in failures for unit in match.missing_units
        )
        extra = Counter(unit for _, match in failures for unit in match.extra_units)
        examples = tuple(
            PlannerExactFailureExample(
                comp_id=trace.comp_id,
                comp_rank=trace.comp_rank,
                comp_name=trace.comp_name,
                demo_level=trace.demo_level,
                match_level=match.level,
                final_level=trace.final_level,
                board_unit_count=match.board_unit_count,
                target_unit_count=match.target_unit_count,
                matched_comp_id=match.comp_id,
                recall=match.recall,
                jaccard=match.jaccard,
                missing_units=match.missing_units,
                extra_units=match.extra_units,
            )
            for trace, match in failures[:max_examples]
        )
        summaries.append(
            PlannerExactFailureSummary(
                level=level,
                failed_count=len(failures),
                underleveled_count=sum(1 for _, match in failures if not match.eligible),
                underfilled_count=sum(
                    1
                    for _, match in failures
                    if match.eligible and match.board_unit_count < match.target_unit_count
                ),
                unit_mismatch_count=sum(
                    1
                    for _, match in failures
                    if match.eligible and (match.missing_units or match.extra_units)
                ),
                top_missing_units=_top_units(missing, max_units),
                top_extra_units=_top_units(extra, max_units),
                examples=examples,
            )
        )
    return tuple(summaries)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0


def _top_units(counter: Counter[str], limit: int) -> tuple[PlannerUnitFrequency, ...]:
    return tuple(
        PlannerUnitFrequency(unit_key=unit_key, count=count)
        for unit_key, count in counter.most_common(limit)
    )
