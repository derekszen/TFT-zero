"""Batch evaluation for current-patch planner traces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from mini_tft.metatft.catalog import MetaTFTCatalog
from mini_tft.metatft.metrics import TopCompMatch, top_comp_match_report
from mini_tft.metatft.policy import PolicyTurnPlan
from mini_tft.metatft.schema import CurrentBoardState, CurrentBoardUnit


class TurnPlanner(Protocol):
    def plan_turn(
        self,
        state: CurrentBoardState,
        *,
        shops: Sequence[Sequence[str]],
        unit_costs: Mapping[str, int] | None = None,
        rank_by: str = "after_value",
    ) -> PolicyTurnPlan:
        """Return a deterministic turn plan for a current-patch state."""


@dataclass(frozen=True)
class PlannerTraceEvaluation:
    comp_id: str
    comp_rank: int
    comp_name: str
    demo_level: int
    final_level: int
    stopped: bool
    decisions_count: int
    initial_board: tuple[str, ...]
    final_board: tuple[str, ...]
    matches: tuple[TopCompMatch, ...]


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


@dataclass(frozen=True)
class PlannerBatchEvaluation:
    comp_ids: tuple[str, ...]
    demo_levels: tuple[int, ...]
    match_levels: tuple[int, ...]
    top_k: int
    min_recall: float
    traces: tuple[PlannerTraceEvaluation, ...]
    summaries: tuple[PlannerLevelMatchSummary, ...]


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
) -> PlannerBatchEvaluation:
    """Run deterministic planner traces and summarize top-comp match rates."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be in [0, 1]")
    selected_comp_ids = _select_comp_ids(catalog, comp_ids, comp_limit)
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
            state, shops = demo_state_and_shops(
                comp_id=comp.comp_id,
                unit_keys=comp.unit_keys,
                level=demo_level,
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
                    initial_board=state.board_unit_keys,
                    final_board=plan.final_state.board_unit_keys,
                    matches=matches,
                )
            )

    return PlannerBatchEvaluation(
        comp_ids=selected_comp_ids,
        demo_levels=demo_level_values,
        match_levels=match_level_values,
        top_k=top_k,
        min_recall=min_recall,
        traces=tuple(traces),
        summaries=_summarize_matches(traces, match_level_values),
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


def _summarize_matches(
    traces: Sequence[PlannerTraceEvaluation],
    levels: Sequence[int],
) -> tuple[PlannerLevelMatchSummary, ...]:
    summaries = []
    for level in levels:
        matches = [match for trace in traces for match in trace.matches if match.level == level]
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


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0
