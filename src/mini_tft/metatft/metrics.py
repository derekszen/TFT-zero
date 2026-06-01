"""Planner trace metrics for current-patch MetaTFT board matching."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from mini_tft.metatft.catalog import CompStageLine, MetaTFTCatalog, MetaTFTComp


@dataclass(frozen=True)
class TopCompMatch:
    level: int
    top_k: int
    eligible: bool
    board_unit_count: int
    comp_id: str
    comp_rank: int
    comp_name: str
    target_unit_count: int
    overlap_count: int
    precision: float
    recall: float
    jaccard: float
    exact_match: bool
    partial_match: bool
    good_enough: bool
    missing_units: tuple[str, ...]
    extra_units: tuple[str, ...]


def top_comp_match_report(
    catalog: MetaTFTCatalog,
    board_unit_keys: Sequence[str],
    *,
    board_level: int | None = None,
    levels: Sequence[int] = (8, 9),
    top_k: int = 10,
    min_recall: float = 0.75,
) -> tuple[TopCompMatch, ...]:
    """Compare a final planner board against top MetaTFT comps at target levels.

    The metric is intentionally transparent and symbolic. It uses multiset unit
    overlap so duplicate units still count correctly when they appear in a comp.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError("min_recall must be in [0, 1]")
    if not levels:
        return ()

    board = tuple(unit for unit in board_unit_keys if unit)
    top_comps = catalog.comps[:top_k]
    return tuple(
        _best_level_match(
            top_comps,
            board,
            level=level,
            board_level=board_level,
            top_k=top_k,
            min_recall=min_recall,
        )
        for level in levels
    )


def _best_level_match(
    comps: Sequence[MetaTFTComp],
    board: tuple[str, ...],
    *,
    level: int,
    board_level: int | None,
    top_k: int,
    min_recall: float,
) -> TopCompMatch:
    eligible = board_level is None or board_level >= level
    matches = [
        _score_comp_match(
            comp,
            board,
            level=level,
            eligible=eligible,
            top_k=top_k,
            min_recall=min_recall,
        )
        for comp in comps
    ]
    if not matches:
        raise ValueError("catalog has no comps to match against")
    matches.sort(
        key=lambda match: (
            match.jaccard,
            match.recall,
            match.precision,
            -match.comp_rank,
        ),
        reverse=True,
    )
    return matches[0]


def _score_comp_match(
    comp: MetaTFTComp,
    board: tuple[str, ...],
    *,
    level: int,
    eligible: bool,
    top_k: int,
    min_recall: float,
) -> TopCompMatch:
    target = target_comp_units_for_level(comp, level)
    overlap = _multiset_overlap(board, target)
    precision = overlap / len(board) if board else 0.0
    recall = overlap / len(target) if target else 0.0
    union = len(board) + len(target) - overlap
    jaccard = overlap / union if union else 1.0
    board_counts = Counter(board)
    target_counts = Counter(target)
    missing = _expanded_counter(target_counts - board_counts)
    extra = _expanded_counter(board_counts - target_counts)
    exact = eligible and not missing and not extra
    partial = eligible and recall >= min_recall
    good_enough = exact or partial
    return TopCompMatch(
        level=level,
        top_k=top_k,
        eligible=eligible,
        board_unit_count=len(board),
        comp_id=comp.comp_id,
        comp_rank=comp.rank,
        comp_name=comp.name,
        target_unit_count=len(target),
        overlap_count=overlap,
        precision=precision,
        recall=recall,
        jaccard=jaccard,
        exact_match=exact,
        partial_match=partial,
        good_enough=good_enough,
        missing_units=missing,
        extra_units=extra,
    )


def target_comp_units_for_level(comp: MetaTFTComp, level: int) -> tuple[str, ...]:
    """Return the MetaTFT target board units for a comp at a specific level."""

    stage_line = _best_stage_line_for_level(comp.stage_lines, level)
    if stage_line is not None:
        return stage_line.unit_keys[:level]
    return comp.unit_keys[:level]


def _best_stage_line_for_level(
    stage_lines: Sequence[CompStageLine],
    level: int,
) -> CompStageLine | None:
    matches = [line for line in stage_lines if line.level == level and line.unit_keys]
    if not matches:
        return None
    return max(matches, key=lambda line: (line.score, line.count, -(line.avg_placement or 99.0)))


def _multiset_overlap(board: Sequence[str], target: Sequence[str]) -> int:
    board_counts = Counter(board)
    target_counts = Counter(target)
    return sum((board_counts & target_counts).values())


def _expanded_counter(counter: Counter[str]) -> tuple[str, ...]:
    return tuple(
        unit
        for unit, count in sorted(counter.items())
        for _ in range(count)
    )
